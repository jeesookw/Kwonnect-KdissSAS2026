"""
5주차 실험: 신규 피처 추가 CV 검증
목적: card_debt_total, credit_x_trend 추가 후 LTV/Churn CV 성능 비교
채택 기준:
  - LTV  : CV RMSE(log) -0.005 이상 개선
  - Churn: CV AUC       +0.001 이상 개선
  - 둘 중 하나라도 채택 기준 충족 + 나머지가 악화되지 않으면 채택
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

# ── 0. 설정 ────────────────────────────────────────────────────────
SEED             = 42
N_SPLITS         = 5
DATA_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed', 'train_df.csv')

BASELINE_AUC     = 0.7971   # 문서 기재 베이스라인
BASELINE_RMSE    = 1.5102

ADOPT_AUC_DELTA  =  0.001   # Churn 채택 기준
ADOPT_RMSE_DELTA = -0.005   # LTV 채택 기준 (음수 = 개선)

DROP_FEATURES = [
    'last_month_trans_flag', 'has_loan', 'has_cash_service',
    'has_card_loan', 'has_overdue', 'fin_asset_trend_group'
]

# ── 1. 데이터 로드 및 신규 피처 생성 ─────────────────────────────
train_df = pd.read_csv(DATA_PATH)


def add_finance_features(df):
    """EDA 검증 완료 신규 피처 추가"""
    if {'card_cash_service_amt', 'card_loan_amt'}.issubset(df.columns):
        df['card_debt_total'] = (
            df['card_cash_service_amt'].fillna(0) +
            df['card_loan_amt'].fillna(0)
        )
    if {'credit_score', 'fin_asset_trend_score'}.issubset(df.columns):
        df['credit_x_trend'] = (
            df['credit_score'].fillna(df['credit_score'].median()) *
            df['fin_asset_trend_score'].fillna(0)
        )
    return df


train_new = add_finance_features(train_df.copy())

NEW_FEATURES = ['card_debt_total', 'credit_x_trend']
EXCLUDE_COLS = DROP_FEATURES + ['target_ltv', 'target_churn', 'customer_id']

feat_base = [c for c in train_df.columns  if c not in EXCLUDE_COLS]
feat_new  = [c for c in train_new.columns if c not in EXCLUDE_COLS]

print(f"베이스라인 피처 수: {len(feat_base)}")
print(f"신규 피처 추가 후:  {len(feat_new)} (+{len(feat_new)-len(feat_base)})")
print(f"추가된 피처: {[f for f in feat_new if f not in feat_base]}")


# ── 2. 공통 CV 유틸 ───────────────────────────────────────────────

def cv_churn(X, y, params_lgb, params_xgb, churn_weight, label=''):
    """LGB+XGB 앙상블 Churn CV. OOF 기반 최적 가중치 적용."""
    kf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))

    for fold, (tr, val) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]

        # LGB
        dtrain_lgb = lgb.Dataset(X_tr, y_tr)
        m_lgb = lgb.train(
            params_lgb, dtrain_lgb,
            num_boost_round=1000,
            valid_sets=[lgb.Dataset(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        oof_lgb[val] = m_lgb.predict(X_val)

        # XGB
        m_xgb = xgb.XGBClassifier(**params_xgb, random_state=SEED, n_jobs=-1)
        m_xgb.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        oof_xgb[val] = m_xgb.predict_proba(X_val)[:, 1]

    # OOF 기반 최적 가중치 탐색
    best_w, best_auc = 0.5, 0
    for w in np.arange(0, 1.01, 0.05):
        blend = w * oof_lgb + (1 - w) * oof_xgb
        auc   = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc, best_w = auc, w

    ensemble = best_w * oof_lgb + (1 - best_w) * oof_xgb
    final_auc = roc_auc_score(y, ensemble)
    print(f"  [{label}] 최적가중치 LGB {best_w:.3f}/XGB {1-best_w:.3f} | AUC={final_auc:.4f}")
    return final_auc


def cv_ltv(X, y_log, params_lgb, params_xgb, label=''):
    """LGB+XGB 앙상블 LTV CV."""
    y_bin = pd.qcut(y_log, q=5, labels=False, duplicates='drop')
    kf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_lgb = np.zeros(len(y_log))
    oof_xgb = np.zeros(len(y_log))

    for fold, (tr, val) in enumerate(kf.split(X, y_bin)):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y_log.iloc[tr], y_log.iloc[val]

        # LGB
        dtrain_lgb = lgb.Dataset(X_tr, y_tr)
        m_lgb = lgb.train(
            params_lgb, dtrain_lgb,
            num_boost_round=1000,
            valid_sets=[lgb.Dataset(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        oof_lgb[val] = m_lgb.predict(X_val)

        # XGB
        m_xgb = xgb.XGBRegressor(**params_xgb, random_state=SEED, n_jobs=-1)
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val] = np.clip(m_xgb.predict(X_val), 0, None)

    # OOF 기반 최적 가중치 탐색
    best_w, best_rmse = 0.5, np.inf
    for w in np.arange(0, 1.01, 0.05):
        blend = np.clip(w * oof_lgb + (1 - w) * oof_xgb, 0, None)
        rmse  = np.sqrt(mean_squared_error(y_log, blend))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w

    ensemble  = np.clip(best_w * oof_lgb + (1 - best_w) * oof_xgb, 0, None)
    final_rmse = np.sqrt(mean_squared_error(y_log, ensemble))
    print(f"  [{label}] 최적가중치 LGB {best_w:.3f}/XGB {1-best_w:.3f} | RMSE(log)={final_rmse:.4f}")
    return final_rmse


# ── 3. 모델 파라미터 (기존 train.py와 동일하게 맞춤) ─────────────
CHURN_POS_WEIGHT = 9.1

lgb_churn_params = {
    'objective':        'binary',
    'metric':           'auc',
    'learning_rate':    0.05,
    'num_leaves':       31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'is_unbalance':     False,
    'scale_pos_weight': CHURN_POS_WEIGHT,
    'verbose':          -1,
}
xgb_churn_params = {
    'objective':        'binary:logistic',
    'eval_metric':      'auc',
    'n_estimators':     1000,
    'learning_rate':    0.05,
    'max_depth':        6,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': CHURN_POS_WEIGHT,
    'early_stopping_rounds': 50,
}
lgb_ltv_params = {
    'objective':        'regression',
    'metric':           'rmse',
    'learning_rate':    0.05,
    'num_leaves':       31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'verbose':          -1,
}
xgb_ltv_params = {
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'n_estimators':     1000,
    'learning_rate':    0.05,
    'max_depth':        6,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'early_stopping_rounds': 50,
}

# ── 4. 실험 실행 ──────────────────────────────────────────────────
y_churn = train_df['target_churn']
y_ltv   = np.log1p(train_df['target_ltv'])

X_base = train_df[feat_base]
X_new  = train_new[feat_new]

print("\n" + "="*60)
print("STEP 1. 베이스라인 CV (신규 피처 미포함)")
print("="*60)
print("[Churn]")
base_auc  = cv_churn(X_base, y_churn, lgb_churn_params, xgb_churn_params,
                     CHURN_POS_WEIGHT, label='BASE-CHURN')
print("[LTV]")
base_rmse = cv_ltv(X_base, y_ltv, lgb_ltv_params, xgb_ltv_params,
                   label='BASE-LTV')

print("\n" + "="*60)
print("STEP 2. 신규 피처 추가 CV")
print("="*60)
print("[Churn]")
new_auc  = cv_churn(X_new, y_churn, lgb_churn_params, xgb_churn_params,
                    CHURN_POS_WEIGHT, label='NEW-CHURN')
print("[LTV]")
new_rmse = cv_ltv(X_new, y_ltv, lgb_ltv_params, xgb_ltv_params,
                  label='NEW-LTV')

# ── 5. 채택 판단 ──────────────────────────────────────────────────
print("\n" + "="*60)
print("최종 채택 판단")
print("="*60)

auc_delta  = new_auc  - base_auc
rmse_delta = new_rmse - base_rmse

print(f"\n{'':20s} {'베이스라인':>12} {'신규':>12} {'변화량':>12} {'판정':>10}")
print("-" * 68)
auc_judge  = '✅ 개선' if auc_delta  >= ADOPT_AUC_DELTA   else ('🔶 미세' if auc_delta > 0 else '❌ 악화')
rmse_judge = '✅ 개선' if rmse_delta <= ADOPT_RMSE_DELTA  else ('🔶 미세' if rmse_delta < 0 else '❌ 악화')
print(f"{'Churn AUC':20s} {base_auc:>12.4f} {new_auc:>12.4f} {auc_delta:>+12.4f} {auc_judge:>10}")
print(f"{'LTV RMSE(log)':20s} {base_rmse:>12.4f} {new_rmse:>12.4f} {rmse_delta:>+12.4f} {rmse_judge:>10}")

# 최종 판단 로직
churn_ok = auc_delta  >= ADOPT_AUC_DELTA
ltv_ok   = rmse_delta <= ADOPT_RMSE_DELTA
churn_harm = auc_delta  < -0.001
ltv_harm   = rmse_delta >  0.005

print("\n[최종 결론]")
if (churn_ok or ltv_ok) and not churn_harm and not ltv_harm:
    print("✅ 채택: 신규 피처 features.py에 반영")
    print("   → feature_additions.py의 add_finance_features()를 features.py 파이프라인 마지막에 호출하면 됨")
elif churn_harm or ltv_harm:
    print("❌ 기각: 한쪽 태스크가 악화됨. 현행 유지")
    if churn_harm:
        print(f"   Churn AUC 악화: {auc_delta:+.4f}")
    if ltv_harm:
        print(f"   LTV RMSE 악화: {rmse_delta:+.4f}")
else:
    print(f"   Churn 기준: +{ADOPT_AUC_DELTA} 필요 / 실제: {auc_delta:+.4f}")
    print(f"   LTV 기준:   {ADOPT_RMSE_DELTA} 필요 / 실제: {rmse_delta:+.4f}")
    print("🔶 기준 미달. 현행 유지 권장")