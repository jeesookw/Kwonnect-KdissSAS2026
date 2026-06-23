"""
5주차 최종 실험: credit_score_group 제거
목적: importance 최하위(2) 노이즈 피처 제거 후 CV 성능 비교
채택 기준: Churn AUC +0.001 이상 개선 AND LTV 악화 없음
사용 데이터: data/processed/train_df.csv (병합 완료 원본)
"""

import os, sys
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── 0. 설정 ────────────────────────────────────────────────────────
SEED     = 42
N_SPLITS = 5
DATA_PATH = 'data/processed/train_df.csv'

BASELINE_AUC  = 0.7930   # 신규 피처 추가 후 기준값
BASELINE_RMSE = 1.5089

CHURN_POS_WEIGHT = 9.1
DEPOSIT_CAP_FIXED = 329_791_893

DROP_FEATURES = [
    'last_month_trans_flag', 'has_loan', 'has_cash_service',
    'has_card_loan', 'has_overdue', 'fin_asset_trend_group'
]

# ── 1. 데이터 로드 및 features.py 적용 ───────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from features import build_features

train_raw = pd.read_csv(DATA_PATH)
train_df  = build_features(train_raw, is_train=True, deposit_cap=DEPOSIT_CAP_FIXED)
train_df  = train_df.drop(columns=DROP_FEATURES, errors='ignore')

EXCLUDE_COLS  = ['target_ltv', 'target_churn', 'customer_id']
feat_base     = [c for c in train_df.columns if c not in EXCLUDE_COLS]
feat_drop_csg = [c for c in feat_base if c != 'credit_score_group']

print(f"베이스라인 피처 수: {len(feat_base)}")
print(f"credit_score_group 제거 후: {len(feat_drop_csg)}")
print(f"제거 피처 확인: {'credit_score_group' in feat_base}, "
      f"제거 후 없음: {'credit_score_group' not in feat_drop_csg}")

X_base = train_df[feat_base]
X_drop = train_df[feat_drop_csg]
y_churn = train_df['target_churn']
y_ltv   = np.log1p(train_df['target_ltv'])
y_bin   = pd.qcut(y_ltv, q=5, labels=False, duplicates='drop')


# ── 2. 기존 파라미터 (현행 베이스라인) ───────────────────────────
# Optuna 튜닝은 기각됐으므로 기존 파라미터 그대로 사용
lgb_churn_params = {
    'objective': 'binary', 'metric': 'auc', 'verbose': -1,
    'scale_pos_weight': CHURN_POS_WEIGHT,
    'learning_rate': 0.05, 'num_leaves': 31,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
}
xgb_churn_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'scale_pos_weight': CHURN_POS_WEIGHT, 'early_stopping_rounds': 50,
}
lgb_ltv_params = {
    'objective': 'regression', 'metric': 'rmse', 'verbose': -1,
    'learning_rate': 0.05, 'num_leaves': 31,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
}
xgb_ltv_params = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'early_stopping_rounds': 50,
}


# ── 3. CV 함수 ────────────────────────────────────────────────────
def ensemble_cv_churn(X, y, lgb_params, xgb_params, label=''):
    kf      = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))

    for fold, (tr, val) in enumerate(kf.split(X, y)):
        m_lgb = lgb.train(
            lgb_params, lgb.Dataset(X.iloc[tr], y.iloc[tr]),
            num_boost_round=1000,
            valid_sets=[lgb.Dataset(X.iloc[val], y.iloc[val])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        oof_lgb[val] = m_lgb.predict(X.iloc[val])

        m_xgb = xgb.XGBClassifier(**xgb_params, random_state=SEED, n_jobs=-1)
        m_xgb.fit(X.iloc[tr], y.iloc[tr],
                  eval_set=[(X.iloc[val], y.iloc[val])], verbose=False)
        oof_xgb[val] = m_xgb.predict_proba(X.iloc[val])[:, 1]

    best_w, best_auc = 0.5, 0
    for w in np.arange(0, 1.01, 0.05):
        auc = roc_auc_score(y, w * oof_lgb + (1-w) * oof_xgb)
        if auc > best_auc:
            best_auc, best_w = auc, w

    print(f"  [{label}] LGB {best_w:.3f}/XGB {1-best_w:.3f} | AUC={best_auc:.4f}")
    return best_auc


def ensemble_cv_ltv(X, y, y_bin, lgb_params, xgb_params, label=''):
    kf      = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))

    for fold, (tr, val) in enumerate(kf.split(X, y_bin)):
        m_lgb = lgb.train(
            lgb_params, lgb.Dataset(X.iloc[tr], y.iloc[tr]),
            num_boost_round=1000,
            valid_sets=[lgb.Dataset(X.iloc[val], y.iloc[val])],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        oof_lgb[val] = np.clip(m_lgb.predict(X.iloc[val]), 0, None)

        m_xgb = xgb.XGBRegressor(**xgb_params, random_state=SEED, n_jobs=-1)
        m_xgb.fit(X.iloc[tr], y.iloc[tr],
                  eval_set=[(X.iloc[val], y.iloc[val])], verbose=False)
        oof_xgb[val] = np.clip(m_xgb.predict(X.iloc[val]), 0, None)

    best_w, best_rmse = 0.5, np.inf
    for w in np.arange(0, 1.01, 0.05):
        rmse = np.sqrt(mean_squared_error(
            y, np.clip(w * oof_lgb + (1-w) * oof_xgb, 0, None)))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w

    print(f"  [{label}] LGB {best_w:.3f}/XGB {1-best_w:.3f} | RMSE(log)={best_rmse:.4f}")
    return best_rmse


# ── 4. 실험 실행 ──────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1. 베이스라인 CV (credit_score_group 포함)")
print("="*60)
print("[Churn]")
base_auc  = ensemble_cv_churn(X_base, y_churn,
                               lgb_churn_params, xgb_churn_params, 'BASE-CHURN')
print("[LTV]")
base_rmse = ensemble_cv_ltv(X_base, y_ltv, y_bin,
                             lgb_ltv_params, xgb_ltv_params, 'BASE-LTV')

print("\n" + "="*60)
print("STEP 2. 실험 CV (credit_score_group 제거)")
print("="*60)
print("[Churn]")
new_auc  = ensemble_cv_churn(X_drop, y_churn,
                              lgb_churn_params, xgb_churn_params, 'DROP-CHURN')
print("[LTV]")
new_rmse = ensemble_cv_ltv(X_drop, y_ltv, y_bin,
                            lgb_ltv_params, xgb_ltv_params, 'DROP-LTV')


# ── 5. 채택 판단 ──────────────────────────────────────────────────
print("\n" + "="*60)
print("최종 결과")
print("="*60)

auc_delta  = new_auc  - base_auc
rmse_delta = new_rmse - base_rmse

print(f"\n{'':20s} {'베이스라인':>12} {'제거 후':>12} {'변화량':>12} {'판정':>10}")
print("-" * 68)

auc_judge  = '✅ 개선' if auc_delta  >  0.001 else \
             ('🔶 미세' if auc_delta  >  0     else '❌ 악화')
rmse_judge = '✅ 개선' if rmse_delta < -0.005 else \
             ('🔶 미세' if rmse_delta <  0     else '❌ 악화')

print(f"{'Churn AUC':20s} {base_auc:>12.4f} {new_auc:>12.4f} "
      f"{auc_delta:>+12.4f} {auc_judge:>10}")
print(f"{'LTV RMSE(log)':20s} {base_rmse:>12.4f} {new_rmse:>12.4f} "
      f"{rmse_delta:>+12.4f} {rmse_judge:>10}")

print("\n[최종 결론]")
if auc_delta > 0.001 and rmse_delta <= 0.005:
    print("✅ 채택: run_pipeline.py의 DROP_FEATURES에 'credit_score_group' 추가")
elif auc_delta > 0 and rmse_delta <= 0:
    print("🔶 둘 다 미세 개선: 악화 없음 확인 → 채택 권장")
    print("   run_pipeline.py의 DROP_FEATURES에 'credit_score_group' 추가")
elif rmse_delta > 0.005 or auc_delta < -0.001:
    print("❌ 기각: 한쪽 태스크가 악화됨. 현행 유지.")
else:
    print("🔶 기준 미달. 현행 유지 권장")