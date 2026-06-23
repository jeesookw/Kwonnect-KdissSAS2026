"""
5주차 실험: Optuna 하이퍼파라미터 튜닝
목적: 신규 피처 반영 후 LGB/XGB 파라미터 최적화
구조: Churn / LTV 각각 독립 튜닝 → 기존 파라미터와 CV 비교
채택 기준:
  - Churn: CV AUC  +0.001 이상 개선
  - LTV  : CV RMSE -0.005 이상 개선
주의: n_trials는 시간 여유에 따라 조절 (기본 50회, 빠른 확인은 20회)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 0. 설정 ────────────────────────────────────────────────────────
SEED     = 42
N_SPLITS = 5
N_TRIALS = 50    # 시간 부족 시 20으로 줄여도 됨
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed', 'train_df.csv')

# 신규 피처 반영된 features.py 사용 시 이미 포함되어 있음
# 없으면 직접 생성
NEW_FEATURES_AVAILABLE = True  # features.py 수정 반영 완료 시 True

BASELINE_AUC  = 0.7930   # 신규 피처 추가 후 기준값
BASELINE_RMSE = 1.5089   # 신규 피처 추가 후 기준값

ADOPT_AUC_DELTA  =  0.001
ADOPT_RMSE_DELTA = -0.005

DROP_FEATURES = [
    'last_month_trans_flag', 'has_loan', 'has_cash_service',
    'has_card_loan', 'has_overdue', 'fin_asset_trend_group'
]
CHURN_POS_WEIGHT = 9.1

# ── 1. 데이터 로드 ────────────────────────────────────────────────
train_df = pd.read_csv(DATA_PATH)

# features.py 파이프라인 적용
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from features import build_features

DEPOSIT_CAP_FIXED = 329_791_893  # train 상위 1% 고정값 (run_pipeline.py와 동일)
train_df = build_features(train_df, is_train=True, deposit_cap=DEPOSIT_CAP_FIXED)

# build_features() 이후 DROP_FEATURES 제거 (run_pipeline.py와 동일)
DROP_FEATURES = [
    'last_month_trans_flag', 'has_loan', 'has_cash_service',
    'has_card_loan', 'has_overdue', 'fin_asset_trend_group'
]
train_df = train_df.drop(columns=DROP_FEATURES, errors='ignore')
print(f"features.py 적용 후 shape: {train_df.shape}")


EXCLUDE_COLS = DROP_FEATURES + ['target_ltv', 'target_churn', 'customer_id']
feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]

X       = train_df[feature_cols]
y_churn = train_df['target_churn']
y_ltv   = np.log1p(train_df['target_ltv'])
y_bin   = pd.qcut(y_ltv, q=5, labels=False, duplicates='drop')  # LTV CV용 bin

print(f"피처 수: {len(feature_cols)}")
print(f"신규 피처 포함 여부: "
      f"card_debt_total={'card_debt_total' in feature_cols}, "
      f"credit_x_trend={'credit_x_trend' in feature_cols}")


# ── 2. CV 유틸 ────────────────────────────────────────────────────

def cv_single_lgb_churn(X, y, params):
    """LGB 단독 Churn CV — Optuna objective용"""
    kf  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, val in kf.split(X, y):
        m = lgb.train(
            params,
            lgb.Dataset(X.iloc[tr], y.iloc[tr]),
            num_boost_round=500,
            valid_sets=[lgb.Dataset(X.iloc[val], y.iloc[val])],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        oof[val] = m.predict(X.iloc[val])
    return roc_auc_score(y, oof)


def cv_single_xgb_churn(X, y, params):
    """XGB 단독 Churn CV — Optuna objective용"""
    kf  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, val in kf.split(X, y):
        m = xgb.XGBClassifier(**params, random_state=SEED, n_jobs=-1)
        m.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[val], y.iloc[val])], verbose=False)
        oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    return roc_auc_score(y, oof)


def cv_single_lgb_ltv(X, y, y_bin, params):
    """LGB 단독 LTV CV — Optuna objective용"""
    kf  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, val in kf.split(X, y_bin):
        m = lgb.train(
            params,
            lgb.Dataset(X.iloc[tr], y.iloc[tr]),
            num_boost_round=500,
            valid_sets=[lgb.Dataset(X.iloc[val], y.iloc[val])],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        oof[val] = np.clip(m.predict(X.iloc[val]), 0, None)
    return np.sqrt(mean_squared_error(y, oof))


def cv_single_xgb_ltv(X, y, y_bin, params):
    """XGB 단독 LTV CV — Optuna objective용"""
    kf  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, val in kf.split(X, y_bin):
        m = xgb.XGBRegressor(**params, random_state=SEED, n_jobs=-1)
        m.fit(X.iloc[tr], y.iloc[tr],
              eval_set=[(X.iloc[val], y.iloc[val])], verbose=False)
        oof[val] = np.clip(m.predict(X.iloc[val]), 0, None)
    return np.sqrt(mean_squared_error(y, oof))


# ── 3. Optuna Objectives ──────────────────────────────────────────

def objective_lgb_churn(trial):
    params = {
        'objective':        'binary',
        'metric':           'auc',
        'verbose':          -1,
        'scale_pos_weight': CHURN_POS_WEIGHT,
        'learning_rate':    trial.suggest_float('lgb_c_lr',      0.01, 0.1,  log=True),
        'num_leaves':       trial.suggest_int(  'lgb_c_leaves',  20,   150),
        'min_child_samples':trial.suggest_int(  'lgb_c_mcs',     10,   100),
        'feature_fraction': trial.suggest_float('lgb_c_ff',      0.5,  1.0),
        'bagging_fraction': trial.suggest_float('lgb_c_bf',      0.5,  1.0),
        'bagging_freq':     trial.suggest_int(  'lgb_c_bfreq',   1,    10),
        'lambda_l1':        trial.suggest_float('lgb_c_l1',      1e-4, 10.0, log=True),
        'lambda_l2':        trial.suggest_float('lgb_c_l2',      1e-4, 10.0, log=True),
    }
    return cv_single_lgb_churn(X, y_churn, params)


def objective_xgb_churn(trial):
    params = {
        'objective':          'binary:logistic',
        'eval_metric':        'auc',
        'n_estimators':       500,
        'scale_pos_weight':   CHURN_POS_WEIGHT,
        'early_stopping_rounds': 30,
        'learning_rate':      trial.suggest_float('xgb_c_lr',    0.01, 0.1,  log=True),
        'max_depth':          trial.suggest_int(  'xgb_c_depth',  3,    8),
        'subsample':          trial.suggest_float('xgb_c_ss',     0.5,  1.0),
        'colsample_bytree':   trial.suggest_float('xgb_c_csbt',   0.5,  1.0),
        'min_child_weight':   trial.suggest_int(  'xgb_c_mcw',    1,    20),
        'gamma':              trial.suggest_float('xgb_c_gamma',  0.0,  5.0),
        'reg_alpha':          trial.suggest_float('xgb_c_alpha',  1e-4, 10.0, log=True),
        'reg_lambda':         trial.suggest_float('xgb_c_lambda', 1e-4, 10.0, log=True),
    }
    return cv_single_xgb_churn(X, y_churn, params)


def objective_lgb_ltv(trial):
    params = {
        'objective':        'regression',
        'metric':           'rmse',
        'verbose':          -1,
        'learning_rate':    trial.suggest_float('lgb_l_lr',      0.01, 0.1,  log=True),
        'num_leaves':       trial.suggest_int(  'lgb_l_leaves',  20,   150),
        'min_child_samples':trial.suggest_int(  'lgb_l_mcs',     10,   100),
        'feature_fraction': trial.suggest_float('lgb_l_ff',      0.5,  1.0),
        'bagging_fraction': trial.suggest_float('lgb_l_bf',      0.5,  1.0),
        'bagging_freq':     trial.suggest_int(  'lgb_l_bfreq',   1,    10),
        'lambda_l1':        trial.suggest_float('lgb_l_l1',      1e-4, 10.0, log=True),
        'lambda_l2':        trial.suggest_float('lgb_l_l2',      1e-4, 10.0, log=True),
    }
    return cv_single_lgb_ltv(X, y_ltv, y_bin, params)


def objective_xgb_ltv(trial):
    params = {
        'objective':          'reg:squarederror',
        'eval_metric':        'rmse',
        'n_estimators':       500,
        'early_stopping_rounds': 30,
        'learning_rate':      trial.suggest_float('xgb_l_lr',    0.01, 0.1,  log=True),
        'max_depth':          trial.suggest_int(  'xgb_l_depth',  3,    8),
        'subsample':          trial.suggest_float('xgb_l_ss',     0.5,  1.0),
        'colsample_bytree':   trial.suggest_float('xgb_l_csbt',   0.5,  1.0),
        'min_child_weight':   trial.suggest_int(  'xgb_l_mcw',    1,    20),
        'gamma':              trial.suggest_float('xgb_l_gamma',  0.0,  5.0),
        'reg_alpha':          trial.suggest_float('xgb_l_alpha',  1e-4, 10.0, log=True),
        'reg_lambda':         trial.suggest_float('xgb_l_lambda', 1e-4, 10.0, log=True),
    }
    return cv_single_xgb_ltv(X, y_ltv, y_bin, params)


# ── 4. 튜닝 실행 ──────────────────────────────────────────────────

def run_study(objective, direction, label, n_trials=N_TRIALS):
    print(f"\n  [{label}] {n_trials}회 탐색 시작...")
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_trial
    print(f"  [{label}] 최적값: {best.value:.4f}")
    return study


print("\n" + "="*60)
print("STEP 1. Churn 튜닝 (LGB)")
print("="*60)
study_lgb_churn = run_study(objective_lgb_churn, 'maximize', 'LGB-CHURN')

print("\n" + "="*60)
print("STEP 2. Churn 튜닝 (XGB)")
print("="*60)
study_xgb_churn = run_study(objective_xgb_churn, 'maximize', 'XGB-CHURN')

print("\n" + "="*60)
print("STEP 3. LTV 튜닝 (LGB)")
print("="*60)
study_lgb_ltv = run_study(objective_lgb_ltv, 'minimize', 'LGB-LTV')

print("\n" + "="*60)
print("STEP 4. LTV 튜닝 (XGB)")
print("="*60)
study_xgb_ltv = run_study(objective_xgb_ltv, 'minimize', 'XGB-LTV')


# ── 5. 최적 파라미터로 앙상블 CV ──────────────────────────────────

def ensemble_cv_churn(X, y, lgb_params, xgb_params):
    kf      = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))

    for tr, val in kf.split(X, y):
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

    print(f"  [CHURN] 최적 가중치: LGB {best_w:.3f} / XGB {1-best_w:.3f} | "
          f"Ensemble AUC={best_auc:.4f}")
    return best_auc, best_w


def ensemble_cv_ltv(X, y, y_bin, lgb_params, xgb_params):
    kf      = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))

    for tr, val in kf.split(X, y_bin):
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
        rmse = np.sqrt(mean_squared_error(y, np.clip(w*oof_lgb+(1-w)*oof_xgb, 0, None)))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w

    print(f"  [LTV] 최적 가중치: LGB {best_w:.3f} / XGB {1-best_w:.3f} | "
          f"Ensemble RMSE(log)={best_rmse:.4f}")
    return best_rmse, best_w


# 최적 파라미터 추출
best_lgb_churn = {**study_lgb_churn.best_params,
                  'objective': 'binary', 'metric': 'auc',
                  'verbose': -1, 'scale_pos_weight': CHURN_POS_WEIGHT}

best_xgb_churn = {**study_xgb_churn.best_params,
                  'objective': 'binary:logistic', 'eval_metric': 'auc',
                  'n_estimators': 1000, 'early_stopping_rounds': 50,
                  'scale_pos_weight': CHURN_POS_WEIGHT}

best_lgb_ltv   = {**study_lgb_ltv.best_params,
                  'objective': 'regression', 'metric': 'rmse', 'verbose': -1}

best_xgb_ltv   = {**study_xgb_ltv.best_params,
                  'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                  'n_estimators': 1000, 'early_stopping_rounds': 50}

print("\n" + "="*60)
print("STEP 5. 최적 파라미터 앙상블 CV")
print("="*60)
tuned_auc,  churn_w = ensemble_cv_churn(X, y_churn, best_lgb_churn, best_xgb_churn)
tuned_rmse, ltv_w   = ensemble_cv_ltv(X, y_ltv, y_bin, best_lgb_ltv, best_xgb_ltv)


# ── 6. 최종 결과 및 채택 판단 ─────────────────────────────────────
print("\n" + "="*60)
print("최종 결과 요약")
print("="*60)

auc_delta  = tuned_auc  - BASELINE_AUC
rmse_delta = tuned_rmse - BASELINE_RMSE

print(f"\n{'':20s} {'베이스라인':>12} {'튜닝 후':>12} {'변화량':>12} {'판정':>10}")
print("-" * 68)

auc_judge  = '✅ 채택' if auc_delta  >= ADOPT_AUC_DELTA  else \
             ('🔶 미세' if auc_delta  >  0               else '❌ 악화')
rmse_judge = '✅ 채택' if rmse_delta <= ADOPT_RMSE_DELTA else \
             ('🔶 미세' if rmse_delta <  0               else '❌ 악화')

print(f"{'Churn AUC':20s} {BASELINE_AUC:>12.4f} {tuned_auc:>12.4f} "
      f"{auc_delta:>+12.4f} {auc_judge:>10}")
print(f"{'LTV RMSE(log)':20s} {BASELINE_RMSE:>12.4f} {tuned_rmse:>12.4f} "
      f"{rmse_delta:>+12.4f} {rmse_judge:>10}")

churn_ok  = auc_delta  >= ADOPT_AUC_DELTA
ltv_ok    = rmse_delta <= ADOPT_RMSE_DELTA
any_harm  = auc_delta < -0.001 or rmse_delta > 0.005

print("\n[최종 결론]")
if (churn_ok or ltv_ok) and not any_harm:
    print("✅ 채택: 아래 파라미터를 train.py에 반영")
elif not any_harm and (auc_delta > 0 or rmse_delta < 0):
    print("🔶 기준 미달이나 악화 없음")
else:
    print("❌ 기각. 현행 파라미터 유지")

# 최적 파라미터 출력 (train.py 반영용)
print("\n" + "="*60)
print("train.py 반영용 최적 파라미터")
print("="*60)

print(f"\n# Churn 앙상블 최적 가중치: LGB {churn_w:.3f} / XGB {1-churn_w:.3f}")
print(f"# LTV   앙상블 최적 가중치: LGB {ltv_w:.3f} / XGB {1-ltv_w:.3f}")

print("\n# [LGB Churn 파라미터]")
for k, v in best_lgb_churn.items():
    print(f"  '{k}': {repr(v)},")

print("\n# [XGB Churn 파라미터]")
for k, v in best_xgb_churn.items():
    print(f"  '{k}': {repr(v)},")

print("\n# [LGB LTV 파라미터]")
for k, v in best_lgb_ltv.items():
    print(f"  '{k}': {repr(v)},")

print("\n# [XGB LTV 파라미터]")
for k, v in best_xgb_ltv.items():
    print(f"  '{k}': {repr(v)},")