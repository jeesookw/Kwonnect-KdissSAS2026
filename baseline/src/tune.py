"""
Optuna로 LightGBM / XGBoost 하이퍼파라미터 탐색.
학습 완료 후 최적 파라미터를 train.py에 붙여넣기 용도로 출력.

실행: python src/tune.py

소요 시간: 각 모델 약 20~40분 (n_trials=50 기준)
빠르게 테스트하려면 N_TRIALS = 20 으로 줄이기
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

from features import build_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_DIR  = "data/processed"
SEED      = 42
N_SPLITS  = 4       # 튜닝 중엔 3-Fold로 빠르게
N_TRIALS  = 100      # 시간 여유 있으면 100
CHURN_POS_WEIGHT = (1 - 0.099) / 0.099

DROP_COLS = ["customer_id", "target_churn", "target_ltv"]

def _get_features(df):
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])

def _cv_auc(params, X, y, model_type="lgb"):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if model_type == "lgb":
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
        else:
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        scores.append(roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]))

    return float(np.mean(scores))


# ─────────────────────────────────────────────
# LightGBM Objective
# ─────────────────────────────────────────────
def lgb_objective(trial, X, y):
    params = dict(
        objective="binary",
        metric="auc",
        scale_pos_weight=CHURN_POS_WEIGHT,
        n_estimators=1000,
        random_state=SEED,
        verbose=-1,
        # 탐색 범위
        learning_rate     = trial.suggest_float("learning_rate",      0.01, 0.1,   log=True),
        num_leaves        = trial.suggest_int(  "num_leaves",         31,   255),
        min_child_samples = trial.suggest_int(  "min_child_samples",  10,   100),
        feature_fraction  = trial.suggest_float("feature_fraction",   0.5,  1.0),
        bagging_fraction  = trial.suggest_float("bagging_fraction",   0.5,  1.0),
        bagging_freq      = trial.suggest_int(  "bagging_freq",       1,    10),
        reg_alpha         = trial.suggest_float("reg_alpha",          1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",         1e-3, 10.0, log=True),
    )
    return _cv_auc(params, X, y, model_type="lgb")


# ─────────────────────────────────────────────
# XGBoost Objective
# ─────────────────────────────────────────────
def xgb_objective(trial, X, y):
    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=CHURN_POS_WEIGHT,
        n_estimators=1000,
        random_state=SEED,
        verbosity=0,
        tree_method="hist",
        early_stopping_rounds=50,
        # 탐색 범위
        learning_rate     = trial.suggest_float("learning_rate",    0.01, 0.1,   log=True),
        max_depth         = trial.suggest_int(  "max_depth",        3,    10),
        min_child_weight  = trial.suggest_int(  "min_child_weight", 1,    20),
        subsample         = trial.suggest_float("subsample",        0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5,  1.0),
        reg_alpha         = trial.suggest_float("reg_alpha",        1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",       1e-3, 10.0, log=True),
        gamma             = trial.suggest_float("gamma",            0.0,  5.0),
    )
    return _cv_auc(params, X, y, model_type="xgb")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("데이터 로드 및 피처 엔지니어링")
    print("=" * 60)
    train_raw = pd.read_csv(f"{DATA_DIR}/train_df.csv")
    DEPOSIT_CAP = train_raw["total_deposit_balance"].quantile(0.99)
    train_df = build_features(train_raw, is_train=True, deposit_cap=DEPOSIT_CAP)

    X = _get_features(train_df)
    y = train_df["target_churn"]

    # ── LightGBM 튜닝 ──
    print(f"\n{'='*60}")
    print(f"LightGBM 튜닝 ({N_TRIALS} trials × {N_SPLITS}-Fold)")
    print("=" * 60)
    lgb_study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
    )
    lgb_study.optimize(
        lambda trial: lgb_objective(trial, X, y),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    lgb_best = lgb_study.best_trial
    print(f"\n✅ LGB 최고 AUC : {lgb_best.value:.4f}")
    print("─ 최적 파라미터 (train.py 의 LGB_CHURN_PARAMS에 붙여넣기) ─")
    for k, v in lgb_best.params.items():
        if isinstance(v, float):
            print(f"    {k:<25} = {v:.6f},")
        else:
            print(f"    {k:<25} = {v},")

    # ── XGBoost 튜닝 ──
    print(f"\n{'='*60}")
    print(f"XGBoost 튜닝 ({N_TRIALS} trials × {N_SPLITS}-Fold)")
    print("=" * 60)
    xgb_study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
    )
    xgb_study.optimize(
        lambda trial: xgb_objective(trial, X, y),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    xgb_best = xgb_study.best_trial
    print(f"\n✅ XGB 최고 AUC : {xgb_best.value:.4f}")
    print("─ 최적 파라미터 (train.py 의 XGB_CHURN_PARAMS에 붙여넣기) ─")
    for k, v in xgb_best.params.items():
        if isinstance(v, float):
            print(f"    {k:<25} = {v:.6f},")
        else:
            print(f"    {k:<25} = {v},")

    print("\n" + "=" * 60)
    print("완료! 위 파라미터를 train.py에 붙여넣고 run_pipeline.py 재실행하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()