"""
LightGBM + XGBoost 앙상블 (Soft Averaging)
- target_churn : 이진 분류 → roc_auc
- target_ltv   : 회귀 (log1p 변환) → RMSE

run_pipeline.py 호환 함수명:
  train_churn_model / predict_churn
  train_ltv_model   / predict_ltv
  make_submission
"""

import os
import random, numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb

# ─────────────────────────────────────────────
# 피처 / 타겟 설정
# ─────────────────────────────────────────────
DROP_COLS = ["customer_id", "target_churn", "target_ltv"]

def _get_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ─────────────────────────────────────────────
# 공통 설정
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
CHURN_POS_WEIGHT = (1 - 0.099) / 0.099   # ≈ 9.1

# ─────────────────────────────────────────────
# 파라미터
# ─────────────────────────────────────────────
LGB_CHURN_PARAMS = dict(
    objective="binary", metric="auc",
    scale_pos_weight=CHURN_POS_WEIGHT,
    learning_rate             = 0.036376,
    num_leaves                = 31,
    min_child_samples         = 70,
    feature_fraction          = 0.708960,
    bagging_fraction          = 0.933145,
    bagging_freq              = 7,
    reg_alpha                 = 0.045768,
    reg_lambda                = 2.967245,
    n_estimators=1000, random_state=SEED, verbose=-1,
)

LGB_LTV_PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.05, num_leaves=63,
    min_child_samples=30, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=1.0,
    n_estimators=1000, random_state=SEED, verbose=-1,
)

XGB_CHURN_PARAMS = dict(
    objective="binary:logistic", eval_metric="auc",
    scale_pos_weight=CHURN_POS_WEIGHT,
    learning_rate             = 0.028415,
    max_depth                 = 5,
    min_child_weight          = 13,
    subsample                 = 0.809486,
    colsample_bytree          = 0.870844,
    reg_alpha                 = 0.053663,
    reg_lambda                = 0.002152,
    gamma                     = 3.120421,
    n_estimators=1000, random_state=SEED,
    verbosity=0, tree_method="hist",
    early_stopping_rounds=50,   # fit()이 아닌 생성자에 지정 (XGBoost ≥ 2.0)
)

XGB_LTV_PARAMS = dict(
    objective="reg:squarederror", eval_metric="rmse",
    learning_rate=0.05, max_depth=6, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_estimators=1000, random_state=SEED,
    verbosity=0, tree_method="hist",
    early_stopping_rounds=50,   # fit()이 아닌 생성자에 지정 (XGBoost ≥ 2.0)
)

LGB_WEIGHT = 0.5   # train_churn_model 내에서 OOF 기반으로 자동 재계산됨
XGB_WEIGHT = 0.5


# ─────────────────────────────────────────────
# 유틸: OOF 기반 최적 가중치 계산
# ─────────────────────────────────────────────
def _optimize_blend_weights(oof_a: np.ndarray, oof_b: np.ndarray,
                             y_true: np.ndarray, metric: str = "auc") -> float:
    """
    oof_a * w + oof_b * (1-w) 에서 metric을 최대화하는 w 반환.
    metric: 'auc' (분류) 또는 'rmse' (회귀, 최소화)
    """
    if metric == "auc":
        def loss(w): return -roc_auc_score(y_true, w * oof_a + (1 - w) * oof_b)
    else:
        def loss(w): return mean_squared_error(y_true, w * oof_a + (1 - w) * oof_b)
    result = minimize_scalar(loss, bounds=(0, 1), method="bounded")
    return float(result.x)


# ─────────────────────────────────────────────
# 유틸: Isotonic Regression으로 확률 보정
# ─────────────────────────────────────────────
def _calibrate_proba(oof_pred: np.ndarray, y_true: np.ndarray,
                     test_pred: np.ndarray) -> tuple:
    """
    OOF 예측으로 보정 모델 학습 → test 예측에 적용.
    scale_pos_weight로 부풀려진 확률을 실제 비율에 맞게 조정.
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof_pred, y_true)
    oof_cal  = ir.predict(oof_pred)
    test_cal = ir.predict(test_pred)
    return oof_cal, test_cal


# ─────────────────────────────────────────────
# train_churn_model
# ─────────────────────────────────────────────
def train_churn_model(train_df: pd.DataFrame, n_splits: int = 5):
    """
    Returns
    -------
    models   : dict  {"lgb": [모델×n_splits], "xgb": [모델×n_splits]}
    oof_pred : np.ndarray  앙상블 OOF 예측 확률
    cv_scores: dict  {"lgb": [...], "xgb": [...]}
    """
    X = _get_features(train_df)
    y = train_df["target_churn"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    lgb_models, xgb_models = [], []
    lgb_scores, xgb_scores = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # LightGBM
        lgb_m = lgb.LGBMClassifier(**LGB_CHURN_PARAMS)

        # LightGBM
        lgb_m = lgb.LGBMClassifier(**LGB_CHURN_PARAMS)
        lgb_m.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
        oof_lgb[va_idx] = lgb_m.predict_proba(X_va)[:, 1]
        lgb_models.append(lgb_m)
        lgb_scores.append(roc_auc_score(y_va, oof_lgb[va_idx]))

        # XGBoost
        xgb_m = xgb.XGBClassifier(**XGB_CHURN_PARAMS)
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va_idx] = xgb_m.predict_proba(X_va)[:, 1]
        xgb_models.append(xgb_m)
        xgb_scores.append(roc_auc_score(y_va, oof_xgb[va_idx]))

        print(f"  Fold {fold} | LGB AUC: {lgb_scores[-1]:.4f} | XGB AUC: {xgb_scores[-1]:.4f}")

    # ① OOF 기반 최적 가중치 계산
    w_lgb = _optimize_blend_weights(oof_lgb, oof_xgb, y.values, metric="auc")
    w_xgb = 1.0 - w_lgb

    oof_pred = w_lgb * oof_lgb + w_xgb * oof_xgb

    print(f"\n[CHURN] LGB  CV AUC : {np.mean(lgb_scores):.4f} ± {np.std(lgb_scores):.4f}")
    print(f"[CHURN] XGB  CV AUC : {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}")
    print(f"[CHURN] 최적 가중치  : LGB {w_lgb:.3f} / XGB {w_xgb:.3f}")
    print(f"[CHURN] Ensemble AUC: {roc_auc_score(y, oof_pred):.4f}")
    print(f"[CHURN] OOF 보정 전 mean: {oof_pred.mean():.4f}  (실제 이탈률: {y.mean():.4f})")

    # ② Isotonic Regression으로 확률 보정
    # ② Isotonic Regression으로 확률 보정 + calibrator 저장
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof_pred, y.values)                        # ← 여기서 한 번만 fit
    oof_cal = ir.predict(oof_pred)
    print(f"[CHURN] OOF 보정 후 mean: {oof_cal.mean():.4f}")

    models = {"lgb": lgb_models, "xgb": xgb_models,
              "w_lgb": w_lgb, "w_xgb": w_xgb,
              "calibrator": ir}                       # ← calibrator 저장
    return models, oof_pred, {"lgb": lgb_scores, "xgb": xgb_scores}


# ─────────────────────────────────────────────
# predict_churn
# ─────────────────────────────────────────────
def predict_churn(models: dict, test_df: pd.DataFrame,
                  oof_pred: np.ndarray = None, y_true: np.ndarray = None) -> np.ndarray:
    """
    oof_pred와 y_true를 넘기면 Isotonic 보정까지 적용.
    run_pipeline.py에서: predict_churn(churn_models, test_df, churn_oof, train_df["target_churn"].values)
    """
    X = _get_features(test_df)

    w_lgb = models.get("w_lgb", 0.5)
    w_xgb = models.get("w_xgb", 0.5)

    lgb_preds = np.mean([m.predict_proba(X)[:, 1] for m in models["lgb"]], axis=0)
    xgb_preds = np.mean([m.predict_proba(X)[:, 1] for m in models["xgb"]], axis=0)
    raw = w_lgb * lgb_preds + w_xgb * xgb_preds

    # 보정 적용 (OOF + 정답 레이블이 있을 때)
    # 보정 적용 (저장된 calibrator 사용)
    if "calibrator" in models:
        cal = np.clip(models["calibrator"].predict(raw), 0.0, 1.0)  # ← transform만
        print(f"  [보정 후] Churn 예측 mean: {cal.mean():.4f}")
        return cal

    return raw


# ─────────────────────────────────────────────
# train_ltv_model
# ─────────────────────────────────────────────
def train_ltv_model(train_df: pd.DataFrame, n_splits: int = 5):
    """
    target_ltv를 log1p 변환 후 학습. OOF는 역변환(expm1) 상태로 반환.

    Returns
    -------
    models   : dict  {"lgb": [...], "xgb": [...]}
    oof_pred : np.ndarray  역변환된 OOF 예측값
    cv_scores: dict  {"lgb": [...], "xgb": [...]}
    """
    X = _get_features(train_df)
    y_raw = train_df["target_ltv"]
    y_log = np.log1p(y_raw.values)

    # 회귀 fold: 분위수 기반 층화
    y_bin = pd.qcut(y_log, q=n_splits, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    lgb_models, xgb_models = [], []
    lgb_scores, xgb_scores = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_bin), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_log[tr_idx], y_log[va_idx]

        # LightGBM
        lgb_m = lgb.LGBMRegressor(**LGB_LTV_PARAMS)
        lgb_m.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
        oof_lgb[va_idx] = lgb_m.predict(X_va)
        lgb_models.append(lgb_m)
        lgb_scores.append(np.sqrt(mean_squared_error(y_va, oof_lgb[va_idx])))

        # XGBoost
        xgb_m = xgb.XGBRegressor(**XGB_LTV_PARAMS)
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va_idx] = xgb_m.predict(X_va)
        xgb_models.append(xgb_m)
        xgb_scores.append(np.sqrt(mean_squared_error(y_va, oof_xgb[va_idx])))

        print(f"  Fold {fold} | LGB RMSE(log): {lgb_scores[-1]:.4f} | XGB RMSE(log): {xgb_scores[-1]:.4f}")

    # OOF 기반 최적 가중치 계산 (RMSE 최소화)
    w_lgb = _optimize_blend_weights(oof_lgb, oof_xgb, y_log, metric="rmse")
    w_xgb = 1.0 - w_lgb

    oof_log  = w_lgb * oof_lgb + w_xgb * oof_xgb
    oof_pred = np.expm1(oof_log)

    print(f"\n[LTV] LGB  CV RMSE(log): {np.mean(lgb_scores):.4f} ± {np.std(lgb_scores):.4f}")
    print(f"[LTV] XGB  CV RMSE(log): {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}")
    print(f"[LTV] 최적 가중치      : LGB {w_lgb:.3f} / XGB {w_xgb:.3f}")
    print(f"[LTV] Ensemble RMSE(log): {np.sqrt(mean_squared_error(y_log, oof_log)):.4f}")

    models = {"lgb": lgb_models, "xgb": xgb_models, "w_lgb": w_lgb, "w_xgb": w_xgb}
    return models, oof_pred, {"lgb": lgb_scores, "xgb": xgb_scores}


# ─────────────────────────────────────────────
# predict_ltv
# ─────────────────────────────────────────────
def predict_ltv(models: dict, test_df: pd.DataFrame) -> np.ndarray:
    X = _get_features(test_df)
    w_lgb = models.get("w_lgb", 0.5)
    w_xgb = models.get("w_xgb", 0.5)
    lgb_preds = np.mean([m.predict(X) for m in models["lgb"]], axis=0)
    xgb_preds = np.mean([m.predict(X) for m in models["xgb"]], axis=0)
    return np.expm1(w_lgb * lgb_preds + w_xgb * xgb_preds)


# ─────────────────────────────────────────────
# make_submission
# ─────────────────────────────────────────────
def make_submission(test_df: pd.DataFrame,
                    churn_pred: np.ndarray,
                    ltv_pred: np.ndarray,
                    output_dir: str = "output",
                    team_name: str = "team",
                    week: int = 1) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{team_name}_week{week}_submission.csv")
    pd.DataFrame({
        "customer_id": test_df["customer_id"],
        "target_churn": churn_pred,
        "target_ltv": ltv_pred,
    }).to_csv(path, index=False)
    print(f"제출 파일 저장: {path}")
    return path