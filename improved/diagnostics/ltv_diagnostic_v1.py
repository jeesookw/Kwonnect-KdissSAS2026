# -*- coding: utf-8 -*-
"""
 LTV 신호 구조 진단 스크립트 v1
=====================================================================
 목적 : 무엇이 LTV를 설명하고, 무엇이 고액 고객을 가르는지를 데이터로 확정
- 5주차 2-stage 실패(고액분류 AUC 0.54)가 '구분 불가'였는지 '피처를 잘못 골라서'였는지 분리 검증
 입력 : 원본 7개 CSV
 출력 : (1) 콘솔 요약 (2) ltv_diagnostic_summary.json
 ※ diagnostic_v1.py와 동일한 피처 생성 로직을 사용하므로 두 진단 결과를 직접 비교할 수 있음.
"""

# ----------------------------------------------------------------------
# 0. 환경 준비
# ----------------------------------------------------------------------
import sys, os
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
from src.paths import DATA_RAW, DIAGNOSTICS_RESULTS

import json, warnings, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, skew, kurtosis
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

CONFIG = {
    "train_customer":     str(DATA_RAW / "train_customer_info.csv"),
    "train_transaction":  str(DATA_RAW / "train_transaction_history.csv"),
    "train_finance":      str(DATA_RAW / "train_finance_profile.csv"),
    "train_targets":      str(DATA_RAW / "train_targets.csv"),
    "test_customer":      str(DATA_RAW / "test_customer_info.csv"),
    "test_transaction":   str(DATA_RAW / "test_transaction_history.csv"),
    "test_finance":       str(DATA_RAW / "test_finance_profile.csv"),

    "ref_date":    "2023-12-31",
    "n_folds":     5,
    "seed":        42,
    "high_ltv_quantile": 0.90,   # 상위 10%를 고액으로 정의
    "out_json":    str(DIAGNOSTICS_RESULTS / "ltv_diagnostic_summary.json"),
}

REPORT = {"_meta": {"script": "ltv_diagnostic_v1",
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}}


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)

def jsonable(o):
    if isinstance(o, (np.integer,)):   return int(o)
    if isinstance(o, (np.floating,)):  return round(float(o), 6)
    if isinstance(o, (np.bool_,)):     return bool(o)
    if isinstance(o, np.ndarray):      return o.tolist()
    return o


# ======================================================================
# 1. 로드 + 진단용 핵심 피처
# ======================================================================
section("[LOAD] 원본 CSV 로드 및 진단용 피처 생성")

def load_side(side):
    cust = pd.read_csv(CONFIG[f"{side}_customer"])
    fin  = pd.read_csv(CONFIG[f"{side}_finance"])
    txn  = pd.read_csv(CONFIG[f"{side}_transaction"])

    cust["join_date"] = pd.to_datetime(cust["join_date"])
    txn["trans_date"] = pd.to_datetime(txn["trans_date"])
    ref = pd.to_datetime(CONFIG["ref_date"])

    cust["tenure_days"] = (ref - cust["join_date"]).dt.days
    cust["is_married_bin"] = cust["is_married"].astype(float)

    g = txn.groupby("customer_id")
    txn_feat = pd.DataFrame({
        "trans_count":        g["trans_id"].count(),
        "trans_amount_total": g["trans_amount"].sum(),
        "trans_amount_mean":  g["trans_amount"].mean(),
        "trans_amount_max":   g["trans_amount"].max(),
        "trans_amount_min":   g["trans_amount"].min(),
        "last_trans_date":    g["trans_date"].max(),
    })
    txn_feat["recency_days"] = (ref - txn_feat["last_trans_date"]).dt.days
    if "biz_type" in txn.columns:
        online = (txn.assign(is_online=(txn["biz_type"].astype(str)
                  .str.lower() == "online").astype(int))
                  .groupby("customer_id")["is_online"].mean())
        txn_feat["online_ratio"] = online
    if "is_installment" in txn.columns:
        txn_feat["installment_ratio"] = g["is_installment"].mean()
    txn_feat = txn_feat.drop(columns=["last_trans_date"])

    fin = fin.copy()
    for col in ["total_deposit_balance", "total_loan_balance",
                "card_cash_service_amt", "card_loan_amt"]:
        if col in fin.columns:
            fin["log_" + col] = np.log1p(fin[col].clip(lower=0))

    df = (cust.merge(fin, on="customer_id", how="left")
              .merge(txn_feat.reset_index(), on="customer_id", how="left"))
    return df

df_tr = load_side("train")
tgt = pd.read_csv(CONFIG["train_targets"])
df_tr = df_tr.merge(tgt, on="customer_id", how="left")
print(f"train shape : {df_tr.shape}")

EXCLUDE = {"customer_id", "join_date", "target_churn", "target_ltv",
           "is_married"}
num_cols = [c for c in df_tr.columns
            if pd.api.types.is_numeric_dtype(df_tr[c]) and c not in EXCLUDE]

# 피처 그룹 정의 (고액분류 분리검증용)
FIN_FEATURES = [c for c in num_cols if any(k in c for k in
    ["deposit", "loan", "credit", "card", "overdue", "fin_asset",
     "num_active"])]
TXN_FEATURES = [c for c in num_cols if any(k in c for k in
    ["trans", "recency", "online", "installment", "tenure"])]
print(f"전체 수치형 {len(num_cols)}개 | 금융 {len(FIN_FEATURES)} | 거래 {len(TXN_FEATURES)}")
REPORT["feature_groups"] = {"all": num_cols,
                            "finance": FIN_FEATURES, "transaction": TXN_FEATURES}

y_ltv = df_tr["target_ltv"].values
y_log = np.log1p(y_ltv)


# ======================================================================
# STEP 0 : 타겟 변환 진단  (어떤 변환이 RMSE에 유리한가)
# ======================================================================
section("[STEP 0] LTV 타겟 변환 비교")

transforms = {
    "raw":         y_ltv,
    "log1p":       np.log1p(y_ltv),
    "sqrt":        np.sqrt(y_ltv),
    "cbrt":        np.cbrt(y_ltv),
    "boxcox_like_log_then_std": (np.log1p(y_ltv) - np.log1p(y_ltv).mean()),
}
tf_report = {}
for name, arr in transforms.items():
    tf_report[name] = {"skew": jsonable(skew(arr)),
                       "kurtosis": jsonable(kurtosis(arr))}
    print(f"  {name:28s} skew={skew(arr):+.3f}  kurt={kurtosis(arr):+.3f}")
print("\n  → skew가 0에 가까울수록 RMSE 학습에 유리. log1p 좌편향(-1.14) 과교정 여부 확인.")
REPORT["step0_target_transform"] = tf_report
# 분위수 정보
REPORT["step0_ltv_quantiles"] = {
    q: jsonable(np.quantile(y_ltv, q))
    for q in [0.01, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
}
print("  LTV 분위수:", {k: round(v) for k, v in
      REPORT["step0_ltv_quantiles"].items()})


# ======================================================================
# STEP 1 : LTV 단변량 연관 — Spearman 상관
# ======================================================================
section("[STEP 1] LTV ~ 피처 Spearman 상관 (단조 관계)")

sp_rows = []
for c in num_cols:
    x = df_tr[c]
    mask = x.notna()
    if mask.sum() < 100:
        continue
    rho, p = spearmanr(x[mask], y_log[mask.values])
    sp_rows.append((c, rho, p))

sp_rows.sort(key=lambda r: abs(r[1]), reverse=True)
print("LTV와 상관 강한 순 Top 15 (|rho|):")
for c, rho, p in sp_rows[:15]:
    sig = "유의" if p < 0.05 else "  - "
    print(f"  [{sig}] {c:28s} rho={rho:+.4f}  p={p:.2e}")
REPORT["step1_spearman"] = [
    {"feature": c, "spearman_rho": jsonable(rho), "p_value": jsonable(p),
     "significant": bool(p < 0.05)}
    for c, rho, p in sp_rows
]


# ======================================================================
# STEP 2 : Null Importance (LTV 회귀)  진짜 신호 vs 우연
# ======================================================================
section("[STEP 2] LTV Null Importance — 진짜 신호 가려내기")

X = df_tr[num_cols].fillna(-999)

def lgb_reg_importance(Xz, yz, seed):
    m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05,
                          num_leaves=31, subsample=0.8,
                          colsample_bytree=0.8, random_state=seed,
                          n_jobs=-1, verbose=-1)
    m.fit(Xz, yz)
    return m.feature_importances_

real_imp = lgb_reg_importance(X, y_log, CONFIG["seed"])
N_NULL = 15
null_imps = np.zeros((N_NULL, len(num_cols)))
for i in range(N_NULL):
    null_imps[i] = lgb_reg_importance(X, np.random.permutation(y_log), 2000 + i)
null_mean = null_imps.mean(axis=0)
null_std = null_imps.std(axis=0) + 1e-9
null_z = (real_imp - null_mean) / null_std

ni_rank = sorted(zip(num_cols, real_imp, null_mean, null_z),
                 key=lambda x: x[3], reverse=True)
print("실제 importance가 '우연' 대비 강한 순 (z-score):")
for c, ri, nm, z in ni_rank:
    v = "진짜신호" if z > 3 else ("애매" if z > 1 else "우연수준")
    print(f"  {c:28s} real={ri:6.0f} null={nm:6.0f} z={z:6.2f}  [{v}]")
REPORT["step2_null_importance"] = [
    {"feature": c, "real": jsonable(ri), "null_mean": jsonable(nm),
     "z_score": jsonable(z),
     "verdict": ("real_signal" if z > 3 else "ambiguous" if z > 1 else "noise")}
    for c, ri, nm, z in ni_rank
]


# ======================================================================
# STEP 3 : 고액 고객 분류 가능성 (5주차 실패 핵심 재검증)
#   - 거래 피처 입력 vs 금융 피처 입력 vs 전체 입력 분리 비교
# ======================================================================
section("[STEP 3] 고액 LTV 고객 분류 — 입력 그룹별 분리 검증")

q = CONFIG["high_ltv_quantile"]
thr = np.quantile(y_ltv, q)
y_high = (y_ltv >= thr).astype(int)
print(f"고액 기준: 상위 {int((1-q)*100)}% (LTV >= {thr:,.0f}), "
      f"고액 고객 {y_high.sum()}명")

skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                      random_state=CONFIG["seed"])

def classify_auc(feature_cols, label):
    Xs = df_tr[feature_cols].fillna(-999)
    oof = np.zeros(len(Xs))
    for tr_i, va_i in skf.split(Xs, y_high):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03,
                               num_leaves=31, subsample=0.8,
                               colsample_bytree=0.8,
                               random_state=CONFIG["seed"], n_jobs=-1,
                               verbose=-1)
        m.fit(Xs.iloc[tr_i], y_high[tr_i])
        oof[va_i] = m.predict_proba(Xs.iloc[va_i])[:, 1]
    auc = roc_auc_score(y_high, oof)
    print(f"  [{label:18s}] 고액분류 AUC = {auc:.4f}  (피처 {len(feature_cols)}개)")
    return auc

auc_all = classify_auc(num_cols, "전체피처")
auc_fin = classify_auc(FIN_FEATURES, "금융피처만")
auc_txn = classify_auc(TXN_FEATURES, "거래피처만")
REPORT["step3_high_ltv_classification"] = {
    "threshold": jsonable(thr),
    "n_high": jsonable(int(y_high.sum())),
    "auc_all_features": jsonable(auc_all),
    "auc_finance_only": jsonable(auc_fin),
    "auc_transaction_only": jsonable(auc_txn),
    "week5_baseline": 0.5421,
}


# ======================================================================
# STEP 4 : LTV 단일모델 CV 재현 (기준점) + Churn 신호와 교집합
# ======================================================================
section("[STEP 4] LTV CV 재현 + Churn/LTV 신호 비교")

kf = KFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["seed"])
oof = np.zeros(len(X))
fold_rmse = []
for k, (tr_i, va_i) in enumerate(kf.split(X)):
    m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03,
                          num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                          random_state=CONFIG["seed"], n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr_i], y_log[tr_i],
          eval_set=[(X.iloc[va_i], y_log[va_i])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    oof[va_i] = m.predict(X.iloc[va_i])
    r = np.sqrt(mean_squared_error(y_log[va_i], oof[va_i]))
    fold_rmse.append(r)
    print(f"  fold {k+1}: RMSE(log) = {r:.4f}")
cv_rmse = np.sqrt(mean_squared_error(y_log, oof))
print(f"\nOOF RMSE(log) = {cv_rmse:.4f}  (fold std={np.std(fold_rmse):.4f})")
print("  → 5주차 LTV RMSE 1.5089와 비교. 진단용 핵심피처 기준점.")
REPORT["step4_cv"] = {
    "oof_rmse_log": jsonable(cv_rmse),
    "fold_rmse": [jsonable(r) for r in fold_rmse],
    "fold_std": jsonable(np.std(fold_rmse)),
    "week5_reference": 1.5089,
}

# LTV 진짜 신호 (z>3) 목록 추출
ltv_signals = set(d["feature"] for d in REPORT["step2_null_importance"]
                  if d["verdict"] == "real_signal")
REPORT["step4_ltv_real_signals"] = sorted(ltv_signals)
print(f"\nLTV 진짜 신호 피처(z>3): {sorted(ltv_signals)}")
print("  (다음 턴에서 A단계 Churn 신호와 교집합/차집합 분석 예정)")


# ======================================================================
# 저장
# ======================================================================
section("[SAVE] 결과 저장")
with open(CONFIG["out_json"], "w", encoding="utf-8") as f:
    json.dump(REPORT, f, ensure_ascii=False, indent=2, default=jsonable)
print(f"saved: {os.path.abspath(CONFIG['out_json'])}")
