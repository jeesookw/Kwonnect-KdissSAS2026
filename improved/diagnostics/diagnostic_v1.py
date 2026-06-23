# -*- coding: utf-8 -*-
"""
=====================================================================
 통합 진단 스크립트  (SAS/KDIS Churn-LTV 대회 사후 개선)
---------------------------------------------------------------------
 목적 : 5주차까지의 의사결정이 올바른 전제 위에 있었는지 검증.
        - 모델 성능을 올리는 코드가 아니라, "무엇이 문제였는지"를
          데이터로 확정하는 코드.
 입력 : 원본 7개 CSV (병합 전)
 출력 : (1) 콘솔에 사람이 읽을 요약
        (2) diagnostic_summary.json  ← 이 파일만 챗에 올리면 됨
 환경 : Google Colab
=====================================================================
실행 방법
 1) 아래 CONFIG의 경로만 본인 환경에 맞게 수정
 2) 셀에 전체 붙여넣고 실행
 3) 끝나면 diagnostic_summary.json 다운로드 → 챗에 첨부
=====================================================================
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
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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

    "ref_date":    "2023-12-31",   # 거래 데이터 기준일 (명세서 기준)
    "n_folds":     5,
    "seed":        42,
    "out_json":    str(DIAGNOSTICS_RESULTS / "diagnostic_summary.json"),
}

REPORT = {"_meta": {"script": "diagnostic_v1",
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
# 1. 로드 + 진단용 최소 피처 생성
#    (※ 5주차의 풀 피처 엔지니어링이 아님. 진단에 필요한 핵심만.)
# ======================================================================
section("[LOAD] 원본 CSV 로드 및 진단용 피처 생성")

def load_side(side):
    cust = pd.read_csv(CONFIG[f"{side}_customer"])
    fin  = pd.read_csv(CONFIG[f"{side}_finance"])
    txn  = pd.read_csv(CONFIG[f"{side}_transaction"])

    cust["join_date"] = pd.to_datetime(cust["join_date"])
    txn["trans_date"] = pd.to_datetime(txn["trans_date"])
    ref = pd.to_datetime(CONFIG["ref_date"])

    # ---- customer 파생 ----
    cust["tenure_days"] = (ref - cust["join_date"]).dt.days
    cust["is_married_bin"] = cust["is_married"].astype(float)

    # ---- transaction 집계 (진단용 핵심만) ----
    g = txn.groupby("customer_id")
    txn_feat = pd.DataFrame({
        "trans_count":        g["trans_id"].count(),
        "trans_amount_total": g["trans_amount"].sum(),
        "trans_amount_mean":  g["trans_amount"].mean(),
        "trans_amount_min":   g["trans_amount"].min(),
        "last_trans_date":    g["trans_date"].max(),
        "first_trans_date":   g["trans_date"].min(),
    })
    txn_feat["recency_days"] = (ref - txn_feat["last_trans_date"]).dt.days
    # online 비율
    if "biz_type" in txn.columns:
        online = (txn.assign(is_online=(txn["biz_type"]
                  .astype(str).str.lower() == "online").astype(int))
                  .groupby("customer_id")["is_online"].mean())
        txn_feat["online_ratio"] = online
    txn_feat = txn_feat.drop(columns=["last_trans_date", "first_trans_date"])

    # ---- finance 파생 (로그 변환만, 캡핑 없음=진단 단계) ----
    fin = fin.copy()
    for col in ["total_deposit_balance", "total_loan_balance",
                "card_cash_service_amt", "card_loan_amt"]:
        if col in fin.columns:
            fin["log_" + col] = np.log1p(fin[col].clip(lower=0))

    # ---- 병합 ----
    df = (cust.merge(fin, on="customer_id", how="left")
              .merge(txn_feat.reset_index(), on="customer_id", how="left"))
    return df

df_tr = load_side("train")
df_te = load_side("test")

# 타겟 부착
tgt = pd.read_csv(CONFIG["train_targets"])
df_tr = df_tr.merge(tgt, on="customer_id", how="left")

print(f"train shape : {df_tr.shape}")
print(f"test  shape : {df_te.shape}")

REPORT["shapes"] = {"train": list(df_tr.shape), "test": list(df_te.shape)}

# 분석에 쓸 수치형 피처 자동 선별 (id/target/날짜 제외)
EXCLUDE = {"customer_id", "join_date", "target_churn", "target_ltv",
           "is_married"}  # is_married는 _bin으로 따로 봄
num_cols = [c for c in df_tr.columns
            if c in df_te.columns
            and pd.api.types.is_numeric_dtype(df_tr[c])
            and c not in EXCLUDE]
print(f"\n진단 대상 수치형 피처 {len(num_cols)}개:")
print("  " + ", ".join(num_cols))
REPORT["feature_list"] = num_cols


# ======================================================================
# STEP 0 : 데이터 신뢰성  (is_married 진실 + Train/Test 분포 비교)
# ======================================================================
section("[STEP 0] 데이터 신뢰성 — is_married 진실 + 분포 비교")

# ---- (0-a) is_married 실제 비율 : 95% vs 65% 종결 ----
tr_married = df_tr["is_married"].mean()
te_married = df_te["is_married"].mean()
print(f"is_married 기혼 비율  Train={tr_married:.4f}  "
      f"Test={te_married:.4f}  차이={abs(tr_married-te_married)*100:.2f}%p")

# 기혼/비기혼별 이탈률
churn_by_married = (df_tr.groupby("is_married")["target_churn"]
                    .agg(["mean", "count"]).to_dict("index"))
print("기혼별 이탈률:", {k: round(v["mean"], 4) for k, v in churn_by_married.items()})

REPORT["step0_is_married"] = {
    "train_ratio": jsonable(tr_married),
    "test_ratio":  jsonable(te_married),
    "diff_pp":     jsonable(abs(tr_married - te_married) * 100),
    "churn_rate_by_group": {int(k): {"churn": jsonable(v["mean"]),
                                     "n": jsonable(v["count"])}
                            for k, v in churn_by_married.items()},
    "verdict": ("문서의 95% vs 65%가 맞음(분포 불일치 실재)"
                if abs(tr_married - te_married) > 0.15
                else "문서 기록 오류 가능 — Train/Test 거의 동일"),
}
print(">>", REPORT["step0_is_married"]["verdict"])

# ---- (0-b) 전 수치형 컬럼 Train/Test 분포 비교 (KS + 평균차) ----
print("\n[Train/Test 분포 비교]  KS 통계량 큰 순 Top 컬럼")
drift_rows = []
for c in num_cols:
    a = df_tr[c].dropna()
    b = df_te[c].dropna()
    if len(a) < 100 or len(b) < 100:
        continue
    ks = ks_2samp(a, b).statistic
    # 표준화 평균차
    pooled_std = np.sqrt((a.var() + b.var()) / 2) + 1e-9
    smd = abs(a.mean() - b.mean()) / pooled_std
    drift_rows.append((c, ks, smd, a.mean(), b.mean()))

drift_rows.sort(key=lambda x: x[1], reverse=True)
for c, ks, smd, ma, mb in drift_rows[:12]:
    flag = "  <-- 주의" if ks > 0.1 else ""
    print(f"  {c:28s} KS={ks:.4f}  SMD={smd:.3f}  "
          f"meanTr={ma:.2f} meanTe={mb:.2f}{flag}")

REPORT["step0_distribution_drift"] = [
    {"feature": c, "ks": jsonable(ks), "smd": jsonable(smd),
     "mean_train": jsonable(ma), "mean_test": jsonable(mb)}
    for c, ks, smd, ma, mb in drift_rows
]

# ---- (0-c) 결측/범위/타겟 분포 재확인 ----
miss_tr = df_tr[num_cols].isna().mean()
miss_te = df_te[num_cols].isna().mean()
miss_report = {c: {"train": jsonable(miss_tr[c]), "test": jsonable(miss_te[c])}
               for c in num_cols if miss_tr[c] > 0 or miss_te[c] > 0}
print(f"\n결측 있는 피처 수: {len(miss_report)}")
for c, v in miss_report.items():
    print(f"  {c:28s} train={v['train']:.4f} test={v['test']:.4f}")

REPORT["step0_missing"] = miss_report
REPORT["step0_target"] = {
    "churn_rate":  jsonable(df_tr["target_churn"].mean()),
    "ltv_min":     jsonable(df_tr["target_ltv"].min()),
    "ltv_max":     jsonable(df_tr["target_ltv"].max()),
    "ltv_mean":    jsonable(df_tr["target_ltv"].mean()),
    "ltv_median":  jsonable(df_tr["target_ltv"].median()),
    "ltv_skew":    jsonable(df_tr["target_ltv"].skew()),
    "log_ltv_skew": jsonable(np.log1p(df_tr["target_ltv"]).skew()),
}
print("\n타겟:", REPORT["step0_target"])


# ======================================================================
# STEP 1 : Adversarial Validation  (Train vs Test 구분 가능한가)
# ======================================================================
section("[STEP 1] Adversarial Validation — Train/Test drift 정량화")

adv = pd.concat([
    df_tr[num_cols].assign(_is_test=0),
    df_te[num_cols].assign(_is_test=1),
], ignore_index=True)
X_adv = adv[num_cols].fillna(-999)
y_adv = adv["_is_test"].values

skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                      random_state=CONFIG["seed"])
adv_oof = np.zeros(len(X_adv))
adv_imp = np.zeros(len(num_cols))
for tr_idx, va_idx in skf.split(X_adv, y_adv):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           num_leaves=31, subsample=0.8,
                           colsample_bytree=0.8, random_state=CONFIG["seed"],
                           n_jobs=-1, verbose=-1)
    m.fit(X_adv.iloc[tr_idx], y_adv[tr_idx])
    adv_oof[va_idx] = m.predict_proba(X_adv.iloc[va_idx])[:, 1]
    adv_imp += m.feature_importances_ / CONFIG["n_folds"]

adv_auc = roc_auc_score(y_adv, adv_oof)
print(f"Adversarial AUC = {adv_auc:.4f}")
print("  (0.5 근처=Train/Test 동일 분포 / 1.0 근처=완전히 구분됨)")

imp_rank = sorted(zip(num_cols, adv_imp), key=lambda x: x[1], reverse=True)
print("\nTrain/Test를 가르는 데 기여한 피처 Top 10 (drift 원인):")
for c, v in imp_rank[:10]:
    print(f"  {c:28s} {v:8.1f}")

REPORT["step1_adversarial"] = {
    "auc": jsonable(adv_auc),
    "interpretation": ("심각한 drift — CV가 Test를 대표 못함"
                       if adv_auc > 0.75 else
                       "중간 drift — 일부 피처 점검 필요"
                       if adv_auc > 0.6 else
                       "drift 거의 없음 — Train/Test 유사"),
    "top_drift_features": [{"feature": c, "importance": jsonable(v)}
                           for c, v in imp_rank[:15]],
}
print(">>", REPORT["step1_adversarial"]["interpretation"])


# ======================================================================
# STEP 2 : 피처 신호 검증  (Mann-Whitney + Null Importance)
# ======================================================================
section("[STEP 2] 피처 신호 검증 — 이탈/유지 단변량 분리력")

# ---- (2-a) Mann-Whitney U + rank-biserial effect size ----
c1 = df_tr[df_tr["target_churn"] == 1]
c0 = df_tr[df_tr["target_churn"] == 0]
mw_rows = []
for c in num_cols:
    a = c1[c].dropna()
    b = c0[c].dropna()
    if len(a) < 30 or len(b) < 30:
        continue
    try:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial correlation = 1 - 2U/(n1*n2)
        rbc = 1 - (2 * u) / (len(a) * len(b))
        mw_rows.append((c, p, abs(rbc)))
    except ValueError:
        continue

mw_rows.sort(key=lambda x: x[2], reverse=True)
print("이탈/유지 분리력 (effect size) 큰 순 Top 12:")
sig_cnt = 0
for c, p, es in mw_rows:
    sig = p < 0.05
    sig_cnt += int(sig)
for c, p, es in mw_rows[:12]:
    mark = "유의" if p < 0.05 else "  - "
    print(f"  [{mark}] {c:28s} effect={es:.4f}  p={p:.2e}")
print(f"\n유의(p<0.05) 피처 수: {sig_cnt} / {len(mw_rows)}")

REPORT["step2_mannwhitney"] = {
    "n_significant": sig_cnt,
    "n_total": len(mw_rows),
    "features": [{"feature": c, "p_value": jsonable(p),
                  "effect_size": jsonable(es), "significant": bool(p < 0.05)}
                 for c, p, es in mw_rows],
}

# ---- (2-b) Null Importance : 진짜 신호 vs 우연 ----
print("\n[Null Importance] 타겟 셔플 대비 실제 importance 비교 중...")
X = df_tr[num_cols].fillna(-999)
y = df_tr["target_churn"].values

def lgb_importance(Xz, yz, seed):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                           num_leaves=31, subsample=0.8,
                           colsample_bytree=0.8, random_state=seed,
                           n_jobs=-1, verbose=-1)
    m.fit(Xz, yz)
    return m.feature_importances_

real_imp = lgb_importance(X, y, CONFIG["seed"])

N_NULL = 15  # Colab에서 1~2분 내. 더 정밀히 하려면 늘리세요.
null_imps = np.zeros((N_NULL, len(num_cols)))
for i in range(N_NULL):
    y_shuf = np.random.permutation(y)
    null_imps[i] = lgb_importance(X, y_shuf, 1000 + i)

null_mean = null_imps.mean(axis=0)
null_std = null_imps.std(axis=0) + 1e-9
# 실제 importance가 null 분포 대비 몇 시그마 위인가
null_score = (real_imp - null_mean) / null_std

ni_rank = sorted(zip(num_cols, real_imp, null_mean, null_score),
                 key=lambda x: x[3], reverse=True)
print("실제 importance가 '우연(null)' 대비 강한 순 (z-score):")
for c, ri, nm, z in ni_rank:
    verdict = "진짜신호" if z > 3 else ("애매" if z > 1 else "우연수준")
    print(f"  {c:28s} real={ri:6.0f} null={nm:6.0f} z={z:6.2f}  [{verdict}]")

REPORT["step2_null_importance"] = [
    {"feature": c, "real": jsonable(ri), "null_mean": jsonable(nm),
     "z_score": jsonable(z),
     "verdict": ("real_signal" if z > 3 else "ambiguous" if z > 1 else "noise")}
    for c, ri, nm, z in ni_rank
]


# ======================================================================
# STEP 3 : 현재 구조 CV 재현  (LB 비교용 기준점)
# ======================================================================
section("[STEP 3] 현재 구조 CV 재현 — Churn LGB 5-fold")

oof = np.zeros(len(X))
fold_aucs = []
pos_weight = (len(y) - y.sum()) / y.sum()
for k, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03,
                           num_leaves=31, subsample=0.8,
                           colsample_bytree=0.8,
                           scale_pos_weight=pos_weight,
                           random_state=CONFIG["seed"], n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr_idx], y[tr_idx],
          eval_set=[(X.iloc[va_idx], y[va_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
    fa = roc_auc_score(y[va_idx], oof[va_idx])
    fold_aucs.append(fa)
    print(f"  fold {k+1}: AUC = {fa:.4f}")

cv_auc = roc_auc_score(y, oof)
print(f"\nOOF AUC = {cv_auc:.4f}  (fold std = {np.std(fold_aucs):.4f})")
print("  → 이 값을 실제 리더보드 점수와 비교하세요.")
print("  → CV는 높은데 LB가 낮으면 = STEP1 drift가 원인일 가능성 큼.")

REPORT["step3_cv"] = {
    "oof_auc": jsonable(cv_auc),
    "fold_aucs": [jsonable(a) for a in fold_aucs],
    "fold_std": jsonable(np.std(fold_aucs)),
    "note": "이 진단용 핵심피처 CV는 5주차 풀피처(0.793)와 다를 수 있음. "
            "절대값보다 drift/신호 진단이 목적.",
}


# ======================================================================
# 저장
# ======================================================================
section("[SAVE] 결과 저장")
with open(CONFIG["out_json"], "w", encoding="utf-8") as f:
    json.dump(REPORT, f, ensure_ascii=False, indent=2, default=jsonable)
print(f"saved: {os.path.abspath(CONFIG['out_json'])}")
