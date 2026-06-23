# -*- coding: utf-8 -*-
"""
=====================================================================
 AB 테스트: 피처셋 검증 및 정돈 여부에 따른 튜닝 효과 비교
---------------------------------------------------------------------
 검증 가설: "튜닝의 성능 개선 효과 차이의 원인은 피처 정리·검증 설계·신호 진단 작업 선행 여부이다."
   → 대회 당시 Optuna 튜닝이 악화(-0.0021)로 기각됐던 것이 튜닝 기법 문제인지 피처셋의 문제였는지를 데이터로 확인

 설계 (2x2):
   X = 5주차 피처셋. is_married 오류, 거래피처 다수, 캡핑/상호작용 포함 (~49개, 미정돈)
   Y = denoised 10개 (진단 정돈)
   각각 대회 당시 고정파라미터(튜닝 전) → 동일 Optuna(튜닝 후)
   두 피처셋에 동일 Optuna 탐색공간/trial수(60)/시드/CV 적용

 판정:
   Y의 튜닝효과(D-C) > X의 튜닝효과(B-A) 유의한 차이 있을 시(>0.001) → 가설 입증
   비슷할 시 → 피처셋 상태 무관, 튜닝은 어떤 환경에서도 정상 작동한다
=====================================================================
"""
import sys, os
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
from src.paths import DATA_PROCESSED, EXPERIMENTS_RESULTS

import json, warnings, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
SEED=42; np.random.seed(SEED)

CONFIG={
    # 5주차 기준 train_df (features.py 로그변환 전 상태)
    "train_df_path":str(DATA_PROCESSED / "train_df_week3.csv"),
    "n_folds":5,"seed":SEED,"optuna_trials":60,
    "out_json":str(EXPERIMENTS_RESULTS / "ab_test_v2_summary.json"),
}
REPORT={"_meta":{"script":"ab_test_v2_correct_control",
                 "generated_at":time.strftime("%Y-%m-%d %H:%M:%S")}}
def section(t): print("\n"+"="*68+"\n"+t+"\n"+"="*68)
def jsonable(o):
    if isinstance(o,(np.integer,)):return int(o)
    if isinstance(o,(np.floating,)):return round(float(o),6)
    if isinstance(o,np.ndarray):return o.tolist()
    return o

# ----------------------------------------------------------------------
section("[LOAD] 5주차 피처셋 train_df 로드 + features.py 상당 로그변환")
df=pd.read_csv(CONFIG["train_df_path"])
print(f"로드된 train_df shape: {df.shape}")

# features.py 가 하던 로그 변환 재현 (trainmerge는 원본 유지 상태로 저장됨)
LOG_SRC=["total_deposit_balance","total_loan_balance","card_cash_service_amt","card_loan_amt"]
for col in LOG_SRC:
    if col in df.columns:
        df["log_"+col]=np.log1p(df[col].clip(lower=0))
if "loan_to_deposit_ratio" in df.columns:
    df["log_loan_to_deposit_ratio"]=np.log1p(df["loan_to_deposit_ratio"].clip(lower=0))

# 타겟 분리
assert "target_churn" in df.columns, "target_churn 컬럼 없음"
y=df["target_churn"].values
pw=(len(y)-y.sum())/y.sum()

# ── X = 5주차 실제 피처셋 (모든 수치 피처, target/id/원본중복 제외) ──
EXCLUDE={"customer_id","target_churn","target_ltv"}
# 로그변환으로 대체된 원본은 제외 (5주차 features.py 방침과 동일)
EXCLUDE |= set(LOG_SRC) | {"loan_to_deposit_ratio"}
X_FULL=[c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in EXCLUDE]

# ── Y = denoised 10개 ──
DENOISED=["region_code_enc","income_group_enc","credit_score","total_deposit_balance",
          "card_cash_service_amt","card_loan_amt","fin_asset_trend_score",
          "log_total_deposit_balance","log_card_cash_service_amt","log_card_loan_amt"]
# denoised는 원본 deposit 등을 쓰므로 EXCLUDE 적용 전 컬럼에서 직접 확인
avail=set(df.columns)
DENOISED=[c for c in DENOISED if c in avail]

# 결측 처리
for c in set(X_FULL)|set(DENOISED):
    if c in df.columns: df[c]=df[c].fillna(-999)

print(f"X(5주차피처셋) {len(X_FULL)}개 | Y(denoised) {len(DENOISED)}개")
print(f"  X 포함 여부 점검 - is_married: {'is_married' in X_FULL}, "
      f"거래피처 일부: {[c for c in X_FULL if 'trans' in c][:3]}")
REPORT["featuresets"]={"X_week5_foundation":X_FULL,"Y_denoised":DENOISED}

skf=StratifiedKFold(n_splits=CONFIG["n_folds"],shuffle=True,random_state=SEED)

# 대회 당시 고정 파라미터 (4주차 기록: num_leaves=31, n_estimators=1000)
LEGACY=dict(n_estimators=1000,learning_rate=0.02,num_leaves=31,
            subsample=0.8,colsample_bytree=0.8)

def cv_auc(cols,params):
    Xc=df[cols]; oof=np.zeros(len(Xc))
    for tr_i,va_i in skf.split(Xc,y):
        m=lgb.LGBMClassifier(**params,scale_pos_weight=pw,random_state=SEED,n_jobs=-1,verbose=-1)
        m.fit(Xc.iloc[tr_i],y[tr_i]); oof[va_i]=m.predict_proba(Xc.iloc[va_i])[:,1]
    return roc_auc_score(y,oof)

def tune(cols):
    def obj(t):
        p=dict(n_estimators=t.suggest_int("n_estimators",400,1000),
               learning_rate=t.suggest_float("learning_rate",0.008,0.05,log=True),
               num_leaves=t.suggest_int("num_leaves",10,48),
               min_child_samples=t.suggest_int("min_child_samples",20,100),
               subsample=t.suggest_float("subsample",0.6,1.0),
               colsample_bytree=t.suggest_float("colsample_bytree",0.5,1.0),
               reg_lambda=t.suggest_float("reg_lambda",0.0,6.0))
        return cv_auc(cols,p)
    st=optuna.create_study(direction="maximize",
                           sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj,n_trials=CONFIG["optuna_trials"],show_progress_bar=False)
    return st.best_value,st.best_params

# ======================================================================
# 2x2 측정
# ======================================================================
section("[측정] 2x2: 5주차피처셋(X) vs denoised(Y) × 튜닝 전/후")
print("A) X(5주차피처셋) + 대회당시 고정파라미터 ...")
A=cv_auc(X_FULL,LEGACY); print(f"   A = {A:.5f}  (대회 0.794 부근 재현되는지 확인)")
print("C) Y(denoised) + 고정파라미터 ...")
C=cv_auc(DENOISED,LEGACY); print(f"   C = {C:.5f}")
print(f"B) X(5주차피처셋) + Optuna({CONFIG['optuna_trials']}) ...")
B,Bp=tune(X_FULL); print(f"   B = {B:.5f}")
print(f"D) Y(denoised) + Optuna({CONFIG['optuna_trials']}) ...")
D,Dp=tune(DENOISED); print(f"   D = {D:.5f}")

dBA=B-A; dDC=D-C; diff=dDC-dBA
section("[결과] 튜닝 효과 비교")
print(f"  X 5주차피처셋 : 튜닝전 A={A:.5f} → 튜닝후 B={B:.5f}  ΔAUC={dBA:+.5f}")
print(f"  Y denoised  : 튜닝전 C={C:.5f} → 튜닝후 D={D:.5f}  ΔAUC={dDC:+.5f}")
print(f"\n  핵심: (D-C)-(B-A) = {diff:+.5f}")

if diff>0.001:
    verdict=("가설 입증: 정돈된 피처셋(Y)에서 튜닝 효과가 유의하게 큼. ")
elif diff<-0.001:
    verdict="주장 반증: 5주차 피처셋에서 오히려 튜닝 효과가 더 큼. 가설 정정 필요."
else:
    verdict=("주장 부분기각: 5주차 피처셋에서도 튜닝 효과가 비슷함. 피처셋 정돈과 무관하게 튜닝은 효과를 냄."
             "→ '정리 선행이 튜닝 전제'가 아니라 '정리와 튜닝은 독립 기여'가 정확.")
print(f"\n  >>> 판정: {verdict}")
print(f"\n  [대회 재현] A={A:.5f} vs 대회 실제 0.7943 → "
      f"{'재현됨' if abs(A-0.7943)<0.01 else '차이 있음(집계/시드 차이)'}")
print(f"  [최종 절대] B(5주차피처셋+튜닝)={B:.5f} vs D(denoised+튜닝)={D:.5f}  D-B={D-B:+.5f}")

REPORT["results"]={
    "A_week5_legacy":jsonable(A),"B_week5_tuned":jsonable(B),
    "C_denoised_legacy":jsonable(C),"D_denoised_tuned":jsonable(D),
    "tuning_gain_week5(B-A)":jsonable(dBA),
    "tuning_gain_denoised(D-C)":jsonable(dDC),
    "diff_of_gains":jsonable(diff),
    "competition_reproduction_A":jsonable(A),
    "final_auc_diff(D-B)":jsonable(D-B),
    "verdict":verdict,
    "best_params_week5":Bp,"best_params_denoised":Dp,
}
with open(CONFIG["out_json"],"w",encoding="utf-8") as f:
    json.dump(REPORT,f,ensure_ascii=False,indent=2,default=jsonable)
print(f"saved: {os.path.abspath(CONFIG['out_json'])}")
