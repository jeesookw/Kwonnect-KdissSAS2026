# -*- coding: utf-8 -*-
"""
=====================================================================
 모델링 스크립트 v1  (Churn AUC 집중)
---------------------------------------------------------------------
 v1 전략 (진단 결론 반영):
   - 채점식 = 0.5*AUC + 0.5/(1+log10(RMSE_원본스케일)) → 역산 결과 LTV term은 점수 기여가 미미함
   - Churn 진짜 신호 3개(card_loan_amt, total_deposit_balance, card_cash_service_amt) 중심.
     fin_asset_trend_score는 noise이므로 신뢰 X
   - Train/Test 분포 일치(adv AUC 0.4995) → CV 신뢰 가능
   - is_married 제거
 구성:
   - 피처셋 3종(full / denoised / core) 준비
   - Churn: 4모델(LGB/XGB/CAT/RF) 개별 CV + 다양성(상관) 측정
   - Churn: 스태킹(메타=LogReg) vs 가중평균 비교
   - LTV: cbrt 변환으로 원본스케일 RMSE 최소화 (점수 영향 작지만 마감용)
   - 최종 Score 계산 + 제출 파일 생성
 출력:
   (1) 콘솔 요약
   (2) modeling_summary.json
   (3) submission_improved.csv
=====================================================================
"""
import sys, os
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
from src.paths import DATA_RAW, MODELING_RESULTS, SUBMISSIONS_DIR

import json, warnings, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

warnings.filterwarnings("ignore")
SEED = 42; np.random.seed(SEED)

CONFIG = {
    "train_customer":str(DATA_RAW/"train_customer_info.csv"),
    "train_transaction":str(DATA_RAW/"train_transaction_history.csv"),
    "train_finance":str(DATA_RAW/"train_finance_profile.csv"),
    "train_targets":str(DATA_RAW/"train_targets.csv"),
    "test_customer":str(DATA_RAW/"test_customer_info.csv"),
    "test_transaction":str(DATA_RAW/"test_transaction_history.csv"),
    "test_finance":str(DATA_RAW/"test_finance_profile.csv"),
    "ref_date":"2023-12-31","n_folds":5,"seed":SEED,
    "out_json":str(MODELING_RESULTS/"modeling_summary.json"),
    "out_csv":str(SUBMISSIONS_DIR/"submission_improved.csv"),
}
REPORT={"_meta":{"script":"modeling_v1","generated_at":time.strftime("%Y-%m-%d %H:%M:%S")}}
def section(t): print("\n"+"="*68+"\n"+t+"\n"+"="*68)
def jsonable(o):
    if isinstance(o,(np.integer,)):return int(o)
    if isinstance(o,(np.floating,)):return round(float(o),6)
    if isinstance(o,(np.bool_,)):return bool(o)
    if isinstance(o,np.ndarray):return o.tolist()
    return o

# ======================================================================
# 피처 생성 (진단과 동일 로직 + 약간의 파생, 단 noise 피처는 최소화)
# ======================================================================
section("[LOAD] 피처 생성")
def load_side(side):
    cust=pd.read_csv(CONFIG[f"{side}_customer"]); fin=pd.read_csv(CONFIG[f"{side}_finance"])
    txn=pd.read_csv(CONFIG[f"{side}_transaction"])
    cust["join_date"]=pd.to_datetime(cust["join_date"]); txn["trans_date"]=pd.to_datetime(txn["trans_date"])
    ref=pd.to_datetime(CONFIG["ref_date"])
    cust["tenure_days"]=(ref-cust["join_date"]).dt.days
    # 범주형 인코딩
    if "gender" in cust: cust["gender_enc"]=(cust["gender"].astype(str).str.upper()=="M").astype(int)
    for col,pre in [("region_code","R"),("income_group","G")]:
        if col in cust:
            cust[col+"_enc"]=cust[col].astype(str).str.extract(r"(\d+)").astype(float)
    cust["is_married_bin"]=cust["is_married"].astype(float)
    g=txn.groupby("customer_id")
    tf=pd.DataFrame({
        "trans_count":g["trans_id"].count(),"trans_amount_total":g["trans_amount"].sum(),
        "trans_amount_mean":g["trans_amount"].mean(),"trans_amount_std":g["trans_amount"].std(),
        "trans_amount_max":g["trans_amount"].max(),"trans_amount_min":g["trans_amount"].min(),
        "last_trans_date":g["trans_date"].max(),
    })
    tf["recency_days"]=(ref-tf["last_trans_date"]).dt.days
    if "biz_type" in txn:
        tf["online_ratio"]=(txn.assign(o=(txn["biz_type"].astype(str).str.lower()=="online").astype(int))
                            .groupby("customer_id")["o"].mean())
    if "is_installment" in txn:
        tf["installment_ratio"]=g["is_installment"].mean()
    tf=tf.drop(columns=["last_trans_date"])
    fin=fin.copy()
    for col in ["total_deposit_balance","total_loan_balance","card_cash_service_amt","card_loan_amt"]:
        if col in fin: fin["log_"+col]=np.log1p(fin[col].clip(lower=0))
    df=(cust.merge(fin,on="customer_id",how="left").merge(tf.reset_index(),on="customer_id",how="left"))
    return df

df_tr=load_side("train"); df_te=load_side("test")
tgt=pd.read_csv(CONFIG["train_targets"]); df_tr=df_tr.merge(tgt,on="customer_id",how="left")

DROP={"customer_id","join_date","gender","region_code","income_group","is_married",
      "target_churn","target_ltv"}
# is_married_bin: 진단에서 무의미 확정 → 제거
DROP.add("is_married_bin")
full_cols=[c for c in df_tr.columns if c in df_te.columns
           and pd.api.types.is_numeric_dtype(df_tr[c]) and c not in DROP]

# 결측 처리(병합 NaN)
for d in (df_tr,df_te):
    for c in full_cols: d[c]=d[c].fillna(-999)

# --- 3종 피처셋 ---
# core: Churn 진짜신호 + 보조 (z>1 수준)
CORE=["card_loan_amt","log_card_loan_amt","total_deposit_balance","log_total_deposit_balance",
      "card_cash_service_amt","log_card_cash_service_amt","credit_score"]
CORE=[c for c in CORE if c in full_cols]
# denoised: full에서 명백 noise(거래량/인구통계 중 z<0) 다수 제거 → 신호+약신호만
NOISE_DROP=["age","tenure_days","trans_count","trans_amount_total","trans_amount_mean",
            "trans_amount_min","trans_amount_max","trans_amount_std","recency_days",
            "online_ratio","installment_ratio","num_active_cards","total_loan_balance",
            "log_total_loan_balance","fin_overdue_days","gender_enc"]
denoised=[c for c in full_cols if c not in NOISE_DROP]

FEATURESETS={"full":full_cols,"denoised":denoised,"core":CORE}
print("피처셋: " + " | ".join(f"{k}={len(v)}" for k,v in FEATURESETS.items()))
REPORT["featuresets"]={k:v for k,v in FEATURESETS.items()}

y=df_tr["target_churn"].values
pw=(len(y)-y.sum())/y.sum()
skf=StratifiedKFold(n_splits=CONFIG["n_folds"],shuffle=True,random_state=SEED)

def churn_models(seed):
    return {
        "lgb":lgb.LGBMClassifier(n_estimators=1200,learning_rate=0.02,num_leaves=31,
              subsample=0.8,colsample_bytree=0.8,scale_pos_weight=pw,
              reg_lambda=1.0,random_state=seed,n_jobs=-1,verbose=-1),
        "xgb":xgb.XGBClassifier(n_estimators=1200,learning_rate=0.02,max_depth=5,
              subsample=0.8,colsample_bytree=0.8,scale_pos_weight=pw,
              reg_lambda=1.0,random_state=seed,n_jobs=-1,eval_metric="auc",verbosity=0),
        "cat":CatBoostClassifier(iterations=1200,learning_rate=0.02,depth=6,
              l2_leaf_reg=3.0,scale_pos_weight=pw,random_seed=seed,verbose=0),
        "rf":RandomForestClassifier(n_estimators=600,max_depth=None,min_samples_leaf=20,
              max_features="sqrt",class_weight="balanced",random_state=seed,n_jobs=-1),
    }

# ======================================================================
# STEP 2 : 개별 모델 CV + 다양성(OOF 상관) — 피처셋별
# ======================================================================
section("[2/3] 피처셋별 개별 모델 CV + 스태킹/가중평균")
def get_oof(model_proto_key, Xdf, seed):
    oof=np.zeros(len(Xdf))
    for tr_i,va_i in skf.split(Xdf,y):
        mdl=churn_models(seed)[model_proto_key]
        mdl.fit(Xdf.iloc[tr_i],y[tr_i])
        oof[va_i]=mdl.predict_proba(Xdf.iloc[va_i])[:,1]
    return oof

results={}
for fs_name,cols in FEATURESETS.items():
    Xdf=df_tr[cols]
    oofs={}; aucs={}
    for mk in ["lgb","xgb","cat","rf"]:
        o=get_oof(mk,Xdf,SEED); oofs[mk]=o; aucs[mk]=roc_auc_score(y,o)
    # 다양성: OOF 예측 간 상관 (낮을수록 앙상블 이득)
    M=np.vstack([oofs[k] for k in ["lgb","xgb","cat","rf"]])
    corr=np.corrcoef(M)
    # 스태킹 (메타=LogReg, OOF에서만 학습)
    Z=np.vstack([oofs[k] for k in ["lgb","xgb","cat","rf"]]).T
    meta=LogisticRegression(max_iter=1000)
    stack_oof=np.zeros(len(y))
    for tr_i,va_i in skf.split(Z,y):
        meta.fit(Z[tr_i],y[tr_i]); stack_oof[va_i]=meta.predict_proba(Z[va_i])[:,1]
    stack_auc=roc_auc_score(y,stack_oof)
    # 가중평균(간단 그리드)
    best_w=None;best_wauc=0
    for wl in np.arange(0,1.01,0.25):
     for wx in np.arange(0,1.01-wl,0.25):
      for wc in np.arange(0,1.01-wl-wx,0.25):
        wr=1-wl-wx-wc
        if wr<0: continue
        blend=wl*oofs["lgb"]+wx*oofs["xgb"]+wc*oofs["cat"]+wr*oofs["rf"]
        a=roc_auc_score(y,blend)
        if a>best_wauc: best_wauc=a;best_w=(wl,wx,wc,wr)
    results[fs_name]={"individual_auc":{k:jsonable(v) for k,v in aucs.items()},
        "oof_corr":jsonable(corr),"stack_auc":jsonable(stack_auc),
        "blend_auc":jsonable(best_wauc),"blend_w":[jsonable(x) for x in best_w]}
    print(f"\n[{fs_name}] 개별 AUC: "+", ".join(f"{k}={v:.4f}" for k,v in aucs.items()))
    print(f"  OOF 상관(lgb-xgb-cat-rf):\n{np.round(corr,3)}")
    print(f"  스태킹 AUC={stack_auc:.4f} | 가중평균 AUC={best_wauc:.4f} w(l,x,c,r)={best_w}")

REPORT["2_3_churn"]=results
# 최고 조합 선택
best_fs=max(results,key=lambda k:max(results[k]["stack_auc"],results[k]["blend_auc"]))
best_auc=max(results[best_fs]["stack_auc"],results[best_fs]["blend_auc"])
print(f"\n>>> 최고: featureset='{best_fs}'  AUC={best_auc:.4f}")
REPORT["A_best_churn"]={"featureset":best_fs,"auc":jsonable(best_auc)}

# ======================================================================
# STEP 4 : LTV — cbrt 변환 (원본스케일 RMSE 최소화)
# ======================================================================
section("[4] LTV cbrt 변환 회귀 (원본스케일 RMSE)")
yl=df_tr["target_ltv"].values
kf=KFold(n_splits=CONFIG["n_folds"],shuffle=True,random_state=SEED)

def ltv_cv(transform_name):
    if transform_name=="log1p":
        yt=np.log1p(yl); inv=lambda p:np.expm1(p)
    else: # cbrt
        yt=np.cbrt(yl); inv=lambda p:np.power(p,3)
    Xl=df_tr[full_cols]; oof=np.zeros(len(yl))
    for tr_i,va_i in kf.split(Xl):
        m=lgb.LGBMRegressor(n_estimators=1200,learning_rate=0.02,num_leaves=31,
            subsample=0.8,colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbose=-1)
        m.fit(Xl.iloc[tr_i],yt[tr_i])
        oof[va_i]=inv(m.predict(Xl.iloc[va_i]))
    oof=np.clip(oof,0,None)
    rmse=np.sqrt(mean_squared_error(yl,oof))  # 원본 스케일
    return rmse,oof

rmse_log,_=ltv_cv("log1p")
rmse_cbrt,ltv_oof=ltv_cv("cbrt")
print(f"  원본스케일 RMSE  log1p={rmse_log:,.0f}  cbrt={rmse_cbrt:,.0f}")
best_ltv_rmse=min(rmse_log,rmse_cbrt)
REPORT["4_ltv"]={"rmse_log1p":jsonable(rmse_log),"rmse_cbrt":jsonable(rmse_cbrt),
                  "chosen":"cbrt" if rmse_cbrt<rmse_log else "log1p"}

# ======================================================================
# STEP 5 : 최종 Score + 제출 생성
# ======================================================================
section("[5] 최종 Score 계산 + 제출 파일")
def score_fn(auc,rmse): return 0.5*auc+0.5/(1+np.log10(rmse))
final_score=score_fn(best_auc,best_ltv_rmse)
base_score=score_fn(0.7971,1347000)  # 5주차 근사 기준
print(f"  목표공식 Score = 0.5*AUC + 0.5/(1+log10(RMSE))")
print(f"  개선 후 : AUC={best_auc:.4f} RMSE={best_ltv_rmse:,.0f} -> Score={final_score:.5f}")
print(f"  (참고 5주차 근사: Score≈{base_score:.5f})")
print(f"  목표 0.469 / 1위팀 0.47027")
REPORT["5_score"]={"final_auc":jsonable(best_auc),"final_rmse":jsonable(best_ltv_rmse),
    "final_score":jsonable(final_score),"target":0.469,"rank1":0.47027}

# 최종 모델 재학습(full train) → test 예측 → 제출
# Churn: best_fs 기준 스태킹
cols=FEATURESETS[best_fs]; Xtr=df_tr[cols]; Xte=df_te[cols]
test_oofs={}
for mk in ["lgb","xgb","cat","rf"]:
    # OOF로 메타학습용, 동시에 test 예측 평균
    test_pred=np.zeros(len(Xte)); oof=np.zeros(len(Xtr))
    for tr_i,va_i in skf.split(Xtr,y):
        m=churn_models(SEED)[mk]; m.fit(Xtr.iloc[tr_i],y[tr_i])
        oof[va_i]=m.predict_proba(Xtr.iloc[va_i])[:,1]
        test_pred+=m.predict_proba(Xte)[:,1]/CONFIG["n_folds"]
    test_oofs[mk]=(oof,test_pred)
Ztr=np.vstack([test_oofs[k][0] for k in ["lgb","xgb","cat","rf"]]).T
Zte=np.vstack([test_oofs[k][1] for k in ["lgb","xgb","cat","rf"]]).T
meta=LogisticRegression(max_iter=1000); meta.fit(Ztr,y)
churn_test=meta.predict_proba(Zte)[:,1]
# Isotonic 보정(평균을 실제 이탈률에 맞춤)
from sklearn.isotonic import IsotonicRegression
ir=IsotonicRegression(out_of_bounds="clip"); ir.fit(meta.predict_proba(Ztr)[:,1],y)
churn_test=np.clip(ir.predict(churn_test),0,1)

# LTV: cbrt 최종
yt=np.cbrt(yl); m=lgb.LGBMRegressor(n_estimators=1200,learning_rate=0.02,num_leaves=31,
    subsample=0.8,colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbose=-1)
m.fit(df_tr[full_cols],yt); ltv_test=np.clip(np.power(m.predict(df_te[full_cols]),3),0,None)

sub=pd.DataFrame({"customer_id":df_te["customer_id"],
                  "target_churn":churn_test,"target_ltv":ltv_test})
sub.to_csv(CONFIG["out_csv"],index=False)
print(f"  제출 파일 저장: {CONFIG['out_csv']}  (churn mean={churn_test.mean():.4f}, "
      f"ltv mean={ltv_test.mean():,.0f})")
REPORT["5_submission"]={"churn_mean":jsonable(churn_test.mean()),
    "ltv_mean":jsonable(ltv_test.mean()),"n_rows":int(len(sub))}

with open(CONFIG["out_json"],"w",encoding="utf-8") as f:
    json.dump(REPORT,f,ensure_ascii=False,indent=2,default=jsonable)
print(f"saved: {os.path.abspath(CONFIG['out_json'])}")
