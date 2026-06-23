# -*- coding: utf-8 -*-
"""
=====================================================================
 모델링 v3  (다양성주입 폐기 + Optuna확대+RF + 시드배깅)
---------------------------------------------------------------------
 v2 결과 반영:
   - 다양성 강제주입 실패(상관 안 내려감, XGB 0.796->0.788 역효과) → 피처 서브셋 폐기
   - Optuna가 개선 주동력(LGB +0.012) → trial 확대 + RF도 튜닝
   - 스태킹 < 가중평균 → 가중평균 기본, 스태킹은 비교만
   - cbrt LTV 유지. (점수 기여 작아 손대지 않음)
 목표: AUC 0.7973 -> 0.798+, Score 0.46860 -> 0.469+
 시간통제: 튜닝은 단일시드, 배깅은 최종예측에서만
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
SEED=42; np.random.seed(SEED)

CONFIG={
    "train_customer":str(DATA_RAW/"train_customer_info.csv"),
    "train_transaction":str(DATA_RAW/"train_transaction_history.csv"),
    "train_finance":str(DATA_RAW/"train_finance_profile.csv"),
    "train_targets":str(DATA_RAW/"train_targets.csv"),
    "test_customer":str(DATA_RAW/"test_customer_info.csv"),
    "test_transaction":str(DATA_RAW/"test_transaction_history.csv"),
    "test_finance":str(DATA_RAW/"test_finance_profile.csv"),
    "ref_date":"2023-12-31","n_folds":5,"seed":SEED,
    "optuna_trials":60,            # 시간 빠듯하면 40
    "rf_trials":20,                # RF는 좁게
    "bag_seeds":[42,202,7,99,1234],# 최종 예측 시드배깅
    "out_json":str(MODELING_RESULTS/"modeling_v3_summary.json"),
    "out_csv":str(SUBMISSIONS_DIR/"submission_v3.csv"),
}
REPORT={"_meta":{"script":"modeling_v3","generated_at":time.strftime("%Y-%m-%d %H:%M:%S")}}
def section(t): print("\n"+"="*68+"\n"+t+"\n"+"="*68)
def jsonable(o):
    if isinstance(o,(np.integer,)):return int(o)
    if isinstance(o,(np.floating,)):return round(float(o),6)
    if isinstance(o,(np.bool_,)):return bool(o)
    if isinstance(o,np.ndarray):return o.tolist()
    return o

# ----------------------------------------------------------------------
section("[LOAD] 피처 생성")
def load_side(side):
    cust=pd.read_csv(CONFIG[f"{side}_customer"]); fin=pd.read_csv(CONFIG[f"{side}_finance"])
    txn=pd.read_csv(CONFIG[f"{side}_transaction"])
    cust["join_date"]=pd.to_datetime(cust["join_date"]); txn["trans_date"]=pd.to_datetime(txn["trans_date"])
    ref=pd.to_datetime(CONFIG["ref_date"])
    cust["tenure_days"]=(ref-cust["join_date"]).dt.days
    for col in ["region_code","income_group"]:
        if col in cust: cust[col+"_enc"]=cust[col].astype(str).str.extract(r"(\d+)").astype(float)
    g=txn.groupby("customer_id")
    tf=pd.DataFrame({"trans_count":g["trans_id"].count(),
        "trans_amount_total":g["trans_amount"].sum(),"trans_amount_mean":g["trans_amount"].mean(),
        "trans_amount_min":g["trans_amount"].min(),"last_trans_date":g["trans_date"].max()})
    tf["recency_days"]=(ref-tf["last_trans_date"]).dt.days
    if "biz_type" in txn:
        tf["online_ratio"]=(txn.assign(o=(txn["biz_type"].astype(str).str.lower()=="online").astype(int))
                            .groupby("customer_id")["o"].mean())
    tf=tf.drop(columns=["last_trans_date"])
    for col in ["total_deposit_balance","total_loan_balance","card_cash_service_amt","card_loan_amt"]:
        if col in fin: fin["log_"+col]=np.log1p(fin[col].clip(lower=0))
    return (cust.merge(fin,on="customer_id",how="left").merge(tf.reset_index(),on="customer_id",how="left"))

df_tr=load_side("train"); df_te=load_side("test")
tgt=pd.read_csv(CONFIG["train_targets"]); df_tr=df_tr.merge(tgt,on="customer_id",how="left")

DENOISED=["region_code_enc","income_group_enc","credit_score",
          "total_deposit_balance","card_cash_service_amt","card_loan_amt",
          "fin_asset_trend_score","log_total_deposit_balance",
          "log_card_cash_service_amt","log_card_loan_amt"]
DENOISED=[c for c in DENOISED if c in df_tr.columns and c in df_te.columns]
for d in (df_tr,df_te):
    for c in DENOISED: d[c]=d[c].fillna(-999)
print(f"피처 {len(DENOISED)}개 (전체 공유): {DENOISED}")
REPORT["featureset"]=DENOISED

y=df_tr["target_churn"].values; pw=(len(y)-y.sum())/y.sum()
skf=StratifiedKFold(n_splits=CONFIG["n_folds"],shuffle=True,random_state=SEED)
X=df_tr[DENOISED]; Xte=df_te[DENOISED]

# ======================================================================
# STEP1 : Optuna 튜닝 (LGB/XGB/CAT/RF) — 전체 피처, 단일 시드
# ======================================================================
section("[STEP1] Optuna 튜닝 (전체 피처 공유)")

def cv_auc(builder):
    oof=np.zeros(len(X))
    for tr_i,va_i in skf.split(X,y):
        m=builder(SEED); m.fit(X.iloc[tr_i],y[tr_i])
        oof[va_i]=m.predict_proba(X.iloc[va_i])[:,1]
    return roc_auc_score(y,oof)

def obj_lgb(t):
    p=dict(n_estimators=t.suggest_int("n_estimators",400,1000),
        learning_rate=t.suggest_float("learning_rate",0.008,0.05,log=True),
        num_leaves=t.suggest_int("num_leaves",10,48),
        min_child_samples=t.suggest_int("min_child_samples",20,100),
        subsample=t.suggest_float("subsample",0.6,1.0),
        colsample_bytree=t.suggest_float("colsample_bytree",0.5,1.0),
        reg_lambda=t.suggest_float("reg_lambda",0.0,6.0))
    return cv_auc(lambda s:lgb.LGBMClassifier(**p,scale_pos_weight=pw,random_state=s,n_jobs=-1,verbose=-1))

def obj_xgb(t):
    p=dict(n_estimators=t.suggest_int("n_estimators",400,1000),
        learning_rate=t.suggest_float("learning_rate",0.008,0.05,log=True),
        max_depth=t.suggest_int("max_depth",3,7),
        min_child_weight=t.suggest_int("min_child_weight",1,12),
        subsample=t.suggest_float("subsample",0.6,1.0),
        colsample_bytree=t.suggest_float("colsample_bytree",0.6,1.0),
        reg_lambda=t.suggest_float("reg_lambda",0.0,6.0))
    return cv_auc(lambda s:xgb.XGBClassifier(**p,scale_pos_weight=pw,random_state=s,
        n_jobs=-1,eval_metric="auc",verbosity=0))

def obj_cat(t):
    p=dict(iterations=t.suggest_int("iterations",300,800),
        learning_rate=t.suggest_float("learning_rate",0.01,0.05,log=True),
        depth=t.suggest_int("depth",3,7),
        l2_leaf_reg=t.suggest_float("l2_leaf_reg",1.0,10.0))
    return cv_auc(lambda s:CatBoostClassifier(**p,scale_pos_weight=pw,random_seed=s,verbose=0))

def obj_rf(t):
    p=dict(n_estimators=t.suggest_int("n_estimators",300,600),
        max_depth=t.suggest_int("max_depth",8,20),
        min_samples_leaf=t.suggest_int("min_samples_leaf",5,40))
    return cv_auc(lambda s:RandomForestClassifier(**p,max_features="sqrt",
        class_weight="balanced",random_state=s,n_jobs=-1))

best={}
for name,obj,nt in [("lgb",obj_lgb,CONFIG["optuna_trials"]),
                    ("xgb",obj_xgb,CONFIG["optuna_trials"]),
                    ("cat",obj_cat,CONFIG["optuna_trials"]),
                    ("rf",obj_rf,CONFIG["rf_trials"])]:
    st=optuna.create_study(direction="maximize",
                           sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj,n_trials=nt,show_progress_bar=False)
    best[name]=st.best_params
    print(f"  {name}: best CV AUC={st.best_value:.4f}  (trials={nt})")
    REPORT.setdefault("step1_optuna",{})[name]={"auc":jsonable(st.best_value),"params":st.best_params}

def build(name,seed):
    if name=="lgb": return lgb.LGBMClassifier(**best["lgb"],scale_pos_weight=pw,random_state=seed,n_jobs=-1,verbose=-1)
    if name=="xgb": return xgb.XGBClassifier(**best["xgb"],scale_pos_weight=pw,random_state=seed,n_jobs=-1,eval_metric="auc",verbosity=0)
    if name=="cat": return CatBoostClassifier(**best["cat"],scale_pos_weight=pw,random_seed=seed,verbose=0)
    if name=="rf":  return RandomForestClassifier(**best["rf"],max_features="sqrt",class_weight="balanced",random_state=seed,n_jobs=-1)

# ======================================================================
# STEP2 : 시드배깅 OOF + test 예측 (전체 피처 공유)
# ======================================================================
section("[STEP2] 시드배깅 OOF/test 생성")
oofs={}; test_preds={}
for mk in ["lgb","xgb","cat","rf"]:
    oof=np.zeros(len(X)); tpred=np.zeros(len(Xte))
    for sd in CONFIG["bag_seeds"]:
        skf_s=StratifiedKFold(n_splits=CONFIG["n_folds"],shuffle=True,random_state=sd)
        for tr_i,va_i in skf_s.split(X,y):
            m=build(mk,sd); m.fit(X.iloc[tr_i],y[tr_i])
            oof[va_i]+=m.predict_proba(X.iloc[va_i])[:,1]/len(CONFIG["bag_seeds"])
            tpred+=m.predict_proba(Xte)[:,1]/(CONFIG["n_folds"]*len(CONFIG["bag_seeds"]))
    oofs[mk]=oof; test_preds[mk]=tpred
    print(f"  {mk}: 배깅 OOF AUC={roc_auc_score(y,oof):.4f}")

M=np.vstack([oofs[k] for k in ["lgb","xgb","cat","rf"]])
corr=np.corrcoef(M)
print(f"\n  OOF 상관:\n{np.round(corr,3)}")
REPORT["step2_diversity"]={"oof_corr":jsonable(corr),
    "individual_auc":{k:jsonable(roc_auc_score(y,oofs[k])) for k in oofs}}

# ======================================================================
# STEP3 : 가중평균(기본) + 스태킹(비교) 자동선택
# ======================================================================
section("[STEP3] 앙상블 결합")
Z=np.vstack([oofs[k] for k in ["lgb","xgb","cat","rf"]]).T
Zte=np.vstack([test_preds[k] for k in ["lgb","xgb","cat","rf"]]).T

# 가중평균 그리드 (0.05 간격, 정밀)
best_w=None; best_wauc=0
grid=np.arange(0,1.01,0.05)
for wl in grid:
 for wx in grid:
  if wl+wx>1: break
  for wc in grid:
    wr=round(1-wl-wx-wc,3)
    if wr<-1e-9 or wr>1: continue
    b=wl*oofs["lgb"]+wx*oofs["xgb"]+wc*oofs["cat"]+wr*oofs["rf"]
    a=roc_auc_score(y,b)
    if a>best_wauc: best_wauc=a; best_w=(round(wl,3),round(wx,3),round(wc,3),wr)
blend_te=(best_w[0]*test_preds["lgb"]+best_w[1]*test_preds["xgb"]
          +best_w[2]*test_preds["cat"]+best_w[3]*test_preds["rf"])

# 스태킹
stack_oof=np.zeros(len(y))
for tr_i,va_i in skf.split(Z,y):
    meta=LogisticRegression(max_iter=1000); meta.fit(Z[tr_i],y[tr_i])
    stack_oof[va_i]=meta.predict_proba(Z[va_i])[:,1]
meta_full=LogisticRegression(max_iter=1000); meta_full.fit(Z,y)
stack_te=meta_full.predict_proba(Zte)[:,1]
stack_auc=roc_auc_score(y,stack_oof)
print(f"  가중평균 AUC={best_wauc:.4f} w(l,x,c,r)={best_w} | 스태킹 AUC={stack_auc:.4f}")

if best_wauc>=stack_auc:
    churn_oof=(best_w[0]*oofs["lgb"]+best_w[1]*oofs["xgb"]+best_w[2]*oofs["cat"]+best_w[3]*oofs["rf"])
    churn_te=blend_te; final_auc=best_wauc; how="blend"
else:
    churn_oof=stack_oof; churn_te=stack_te; final_auc=stack_auc; how="stack"
print(f"  >>> 채택: {how}  AUC={final_auc:.4f}")
REPORT["step3_ensemble"]={"blend_auc":jsonable(best_wauc),"blend_w":[jsonable(x) for x in best_w],
    "stack_auc":jsonable(stack_auc),"chosen":how,"final_auc":jsonable(final_auc)}

# Isotonic 보정
ir=IsotonicRegression(out_of_bounds="clip"); ir.fit(churn_oof,y)
churn_te=np.clip(ir.predict(churn_te),0,1)

# ======================================================================
# STEP4 : LTV(cbrt) + 최종 Score
# ======================================================================
section("[STEP4] LTV(cbrt) + 최종 Score")
yl=df_tr["target_ltv"].values; kf=KFold(5,shuffle=True,random_state=SEED); yt=np.cbrt(yl)
ltv_oof=np.zeros(len(yl))
for tr_i,va_i in kf.split(X):
    m=lgb.LGBMRegressor(n_estimators=800,learning_rate=0.02,num_leaves=31,subsample=0.8,
        colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbose=-1)
    m.fit(X.iloc[tr_i],yt[tr_i]); ltv_oof[va_i]=np.clip(m.predict(X.iloc[va_i])**3,0,None)
ltv_rmse=np.sqrt(mean_squared_error(yl,ltv_oof))
m=lgb.LGBMRegressor(n_estimators=800,learning_rate=0.02,num_leaves=31,subsample=0.8,
    colsample_bytree=0.8,random_state=SEED,n_jobs=-1,verbose=-1)
m.fit(X,yt); ltv_te=np.clip(m.predict(Xte)**3,0,None)

def score_fn(a,r): return 0.5*a+0.5/(1+np.log10(r))
final_score=score_fn(final_auc,ltv_rmse)
print(f"  AUC={final_auc:.4f}  RMSE={ltv_rmse:,.0f}  Score={final_score:.5f}")
print(f"  v2: AUC 0.7973 Score 0.46860 | 목표 0.469 | 1위 0.47027")
REPORT["step4_final"]={"auc":jsonable(final_auc),"rmse":jsonable(ltv_rmse),
    "score":jsonable(final_score),"v2_score":0.46860,"target":0.469}

sub=pd.DataFrame({"customer_id":df_te["customer_id"],"target_churn":churn_te,"target_ltv":ltv_te})
sub.to_csv(CONFIG["out_csv"],index=False)
print(f"\n  제출: churn mean={churn_te.mean():.4f} ltv mean={ltv_te.mean():,.0f}")
with open(CONFIG["out_json"],"w",encoding="utf-8") as f:
    json.dump(REPORT,f,ensure_ascii=False,indent=2,default=jsonable)
print(f"saved: {os.path.abspath(CONFIG['out_json'])}")
