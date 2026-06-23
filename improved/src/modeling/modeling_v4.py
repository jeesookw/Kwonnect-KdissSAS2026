# -*- coding: utf-8 -*-
"""
=====================================================================
 모델링 v4  (거래 피처 재설계 + 상호작용 + 검증내장)
---------------------------------------------------------------------
   1) 거래 피처를 조합/비율/세그먼트로 재설계 (RFM, HHI, recent/prev 비율 등)
   2) 금융 상호작용 피처 명시 생성 (credit×trend, loan/deposit 등)
   3) 신규 피처 전부 null importance로 즉시 검증 (real/ambiguous/noise)
   4) 피처셋 3종 실측 비교: full / denoised(v3) / full+신규
   5) 모델은 v3 베스트 파라미터 재사용 (피처 효과만 분리 측정)
 출력: modeling_v4_summary.json (+ submission_v4.csv)
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
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
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
    "ref_date":"2023-12-31","n_folds":5,"seed":SEED,"n_null":15,
    "bag_seeds":[42,202,7],
    "out_json":str(MODELING_RESULTS/"modeling_v4_summary.json"),
    "out_csv":str(SUBMISSIONS_DIR/"submission_v4.csv"),
}
# v3에서 찾은 베스트 파라미터 (재사용 — 피처 효과만 분리하기 위함)
BEST={
 "lgb":{"n_estimators":600,"learning_rate":0.00877,"num_leaves":11,"min_child_samples":46,
        "subsample":0.7623,"colsample_bytree":0.5057,"reg_lambda":3.5919},
 "xgb":{"n_estimators":541,"learning_rate":0.00828,"max_depth":4,"min_child_weight":4,
        "subsample":0.6385,"colsample_bytree":0.6157,"reg_lambda":3.4384},
 "cat":{"iterations":309,"learning_rate":0.02592,"depth":5,"l2_leaf_reg":2.9293},
}
REPORT={"_meta":{"script":"modeling_v4","generated_at":time.strftime("%Y-%m-%d %H:%M:%S")}}
def section(t): print("\n"+"="*68+"\n"+t+"\n"+"="*68)
def jsonable(o):
    if isinstance(o,(np.integer,)):return int(o)
    if isinstance(o,(np.floating,)):return round(float(o),6)
    if isinstance(o,(np.bool_,)):return bool(o)
    if isinstance(o,np.ndarray):return o.tolist()
    return o

# ======================================================================
# 1. 피처 생성 — 거래 재설계 + 금융 상호작용
# ======================================================================
section("[LOAD] 피처 생성 (거래 재설계 + 상호작용)")
ref=pd.to_datetime(CONFIG["ref_date"])

def build_txn_features(txn):
    txn=txn.copy()
    txn["trans_date"]=pd.to_datetime(txn["trans_date"])
    txn["amt"]=pd.to_numeric(txn["trans_amount"],errors="coerce")
    g=txn.groupby("customer_id")

    feat=pd.DataFrame({
        "trans_count":g["trans_id"].count(),
        "amt_total":g["amt"].sum(),"amt_mean":g["amt"].mean(),
        "amt_std":g["amt"].std(),"amt_min":g["amt"].min(),"amt_max":g["amt"].max(),
        "last_date":g["trans_date"].max(),"first_date":g["trans_date"].min(),
    })
    feat["recency_days"]=(ref-feat["last_date"]).dt.days
    feat["active_span"]=(feat["last_date"]-feat["first_date"]).dt.days+1
    feat["amt_cv"]=feat["amt_std"]/(feat["amt_mean"]+1)            # 변동계수
    feat["freq_per_active"]=feat["trans_count"]/feat["active_span"] # 활동밀도

    # --- 거래 간격 변동성 (이탈 신호 후보) ---
    def gap_stats(s):
        d=np.sort(s.values.astype("datetime64[D]")).astype(int)
        if len(d)<2: return pd.Series({"gap_mean":np.nan,"gap_std":np.nan,"gap_max":np.nan})
        diffs=np.diff(d)
        return pd.Series({"gap_mean":diffs.mean(),"gap_std":diffs.std(),"gap_max":diffs.max()})
    gaps=g["trans_date"].apply(gap_stats).unstack()
    feat=feat.join(gaps)
    feat["gap_cv"]=feat["gap_std"]/(feat["gap_mean"]+1)

    # --- recent3 vs prev3 비율 (트렌드) ---
    cut_recent=pd.to_datetime("2023-10-01"); cut_mid=pd.to_datetime("2023-07-01")
    r3=txn[txn["trans_date"]>=cut_recent].groupby("customer_id").agg(
        r3_cnt=("trans_id","count"),r3_amt=("amt","sum"))
    p3=txn[(txn["trans_date"]<cut_recent)].groupby("customer_id").agg(
        p3_cnt=("trans_id","count"),p3_amt=("amt","sum"))
    feat=feat.join(r3).join(p3)
    feat["cnt_trend_ratio"]=(feat["r3_cnt"]+1)/(feat["p3_cnt"]+1)
    feat["amt_trend_ratio"]=(feat["r3_amt"]+1)/(feat["p3_amt"]+1)

    # --- 채널 (online 비율 + recent 채널 전환) ---
    if "biz_type" in txn.columns:
        txn["is_online"]=(txn["biz_type"].astype(str).str.lower()=="online").astype(int)
        feat["online_ratio"]=g["is_online"].mean()
        on_r=txn[txn["trans_date"]>=cut_recent].groupby("customer_id")["is_online"].mean()
        feat["online_ratio_recent"]=on_r
        feat["channel_shift"]=feat["online_ratio_recent"]-feat["online_ratio"]

    # --- 카테고리 집중도 HHI + 다양성 ---
    if "item_category" in txn.columns:
        piv=txn.pivot_table(index="customer_id",columns="item_category",
                            values="amt",aggfunc="sum",fill_value=0)
        share=piv.div(piv.sum(axis=1)+1e-9,axis=0)
        feat["cat_hhi"]=(share**2).sum(axis=1)            # 1=한 카테고리 집중
        feat["cat_nunique"]=(piv>0).sum(axis=1)            # 이용 카테고리 수
        feat["top_cat_share"]=share.max(axis=1)            # 최다 카테고리 비중

    # --- 할부 ---
    if "is_installment" in txn.columns:
        feat["installment_ratio"]=g["is_installment"].mean()

    feat=feat.drop(columns=["last_date","first_date","amt_std","r3_cnt","r3_amt",
                            "p3_cnt","p3_amt","gap_std"],errors="ignore")
    return feat.reset_index()

def load_side(side):
    cust=pd.read_csv(CONFIG[f"{side}_customer"]); fin=pd.read_csv(CONFIG[f"{side}_finance"])
    txn=pd.read_csv(CONFIG[f"{side}_transaction"])
    cust["join_date"]=pd.to_datetime(cust["join_date"])
    cust["tenure_days"]=(ref-cust["join_date"]).dt.days
    for col in ["region_code","income_group"]:
        if col in cust: cust[col+"_enc"]=cust[col].astype(str).str.extract(r"(\d+)").astype(float)
    if "age" in cust: cust["age"]=cust["age"]

    # 금융 + 로그 + 상호작용
    for col in ["total_deposit_balance","total_loan_balance","card_cash_service_amt","card_loan_amt"]:
        if col in fin: fin["log_"+col]=np.log1p(fin[col].clip(lower=0))
    # 상호작용 (금융 핵심 신호 곱/비)
    eps=1.0
    fin["loan_to_deposit"]=fin["total_loan_balance"]/(fin["total_deposit_balance"]+eps)
    fin["cardloan_to_deposit"]=fin["card_loan_amt"]/(fin["total_deposit_balance"]+eps)
    fin["log_loan_to_deposit"]=np.log1p(fin["loan_to_deposit"].clip(lower=0))
    if "fin_asset_trend_score" in fin and "credit_score" in fin:
        fin["credit_x_trend"]=fin["credit_score"]*fin["fin_asset_trend_score"]
    if "card_loan_amt" in fin and "card_cash_service_amt" in fin:
        fin["card_debt_total"]=fin["card_loan_amt"]+fin["card_cash_service_amt"]
        fin["log_card_debt_total"]=np.log1p(fin["card_debt_total"].clip(lower=0))

    txn_feat=build_txn_features(txn)
    df=(cust.merge(fin,on="customer_id",how="left").merge(txn_feat,on="customer_id",how="left"))
    return df

df_tr=load_side("train"); df_te=load_side("test")
tgt=pd.read_csv(CONFIG["train_targets"]); df_tr=df_tr.merge(tgt,on="customer_id",how="left")

DROP={"customer_id","join_date","gender","region_code","income_group","is_married",
      "target_churn","target_ltv","trans_amount"}
all_cols=[c for c in df_tr.columns if c in df_te.columns
          and pd.api.types.is_numeric_dtype(df_tr[c]) and c not in DROP]
for d in (df_tr,df_te):
    for c in all_cols: d[c]=d[c].fillna(-999)
print(f"전체 피처 {len(all_cols)}개")

# 피처 그룹
DENOISED=["region_code_enc","income_group_enc","credit_score","total_deposit_balance",
          "card_cash_service_amt","card_loan_amt","fin_asset_trend_score",
          "log_total_deposit_balance","log_card_cash_service_amt","log_card_loan_amt"]
DENOISED=[c for c in DENOISED if c in all_cols]
NEW_FEATURES=[c for c in all_cols if c in [
    "amt_cv","freq_per_active","gap_mean","gap_max","gap_cv","cnt_trend_ratio",
    "amt_trend_ratio","online_ratio_recent","channel_shift","cat_hhi","cat_nunique",
    "top_cat_share","loan_to_deposit","cardloan_to_deposit","log_loan_to_deposit",
    "credit_x_trend","card_debt_total","log_card_debt_total","active_span"]]
print(f"denoised {len(DENOISED)} | 신규 후보 {len(NEW_FEATURES)}")
REPORT["featuresets"]={"all":all_cols,"denoised":DENOISED,"new_candidates":NEW_FEATURES}

y=df_tr["target_churn"].values; pw=(len(y)-y.sum())/y.sum()
skf=StratifiedKFold(n_splits=CONFIG["n_folds"],shuffle=True,random_state=SEED)

# ======================================================================
# STEP1 : 신규 피처 null importance 검증
# ======================================================================
section("[STEP1] 신규 피처 null importance 검증")
X_all=df_tr[all_cols]
def imp(Xz,yz,seed):
    m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,
        min_child_samples=40,scale_pos_weight=pw,random_state=seed,n_jobs=-1,verbose=-1)
    m.fit(Xz,yz); return m.feature_importances_
real=imp(X_all,y,SEED)
nulls=np.zeros((CONFIG["n_null"],len(all_cols)))
for i in range(CONFIG["n_null"]):
    nulls[i]=imp(X_all,np.random.permutation(y),3000+i)
nm=nulls.mean(0); ns=nulls.std(0)+1e-9; z=(real-nm)/ns
zmap=dict(zip(all_cols,z))
print("신규 피처 검증 결과 (z>3=진짜신호):")
new_signal=[]
for c in NEW_FEATURES:
    zz=zmap[c]; v="진짜신호" if zz>3 else("애매" if zz>1 else "noise")
    if zz>1: new_signal.append(c)
    print(f"  {c:24s} z={zz:6.2f} [{v}]")
REPORT["step1_new_feature_validation"]=[
    {"feature":c,"z_score":jsonable(zmap[c]),
     "verdict":("real_signal" if zmap[c]>3 else "ambiguous" if zmap[c]>1 else "noise")}
    for c in NEW_FEATURES]
print(f"\n  z>1 살아남은 신규 피처: {new_signal}")

# ======================================================================
# STEP2 : 피처셋 3종 CV 비교 (LGB 단독, 빠른 비교)
# ======================================================================
section("[STEP2] 피처셋 3종 CV 비교")
DENOISED_PLUS=DENOISED+new_signal   # denoised + 살아남은 신규
SETS={"all":all_cols,"denoised":DENOISED,"denoised+new":DENOISED_PLUS}
def quick_cv(cols):
    Xc=df_tr[cols]; oof=np.zeros(len(Xc))
    for tr_i,va_i in skf.split(Xc,y):
        m=lgb.LGBMClassifier(**BEST["lgb"],scale_pos_weight=pw,random_state=SEED,n_jobs=-1,verbose=-1)
        m.fit(Xc.iloc[tr_i],y[tr_i]); oof[va_i]=m.predict_proba(Xc.iloc[va_i])[:,1]
    return roc_auc_score(y,oof)
set_auc={}
for name,cols in SETS.items():
    a=quick_cv(cols); set_auc[name]=a
    print(f"  {name:16s} ({len(cols):2d}개) LGB CV AUC = {a:.4f}")
best_set=max(set_auc,key=set_auc.get)
print(f"  >>> 최고 피처셋: {best_set} (AUC {set_auc[best_set]:.4f})")
REPORT["step2_featureset_cv"]={k:jsonable(v) for k,v in set_auc.items()}
REPORT["step2_best_set"]=best_set
FINAL_COLS=SETS[best_set]

# ======================================================================
# STEP3 : 3모델 시드배깅 앙상블 (best 피처셋)
# ======================================================================
section(f"[STEP3] 앙상블 ({best_set}, {len(FINAL_COLS)}개 피처)")
X=df_tr[FINAL_COLS]; Xte=df_te[FINAL_COLS]
def build(name,seed):
    if name=="lgb": return lgb.LGBMClassifier(**BEST["lgb"],scale_pos_weight=pw,random_state=seed,n_jobs=-1,verbose=-1)
    if name=="xgb": return xgb.XGBClassifier(**BEST["xgb"],scale_pos_weight=pw,random_state=seed,n_jobs=-1,eval_metric="auc",verbosity=0)
    if name=="cat": return CatBoostClassifier(**BEST["cat"],scale_pos_weight=pw,random_seed=seed,verbose=0)
oofs={}; tps={}
for mk in ["lgb","xgb","cat"]:
    oof=np.zeros(len(X)); tp=np.zeros(len(Xte))
    for sd in CONFIG["bag_seeds"]:
        sk=StratifiedKFold(n_splits=CONFIG["n_folds"],shuffle=True,random_state=sd)
        for tr_i,va_i in sk.split(X,y):
            m=build(mk,sd); m.fit(X.iloc[tr_i],y[tr_i])
            oof[va_i]+=m.predict_proba(X.iloc[va_i])[:,1]/len(CONFIG["bag_seeds"])
            tp+=m.predict_proba(Xte)[:,1]/(CONFIG["n_folds"]*len(CONFIG["bag_seeds"]))
    oofs[mk]=oof; tps[mk]=tp
    print(f"  {mk}: AUC={roc_auc_score(y,oof):.4f}")
# 가중평균 그리드
best_w=None;best_a=0
for wl in np.arange(0,1.01,0.05):
 for wx in np.arange(0,1.01-wl,0.05):
    wc=round(1-wl-wx,3)
    if wc<-1e-9: continue
    a=roc_auc_score(y,wl*oofs["lgb"]+wx*oofs["xgb"]+wc*oofs["cat"])
    if a>best_a: best_a=a;best_w=(round(wl,3),round(wx,3),wc)
churn_oof=best_w[0]*oofs["lgb"]+best_w[1]*oofs["xgb"]+best_w[2]*oofs["cat"]
churn_te=best_w[0]*tps["lgb"]+best_w[1]*tps["xgb"]+best_w[2]*tps["cat"]
print(f"  가중평균 AUC={best_a:.4f} w(l,x,c)={best_w}")
ir=IsotonicRegression(out_of_bounds="clip"); ir.fit(churn_oof,y)
churn_te=np.clip(ir.predict(churn_te),0,1)
REPORT["step3_ensemble"]={"individual":{k:jsonable(roc_auc_score(y,oofs[k])) for k in oofs},
    "blend_auc":jsonable(best_a),"blend_w":[jsonable(x) for x in best_w]}

# ======================================================================
# STEP4 : LTV(cbrt) + 최종 Score
# ======================================================================
section("[STEP4] LTV + 최종 Score")
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
fs=score_fn(best_a,ltv_rmse)
print(f"  AUC={best_a:.4f} RMSE={ltv_rmse:,.0f} Score={fs:.5f}")
print(f"  v3: AUC 0.79817 Score 0.46901 | 24위 0.798036 | 10위 0.799111")
REPORT["step4_final"]={"auc":jsonable(best_a),"rmse":jsonable(ltv_rmse),"score":jsonable(fs),
    "v3_auc":0.79817,"v3_score":0.46901}

sub=pd.DataFrame({"customer_id":df_te["customer_id"],"target_churn":churn_te,"target_ltv":ltv_te})
sub.to_csv(CONFIG["out_csv"],index=False)
print(f"\n  제출: churn mean={churn_te.mean():.4f} ltv mean={ltv_te.mean():,.0f} rows={len(sub)}")
with open(CONFIG["out_json"],"w",encoding="utf-8") as f:
    json.dump(REPORT,f,ensure_ascii=False,indent=2,default=jsonable)
print(f"saved: {os.path.abspath(CONFIG['out_json'])}")
