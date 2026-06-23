# -*- coding: utf-8 -*-
"""
features.py — 피처 엔지니어링 함수 모음

diagnostics/ 와 src/modeling/ 의 각 스크립트가 자체적으로 load_side()를 포함하고 있어 단독 실행이 가능하도록 설계됨
이 모듈은 그 공통 로직을 한 곳에 모아 유지보수하기 위한 참조 구현

핵심 설계 원칙(진단 결론 반영):
  - 진단에서 거래 피처 31종이 전부 noise로 확정 → 거래 집계 최소화
  - 진짜 신호는 금융 3종(card_loan_amt, total_deposit_balance, card_cash_service_amt) + 로그 변환
  - is_married 는 무의미 피처로 확정 → 기본 제외
"""
import numpy as np
import pandas as pd

REF_DATE = "2023-12-31"

# 검증을 통과한 핵심 피처셋 (modeling_v3/v4에서 최고 성능)
DENOISED_FEATURES = [
    "region_code_enc", "income_group_enc", "credit_score",
    "total_deposit_balance", "card_cash_service_amt", "card_loan_amt",
    "fin_asset_trend_score", "log_total_deposit_balance",
    "log_card_cash_service_amt", "log_card_loan_amt",
]


def build_features(customer_csv, finance_csv, transaction_csv, ref_date=REF_DATE):
    cust = pd.read_csv(customer_csv)
    fin = pd.read_csv(finance_csv)
    txn = pd.read_csv(transaction_csv)

    cust["join_date"] = pd.to_datetime(cust["join_date"])
    txn["trans_date"] = pd.to_datetime(txn["trans_date"])
    ref = pd.to_datetime(ref_date)

    cust["tenure_days"] = (ref - cust["join_date"]).dt.days
    for col in ["region_code", "income_group"]:
        if col in cust:
            cust[col + "_enc"] = cust[col].astype(str).str.extract(r"(\d+)").astype(float)

    # 금융 로그 변환 (진짜 신호)
    for col in ["total_deposit_balance", "total_loan_balance",
                "card_cash_service_amt", "card_loan_amt"]:
        if col in fin:
            fin["log_" + col] = np.log1p(fin[col].clip(lower=0))

    # 거래 최소 집계 (참조용 — importance 하위 확정)
    g = txn.groupby("customer_id")
    tf = pd.DataFrame({
        "trans_count": g["trans_id"].count(),
        "trans_amount_total": g["trans_amount"].sum(),
        "trans_amount_mean": g["trans_amount"].mean(),
    })

    df = (cust.merge(fin, on="customer_id", how="left")
              .merge(tf.reset_index(), on="customer_id", how="left"))
    return df


def cbrt_transform(y):
    return np.cbrt(y)


def inverse_cbrt(pred):
    return np.clip(np.power(pred, 3), 0, None)
