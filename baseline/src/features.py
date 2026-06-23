"""
집계·병합 완료된 train.csv / test.csv 를 받아 추가 피처 엔지니어링 + 인코딩

입력: 병합 완료된 DataFrame
출력: 모델 학습에 바로 투입 가능한 DataFrame

수정 이력:
- prefer_category 이중 처리 제거 (features.py에서 인코딩 후 drop으로 통일)
- target_ltv_log 생성 코드 제거 (train.py에서 직접 변환으로 통일)
- total_deposit_balance 캡핑 추가 (train 기준 상위 1% cap 값을 파라미터로 받음)
- [5주차] 신규 파생 피처 추가: card_debt_total, credit_x_trend
  (EDA 검증 완료 — LTV Spearman r=-0.08/+0.07, Churn 유의)
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. 인코딩
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 성별: F=0, M=1
    if "gender" in df.columns:
        df["gender_enc"] = (df["gender"] == "M").astype(int)
        df = df.drop(columns=["gender"])

    # 소득 구간 순서형 인코딩 (G1=1 ~ G5=5)
    if "income_group" in df.columns:
        income_map = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
        df["income_group_enc"] = df["income_group"].map(income_map)
        df = df.drop(columns=["income_group"])

    # 지역 코드: R01~R07 → 1~7
    if "region_code" in df.columns:
        region_map = {f"R0{i}": i for i in range(1, 8)}
        df["region_enc"] = df["region_code"].map(region_map)
        df = df.drop(columns=["region_code"])

    # 선호 카테고리: 레이블 인코딩 후 drop
    # ※ train.py DROP_COLS에서 prefer_category 제거했으므로 여기서만 처리
    # prefer_category 블록 교체
    if "prefer_category" in df.columns:
        df = df.drop(columns=["prefer_category"])  # enc 생성 없이 drop만

    # 최다 구매 카테고리: 레이블 인코딩
    if "most_purchased_category" in df.columns:
        cat_map = {"Beauty": 0, "Electronics": 1, "Fashion": 2,
                   "Grocery": 3, "Home": 4}
        df["most_purchased_category"] = df["most_purchased_category"].map(cat_map)

    return df


# ─────────────────────────────────────────────
# 2. finance_profile 파생 피처
# ─────────────────────────────────────────────
"""
total_deposit_balance 등 4개 컬럼 로그 변환 + 이진 피처 생성.

Parameters
----------
df: 병합 완료된 DataFrame
deposit_cap: total_deposit_balance 캡핑 기준값.
            train에서 계산한 상위 1% 값을 넘겨받아 train/test 동일하게 적용 (데이터 누수 방지).
            None이면 캡핑 미적용.
"""
def build_finance_features(df: pd.DataFrame,
                           deposit_cap: float = None) -> pd.DataFrame:
    df = df.copy()

    # total_deposit_balance 캡핑 (train 기준 상위 1%)
    if deposit_cap is not None and "total_deposit_balance" in df.columns:
        df["total_deposit_balance"] = df["total_deposit_balance"].clip(
            upper=deposit_cap)

    # 로그 변환
    for col in ["total_deposit_balance", "total_loan_balance",
                "card_cash_service_amt", "card_loan_amt", "loan_to_deposit_ratio"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # 이진 피처
    if "total_loan_balance" in df.columns:
        df["has_loan"] = (df["total_loan_balance"] > 0).astype(int)
    if "card_cash_service_amt" in df.columns:
        df["has_cash_service"] = (df["card_cash_service_amt"] > 0).astype(int)
    if "card_loan_amt" in df.columns:
        df["has_card_loan"] = (df["card_loan_amt"] > 0).astype(int)

    # ── [5주차 추가] 신규 파생 피처 ──────────────────────────────
    # card_debt_total: 카드 부채 합산 (LTV r=-0.08, Churn 유의)
    # 원본 컬럼 drop 이전에 계산해야 하므로 이 위치에 배치
    if {"card_cash_service_amt", "card_loan_amt"}.issubset(df.columns):
        df["card_debt_total"] = (
            df["card_cash_service_amt"].fillna(0) +
            df["card_loan_amt"].fillna(0)
        )

    # credit_x_trend: 신용점수 × 자산트렌드 상호작용 (LTV r=0.07, Churn 유의)
    if {"credit_score", "fin_asset_trend_score"}.issubset(df.columns):
        df["credit_x_trend"] = (
            df["credit_score"].fillna(df["credit_score"].median()) *
            df["fin_asset_trend_score"].fillna(0)
        )
    # ──────────────────────────────────────────────────────────────

    # 원본 고액 컬럼 drop
    drop_cols = [c for c in [
        "total_deposit_balance", "total_loan_balance",
        "card_cash_service_amt", "card_loan_amt", "loan_to_deposit_ratio"
    ] if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


# ─────────────────────────────────────────────
# 3. 전체 파이프라인
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   is_train: bool = True,
                   deposit_cap: float = None) -> pd.DataFrame:
    df = df.copy()

    # 1) 인코딩
    df = encode_categoricals(df)

    # 2) finance 파생 피처
    df = build_finance_features(df, deposit_cap=deposit_cap)

    # 3) 결측치 처리
    if "trans_count_monthly_std" in df.columns:
        df["trans_count_monthly_std"] = df["trans_count_monthly_std"].fillna(0)

    # ─────────────────────────────────────────────
    # 4) 검증 (train/test 공통)
    # ─────────────────────────────────────────────
    expected_log_cols = [
        'log_total_deposit_balance',
        'log_total_loan_balance',
        'log_card_cash_service_amt',
        'log_card_loan_amt',
        'log_loan_to_deposit_ratio',
    ]
    gone_cols = [
        'total_deposit_balance',
        'total_loan_balance',
        'card_cash_service_amt',
        'card_loan_amt',
        'prefer_category',
        'most_purchased_category',
    ]

    for col in expected_log_cols:
        assert col in df.columns, f"❌ 누락: {col}"
        assert df[col].dtype == 'float64', f"❌ dtype 오류: {col}"

    for col in gone_cols:
        assert col not in df.columns, f"❌ 잔존: {col}"

    print("✅ features.py 검증 통과")

    return df