"""
run_pipeline.py
===============
수정 이력:
- [5주차] WEEK 4 → 5 변경
- [5주차] DEPOSIT_CAP 고정값 상수로 명시 (문서 기재값 329,791,893)
          train 분포가 바뀌지 않는 한 매 실행마다 동일값이 나오나,
          명시적 상수로 고정해 train/test 누수 가능성을 차단
- [5주차] Optuna 튜닝 파라미터는 튜닝 완료 후 train.py에 직접 반영
"""

import os
import numpy as np
import pandas as pd

# src 폴더를 import 경로에 추가
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from features import build_features
from train import (
    train_churn_model, predict_churn,
    train_ltv_model,   predict_ltv,
    make_submission,
)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_DIR   = "data/processed"
PROC_DIR   = "data/processed"
OUTPUT_DIR = "output"

TEAM_NAME  = "Kwonnect"
WEEK       = 5
SAVE_FEATURES = True   # processed CSV 저장 여부 (재실행 시 시간 절약)

# [5주차] train 상위 1% 고정값 — train.py 문서 기재값과 동일
# quantile()로 매번 계산하면 train 데이터가 바뀔 때 값이 달라질 수 있으므로 상수로 고정
DEPOSIT_CAP_FIXED = 329_791_893

os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Step 1. 데이터 로드
# ─────────────────────────────────────────────
print("=" * 60)
print("Step 1. 데이터 로드")
print("=" * 60)

train_raw = pd.read_csv(f"{DATA_DIR}/train_df.csv")
test_raw  = pd.read_csv(f"{DATA_DIR}/test_df.csv")

print(f"  Train : {train_raw.shape}")
print(f"  Test  : {test_raw.shape}")


# ─────────────────────────────────────────────
# Step 2. 피처 엔지니어링
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2. 피처 엔지니어링")
print("=" * 60)

# [5주차] quantile() 대신 고정값 사용 — train/test 동일 적용 보장
# 기존 코드: DEPOSIT_CAP = train_raw["total_deposit_balance"].quantile(0.99)
train_df = build_features(train_raw, is_train=True,  deposit_cap=DEPOSIT_CAP_FIXED)
test_df  = build_features(test_raw,  is_train=False, deposit_cap=DEPOSIT_CAP_FIXED)

# run_pipeline.py — Step 2 build_features() 호출 직후
DROP_FEATURES = [
    'last_month_trans_flag',
    'has_loan',
    'has_cash_service',
    'has_card_loan',
    'has_overdue',
    'fin_asset_trend_group',
]

train_df = train_df.drop(columns=DROP_FEATURES, errors='ignore')
test_df  = test_df.drop(columns=DROP_FEATURES, errors='ignore')

print(f"피처 제거 후 train shape: {train_df.shape}")
print(f"피처 제거 후 test shape:  {test_df.shape}")

# ─────────────────────────────────────────────
# Step 3. Churn 모델 학습
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3. Churn 모델 학습")
print("=" * 60)

churn_models, churn_oof, churn_cv = train_churn_model(train_df, n_splits=5)


# ─────────────────────────────────────────────
# Step 4. LTV 모델 학습
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4. LTV 모델 학습 (회귀, log1p 변환)")
print("=" * 60)

ltv_models, ltv_oof, ltv_cv = train_ltv_model(train_df, n_splits=5)


# ─────────────────────────────────────────────
# Step 5. Test 예측
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5. Test 예측")
print("=" * 60)

churn_pred = predict_churn(churn_models, test_df)
ltv_preds  = predict_ltv(ltv_models, test_df)

print(f"  Churn 예측 완료 | mean={churn_pred.mean():.4f} | std={churn_pred.std():.4f}")
print(f"  LTV   예측 완료 | mean={ltv_preds.mean():,.0f}  | min={ltv_preds.min():.2f}")

# churn_oof는 예측이 아닌 train 검증용이므로 아래처럼 구분
print(f"OOF Churn mean: {churn_oof.mean():.4f}")
print(f"OOF Churn > 0.5 비율: {(churn_oof > 0.5).mean():.4f}")


# ─────────────────────────────────────────────
# Step 6. 제출 파일 생성
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 6. 제출 파일 생성")
print("=" * 60)

submission_path = make_submission(
    test_df,
    churn_pred,
    ltv_preds,
    output_dir=OUTPUT_DIR,
    team_name=TEAM_NAME,
    week=WEEK,
)

print("\n" + "=" * 60)
print("✅ 전체 파이프라인 완료")
print(f"   제출 파일: {submission_path}")
print("=" * 60)