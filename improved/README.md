# Kdiss & SAS KOREA 데이터 분석 경진대회 — 사후 개선 실험 directory

제2회 한국데이터정보과학회 & SAS KOREA 데이터 분석 경진대회(이탈 예측 + LTV 회귀)의 대회 종료 후 모델 재설계 기록

**자세한 진행 과정 및 개선 결과: [`reports/improvement_report.md`](reports/improvement_report.md)**

## 결과 요약

| 구분 | Churn AUC | Total Score | 비고 |
|---|---|---|---|
| 대회 종료 시점 | 0.7943 | 0.46683 | 50팀 중 46위 |
| 재설계 후 (최종) | **0.79824** | **0.46904** | 목표(0.469) 달성 |

개선폭 **ΔAUC +0.0039**. 기여도 대부분은 하이퍼파라미터 튜닝(+0.0045)에서 발생했고, 거래 피처 재설계(+0.00006)는 데이터에 신호가 없어 기각

## 데이터 진단 결과 핵심 발견

1. **`is_married` 분포 불일치**: 대회 진행 당시 "Train 95% vs Test 65%"로 기록되어 최우선 검증 과제였으나, 실제 집계 결과 Train 65.1% / Test 64.7%로 거의 동일하여 집계 오류로 확인됨
2. **Train/Test 분포 일치**: adversarial validation AUC 0.4995 (구분 불가). CV-LB 괴리의 원인은 분포 차이가 아님
3. **이탈 신호는 금융 자산 3개에만 존재**: `card_loan_amt`, `total_deposit_balance`, `card_cash_service_amt`. 거래 이력 31종은 단변량·다변량 양쪽 검증에서 모두 noise
4. **평가식의 AUC/RMSE 비중**: 채점식의 로그는 log10, RMSE는 원본 스케일. 역산 결과 LTV 개선의 점수 기여는 미미하므로 개선 실험을 Churn AUC에 집중
5. **5주차 튜닝 실패 원인**: 정돈된 피처셋에서 Optuna 튜닝 효과(+0.0130)가 5주차 미정돈 피처셋(+0.0107)보다 유의하게 컸음. 모델 설계 이전에 피처셋 정돈이 충분히 선행되지 않았던 것이 대회 당시 튜닝 실패 원인임을 입증

## 폴더 구조

```
.
├── data/
│   ├── raw/              # 원본 7개 CSV
│   └── processed/        # 병합·피처 결과
├── diagnostics/          # 데이터 검증
│   ├── diagnostic_v1.py      # 분포·adversarial·null importance·CV 재현 (Churn)
│   ├── ltv_diagnostic_v1.py  # LTV 신호구조·고액분류·타겟변환 진단
│   └── results/              # 진단 출력 JSON
├── src/
│   ├── features.py           # 피처 엔지니어링 함수
│   └── modeling/             # 모델링 버전 누적
│       ├── modeling_v1.py        # 4모델 다양성 앙상블 베이스라인
│       ├── modeling_v2.py        # Optuna + 다양성주입
│       ├── modeling_v3.py        # 다양성주입 제거 + RF튜닝 + 배깅
│       └── modeling_v4.py        # 거래피처 재설계 + 검증내장
├── output/
│   ├── submissions/      # 제출 CSV
│   └── results/          # 모델링 요약 JSON
├── experiments/
│   ├── ab_test_foundation_tuning.py      # 가설 검증 (AB 테스트)
│   └── results/                          # AB 테스트 결과 JSON
├── reports/
│   └── improvement_report.md # 진행과정 기록 및 회고 보고서
├── paths.py              # 경로 설정
├── run_pipeline.py       # 최종 파이프라인 (v3 기준)
├── requirements.txt
└── README.md
```

## 실행 순서

```bash
# 0) 환경 설치
pip install -r requirements.txt

# 1) data/raw/ 에 원본 CSV 파일 7개 배치 (대회 제공 데이터)

# 2) 데이터 진단 우선
python diagnostics/diagnostic_v1.py        # Churn 신호·분포·CV
#   → 결과: diagnostics/results/diagnostic_summary.json
python diagnostics/ltv_diagnostic_v1.py    # LTV 신호구조
#   → 결과: diagnostics/results/ltv_diagnostic_summary.json

# 3) 모델링
python src/modeling/modeling_v3.py         # 목표 달성 버전
#   → 결과: output/results/modeling_v3_summary.json
#   → 제출: output/submissions/submission_v3.csv
#   modeling_v1~v4는 통제된 실험 기록. 버전별 변경점은 보고서 참고
```

각 스크립트는 콘솔 요약과 함께 `*_summary.json`을 저장하고 절대경로를 출력함 (보고서 수치 근거)

## 사용 모델

LightGBM · XGBoost · CatBoost · RandomForest 4종 앙상블 (가중평균 + Isotonic 보정)
RandomForest는 단독 최고 성능은 아니나 앙상블 다양성 부품으로 도입