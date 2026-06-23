# 한국데이터정보과학회 × SAS Korea 제2회 데이터 분석 경진대회

## 환경 세팅

```bash
# 아나콘다 가상환경 생성 및 활성화
conda create -n kdissSas2026 python=3.11
conda activate kdissSas2026

# 패키지 설치
pip install pandas lightgbm scikit-learn numpy
```

## 폴더 구조

```
├── data/
│   ├── raw/          # 원본 CSV (git에 올리지 않음)
│   └── processed/    # 피처 엔지니어링 결과 CSV
├── notebooks/        # 탐색·실험용 .ipynb
├── src/
│   ├── features.py   # 피처 엔지니어링 함수
│   └── train.py      # 모델 학습·예측·제출 함수
├── output/           # 제출용 CSV
├── run_pipeline.py   # 전체 파이프라인 실행 스크립트
├── .gitignore
└── README.md
```

## 실행 방법

1. `data/raw/` 폴더에 대회에서 받은 CSV 파일 복사
2. `run_pipeline.py` 상단의 `TEAM_NAME`을 실제 팀명으로 변경
3. 실행:

```bash
python run_pipeline.py
```

4. `output/팀명_submission_N주차.csv` 파일이 생성된다.

## 모델 구성

| 타겟         | 모델           | 특이사항                                 |
| ------------ | -------------- | ---------------------------------------- |
| target_churn | LGBMClassifier | scale_pos_weight≈9.1 (이탈 9.9% 불균형)  |
| target_ltv   | LGBMRegressor  | log1p 변환 후 학습, expm1 역변환 후 제출 |

## 제출 규칙

- 파일명: `팀명_submission_N주차.csv`
- 인코딩: UTF-8
- 컬럼: customer_id, target_churn, target_ltv
- target_churn: 0.0~1.0 float
- target_ltv: float, 음수 불가
- 메일 제목: `[N주차] 팀명 결과제출`