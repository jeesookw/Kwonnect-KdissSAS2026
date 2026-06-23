# -*- coding: utf-8 -*-
"""
run_pipeline.py — 최종 파이프라인 진입점

진단(diagnostics/) → 모델링(src/modeling/)의 결론을 반영한 최종 실행 스크립트.
정식 채택 버전은 modeling_v3 (목표 Score 0.469 달성, AUC 0.79817).
v4는 거래피처 재설계 실험으로 수렴 확인용이며 정식 파이프라인에는 미포함.

실행:
    1) data/raw/ 에 원본 7개 CSV 배치
    2) python run_pipeline.py
    3) output/submissions/submission_v3.csv 생성
"""
import runpy
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELING = os.path.join(ROOT, "src", "modeling", "modeling_v3.py")

if __name__ == "__main__":
    if not os.path.exists(MODELING):
        sys.exit("modeling_v3.py 를 찾을 수 없습니다.")
    print("=" * 60)
    print(" 최종 파이프라인 실행 (modeling_v3 기준)")
    print(" 원본 CSV 경로를 data/raw/ 로 맞췄는지 확인하세요.")
    print("=" * 60)
    # modeling_v3.py 내부 CONFIG의 경로만 data/raw/ 로 수정 후 실행 권장.
    runpy.run_path(MODELING, run_name="__main__")