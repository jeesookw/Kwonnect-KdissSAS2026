from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DIAGNOSTICS_RESULTS = PROJECT_ROOT / "diagnostics" / "results"
MODELING_RESULTS = PROJECT_ROOT / "output" / "results"
SUBMISSIONS_DIR = PROJECT_ROOT / "output" / "submissions"
EXPERIMENTS_RESULTS = PROJECT_ROOT / "experiments" / "results"
