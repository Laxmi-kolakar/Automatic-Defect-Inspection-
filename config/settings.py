from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
APP_NAME = os.getenv("APP_NAME", "Defect Doctor AI")
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(ROOT_DIR / "data" / "defect_doctor.db")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT_DIR / "models" / "brake_disc_cnn.pt")))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "128"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
DATASET_DIR = ROOT_DIR / "data" / "dataset"
for directory in (DATABASE_PATH.parent, MODEL_PATH.parent, DATASET_DIR):
    directory.mkdir(parents=True, exist_ok=True)
