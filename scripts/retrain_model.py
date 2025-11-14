# scripts/retrain_model.py

import os
import sys
import csv
import subprocess
from datetime import datetime

# Project root: .../movie-recs
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_PATH = os.path.join(ROOT_DIR, "logs", "model_updates.csv")


def run_training():
    """
    Call the existing ALS training script as a CLI.
    We only pass the required argument: --ratings_csv.
    """
    ratings_path = os.path.join(ROOT_DIR, "src", "data", "interactions_train.csv")

    cmd = [
        sys.executable,
        os.path.join(ROOT_DIR, "scripts", "train_als.py"),
        "--ratings_csv",
        ratings_path,
    ]

    print("[Milestone 4] Running training command:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[Milestone 4] Training run completed.")


def log_update(note: str = "ALS retrain via train_als.py"):
    """Append a timestamped entry to logs/model_updates.csv."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = timestamp.replace(" ", "_").replace(":", "").replace("-", "")

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "run_id", "note"]
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": timestamp,
                "run_id": run_id,
                "note": note,
            }
        )

    print(f"[Milestone 4] Logged model update at {timestamp}")


def main():
    # 1. Train the model (this is your “model update”)
    run_training()

    # 2. Log that an update happened
    log_update()


if __name__ == "__main__":
    main()
