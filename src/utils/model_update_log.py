import os
import csv
from datetime import datetime
from typing import Optional

LOG_PATH = os.path.join("logs", "model_updates.csv")


def log_model_update(
    run_id: str,
    model_name: str,
    hr_at_10: Optional[float] = None,
    ndcg_at_10: Optional[float] = None,
) -> None:
    """
    Append a row to the model-update window log.

    Each call = one 'model update' (retraining event).

    Columns:
    - timestamp: ISO string of when this update happened
    - run_id: free-form identifier (timestamp, config name, etc.)
    - model_name: e.g., 'ALS', 'Neumf', 'ItemCF'
    - hr_at_10: Hit Rate@10 (if available)
    - ndcg_at_10: NDCG@10 (if available)
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "run_id",
                "model_name",
                "hr_at_10",
                "ndcg_at_10",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id,
                "model_name": model_name,
                "hr_at_10": hr_at_10,
                "ndcg_at_10": ndcg_at_10,
            }
        )