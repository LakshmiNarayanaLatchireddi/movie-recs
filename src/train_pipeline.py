# src/train_pipeline.py
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from src.config_loader import load_config
from src.evaluation.evaluator import evaluate_topk
from src.utils.model_update_log import log_model_update
from src.recommender.baseline import PopularityRecommender


# Resolve repo root (.../movie-recs) regardless of where the script is run from
REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------
# Helpers
# ----------------------------

def _data_path(relpath: str) -> Path:
    """Return an absolute path for a repo-relative path."""
    return (REPO_ROOT / relpath).resolve()


def _load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test CSVs based on config paths.
    Tolerant: if keys are missing, fall back to src/data/*.csv
    """
    data_cfg = (config.get("data") or {})
    train_rel = data_cfg.get("train") or "src/data/interactions_train.csv"
    test_rel = data_cfg.get("test") or "src/data/interactions_test.csv"

    train_path = _data_path(train_rel)
    test_path = _data_path(test_rel)

    logging.info(f"Loading train interactions from: {train_path}")
    logging.info(f"Loading test interactions from:  {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def _train_model(config: Dict[str, Any], train_df: pd.DataFrame):
    """
    Fit the baseline Popularity model using config-driven column names.
    """
    cols_cfg = (config.get("data", {}).get("columns", {}) or {})
    user_col = cols_cfg.get("user", "userId")
    item_col = cols_cfg.get("item", "movieId")
    rating_col = cols_cfg.get("rating", "rating")
    ts_col = cols_cfg.get("timestamp", "timestamp")

    logging.info("Training PopularityRecommender baseline model...")
    model = PopularityRecommender(
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,      # accepted (unused) by baseline
        timestamp_col=ts_col,       # accepted (unused) by baseline
    )
    model.fit(train_df)
    return model, user_col, item_col


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    config = load_config()
    logging.info(f"APP_ENV={config.get('env')}")
    logging.info(f"Loaded config keys: {list(config.keys())}")

    # 1) Load data
    train_df, test_df = _load_data(config)

    # 2) Train model
    model, user_col, item_col = _train_model(config, train_df)

    # 3) Evaluate
    eval_cfg = (config.get("eval") or {})
    k = int(eval_cfg.get("k", 10))
    negatives = int(eval_cfg.get("negatives_per_user", 99))

    res = evaluate_topk(
        model,
        test_df=test_df,
        user_col=user_col,
        item_col=item_col,
        k=k,
        train_df=train_df,
        negatives_per_user=negatives,
    )

    hr_at_10 = float(res.hr)
    ndcg_at_10 = float(res.ndcg)

    # 4) Milestone 4 window log (≥2 updates within 7 days)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_model_update(
        run_id=run_id,
        model_name="Popularity",
        hr_at_10=hr_at_10,
        ndcg_at_10=ndcg_at_10,
    )

    print(
        f"[Milestone 4] Model update logged: run_id={run_id}, "
        f"HR@10={hr_at_10:.4f}, NDCG@10={ndcg_at_10:.4f}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%.asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    main()
