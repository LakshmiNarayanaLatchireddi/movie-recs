# src/eval/run_offline_eval.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Iterable, List, Dict, Set

import numpy as np
import pandas as pd


# -----------------------------
# Models: plug in your own later
# -----------------------------
class RandomModel:
    """Scores items randomly (reproducible via seed)."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def score_items(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        n = len(list(item_ids))
        return self.rng.random(n)


class PopularityModel:
    """Scores items by item frequency in the TRAIN set (higher = better)."""
    def __init__(self, item_freq: Dict[int, int]):
        self.item_freq = item_freq

    def score_items(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        ids = list(item_ids)
        return np.array([self.item_freq.get(i, 0) for i in ids], dtype=float)


# -----------------------------
# Data loading and utilities
# -----------------------------
def load_csv_rows(path: str, required_cols: List[str]) -> pd.DataFrame:
    """
    Robust CSV loader using pandas. Raises a helpful error if columns are missing.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {path}. Error: {e}")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df[required_cols]


def build_seen_dict(df: pd.DataFrame, user_col: str, item_col: str) -> Dict[int, Set[int]]:
    seen: Dict[int, Set[int]] = {}
    for u, i in zip(df[user_col].tolist(), df[item_col].tolist()):
        seen.setdefault(int(u), set()).add(int(i))
    return seen


def sample_negatives_for_user(
    all_items: List[int],
    seen_items: Set[int],
    n_neg: int,
    rng: random.Random,
) -> List[int]:
    """
    Sample negatives from all_items excluding seen_items.
    If the candidate pool is too small, we sample without replacement as much as possible
    and then fill by allowing repeats (rare; indicates tiny catalog).
    """
    candidates = [it for it in all_items if it not in seen_items]
    if len(candidates) >= n_neg:
        return rng.sample(candidates, n_neg)
    # not enough candidates -> take all and then repeat randomly
    res = candidates[:]
    if len(candidates) > 0:
        while len(res) < n_neg:
            res.append(rng.choice(candidates))
    else:
        # degenerate case: every item is seen; just repeat something from all_items
        while len(res) < n_neg:
            res.append(rng.choice(all_items))
    return res


def hr_at_k(rank: int, k: int) -> float:
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    if rank > k:
        return 0.0
    # DCG with a single relevant item of gain 1
    return 1.0 / math.log2(rank + 1.0)


@dataclass
class EvalResult:
    users: int
    k: int
    negatives: int
    hr: float
    ndcg: float


# -----------------------------
# Evaluation core
# -----------------------------
def evaluate_topk(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str,
    item_col: str,
    k: int,
    negatives_per_user: int,
    all_items: List[int],
    seed: int = 42,
) -> EvalResult:

    rng = random.Random(seed)

    # Build user->seen items from TRAIN + TEST to prevent leakage in negative sampling
    seen_train = build_seen_dict(train_df, user_col, item_col)
    seen_test = build_seen_dict(test_df, user_col, item_col)

    # Iterate each user in test set; each row is the positive item
    hrs, ndcgs = [], []
    n_users = 0

    # groupby in case multiple rows per user exist in test; we evaluate each row independently
    for _, row in test_df.iterrows():
        u = int(row[user_col])
        pos = int(row[item_col])

        seen = set()
        if u in seen_train:
            seen |= seen_train[u]
        if u in seen_test:
            seen |= seen_test[u]

        negs = sample_negatives_for_user(
            all_items=all_items,
            seen_items=seen,
            n_neg=negatives_per_user,
            rng=rng,
        )

        items = [pos] + negs
        scores = model.score_items(u, items)

        # higher score = better
        # argsort descending
        ranked_idx = np.argsort(-scores)
        # rank is 1-based index of the positive
        rank_of_pos = int(np.where(ranked_idx == 0)[0][0]) + 1

        hrs.append(hr_at_k(rank_of_pos, k))
        ndcgs.append(ndcg_at_k(rank_of_pos, k))
        n_users += 1

    hr = float(np.mean(hrs)) if hrs else 0.0
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

    return EvalResult(users=n_users, k=k, negatives=negatives_per_user, hr=hr, ndcg=ndcg)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline evaluation: compute HR@K and NDCG@K on a held-out test set."
    )
    p.add_argument("--train", required=True, help="Path to interactions_train.csv")
    p.add_argument("--test", required=True, help="Path to interactions_test.csv")
    p.add_argument("--items", required=False, default="", help="Path to items.csv (optional)")
    p.add_argument("--outdir", required=False, default="evaluation/offline", help="Output directory")
    p.add_argument("--k", type=int, default=5, help="Top-K for HR/NDCG")
    p.add_argument("--negatives", type=int, default=99, help="Negatives per user")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    # Column names (defaults match the guidance we used)
    p.add_argument("--user-col", default="user_id")
    p.add_argument("--item-col", default="item_id")
    p.add_argument("--rating-col", default="rating")
    p.add_argument("--time-col", default="timestamp")

    # Model options
    p.add_argument(
        "--baseline",
        choices=["popularity", "random"],
        default="popularity",
        help="Built-in baseline to use if no custom model is provided."
    )
    p.add_argument(
        "--model-pickle",
        default="",
        help="Optional path to a pickled model with .score_items(user_id, item_ids)->np.array."
    )

    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_items_catalog(
    items_path: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_col: str,
) -> List[int]:
    """
    Returns the catalog used for negative sampling.
    Priority:
      1) items.csv if provided and has item_col
      2) union of item ids from train+test
    """
    if items_path and os.path.exists(items_path):
        try:
            idf = pd.read_csv(items_path)
            if item_col in idf.columns:
                catalog = sorted(map(int, idf[item_col].unique().tolist()))
                if len(catalog) > 0:
                    return catalog
        except Exception:
            pass  # fallback below

    # fallback: union of seen items
    seen_items = pd.concat([train_df[[item_col]], test_df[[item_col]]], axis=0)[item_col].unique().tolist()
    catalog = sorted(map(int, seen_items))
    return catalog


def build_baseline_model(
    which: str,
    train_df: pd.DataFrame,
    item_col: str,
) -> object:
    if which == "random":
        return RandomModel(seed=42)
    # popularity
    freq = train_df[item_col].value_counts().to_dict()
    freq = {int(k): int(v) for k, v in freq.items()}
    return PopularityModel(freq)


def load_pickled_model(path: str):
    try:
        import joblib
    except Exception as e:
        raise RuntimeError(f"To use --model-pickle, install joblib. Error: {e}")
    try:
        model = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load pickled model from {path}. Error: {e}")
    # must have score_items(user_id, item_ids)
    if not hasattr(model, "score_items"):
        raise AttributeError(
            "Loaded object does not define .score_items(user_id, item_ids)->np.ndarray"
        )
    return model


def main():
    args = parse_args()

    # 1) Load data (robust, header-checked)
    train_df = load_csv_rows(args.train, [args.user_col, args.item_col, args.rating_col, args.time_col])
    test_df  = load_csv_rows(args.test,  [args.user_col, args.item_col, args.rating_col, args.time_col])

    # 2) Items catalog for negative sampling
    all_items = load_items_catalog(args.items, train_df, test_df, args.item_col)
    if not all_items:
        raise RuntimeError("Item catalog is empty. Provide items.csv or ensure train/test contain items.")

    # 3) Model (pickled or baseline)
    if args.model_pickle:
        model = load_pickled_model(args.model_pickle)
        model_name = f"custom:{os.path.basename(args.model_pickle)}"
    else:
        model = build_baseline_model(args.baseline, train_df, args.item_col)
        model_name = f"baseline:{args.baseline}"

    # 4) Evaluate
    res = evaluate_topk(
        model=model,
        train_df=train_df,
        test_df=test_df,
        user_col=args.user_col,
        item_col=args.item_col,
        k=args.k,
        negatives_per_user=args.negatives,
        all_items=all_items,
        seed=args.seed,
    )

    # 5) Save + print
    ensure_dir(args.outdir)
    metrics_json = {
        "users": res.users,
        "k": res.k,
        "negatives": res.negatives,
        "hr": round(res.hr, 6),
        "ndcg": round(res.ndcg, 6),
        "model": model_name,
        "train_path": args.train,
        "test_path": args.test,
        "items_path": args.items if args.items else None,
        "user_col": args.user_col,
        "item_col": args.item_col,
        "rating_col": args.rating_col,
        "time_col": args.time_col,
        "seed": args.seed,
    }
    out_json = os.path.join(args.outdir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    # Optional CSV for quick comparisons
    out_csv = os.path.join(args.outdir, "metrics.csv")
    pd.DataFrame([metrics_json]).to_csv(out_csv, index=False)

    print(f"Users evaluated: {res.users}")
    print(f"HR@{res.k}:   {res.hr:.4f}")
    print(f"NDCG@{res.k}: {res.ndcg:.4f}")
    print(f"Model:        {model_name}")
    print(f"Saved:        {out_json}")
    print(f"Saved:        {out_csv}")


if __name__ == "__main__":
    main()