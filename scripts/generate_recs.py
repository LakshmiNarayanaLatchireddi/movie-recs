import argparse
import os
import joblib
import pandas as pd
from evaluation.evaluator import evaluate_topk

def topk_for_users(model, users, items, k):
    rows = []
    for u in users:
        scores = model.score_items(int(u), items)
        order = scores.argsort()[::-1][:k]
        top_items = items[order]
        rows.append({"user": int(u), "items": ",".join(map(str, top_items))})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--item-col", default="item_id")
    ap.add_argument("--item-id-col", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--negatives", type=int, default=99)
    ap.add_argument("--out-recs", default="data/topk_recommendations.csv")
    ap.add_argument("--out-report", default="data/metrics_report.csv")
    args = ap.parse_args()

    model = joblib.load(args.model)

    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)
    items_df = pd.read_csv(args.items)

    # Evaluate
    res = evaluate_topk(
        model,
        test_df=test,
        user_col=args.user_col,
        item_col=args.item_col,
        k=args.k,
        train_df=train,
        negatives_per_user=args.negatives,
        items_df=items_df,
        item_id_col=(args.item_id_col or args.item_col),
    )

    # Save metrics
    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    pd.DataFrame([{
        "users": res.users,
        "k": res.k,
        "hr_at_k": round(res.hr, 6),
        "ndcg_at_k": round(res.ndcg, 6),
    }]).to_csv(args.out_report, index=False)

    # Save Top-K recs for all users seen in test (demo)
    all_items = items_df[(args.item_id_col or args.item_col)].astype(int).values
    users = test[args.user_col].astype(int).unique()
    recs = topk_for_users(model, users, all_items, args.k)
    recs.to_csv(args.out_recs, index=False)

    print(f"Saved metrics to {args.out_report}")
    print(f"Saved top-{args.k} recommendations to {args.out_recs}")
    print(f"HR@{res.k}={res.hr:.4f}  NDCG@{res.k}={res.ndcg:.4f}")

if __name__ == "__main__":
    main()