import argparse
import pandas as pd
import joblib
from recommender.baseline import PopularityModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--item-col", default="item_id")
    ap.add_argument("--rating-col", default=None)
    ap.add_argument("--out", default="model_registry/popularity.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    model = PopularityModel().fit(df, args.user_col, args.item_col, args.rating_col)
    # ensure folder exists
    out_path = args.out
    import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    main()