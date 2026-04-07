"""
Phase 1 — Data Preprocessing
- Load CSV (works with Amazon reviews OR MovieLens ratings.csv)
- Drop duplicates and nulls
- Normalize ratings
- Build user-item matrix
- Report stats
"""

import pandas as pd
import numpy as np
import os

# ── Path to your dataset ──────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/amazon_reviews.csv")



def load_and_preprocess(path=DATA_PATH, min_user_ratings=3, min_product_ratings=3):
    print("=" * 62)
    print("  PHASE 1 — DATA PREPROCESSING")
    print("=" * 62)

    # ── Load CSV ──────────────────────────────────────────────────
    df = pd.read_csv(path)
    print(f"\n[1] Raw records loaded   : {len(df):>10,}")

    # ── Rename columns if using MovieLens ─────────────────────────
    # MovieLens uses: userId, movieId, rating, timestamp
    # We rename movieId → productId so rest of code works
    if "movieId" in df.columns:
        df = df.rename(columns={"movieId": "productId"})
        print("    (MovieLens detected — renamed movieId to productId)")

    # ── Keep only needed columns ──────────────────────────────────
    df = df[["userId", "productId", "rating", "timestamp"]].copy()

    print(f"    Unique users         : {df['userId'].nunique():>10,}")
    print(f"    Unique products      : {df['productId'].nunique():>10,}")
    print(f"    Rating range         : {df['rating'].min()} to {df['rating'].max()}")

    # ── Drop nulls & duplicates ───────────────────────────────────
    df = df.dropna()
    df = df.drop_duplicates(subset=["userId", "productId"])
    print(f"\n[2] After dedup/null drop: {len(df):>10,}")

    # ── For large datasets, sample to keep it fast ─────────────────
    MAX_RECORDS = 200_000
    if len(df) > MAX_RECORDS:
        df = df.sample(n=MAX_RECORDS, random_state=42).reset_index(drop=True)
        print(f"[3] Sampled to           : {len(df):>10,} records (for speed)")

    # ── Filter sparse users / products ───────────────────────────
    u_counts = df["userId"].value_counts()
    p_counts = df["productId"].value_counts()
    df = df[df["userId"].isin(u_counts[u_counts >= min_user_ratings].index)]
    df = df[df["productId"].isin(p_counts[p_counts >= min_product_ratings].index)]
    df = df.reset_index(drop=True)
    print(f"[4] After sparsity filter: {len(df):>10,}")
    print(f"    Active users         : {df['userId'].nunique():>10,}")
    print(f"    Active products      : {df['productId'].nunique():>10,}")

    # ── Encode IDs to integers ────────────────────────────────────
    users    = sorted(df["userId"].unique())
    products = sorted(df["productId"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    prod2idx = {p: i for i, p in enumerate(products)}
    df["user_idx"] = df["userId"].map(user2idx)
    df["prod_idx"] = df["productId"].map(prod2idx)

    # ── Normalize ratings ─────────────────────────────────────────
    r_min = df["rating"].min()
    r_max = df["rating"].max()
    df["rating_norm"] = (df["rating"] - r_min) / (r_max - r_min)

    # ── Rating distribution ───────────────────────────────────────
    print("\n[5] Rating distribution:")
    dist = df["rating"].value_counts().sort_index()
    for star, cnt in dist.items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {star:<4} : {cnt:>7,}  ({pct:5.1f}%)  {bar}")

    # ── Sparsity ──────────────────────────────────────────────────
    n_u = df["userId"].nunique()
    n_p = df["productId"].nunique()
    sparsity = 1 - len(df) / (n_u * n_p)
    print(f"\n[6] Matrix sparsity      : {sparsity * 100:.2f}%")
    print(f"    (Higher sparsity = harder recommendation problem)")

    return df, user2idx, prod2idx, users, products


def build_user_item_matrix(df, n_users, n_products):
    """Build dense user-item matrix. 0 = unrated."""
    mat = np.zeros((n_users, n_products), dtype=np.float32)
    for row in df.itertuples(index=False):
        mat[row.user_idx, row.prod_idx] = row.rating
    return mat


if __name__ == "__main__":
    df, u2i, p2i, users, products = load_and_preprocess()
    print(f"\nUser-item matrix would be: {len(users)} x {len(products)}")
