"""
Phase 4 — Collaborative Filtering
Implements from scratch:
  - SVD  (Singular Value Decomposition)
  - SVD++ (SVD with Implicit Feedback)
  - ALS  (Alternating Least Squares)
  - KNNBasic (Item-based Collaborative Filtering)
3-fold cross validation + RMSE & MAE evaluation.
"""

import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_and_preprocess

GRAPH_DIR   = os.path.join(os.path.dirname(__file__), "../graphs/collaborative_filtering")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(GRAPH_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════════
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


# ════════════════════════════════════════════════════════════════
#  SVD — Singular Value Decomposition
# ════════════════════════════════════════════════════════════════
class SVDModel:
    name = "SVD"

    def __init__(self, n_factors=20, n_epochs=20, lr=0.005, reg=0.02, seed=42):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr        = lr
        self.reg       = reg
        self.seed      = seed

    def fit(self, train_df, n_users, n_items):
        rng      = np.random.default_rng(self.seed)
        self.mu  = train_df["rating"].mean()
        self.P   = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q   = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.bu  = np.zeros(n_users)
        self.bi  = np.zeros(n_items)

        for _ in range(self.n_epochs):
            for row in train_df.itertuples(index=False):
                u, i, r = row.user_idx, row.prod_idx, row.rating
                err      = r - (self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                Pu          = self.P[u].copy()
                self.P[u]  += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i]  += self.lr * (err * Pu        - self.reg * self.Q[i])
        return self

    def predict(self, test_df):
        preds = []
        for row in test_df.itertuples(index=False):
            p = self.mu + self.bu[row.user_idx] + self.bi[row.prod_idx] + \
                self.P[row.user_idx] @ self.Q[row.prod_idx]
            preds.append(float(np.clip(p, 1, 5)))
        return preds


# ════════════════════════════════════════════════════════════════
#  SVD++ — SVD with Implicit Feedback
# ════════════════════════════════════════════════════════════════
class SVDppModel:
    name = "SVD++"

    def __init__(self, n_factors=20, n_epochs=15, lr=0.005, reg=0.02, seed=42):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr        = lr
        self.reg       = reg
        self.seed      = seed

    def fit(self, train_df, n_users, n_items):
        rng      = np.random.default_rng(self.seed)
        self.mu  = train_df["rating"].mean()
        self.P   = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q   = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.Y   = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.bu  = np.zeros(n_users)
        self.bi  = np.zeros(n_items)
        self.Iu  = {u: list(g["prod_idx"])
                    for u, g in train_df.groupby("user_idx")}

        for _ in range(self.n_epochs):
            for row in train_df.itertuples(index=False):
                u, i, r = row.user_idx, row.prod_idx, row.rating
                Iu       = self.Iu.get(u, [])
                norm     = max(1, len(Iu)) ** -0.5
                impl     = norm * sum(self.Y[j] for j in Iu)
                pu_hat   = self.P[u] + impl
                err      = r - (self.mu + self.bu[u] + self.bi[i] + pu_hat @ self.Q[i])
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                self.P[u]  += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i]  += self.lr * (err * pu_hat    - self.reg * self.Q[i])
                for j in Iu:
                    self.Y[j] += self.lr * (err * norm * self.Q[i] - self.reg * self.Y[j])
        return self

    def predict(self, test_df):
        preds = []
        for row in test_df.itertuples(index=False):
            u   = row.user_idx
            i   = row.prod_idx
            Iu  = self.Iu.get(u, [])
            norm = max(1, len(Iu)) ** -0.5
            impl = norm * sum(self.Y[j] for j in Iu)
            p    = self.mu + self.bu[u] + self.bi[i] + (self.P[u] + impl) @ self.Q[i]
            preds.append(float(np.clip(p, 1, 5)))
        return preds


# ════════════════════════════════════════════════════════════════
#  ALS — Alternating Least Squares
# ════════════════════════════════════════════════════════════════
class ALSModel:
    name = "ALS"

    def __init__(self, n_factors=20, n_epochs=15, reg=0.06, seed=42):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.reg       = reg
        self.seed      = seed

    def fit(self, train_df, n_users, n_items):
        rng      = np.random.default_rng(self.seed)
        self.P   = np.abs(rng.normal(0, 0.5, (n_users, self.n_factors)))
        self.Q   = np.abs(rng.normal(0, 0.5, (n_items, self.n_factors)))
        self.mu  = train_df["rating"].mean()

        user_items   = {u: [] for u in range(n_users)}
        item_users   = {i: [] for i in range(n_items)}
        user_ratings = {u: {} for u in range(n_users)}

        for row in train_df.itertuples(index=False):
            user_items[row.user_idx].append(row.prod_idx)
            item_users[row.prod_idx].append(row.user_idx)
            user_ratings[row.user_idx][row.prod_idx] = row.rating

        I = np.eye(self.n_factors)
        for _ in range(self.n_epochs):
            for u in range(n_users):
                items = user_items[u]
                if not items:
                    continue
                Qi = self.Q[items]
                ru = np.array([user_ratings[u][i] for i in items])
                self.P[u] = np.linalg.solve(Qi.T @ Qi + self.reg * I, Qi.T @ ru)

            for i in range(n_items):
                users = item_users[i]
                if not users:
                    continue
                Pu = self.P[users]
                ri = np.array([user_ratings[u][i] for u in users])
                self.Q[i] = np.linalg.solve(Pu.T @ Pu + self.reg * I, Pu.T @ ri)
        return self

    def predict(self, test_df):
        preds = []
        for row in test_df.itertuples(index=False):
            p = float(self.P[row.user_idx] @ self.Q[row.prod_idx])
            preds.append(float(np.clip(p, 1, 5)))
        return preds


# ════════════════════════════════════════════════════════════════
#  KNNBasic — Item-Based Collaborative Filtering
# ════════════════════════════════════════════════════════════════
class KNNBasicModel:
    name = "KNNBasic"

    def __init__(self, k=20, seed=42):
        self.k = k

    def fit(self, train_df, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        self.mu      = train_df["rating"].mean()

        mat = np.zeros((n_items, n_users), dtype=np.float32)
        for row in train_df.itertuples(index=False):
            mat[row.prod_idx, row.user_idx] = row.rating

        norms    = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat_norm = mat / norms
        self.sim = mat_norm @ mat_norm.T
        self.mat = mat
        self.user_mean = {
            u: train_df[train_df["user_idx"] == u]["rating"].mean()
            for u in train_df["user_idx"].unique()
        }
        return self

    def _predict_one(self, u, i):
        sims       = self.sim[i]
        rated_mask = self.mat[:, u] > 0
        rated_mask[i] = False
        if not rated_mask.any():
            return self.user_mean.get(u, self.mu)
        neighbors = np.argsort(-sims)
        neighbors = [j for j in neighbors if rated_mask[j]][:self.k]
        if not neighbors:
            return self.user_mean.get(u, self.mu)
        w = np.array([abs(sims[j]) for j in neighbors]) + 1e-9
        r = np.array([self.mat[j, u]  for j in neighbors])
        return float(np.dot(w, r) / w.sum())

    def predict(self, test_df):
        return [float(np.clip(self._predict_one(r.user_idx, r.prod_idx), 1, 5))
                for r in test_df.itertuples(index=False)]


# ════════════════════════════════════════════════════════════════
#  3-FOLD CROSS VALIDATION
# ════════════════════════════════════════════════════════════════
def kfold_cv(model_cls, df, n_folds=3, **kwargs):
    df      = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_users = df["user_idx"].max() + 1
    n_items = df["prod_idx"].max() + 1
    fold_sz = len(df) // n_folds
    results = []

    for fold in range(n_folds):
        val   = df.iloc[fold * fold_sz:(fold + 1) * fold_sz]
        train = pd.concat([df.iloc[:fold * fold_sz],
                           df.iloc[(fold + 1) * fold_sz:]])
        # Only keep val rows whose user/item appeared in train
        train_u = set(train["user_idx"])
        train_i = set(train["prod_idx"])
        val     = val[val["user_idx"].isin(train_u) & val["prod_idx"].isin(train_i)]
        if len(val) == 0:
            continue

        model = model_cls(**kwargs)
        t0       = time.time()
        model.fit(train, n_users, n_items)
        fit_time = time.time() - t0

        t0        = time.time()
        preds     = model.predict(val)
        test_time = time.time() - t0

        y_true = val["rating"].tolist()
        r      = rmse(y_true, preds)
        m      = mae(y_true,  preds)
        results.append({
            "fold": fold + 1, "rmse": r, "mae": m,
            "fit_time": fit_time, "test_time": test_time
        })
        print(f"    Fold {fold + 1}: RMSE={r:.4f}  MAE={m:.4f}  "
              f"fit={fit_time:.1f}s  test={test_time:.2f}s")

    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════
#  GRAPHS
# ════════════════════════════════════════════════════════════════
def plot_rmse_mae_comparison(all_results):
    import matplotlib.pyplot as plt

    algos  = list(all_results.keys())
    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF6B6B"]
    paper  = {
        "SVD":      {"rmse": 1.3116, "mae": 1.0414},
        "SVD++":    {"rmse": 1.3253, "mae": 1.0514},
        "ALS":      {"rmse": 1.4485, "mae": 1.1518},
        "KNNBasic": {"rmse": 1.4071, "mae": 1.1115},
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Collaborative Filtering — RMSE & MAE Comparison",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(algos))
    for ax, metric, title in [
        (axes[0], "mean_rmse", "Average RMSE (Lower = Better)"),
        (axes[1], "mean_mae",  "Average MAE  (Lower = Better)"),
    ]:
        vals  = [all_results[a][metric] for a in algos]
        pvals = [paper[a]["rmse" if "rmse" in metric else "mae"] for a in algos]
        bars  = ax.bar(x, vals, color=colors[:len(algos)],
                       edgecolor="white", linewidth=1.2, width=0.5)
        ax.plot(x, pvals, "k--o", label="Paper value", markersize=7, linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.set_ylabel("Error")
        ax.legend()
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, "rmse_mae_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graphs/collaborative_filtering/rmse_mae_comparison.png")


def plot_per_fold(all_fold_dfs):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Fold RMSE & MAE for Each Algorithm",
                 fontsize=14, fontweight="bold")
    axes   = axes.flatten()
    colors = ["#4472C4", "#ED7D31", "#A9D18E"]
    labels = ["Fold 1", "Fold 2", "Fold 3"]

    for idx, (name, df) in enumerate(all_fold_dfs.items()):
        ax = axes[idx]
        x  = np.arange(len(df))
        w  = 0.35
        ax.bar(x - w / 2, df["rmse"], w, label="RMSE",
               color=colors[0], edgecolor="white")
        ax.bar(x + w / 2, df["mae"],  w, label="MAE",
               color=colors[1], edgecolor="white")
        ax.set_title(f"{name} — Per-Fold Metrics", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels[:len(df)])
        ax.set_ylabel("Error")
        ax.legend()

    if len(all_fold_dfs) < 4:
        axes[-1].set_visible(False)

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, "per_fold_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graphs/collaborative_filtering/per_fold_metrics.png")


def plot_radar(all_results):
    import matplotlib.pyplot as plt

    algos      = list(all_results.keys())
    colors     = ["#4472C4", "#ED7D31", "#A9D18E", "#FF6B6B"]
    categories = ["Low RMSE", "Low MAE", "Fast Fit", "Fast Test"]

    def norm_inv(vals):
        v  = np.array(vals, dtype=float)
        v2 = v.max() - v + v.min()
        return v2 / v2.max() if v2.max() > 0 else v2

    rmse_n = norm_inv([all_results[a]["mean_rmse"] for a in algos])
    mae_n  = norm_inv([all_results[a]["mean_mae"]  for a in algos])
    fit_n  = norm_inv([all_results[a]["mean_fit"]  for a in algos])
    test_n = norm_inv([all_results[a]["mean_test"] for a in algos])

    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_title("Algorithm Comparison — Radar Chart",
                 fontsize=13, fontweight="bold", pad=20)

    for idx, algo in enumerate(algos):
        vals  = [rmse_n[idx], mae_n[idx], fit_n[idx], test_n[idx]]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=colors[idx], label=algo)
        ax.fill(angles, vals, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, "radar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graphs/collaborative_filtering/radar_comparison.png")


# ════════════════════════════════════════════════════════════════
#  TOP-N RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════
def get_top_n(model, user_idx, all_product_idxs, rated_idxs, n=10):
    candidates = [i for i in all_product_idxs if i not in rated_idxs]
    tmp        = pd.DataFrame({
        "user_idx": user_idx,
        "prod_idx": candidates,
        "rating":   [0] * len(candidates)
    })
    preds  = model.predict(tmp)
    scores = sorted(zip(candidates, preds), key=lambda x: -x[1])
    return scores[:n]


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def run_collaborative_filtering(df):
    print("\n" + "=" * 62)
    print("  PHASE 4 — COLLABORATIVE FILTERING")
    print("=" * 62)

    models = [
        (SVDModel,     "SVD",      dict(n_factors=20, n_epochs=20, lr=0.005, reg=0.02)),
        (SVDppModel,   "SVD++",    dict(n_factors=20, n_epochs=15, lr=0.005, reg=0.02)),
        (ALSModel,     "ALS",      dict(n_factors=20, n_epochs=12, reg=0.06)),
        (KNNBasicModel,"KNNBasic", dict(k=20)),
    ]

    all_results  = {}
    all_fold_dfs = {}
    best_rmse    = 999
    best_name    = ""

    for model_cls, name, kwargs in models:
        print(f"\n  ── {name} " + "─" * (40 - len(name)))
        fold_df   = kfold_cv(model_cls, df, n_folds=3, **kwargs)
        mean_rmse = fold_df["rmse"].mean()
        mean_mae  = fold_df["mae"].mean()
        mean_fit  = fold_df["fit_time"].mean()
        mean_test = fold_df["test_time"].mean()
        std_rmse  = fold_df["rmse"].std()
        std_mae   = fold_df["mae"].std()

        print(f"    Mean RMSE : {mean_rmse:.4f}  (±{std_rmse:.4f})")
        print(f"    Mean MAE  : {mean_mae:.4f}  (±{std_mae:.4f})")

        all_results[name]  = {
            "mean_rmse": mean_rmse, "mean_mae":  mean_mae,
            "mean_fit":  mean_fit,  "mean_test": mean_test,
            "std_rmse":  std_rmse,  "std_mae":   std_mae,
        }
        all_fold_dfs[name] = fold_df

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_name = name

    print(f"\n  BEST MODEL: {best_name}  (RMSE={best_rmse:.4f})")

    # ── Retrain best model and show top-10 recommendations ───────
    n_users    = df["user_idx"].max() + 1
    n_items    = df["prod_idx"].max() + 1
    best_model = SVDModel(n_factors=20, n_epochs=25, lr=0.005, reg=0.02)
    best_model.fit(df, n_users, n_items)

    sample_user = int(df["user_idx"].value_counts().index[0])
    rated       = set(df[df["user_idx"] == sample_user]["prod_idx"].tolist())
    recs        = get_top_n(best_model, sample_user, list(range(n_items)), rated)

    print(f"\n  Top-10 Product Recommendations (user_idx={sample_user}):")
    print(f"  {'Rank':<6} {'Product Index':<18} {'Predicted Rating'}")
    print(f"  {'----':<6} {'-------------':<18} {'----------------'}")
    for rank, (pid, score) in enumerate(recs, 1):
        print(f"  {rank:<6} {pid:<18} {score:.2f} ★")

    # ── Paper comparison table ────────────────────────────────────
    paper = {
        "SVD":      {"rmse": 1.3116, "mae": 1.0414},
        "SVD++":    {"rmse": 1.3253, "mae": 1.0514},
        "ALS":      {"rmse": 1.4485, "mae": 1.1518},
        "KNNBasic": {"rmse": 1.4071, "mae": 1.1115},
    }
    print("\n  ┌──────────────┬────────────┬────────────┬────────────┬────────────┐")
    print(  "  │ Algorithm    │ Our RMSE   │ Paper RMSE │ Our MAE    │ Paper MAE  │")
    print(  "  ├──────────────┼────────────┼────────────┼────────────┼────────────┤")
    for name, r in all_results.items():
        pv = paper.get(name, {})
        print(f"  │ {name:<12} │ {r['mean_rmse']:>10.4f} │ "
              f"{pv.get('rmse', 0):>10.4f} │ {r['mean_mae']:>10.4f} │ "
              f"{pv.get('mae', 0):>10.4f} │")
    print(  "  └──────────────┴────────────┴────────────┴────────────┴────────────┘")

    # ── Save results ──────────────────────────────────────────────
    rows = [{"Algorithm": n, **r} for n, r in all_results.items()]
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "cf_results.csv"), index=False)

    # ── Graphs ────────────────────────────────────────────────────
    print("\n  Generating collaborative filtering graphs...")
    plot_rmse_mae_comparison(all_results)
    plot_per_fold(all_fold_dfs)
    plot_radar(all_results)

    return all_results, best_model


if __name__ == "__main__":
    df, *_ = load_and_preprocess()
    run_collaborative_filtering(df)
