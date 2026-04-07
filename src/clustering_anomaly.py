"""
Phase 5 & 6 — K-Means Clustering + Anomaly Detection
Implements from scratch:
  - K-Means Clustering (user segmentation)
  - Isolation Forest  (anomaly detection)
  - KNN-based anomaly detection
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_and_preprocess

CLUSTER_DIR = os.path.join(os.path.dirname(__file__), "../graphs/clustering")
ANOMALY_DIR = os.path.join(os.path.dirname(__file__), "../graphs/anomaly_detection")
os.makedirs(CLUSTER_DIR, exist_ok=True)
os.makedirs(ANOMALY_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
#  K-MEANS CLUSTERING (from scratch)
# ════════════════════════════════════════════════════════════════
class KMeans:
    def __init__(self, k=5, max_iter=100, seed=42):
        self.k        = k
        self.max_iter = max_iter
        self.seed     = seed

    def fit(self, X):
        rng            = np.random.default_rng(self.seed)
        idx            = rng.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].copy().astype(float)

        for _ in range(self.max_iter):
            dists  = np.linalg.norm(
                X[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = dists.argmin(axis=1)
            new_c  = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any()
                else self.centroids[k]
                for k in range(self.k)
            ])
            if np.allclose(new_c, self.centroids, atol=1e-4):
                break
            self.centroids = new_c

        self.labels_   = labels
        self.inertia_  = sum(
            np.sum((X[labels == k] - self.centroids[k]) ** 2)
            for k in range(self.k) if (labels == k).any()
        )
        return self


# ════════════════════════════════════════════════════════════════
#  PCA (2D projection for visualization)
# ════════════════════════════════════════════════════════════════
def pca_2d(X):
    X_     = X - X.mean(axis=0)
    cov    = X_.T @ X_ / (len(X_) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order  = np.argsort(-eigvals)
    return X_ @ eigvecs[:, order[:2]]


# ════════════════════════════════════════════════════════════════
#  ISOLATION FOREST (from scratch)
# ════════════════════════════════════════════════════════════════
def _c(n):
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.left = self.right = None
        self.split_col = self.split_val = None
        self.size      = 0
        self.is_leaf   = False

    def fit(self, X, depth=0):
        self.size = len(X)
        if depth >= self.max_depth or len(X) <= 1:
            self.is_leaf = True
            return self
        self.split_col = np.random.randint(X.shape[1])
        col            = X[:, self.split_col]
        mn, mx         = col.min(), col.max()
        if mn == mx:
            self.is_leaf = True
            return self
        self.split_val = np.random.uniform(mn, mx)
        mask           = col < self.split_val
        self.left      = IsolationTree(self.max_depth).fit(X[mask],  depth + 1)
        self.right     = IsolationTree(self.max_depth).fit(X[~mask], depth + 1)
        return self

    def path_length(self, x, depth=0):
        if self.is_leaf:
            return depth + _c(self.size)
        if x[self.split_col] < self.split_val:
            return self.left.path_length(x, depth + 1)
        return self.right.path_length(x, depth + 1)


class IsolationForest:
    def __init__(self, n_trees=50, max_samples=128, contamination=0.05, seed=42):
        self.n_trees       = n_trees
        self.max_samples   = max_samples
        self.contamination = contamination
        np.random.seed(seed)

    def fit(self, X):
        max_depth  = int(np.ceil(np.log2(self.max_samples)))
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(len(X),
                                   min(self.max_samples, len(X)),
                                   replace=False)
            self.trees.append(IsolationTree(max_depth).fit(X[idx]))
        self.c_n = _c(self.max_samples)
        scores   = self.score_samples(X)
        self.threshold_ = np.percentile(scores,
                                        100 * (1 - self.contamination))
        return self

    def score_samples(self, X):
        avg = np.array([
            np.mean([t.path_length(x) for t in self.trees]) for x in X
        ])
        return -2 ** (-avg / self.c_n)

    def predict(self, X):
        scores = self.score_samples(X)
        return np.where(scores < self.threshold_, -1, 1)


# ════════════════════════════════════════════════════════════════
#  KNN ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════
def knn_anomaly(X, k=5, contamination=0.05):
    from sklearn.neighbors import NearestNeighbors
    nn     = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, _ = nn.kneighbors(X)
    scores = dists[:, 1:].mean(axis=1)
    thresh = np.percentile(scores, 100 * (1 - contamination))
    return np.where(scores > thresh, -1, 1), scores


# ════════════════════════════════════════════════════════════════
#  BUILD USER FEATURE MATRIX
# ════════════════════════════════════════════════════════════════
def build_user_features(df):
    feats = df.groupby("user_idx").agg(
        mean_rating=("rating",   "mean"),
        rating_std =("rating",   "std"),
        n_ratings  =("rating",   "count"),
        n_products =("prod_idx", "nunique"),
        pct_5star  =("rating",   lambda x: (x == 5).mean()),
        pct_1star  =("rating",   lambda x: (x == 1).mean()),
    ).fillna(0).reset_index()

    X = feats[["mean_rating", "rating_std", "n_ratings",
               "n_products",  "pct_5star",  "pct_1star"]].values.astype(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    return X, feats


# ════════════════════════════════════════════════════════════════
#  GRAPHS — CLUSTERING
# ════════════════════════════════════════════════════════════════
def plot_elbow(ks, inertias):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia (WCSS)",          fontsize=12)
    ax.set_title("Elbow Method — Optimal k for K-Means",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CLUSTER_DIR, "elbow_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graphs/clustering/elbow_curve.png")


def plot_clusters(X2d, labels):
    import matplotlib.pyplot as plt
    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF6B6B", "#9B59B6"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("K-Means User Clustering", fontsize=14, fontweight="bold")

    axes[0].scatter(X2d[:, 0], X2d[:, 1], s=10, alpha=0.4, color="#aaaaaa")
    axes[0].set_title("Before Clustering (PCA 2D)", fontsize=11)

    k = labels.max() + 1
    for i in range(k):
        mask = labels == i
        axes[1].scatter(X2d[mask, 0], X2d[mask, 1], s=12, alpha=0.6,
                        color=colors[i % len(colors)], label=f"Cluster {i + 1}")
    axes[1].set_title("After K-Means Clustering", fontsize=11)
    axes[1].legend(markerscale=2)

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    plt.tight_layout()
    path = os.path.join(CLUSTER_DIR, "cluster_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graphs/clustering/cluster_scatter.png")


def plot_cluster_profiles(feats_df, labels):
    import matplotlib.pyplot as plt
    feats_df           = feats_df.copy()
    feats_df["cluster"] = labels
    cols  = ["mean_rating", "rating_std", "n_ratings", "pct_5star", "pct_1star"]
    group = feats_df.groupby("cluster")[cols].mean()
    group.T.plot(kind="bar", figsize=(12, 6), colormap="tab10", edgecolor="white")
    plt.title("Cluster Profiles — Mean Feature Values",
              fontsize=13, fontweight="bold")
    plt.xlabel("Feature")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=20)
    plt.legend(title="Cluster", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    path = os.path.join(CLUSTER_DIR, "cluster_profiles.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graphs/clustering/cluster_profiles.png")


# ════════════════════════════════════════════════════════════════
#  GRAPHS — ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════
def plot_isolation_forest(X2d, labels_if):
    import matplotlib.pyplot as plt
    normal  = X2d[labels_if == 1]
    outlier = X2d[labels_if == -1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Anomaly Detection — Isolation Forest",
                 fontsize=14, fontweight="bold")

    for ax in axes:
        ax.scatter(normal[:, 0],  normal[:, 1],  s=10, alpha=0.4,
                   color="#4472C4", label="Inlier (normal)")
        ax.scatter(outlier[:, 0], outlier[:, 1], s=30, alpha=0.9,
                   color="#C00000", label="Outlier (anomaly)", marker="x")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()

    axes[0].set_title("Predicted Outliers", fontsize=11)
    axes[1].set_title("Data Points View",   fontsize=11)
    plt.tight_layout()
    path = os.path.join(ANOMALY_DIR, "isolation_forest.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graphs/anomaly_detection/isolation_forest.png")


def plot_knn_anomaly(X2d, labels_knn):
    import matplotlib.pyplot as plt
    normal  = X2d[labels_knn == 1]
    outlier = X2d[labels_knn == -1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("K-Nearest Neighbours — Anomaly Detection",
                 fontsize=14, fontweight="bold")

    for ax in axes:
        ax.scatter(normal[:, 0],  normal[:, 1],  s=10, alpha=0.4,
                   color="#4472C4", label="True inlier")
        ax.scatter(outlier[:, 0], outlier[:, 1], s=30, alpha=0.9,
                   color="#C00000", label="True outlier", marker="o")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend()

    axes[0].set_title("KNN Anomaly — Full View",   fontsize=11)
    axes[1].set_title("KNN Anomaly — Zoomed View", fontsize=11)
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    plt.tight_layout()
    path = os.path.join(ANOMALY_DIR, "knn_anomaly.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graphs/anomaly_detection/knn_anomaly.png")


def plot_violin(df):
    import matplotlib.pyplot as plt
    normal  = df[df["is_anomaly"] == 0]["rating"].values
    anomaly = df[df["is_anomaly"] == 1]["rating"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Rating Distribution: Normal vs Anomaly",
                 fontsize=13, fontweight="bold")

    for ax, title, color in [
        (axes[0], "Naive Bayes",      "#4472C4"),
        (axes[1], "K-Nearest Neigh.", "#ED7D31"),
    ]:
        parts = ax.violinplot([normal, anomaly], positions=[1, 2],
                              showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Rating")

    plt.tight_layout()
    path = os.path.join(ANOMALY_DIR, "violin_anomaly.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: graphs/anomaly_detection/violin_anomaly.png")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def run_clustering_and_anomaly(df):
    # ── Phase 5: K-Means ─────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  PHASE 5 — K-MEANS CLUSTERING")
    print("=" * 62)

    X, feats_df = build_user_features(df)
    X2d         = pca_2d(X)

    # Elbow method to find best k
    ks, inertias = list(range(2, 10)), []
    for k in ks:
        km = KMeans(k=k).fit(X)
        inertias.append(km.inertia_)
    plot_elbow(ks, inertias)

    # Final clustering with k=5
    K_BEST = 5
    km     = KMeans(k=K_BEST).fit(X)
    print(f"  K-Means k={K_BEST}: Inertia={km.inertia_:.2f}")
    plot_clusters(X2d, km.labels_)
    plot_cluster_profiles(feats_df, km.labels_)

    sizes = pd.Series(km.labels_).value_counts().sort_index()
    print("  Cluster sizes:")
    for ci, sz in sizes.items():
        print(f"    Cluster {ci + 1}: {sz} users")

    # ── Phase 6: Anomaly Detection ────────────────────────────────
    print("\n" + "=" * 62)
    print("  PHASE 6 — ANOMALY DETECTION")
    print("=" * 62)

    # User-level features for anomaly detection
    ua = df.groupby("user_idx").agg(
        mean_r=("rating", "mean"),
        std_r =("rating", "std"),
        cnt   =("rating", "count")
    ).fillna(0).values.astype(float)
    ua = (ua - ua.mean(axis=0)) / (ua.std(axis=0) + 1e-9)

    # KNN anomaly
    labels_knn, _ = knn_anomaly(ua, k=5, contamination=0.05)
    print(f"  KNN anomalies detected   : {(labels_knn == -1).sum()} / {len(ua)} users")

    # Isolation Forest
    print("  Running Isolation Forest...")
    ifo       = IsolationForest(n_trees=50,
                                max_samples=min(128, len(ua)),
                                contamination=0.05)
    ifo.fit(ua)
    labels_if = ifo.predict(ua)
    print(f"  Isolation Forest anomalies: {(labels_if == -1).sum()} / {len(ua)} users")

    # Generate graphs
    ua2d = pca_2d(ua)
    plot_isolation_forest(ua2d, labels_if)
    plot_knn_anomaly(ua2d, labels_knn)

    # Violin plot
    df2 = df.copy()
    user_anomaly = dict(zip(range(len(labels_knn)),
                            (labels_knn == -1).astype(int)))
    df2["is_anomaly"] = df2["user_idx"].map(user_anomaly).fillna(0).astype(int)
    plot_violin(df2)

    return km, labels_knn, labels_if


if __name__ == "__main__":
    df, *_ = load_and_preprocess()
    run_clustering_and_anomaly(df)
