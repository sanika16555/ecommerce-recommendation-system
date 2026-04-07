"""
Phase 7 — Visualization & Comparison Dashboard
Generates all comparison graphs: Our results vs Paper results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
COMP_DIR = os.path.join(os.path.dirname(__file__), "../graphs/comparison")
os.makedirs(COMP_DIR, exist_ok=True)

# ── Paper values from Tables 4–7 ─────────────────────────────────
PAPER_CF = {
    "SVD":      {"rmse": 1.3116, "mae": 1.0414},
    "SVD++":    {"rmse": 1.3253, "mae": 1.0514},
    "ALS":      {"rmse": 1.4485, "mae": 1.1518},
    "KNNBasic": {"rmse": 1.4071, "mae": 1.1115},
}
PAPER_ASSOC = {"Apriori": 0.0761, "FP-Growth": 0.0662, "Hybrid": 0.0043}


def plot_full_comparison(our_results, our_assoc_times=None):
    algos        = list(our_results.keys())
    colors_ours  = ["#4472C4", "#ED7D31", "#A9D18E", "#FF6B6B"]
    colors_paper = ["#7FAFDC", "#F0A070", "#C8E8B0", "#FF9999"]

    # ── 1. RMSE: Ours vs Paper ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(algos))
    w = 0.35
    b1 = ax.bar(x - w / 2,
                [our_results[a]["mean_rmse"] for a in algos], w,
                label="Our Implementation",
                color=colors_ours, edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + w / 2,
                [PAPER_CF[a]["rmse"] for a in algos], w,
                label="Paper Results",
                color=colors_paper, edgecolor="#666", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("RMSE: Our Implementation vs Research Paper",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(1.0, 2.0)
    ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, "rmse_ours_vs_paper.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: graphs/comparison/rmse_ours_vs_paper.png")

    # ── 2. MAE: Ours vs Paper ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w / 2,
                [our_results[a]["mean_mae"] for a in algos], w,
                label="Our Implementation",
                color=colors_ours, edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + w / 2,
                [PAPER_CF[a]["mae"] for a in algos], w,
                label="Paper Results",
                color=colors_paper, edgecolor="#666", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=12)
    ax.set_ylabel("MAE", fontsize=12)
    ax.set_title("MAE: Our Implementation vs Research Paper",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0.8, 1.7)
    ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, "mae_ours_vs_paper.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: graphs/comparison/mae_ours_vs_paper.png")

    # ── 3. Full 4-panel Summary ───────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Complete Performance Summary — Our Implementation vs Research Paper",
        fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: RMSE
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(x - w / 2, [our_results[a]["mean_rmse"] for a in algos], w,
            color=colors_ours,  label="Ours")
    ax0.bar(x + w / 2, [PAPER_CF[a]["rmse"] for a in algos], w,
            color=colors_paper, label="Paper", edgecolor="#666")
    ax0.set_title("A. RMSE Comparison", fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(algos, fontsize=9)
    ax0.set_ylabel("RMSE")
    ax0.legend(fontsize=9)
    ax0.set_ylim(1.0, 2.1)

    # Panel B: MAE
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(x - w / 2, [our_results[a]["mean_mae"] for a in algos], w,
            color=colors_ours,  label="Ours")
    ax1.bar(x + w / 2, [PAPER_CF[a]["mae"] for a in algos], w,
            color=colors_paper, label="Paper", edgecolor="#666")
    ax1.set_title("B. MAE Comparison", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, fontsize=9)
    ax1.set_ylabel("MAE")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.8, 1.7)

    # Panel C: % Difference heatmap
    ax2    = fig.add_subplot(gs[1, 0])
    metrics = ["RMSE", "MAE"]
    diff_data = np.array([
        [(our_results[a]["mean_rmse"] - PAPER_CF[a]["rmse"]) /
         PAPER_CF[a]["rmse"] * 100 for a in algos],
        [(our_results[a]["mean_mae"]  - PAPER_CF[a]["mae"]) /
         PAPER_CF[a]["mae"]  * 100 for a in algos],
    ])
    im = ax2.imshow(diff_data, cmap="RdYlGn_r", aspect="auto",
                    vmin=-20, vmax=20)
    ax2.set_xticks(range(len(algos)))
    ax2.set_xticklabels(algos, fontsize=9)
    ax2.set_yticks(range(2))
    ax2.set_yticklabels(metrics)
    ax2.set_title("C. % Difference vs Paper\n(green = better, red = worse)",
                  fontweight="bold")
    for i in range(2):
        for j in range(len(algos)):
            ax2.text(j, i, f"{diff_data[i, j]:+.1f}%",
                     ha="center", va="center",
                     fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax2, label="% diff")

    # Panel D: Algorithm ranking
    ax3      = fig.add_subplot(gs[1, 1])
    combined = {a: (our_results[a]["mean_rmse"] + our_results[a]["mean_mae"]) / 2
                for a in algos}
    sorted_a = sorted(combined, key=combined.get)
    bar_c    = ["#FFD700", "#C0C0C0", "#CD7F32", "#AAAAAA"]
    bars     = ax3.barh(sorted_a, [combined[a] for a in sorted_a],
                        color=bar_c, edgecolor="white")
    ax3.set_title("D. Algorithm Ranking\n(lower = better)",
                  fontweight="bold")
    ax3.set_xlabel("Average (RMSE + MAE) / 2")
    for bar, val in zip(bars, [combined[a] for a in sorted_a]):
        ax3.text(bar.get_width() + 0.005,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=10)

    plt.savefig(os.path.join(COMP_DIR, "full_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: graphs/comparison/full_summary.png")

    # ── 4. Association timing ─────────────────────────────────────
    if our_assoc_times:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Association Rule Mining — Time vs Paper",
                     fontsize=13, fontweight="bold")
        names   = ["Apriori", "FP-Growth", "Hybrid"]
        ours_t  = [our_assoc_times.get("apriori",  0),
                   our_assoc_times.get("fpgrowth", 0),
                   our_assoc_times.get("hybrid",   0)]
        paper_t = [PAPER_ASSOC[n] for n in names]
        cx      = np.arange(3)
        cw      = 0.35
        axes[0].bar(cx - cw / 2, ours_t,  cw, label="Ours",  color="#4472C4")
        axes[0].bar(cx + cw / 2, paper_t, cw, label="Paper", color="#7FAFDC",
                    edgecolor="#666")
        axes[0].set_xticks(cx)
        axes[0].set_xticklabels(names)
        axes[0].set_ylabel("Execution Time (s)")
        axes[0].set_title("Execution Time Comparison")
        axes[0].legend()

        speedup = [p / max(o, 1e-6) for o, p in zip(ours_t, paper_t)]
        axes[1].bar(names, speedup,
                    color=["#A9D18E", "#55A868", "#2E7D32"],
                    edgecolor="white")
        axes[1].axhline(1, color="red", linestyle="--", label="Baseline")
        axes[1].set_ylabel("Paper time / Our time")
        axes[1].set_title("Relative Speed vs Paper")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(COMP_DIR, "assoc_time_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: graphs/comparison/assoc_time_comparison.png")


def plot_rating_distribution_dashboard(df):
    """Mimics the Power BI dashboard from the paper."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("E-Commerce Recommendation System — Data Dashboard",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Rating distribution bar
    ax0 = fig.add_subplot(gs[0, 0])
    rc  = df["rating"].value_counts().sort_index()
    ax0.bar(rc.index, rc.values,
            color=["#C00000", "#FF6B6B", "#FFD700", "#92D050", "#00B050"],
            edgecolor="white", linewidth=1.5)
    ax0.set_title("Rating Distribution", fontweight="bold")
    ax0.set_xlabel("Rating")
    ax0.set_ylabel("Count")
    for x, y in zip(rc.index, rc.values):
        ax0.text(x, y + max(rc.values) * 0.01, str(y),
                 ha="center", fontsize=9)

    # 2. Ratings pie chart
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.pie(rc.values,
            labels=[f"★{i}" for i in rc.index],
            autopct="%1.1f%%", startangle=90,
            colors=["#C00000", "#FF6B6B", "#FFD700", "#92D050", "#00B050"])
    ax1.set_title("Rating Share (%)", fontweight="bold")

    # 3. Top-20 most-reviewed products
    ax2       = fig.add_subplot(gs[0, 2])
    top_prod  = df["productId"].value_counts().head(20)
    ax2.barh(range(len(top_prod)), top_prod.values[::-1], color="#4472C4")
    ax2.set_title("Top 20 Most-Reviewed Products", fontweight="bold")
    ax2.set_yticks(range(len(top_prod)))
    ax2.set_yticklabels([f"P{i + 1}" for i in range(len(top_prod))], fontsize=8)
    ax2.set_xlabel("Number of Reviews")

    # 4. Monthly rating trend
    ax3      = fig.add_subplot(gs[1, 0:2])
    df2      = df.copy()
    df2["month"] = pd.to_datetime(df2["timestamp"], unit="s").dt.to_period("M")
    monthly  = df2.groupby("month")["rating"].agg(["mean", "count"])
    months   = [str(m) for m in monthly.index]
    x_idx    = np.arange(len(months))
    ax3_twin = ax3.twinx()
    ax3.bar(x_idx, monthly["count"], alpha=0.4,
            color="#4472C4", label="Review count")
    ax3_twin.plot(x_idx, monthly["mean"], "r-o",
                  linewidth=2, markersize=4, label="Avg rating")
    step = max(1, len(months) // 12)
    ax3.set_xticks(x_idx[::step])
    ax3.set_xticklabels(months[::step], rotation=45, fontsize=7)
    ax3.set_title("Monthly Review Volume & Average Rating", fontweight="bold")
    ax3.set_ylabel("Review Count")
    ax3_twin.set_ylabel("Avg Rating")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")

    # 5. User activity histogram
    ax4           = fig.add_subplot(gs[1, 2])
    user_activity = df.groupby("userId")["rating"].count()
    ax4.hist(user_activity, bins=30, color="#ED7D31", edgecolor="white")
    ax4.set_title("User Activity Distribution", fontweight="bold")
    ax4.set_xlabel("Reviews per User")
    ax4.set_ylabel("Number of Users")

    plt.savefig(os.path.join(COMP_DIR, "data_dashboard.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: graphs/comparison/data_dashboard.png")


if __name__ == "__main__":
    pass
