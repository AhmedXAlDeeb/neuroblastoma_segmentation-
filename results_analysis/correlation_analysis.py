import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# -------------------
# Correlation + Agreement
# -------------------
def correlation_analysis(results_df, manual_df, metric="volume_cm3", save_prefix=None):
    """
    Compare nnUNet results with manual labels.
    results_df : DataFrame with nnUNet metrics (from analyze_all)
    manual_df  : DataFrame with manual metrics (must have same case IDs)
    metric     : Which metric to compare ("volume_cm3", "RECIST_diameter_mm", etc.)
    """

    # merge on case ID
    df = pd.merge(results_df, manual_df, on="case", suffixes=("_pred", "_manual"))

    if df.empty:
        print(f"⚠️ No matching cases for metric={metric}")
        return None

    x = df[f"{metric}_manual"]
    y = df[f"{metric}_pred"]

    # Pearson + Spearman correlation
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    # ensure directory exists
    if save_prefix:
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)

    # Plot correlation
    plt.figure(figsize=(6, 6))
    sns.regplot(x=x, y=y, ci=None, scatter_kws={"s": 60, "alpha": 0.7})
    plt.xlabel(f"Manual {metric}")
    plt.ylabel(f"Predicted {metric}")
    plt.title(f"{metric} correlation\nPearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}")
    if save_prefix:
        plt.savefig(f"{save_prefix}_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Bland-Altman plot
    mean_vals = (x + y) / 2
    diff_vals = y - x
    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals)

    plt.figure(figsize=(6, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.7)
    plt.axhline(mean_diff, color="red", linestyle="--")
    plt.axhline(mean_diff + 1.96 * sd_diff, color="gray", linestyle="--")
    plt.axhline(mean_diff - 1.96 * sd_diff, color="gray", linestyle="--")
    plt.xlabel(f"Mean {metric}")
    plt.ylabel(f"Predicted - Manual {metric}")
    plt.title(f"Bland-Altman for {metric}")
    if save_prefix:
        plt.savefig(f"{save_prefix}_bland_altman.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "bland_altman_mean_diff": mean_diff,
        "bland_altman_sd_diff": sd_diff,
    }


if __name__ == "__main__":
    all_fold_results = []
    metrics_union = set()

    # per fold analysis
    for fold_num in range(4):
        print(f"Processing fold {fold_num}")

        results_dir = f"fold_{fold_num}/results/volumetric_analysis"
        label_csv = os.path.join(results_dir, "label_volumetric.csv")
        pred_csv = os.path.join(results_dir, "pred_volumetric.csv")

        manual_df = pd.read_csv(label_csv)
        results_df = pd.read_csv(pred_csv)

        # Merge to get common cases
        df = pd.merge(results_df, manual_df, on="case", suffixes=("_pred", "_manual"))
        metrics = [col.replace("_manual", "") for col in df.columns if col.endswith("_manual")]
        metrics_union.update(metrics)

        fold_results = {}

        for metric in metrics:
            print(f"  Analyzing: {metric}")
            save_prefix = os.path.join(f"fold_{fold_num}", "plots", metric)
            res = correlation_analysis(results_df, manual_df, metric=metric, save_prefix=save_prefix)
            if res:
                fold_results[metric] = res

        # save fold JSON
        os.makedirs(f"fold_{fold_num}", exist_ok=True)
        with open(f"fold_{fold_num}/all_correlation_results.json", "w") as f:
            json.dump(fold_results, f, indent=4)

        all_fold_results.append((results_df, manual_df))

    # -------- combined analysis across folds --------
    print("Running combined analysis across all folds...")

    combined_preds = pd.concat([r for r, _ in all_fold_results], ignore_index=True)
    combined_labels = pd.concat([m for _, m in all_fold_results], ignore_index=True)

    os.makedirs("all_folds", exist_ok=True)

    combined_results = {}
    for metric in metrics_union:
        print(f"Analyzing combined: {metric}")
        res = correlation_analysis(
            combined_preds, combined_labels, metric=metric,
            save_prefix=os.path.join("all_folds", metric)
        )
        if res:
            combined_results[metric] = res

    with open("all_folds/all_correlation_results.json", "w") as f:
        json.dump(combined_results, f, indent=4)

    # -------- summary plot of correlations --------
    summary_df = pd.DataFrame(combined_results).T.reset_index().rename(columns={"index": "metric"})

    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary_df.melt(id_vars="metric", value_vars=["pearson_r", "spearman_r"]),
                x="metric", y="value", hue="variable")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Correlation Coefficient")
    plt.title("Correlation Summary Across All Folds")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("all_folds/summary_correlations.png", dpi=300)
    plt.close()

    print("✅ Done! Results saved in fold_* and all_folds/")
