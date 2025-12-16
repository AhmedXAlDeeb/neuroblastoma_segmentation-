import json
import numpy as np

with open("segmentation_metrics.json", "r") as f:
    data = json.load(f)

metrics = ["dice", "jaccard", "precision", "recall", "hausdorff95", "asd"]

summary = {"overall": {}, "per_fold": {}}

# Per-fold stats
for fold, cases in data.items():
    fold_stats = {}
    for metric in metrics:
        values = [case[metric] for case in cases if case[metric] is not None and not isinstance(case[metric], str)]
        values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        if values:
            fold_stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values)
            }
        else:
            fold_stats[metric] = None
    summary["per_fold"][fold] = fold_stats

# Overall stats
for metric in metrics:
    all_values = []
    for fold, cases in data.items():
        values = [case[metric] for case in cases if case[metric] is not None and not isinstance(case[metric], str)]
        values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        all_values.extend(values)
    if all_values:
        summary["overall"][metric] = {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "min": float(np.min(all_values)),
            "max": float(np.max(all_values)),
            "n": len(all_values)
        }
    else:
        summary["overall"][metric] = None

# Print summary
import pprint
pprint.pprint(summary)

# Optionally, save to JSON
with open("segmentation_metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=4)