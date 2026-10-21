"""Draw the held-out precision-recall curve from saved validation results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
results = json.loads((ROOT / "validation_results.json").read_text())
curve = results["holdout"]["selected"]["precision_recall_curve"]

fig, ax = plt.subplots(figsize=(3.5, 2.35))
ax.plot(curve["recall"], curve["precision"], color="#183b65", linewidth=1.5)
ax.scatter(
    curve["chosen_recall"],
    curve["chosen_precision"],
    color="#d47928",
    s=36,
    label="Training-F1 cutoff",
    zorder=3,
)
ax.scatter(
    curve["default_recall"],
    curve["default_precision"],
    facecolors="white",
    edgecolors="#183b65",
    s=36,
    label="Probability 0.5",
    zorder=3,
)
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
ax.grid(color="#d6dce3", linewidth=0.6)
ax.legend(loc="upper right", frameon=False, fontsize=7)
fig.tight_layout()
output = ROOT / "figures" / "precision_recall.pdf"
output.parent.mkdir(exist_ok=True)
fig.savefig(output)
print(f"Wrote {output}")
