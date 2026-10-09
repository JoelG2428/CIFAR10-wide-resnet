"""Render saved experimental evidence. No training or dataset access."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
OUT.mkdir(parents=True, exist_ok=True)
history = json.loads((ROOT / "results/selection/history.json").read_text())
final = json.loads((ROOT / "results/final/history.json").read_text())
test = json.loads((ROOT / "results/evaluation/test_results.json").read_text())
best = max(history, key=lambda row: row["validation_accuracy"])
cm = np.array(test["confusion_matrix"])
assert cm.sum() == test["test_images"] == 10000
assert np.trace(cm) == test["correct"] == 9719
assert np.isclose(np.trace(cm) / cm.sum(), test["overall_accuracy"])
assert best["epoch"] == len(final) == test["epoch"] == 196
assert len(history) == 200 and np.isclose(best["validation_accuracy"], .9728)
plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False, "axes.spines.right":False, "figure.facecolor":"#f8fafc", "axes.facecolor":"#f8fafc", "font.size":11})
epochs = [r["epoch"] for r in history]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
for key, label, color in [("train_accuracy", "Augmented training", "#64748b"), ("validation_accuracy", "Clean validation", "#2563eb")]:
    axes[0].plot(epochs, [100*r[key] for r in history], label=label, color=color)
axes[0].scatter([196], [97.28], color="#db2777", zorder=4)
axes[0].set(title="Selection: 97.28% at epoch 196", xlabel="Epoch", ylabel="Accuracy (%)")
axes[0].legend(fontsize=9)
for key, label, color in [("train_loss", "Augmented training", "#64748b"), ("validation_loss", "Clean validation", "#2563eb")]:
    axes[1].plot(epochs, [r[key] for r in history], label=label, color=color)
axes[1].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
axes[2].plot(epochs, [r["learning_rate"] for r in history], color="#7c3aed")
axes[2].set(title="Cosine learning-rate schedule", xlabel="Epoch", ylabel="Learning rate")
fig.savefig(OUT / "training.png", dpi=170); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
classes = test["class_names"]
scores = 100 * cm.diagonal() / cm.sum(axis=1)
ax.barh(classes[::-1], scores[::-1], color="#2563eb")
ax.set(xlim=(0, 105), xlabel="Test accuracy (%)", title="Final model: 97.19% on 10,000 test images")
for i, score in enumerate(scores[::-1]): ax.text(score + .6, i, f"{score:.1f}%", va="center", fontsize=10)
fig.savefig(OUT / "class_accuracy.png", dpi=170); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1000)
ax.set(xticks=range(10), yticks=range(10), xticklabels=classes, yticklabels=classes, xlabel="Predicted class", ylabel="True class", title="Confusion matrix: counts out of 1,000 per class")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
for i in range(10):
    for j in range(10): ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j] > 500 else "#1e293b", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.8)
fig.savefig(OUT / "confusion_matrix.png", dpi=170); plt.close(fig)
with (ROOT / "results/evaluation/per_class_accuracy.csv").open("w") as f:
    writer = csv.writer(f); writer.writerow(["class", "correct", "total", "accuracy"])
    for i, name in enumerate(classes): writer.writerow([name, int(cm[i,i]), int(cm[i].sum()), float(scores[i]/100)])
print("Verified 97.19% test accuracy and epoch-196 selection; generated three visualizations.")
