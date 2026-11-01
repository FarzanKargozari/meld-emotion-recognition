"""Describe where the final ensemble goes wrong. Reporting only: nothing here may change the model."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from meld_emotion.data import LABELS, read_split
from meld_emotion.metrics import scores

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts" / "test_roberta_context3.npz"
OUTPUT = ROOT / "reports" / "test" / "roberta_context3"
LENGTHS = ((1, 3), (4, 8), (9, 15), (16, 10_000))
THRESHOLDS = (0.0, 0.4, 0.6, 0.8)


def group_scores(labels, predictions, mask):
    return {"lines": int(mask.sum()), **scores(labels[mask], predictions[mask])}


def top_confusions(matrix, limit=8):
    rows = []
    for true_index, name in enumerate(LABELS):
        total = int(matrix[true_index].sum())
        for predicted_index, other in enumerate(LABELS):
            if true_index != predicted_index:
                count = int(matrix[true_index, predicted_index])
                rows.append({"true": name, "predicted": other, "lines": count,
                             "share_of_true": count / total})
    return sorted(rows, key=lambda row: -row["lines"])[:limit]


def confusion_figure(matrix, path):
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    colors = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#2a78d6", "#123d73"])
    figure, axis = plt.subplots(figsize=(7.2, 6.2), dpi=160)
    axis.imshow(shares, cmap=colors, vmin=0, vmax=1)
    axis.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
    axis.set_yticks(range(len(LABELS)), [f"{name} ({int(matrix[i].sum())})" for i, name in enumerate(LABELS)])
    axis.set_xlabel("Predicted emotion")
    axis.set_ylabel("True emotion (test lines)")
    axis.set_title("Each row shows where that emotion's lines went", loc="left", fontsize=12)
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            ink = "white" if shares[row, column] > 0.45 else "#1f1f1e"
            axis.text(column, row, f"{shares[row, column]:.0%}", ha="center", va="center", color=ink,
                      fontsize=9)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main():
    with np.load(PREDICTIONS, allow_pickle=False) as saved:
        probabilities, labels, keys = saved["probabilities"], saved["labels"], saved["keys"]
    utterances = {item.key: item for item in read_split("test")}
    lines = [utterances[str(key)] for key in keys]
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    words = np.array([len(item.text.split()) for item in lines])
    first = np.array([item.key == min((other for other in lines if other.dialogue == item.dialogue),
                                      key=lambda other: other.index).key for item in lines])

    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(matrix, (labels, predictions), 1)
    report = {
        "note": "Three-seed ensemble on the test split, described after the model was final.",
        "top_confusions": top_confusions(matrix),
        "by_line_length": {f"{low}+ words" if high > 1000 else f"{low}-{high} words":
                           group_scores(labels, predictions, (words >= low) & (words <= high))
                           for low, high in LENGTHS},
        "by_position": {"first line of a dialogue": group_scores(labels, predictions, first),
                        "later lines": group_scores(labels, predictions, ~first)},
        "by_confidence": [{"minimum_confidence": threshold,
                           "share_of_lines": float((confidence >= threshold).mean()),
                           "accuracy": float((predictions == labels)[confidence >= threshold].mean())}
                          for threshold in THRESHOLDS],
    }
    (OUTPUT / "error_analysis.json").write_text(json.dumps(report, indent=2))
    confusion_figure(matrix, OUTPUT / "confusion_matrix.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
