"""Save per-round history, per-class scores and paired gaps for finished runs. Reads validation data only."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from meld_emotion.data import LABELS

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "reports" / "validation"
DESIGNS = ("distilroberta_utterance", "distilroberta_context3", "distilroberta_context3_speakers",
           "roberta_context3")
SEEDS = (42, 7, 123)
RESAMPLES = 2000


def macro_f1(labels, predictions):
    return f1_score(labels, predictions, average="macro", labels=range(len(LABELS)), zero_division=0)


def softmax(logits):
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def main():
    runs = {design: [np.load(ARTIFACTS / design / f"seed_{seed}" / "validation_logits.npz") for seed in SEEDS]
            for design in DESIGNS}
    labels, keys = runs[DESIGNS[0]][0]["labels"], runs[DESIGNS[0]][0]["keys"]
    for group in runs.values():
        for run in group:
            assert (run["keys"] == keys).all() and (run["labels"] == labels).all()
    predictions = {design: [run["logits"].argmax(axis=1) for run in group] for design, group in runs.items()}

    per_class = {}
    ensembles = {}
    for design, group in runs.items():
        values = np.mean([f1_score(labels, row, average=None, labels=range(len(LABELS)), zero_division=0)
                          for row in predictions[design]], axis=0)
        per_class[design] = dict(zip(LABELS, values.tolist(), strict=True))
        combined = sum(softmax(run["logits"]) for run in group).argmax(axis=1)
        ensembles[design] = {"macro_f1": macro_f1(labels, combined),
                             "accuracy": float((combined == labels).mean())}

    # Lines of one dialogue are related, so whole dialogues are resampled.
    dialogues = np.array([str(key).rsplit("_", 1)[0] for key in keys])
    names = np.unique(dialogues)
    lines = {name: np.flatnonzero(dialogues == name) for name in names}
    random = np.random.default_rng(0)
    samples = [np.concatenate([lines[name] for name in random.choice(names, len(names))])
               for _ in range(RESAMPLES)]

    def mean_score(design, rows):
        return np.mean([macro_f1(labels[rows], row[rows]) for row in predictions[design]])

    gaps = []
    everything = np.arange(len(labels))
    for first, second in ((DESIGNS[1], DESIGNS[0]), (DESIGNS[2], DESIGNS[0]), (DESIGNS[1], DESIGNS[2]),
                          (DESIGNS[3], DESIGNS[1])):
        values = np.array([mean_score(first, rows) - mean_score(second, rows) for rows in samples])
        low, high = np.percentile(values, [2.5, 97.5])
        gaps.append({"first": first, "second": second,
                     "gap": float(mean_score(first, everything) - mean_score(second, everything)),
                     "ci_low": float(low), "ci_high": float(high),
                     "share_above_zero": float((values > 0).mean())})

    def read_history(design, seed):
        return json.loads((ARTIFACTS / design / f"seed_{seed}" / "history.json").read_text())

    history = {design: {str(seed): read_history(design, seed) for seed in SEEDS} for design in DESIGNS}
    (OUTPUT / "gaps.json").write_text(json.dumps({
        "dialogues": len(names), "resamples": RESAMPLES, "gaps": gaps,
        "per_class_f1": per_class, "three_seed_ensembles": ensembles,
    }, indent=2))
    (OUTPUT / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(gaps, indent=2))
    print(json.dumps(ensembles, indent=2))


if __name__ == "__main__":
    main()
