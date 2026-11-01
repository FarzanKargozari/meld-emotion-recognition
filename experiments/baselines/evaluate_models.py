"""Evaluate three fixed checkpoints without training or tuning on the test set."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from inspect_data import load
from train_model import make_model, predict, scores


ROOT = Path(__file__).resolve().parents[2]
# These are the completed experiments chosen before examining test results.
RUNS = {
    "text": "text_20261030_223057_702040",
    "audio": "audio_20261030_223206_949255",
    "both": "both_20261030_222925_430205",
}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output", type=Path)
    for modality, folder in RUNS.items():
        parser.add_argument(f"--{modality}-run", default=folder,
                            help=f"Saved {modality} run folder inside results/, or an absolute path")
    args = parser.parse_args()
    runs = {mode: getattr(args, f"{mode}_run") for mode in RUNS}
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    configs, summaries = {}, {}
    for modality, name in runs.items():
        folder = ROOT / "results" / name
        configs[modality] = json.loads((folder / "config.json").read_text())
        summaries[modality] = json.loads((folder / "summary.json").read_text())
        if configs[modality]["modality"] != modality:
            raise ValueError(f"Wrong modality in {name}")
    selected = max(runs, key=lambda mode: summaries[mode]["validation"]["macro_f1"])
    print(f"Model selected using validation macro-F1: {selected}", flush=True)
    print(f"Evaluating {args.split} data on {device}; no training or tuning.", flush=True)

    rows, _, _, _, _, label_index = load("data_emotion.p")
    feature_index, split_name = (1, "val") if args.split == "validation" else (2, "test")
    subset = [row for row in rows if row["split"] == split_name]
    keys = [f"{row['dialog']}_{row['utterance']}" for row in subset]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate IDs in evaluation data")
    text = load("text_glove_average_emotion.pkl")[feature_index]
    audio = load("audio_embeddings_feature_selection_emotion.pkl")[feature_index]
    if set(keys) != set(text) or set(keys) != set(audio):
        raise ValueError("Text, audio, and label IDs do not match")
    features = {
        "text": np.stack([text[key] for key in keys]).astype(np.float32),
        "audio": np.stack([audio[key] for key in keys]).astype(np.float32),
    }
    labels = np.array([label_index[row["y"]] for row in subset], dtype=np.int64)
    num_classes = len(label_index)
    names = [name for name, index in sorted(label_index.items(), key=lambda pair: pair[1])]
    train_counts = np.bincount([label_index[row["y"]] for row in rows
                               if row["split"] == "train"], minlength=num_classes)
    majority_predictions = np.full_like(labels, train_counts.argmax())
    predictions_by_model = {"majority": majority_predictions}
    comparison = [{"model": "majority", **scores(labels, majority_predictions, num_classes)}]
    reports = {}

    for modality, name in runs.items():
        folder = ROOT / "results" / name
        config = configs[modality]
        if config["label_index"] != label_index:
            raise ValueError(f"Label mapping differs for {name}")
        values = (np.concatenate([features["text"], features["audio"]], axis=1)
                  if modality == "both" else features[modality].copy())
        if not np.isfinite(values).all() or values.shape[1] != config["input_size"]:
            raise ValueError(f"Invalid input features for {name}")
        # Reuse the training-only scaler exactly; never fit on evaluation data.
        with np.load(folder / "scaler.npz") as saved:
            scaler = StandardScaler()
            scaler.mean_ = saved["mean"]
            scaler.scale_ = saved["scale"]
            scaler.n_features_in_ = len(scaler.mean_)
        values = scaler.transform(values).astype(np.float32)
        model = make_model(config["input_size"], num_classes, dropout=config.get("dropout", 0.3))
        model.load_state_dict(torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True))
        model.to(device)
        predictions = predict(model, values, device)
        predictions_by_model[modality] = predictions
        metrics = scores(labels, predictions, num_classes)
        comparison.append({"model": modality, **metrics})
        reports[modality] = classification_report(
            labels, predictions, labels=list(range(num_classes)), target_names=names,
            digits=3, zero_division=0,
        )
        if args.split == "validation":
            for metric, value in metrics.items():
                if abs(value - summaries[modality]["validation"][metric]) > 0.001:
                    raise ValueError(f"Reloaded {modality} {metric} differs from saved validation result")

    output = args.output or ROOT / "results" / f"{args.split}_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "comparison.csv", comparison)
    for modality, report in reports.items():
        (output / f"{modality}_report.txt").write_text(report)
        matrix = confusion_matrix(labels, predictions_by_model[modality], labels=list(range(num_classes)))
        write_csv(output / f"{modality}_confusion.csv", [
            {"true_emotion": name, **dict(zip(names, matrix[index].tolist()))}
            for index, name in enumerate(names)
        ])
    write_csv(output / "predictions.csv", [
        {"split": split_name, "utterance_key": key, "true_emotion": names[labels[index]],
         **{mode: names[predictions[index]] for mode, predictions in predictions_by_model.items()}}
        for index, key in enumerate(keys)
    ])
    (output / "summary.json").write_text(json.dumps({
        "split": args.split, "examples": len(labels), "selected_by_validation": selected,
        "selection_rule": "Highest validation macro-F1 among the three fixed runs",
        "runs": runs, "device": str(device), "results": comparison,
        "scope": "One seed; classification from published precomputed MELD features",
    }, indent=2))
    print(f"\n{args.split.title()} results ({len(labels):,} examples)")
    print(f"{'Model':<12} {'Macro-F1':>10} {'Weighted-F1':>13} {'Accuracy':>10}")
    for result in comparison:
        print(f"{result['model']:<12} {result['macro_f1']:>10.3f} "
              f"{result['weighted_f1']:>13.3f} {result['accuracy']:>9.1%}")
    print(f"\nSelected model remains: {selected} (chosen using validation)")
    print(f"Reports, predictions, and confusion tables saved: {output}")


if __name__ == "__main__":
    main()
