"""Train a small MELD classifier; select checkpoints using validation only."""

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from inspect_data import DATA, load


def make_model(input_size, num_classes, dropout=0.3):
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be at least 0 and less than 1")
    return nn.Sequential(
        nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )


def prepare_minilm_data(rows, label_index):
    folder = DATA / "minilm_features"
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata["label_index"] != label_index or metadata["dimension"] != 384:
        raise ValueError("MiniLM metadata has unexpected labels or feature dimensions")
    digest = hashlib.sha256((DATA / "data_emotion.p").read_bytes()).hexdigest()
    if metadata["source_sha256"] != digest:
        raise ValueError("MiniLM features were extracted from a different source file")
    splits = []
    for split, count in [("train", 9989), ("val", 1109)]:
        subset = [row for row in rows if row["split"] == split]
        expected_keys = [f"{row['dialog']}_{row['utterance']}" for row in subset]
        expected_labels = np.array([label_index[row["y"]] for row in subset], dtype=np.int64)
        with np.load(folder / f"{split}.npz", allow_pickle=False) as saved:
            keys = saved["keys"].tolist()
            features = saved["features"].astype(np.float32)
            labels = saved["labels"]
        if (len(expected_keys) != count or len(set(expected_keys)) != count
                or len(keys) != count or len(set(keys)) != count
                or set(keys) != set(expected_keys)):
            raise ValueError(f"MiniLM {split} IDs do not match the official split")
        if features.shape != (count, 384) or labels.shape != (count,):
            raise ValueError(f"Unexpected MiniLM {split} array dimensions")
        positions = {key: index for index, key in enumerate(keys)}
        order = [positions[key] for key in expected_keys]
        features, labels = features[order], labels[order]
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f"MiniLM {split} labels do not match source labels")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite MiniLM {split} features")
        if not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-4):
            raise ValueError(f"MiniLM {split} features are not normalized as expected")
        splits.append((features, expected_labels))
    return splits, label_index


def prepare_data(modality, text_features="glove"):
    rows, _, _, _, _, label_index = load("data_emotion.p")
    if text_features == "minilm":
        if modality != "text":
            raise ValueError("The MiniLM comparison currently supports --modality text only")
        return prepare_minilm_data(rows, label_index)
    text = load("text_glove_average_emotion.pkl")
    audio = load("audio_embeddings_feature_selection_emotion.pkl")
    splits = []
    for index, split in enumerate(["train", "val"]):
        split_rows = [row for row in rows if row["split"] == split]
        keys = [f"{row['dialog']}_{row['utterance']}" for row in split_rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate IDs in {split}")
        if set(keys) != set(text[index]) or set(keys) != set(audio[index]):
            raise ValueError(f"Mismatched feature IDs in {split}")
        parts = []
        if modality in ("text", "both"):
            parts.append(np.stack([text[index][key] for key in keys]))
        if modality in ("audio", "both"):
            parts.append(np.stack([audio[index][key] for key in keys]))
        features = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features in {split}")
        labels = np.array([label_index[row["y"]] for row in split_rows], dtype=np.int64)
        splits.append((features, labels))
    return splits, label_index


def scores(labels, predictions, num_classes):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(num_classes)),
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=list(range(num_classes)),
                                      average="weighted", zero_division=0)),
    }


@torch.no_grad()
def predict(model, features, device):
    model.eval()
    return torch.cat([
        model(batch.to(device)).argmax(dim=1).cpu()
        for batch in torch.from_numpy(features).split(256)
    ]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=["text", "audio", "both"], default="both")
    parser.add_argument("--text-features", choices=["glove", "minilm"], default="glove",
                        help="Text representation; minilm currently requires --modality text")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Hidden-unit dropout probability (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random initialization and shuffle seed (default: 42)")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be at least 0 and less than 1")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if args.text_features == "minilm" and args.modality != "text":
        parser.error("Use --modality text with --text-features minilm")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device(
        ("mps" if torch.backends.mps.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this session; use --device cpu")
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    print(f"Device: {'Apple GPU (MPS)' if device.type == 'mps' else 'CPU'}", flush=True)
    print(f"Random seed: {seed}", flush=True)
    print(f"Dropout: {args.dropout:g}", flush=True)
    print(f"Loading and matching {args.modality} features and labels...", flush=True)
    if args.text_features == "minilm":
        print("Text features: frozen MiniLM sentence embeddings (384 dimensions)", flush=True)
    ((x_train, y_train), (x_val, y_val)), label_index = prepare_data(args.modality, args.text_features)

    # Only training data determines scaling and class weights.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    num_classes = len(label_index)
    counts = np.bincount(y_train, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    class_weights /= class_weights.mean()
    model = make_model(x_train.shape[1], num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    batches = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=32, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )

    run_prefix = f"minilm_{args.modality}" if args.text_features == "minilm" else args.modality
    output = args.output or (Path(__file__).resolve().parents[2] / "results" /
                            f"{run_prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    config = {
        "modality": args.modality, "input_size": x_train.shape[1], "hidden_size": 256,
        "dropout": args.dropout, "label_index": label_index, "seed": seed,
        "batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.01,
        "class_weighting": "inverse square root of training class count, mean normalized",
        "class_weights": class_weights.tolist(), "max_epochs": args.epochs,
        "patience": 5, "selection_metric": "validation macro-F1",
        "device": str(device), "torch_version": str(torch.__version__),
        "train_examples": len(y_train), "validation_examples": len(y_val),
        "test_evaluated": False,
        "text_features": args.text_features if args.modality != "audio" else None,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    if args.text_features == "minilm":
        config["feature_metadata"] = json.loads((DATA / "minilm_features" / "metadata.json").read_text())
    (output / "config.json").write_text(json.dumps(config, indent=2))
    baseline = scores(y_val, np.full_like(y_val, counts.argmax()), num_classes)
    print(f"Training: {len(y_train):,}; validation: {len(y_val):,}; inputs: {x_train.shape[1]:,}")
    print(f"Always predicting the most common training emotion gives validation "
          f"macro-F1 {baseline['macro_f1']:.3f} and accuracy {baseline['accuracy']:.1%}.")
    print("Training for up to 30 epochs (or your --epochs setting); stopping after 5 without improvement.", flush=True)

    best_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for features, labels in batches:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Training loss became non-finite")
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.detach())
        metrics = scores(y_val, predict(model, x_val, device), num_classes)
        loss_value = torch.stack(batch_losses).mean().item()
        improved = metrics["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, stale = metrics["macro_f1"], epoch, 0
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       output / "best_model.pt")
        else:
            stale += 1
        history.append({"epoch": epoch, "mean_batch_train_loss": loss_value, **metrics})
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch:02d} | loss {loss_value:.3f} | validation accuracy "
              f"{metrics['accuracy']:.1%} | macro-F1 {metrics['macro_f1']:.3f}"
              f"{'  <- saved best model' if improved else ''}", flush=True)
        if stale >= 5:
            print("Stopping: validation macro-F1 has not improved for 5 epochs.")
            break

    model.load_state_dict(torch.load(output / "best_model.pt", map_location="cpu", weights_only=True))
    predictions = predict(model, x_val, device)
    metrics = scores(y_val, predictions, num_classes)
    names = [name for name, index in sorted(label_index.items(), key=lambda item: item[1])]
    report = classification_report(y_val, predictions, labels=list(range(num_classes)),
                                   target_names=names, zero_division=0, digits=3)
    (output / "validation_report.txt").write_text(report)
    (output / "summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "validation": metrics,
        "majority_baseline_validation": baseline, "test_evaluated": False,
    }, indent=2))
    print(f"\nFinished. Best epoch: {best_epoch}")
    print(f"Validation macro-F1: {metrics['macro_f1']:.3f}")
    print(f"Validation weighted-F1: {metrics['weighted_f1']:.3f}")
    print(f"Validation accuracy: {metrics['accuracy']:.1%}")
    print(f"Saved model and results: {output}")
    print("Test data has not been evaluated. Keep it for the final comparison.")


if __name__ == "__main__":
    main()
