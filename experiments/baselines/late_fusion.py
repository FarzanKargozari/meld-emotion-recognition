"""Compare four predeclared text/audio probability mixtures on validation only."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from evaluate_models import RUNS, write_csv
from inspect_data import load
from train_model import make_model, scores


ROOT = Path(__file__).resolve().parents[2]
AUDIO_WEIGHTS = (0.0, 0.05, 0.10, 0.20)


def mix_probabilities(text, audio, audio_weight):
    if text.shape != audio.shape or not 0 <= audio_weight <= 1:
        raise ValueError("Probability arrays must match and weight must be in [0, 1]")
    return (1.0 - audio_weight) * text + audio_weight * audio


@torch.no_grad()
def validation_probabilities(folder, modality, feature_dict, keys, label_index, device):
    config = json.loads((folder / "config.json").read_text())
    if config["modality"] != modality or config["label_index"] != label_index:
        raise ValueError(f"Checkpoint modality or label order mismatch: {folder}")
    if set(feature_dict) != set(keys):
        raise ValueError(f"Feature IDs do not match validation labels: {modality}")
    features = np.stack([feature_dict[key] for key in keys]).astype(np.float32)
    if features.shape[1] != config["input_size"] or not np.isfinite(features).all():
        raise ValueError(f"Invalid {modality} features")
    # Reuse each model's own training-only scaler, without refitting.
    with np.load(folder / "scaler.npz") as saved:
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = saved["mean"], saved["scale"]
        scaler.n_features_in_ = len(scaler.mean_)
    features = scaler.transform(features).astype(np.float32)
    model = make_model(config["input_size"], len(label_index), dropout=config.get("dropout", 0.3))
    model.load_state_dict(torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    probabilities = torch.cat([
        torch.softmax(model(batch.to(device)), dim=1).cpu()
        for batch in torch.from_numpy(features).split(256)
    ]).numpy()
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid probability output")
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-run", default=RUNS["text"])
    parser.add_argument("--audio-run", default=RUNS["audio"])
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(("mps" if torch.backends.mps.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this session; use --device cpu")
    print(f"Device: {device}. Comparing saved models; no retraining.", flush=True)
    print("Validation only. Audio weights: 0%, 5%, 10%, 20%.", flush=True)

    rows, _, _, _, _, label_index = load("data_emotion.p")
    subset = [row for row in rows if row["split"] == "val"]
    keys = [f"{row['dialog']}_{row['utterance']}" for row in subset]
    if len(keys) != 1109 or len(set(keys)) != len(keys):
        raise ValueError("Unexpected validation count or duplicate IDs")
    labels = np.array([label_index[row["y"]] for row in subset], dtype=np.int64)
    names = [name for name, index in sorted(label_index.items(), key=lambda item: item[1])]
    runs = {"text": args.text_run, "audio": args.audio_run}
    filenames = {"text": "text_glove_average_emotion.pkl",
                 "audio": "audio_embeddings_feature_selection_emotion.pkl"}
    probabilities = {}
    for modality, run in runs.items():
        folder = ROOT / "results" / run
        probabilities[modality] = validation_probabilities(
            folder, modality, load(filenames[modality])[1], keys, label_index, device,
        )
        # Confirm the original checkpoints before comparing their mixtures.
        measured = scores(labels, probabilities[modality].argmax(axis=1), len(label_index))
        expected = json.loads((folder / "summary.json").read_text())["validation"]
        if any(abs(measured[key] - expected[key]) > 0.001 for key in measured):
            raise ValueError(f"Reloaded {modality} metrics differ from recorded validation metrics")

    comparison, predictions = [], []
    for audio_weight in AUDIO_WEIGHTS:
        predicted = mix_probabilities(probabilities["text"], probabilities["audio"], audio_weight).argmax(axis=1)
        predictions.append(predicted)
        comparison.append({"text_weight": 1 - audio_weight, "audio_weight": audio_weight,
                           **scores(labels, predicted, len(label_index))})
    # Ties favor the smaller audio contribution; candidates are ordered accordingly.
    best_index = max(range(len(comparison)), key=lambda index: comparison[index]["macro_f1"])
    best, baseline = comparison[best_index], comparison[0]
    delta = best["macro_f1"] - baseline["macro_f1"]
    changed = int(np.count_nonzero(predictions[best_index] != predictions[0]))
    corrected = int(np.count_nonzero((predictions[0] != labels) & (predictions[best_index] == labels)))
    harmed = int(np.count_nonzero((predictions[0] == labels) & (predictions[best_index] != labels)))

    output = args.output or ROOT / "results" / f"late_fusion_validation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "comparison.csv", comparison)
    summary = {
        "split": "validation", "examples": len(labels), "runs": runs,
        "device": str(device), "torch_version": str(torch.__version__),
        "method": "Weighted average of uncalibrated softmax probabilities from frozen models",
        "candidate_audio_weights": list(AUDIO_WEIGHTS),
        "selection_rule": "Highest validation macro-F1; ties favor less audio",
        "selected": best, "delta_macro_f1_vs_text": delta,
        "changed_predictions_vs_text": changed, "corrected_vs_text": corrected,
        "harmed_vs_text": harmed, "comparison": comparison, "test_evaluated": False,
        "limitation": "Exploratory validation selection on one pair of trained checkpoints; not independent evidence of a generalization gain",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "selected_validation_report.txt").write_text(classification_report(
        labels, predictions[best_index], labels=list(range(len(names))), target_names=names,
        digits=3, zero_division=0,
    ))
    write_csv(output / "validation_predictions.csv", [
        {"utterance_key": key, "true_emotion": names[labels[index]],
         **{f"audio_{int(weight * 100)}pct": names[predictions[mixture][index]]
            for mixture, weight in enumerate(AUDIO_WEIGHTS)}}
        for index, key in enumerate(keys)
    ])
    print("\nValidation late-fusion comparison")
    print(f"{'Text / audio':<16} {'Macro-F1':>10} {'Weighted-F1':>13} {'Accuracy':>10}")
    for result in comparison:
        label = f"{result['text_weight']:.0%} / {result['audio_weight']:.0%}"
        print(f"{label:<16} {result['macro_f1']:>10.3f} {result['weighted_f1']:>13.3f} {result['accuracy']:>9.1%}")
    print(f"\nBest validation mixture: {best['text_weight']:.0%} text + {best['audio_weight']:.0%} audio")
    print(f"Macro-F1 change versus text: {delta:+.4f}")
    print(f"Changed predictions: {changed}; corrected: {corrected}; newly incorrect: {harmed}")
    print("This is a validation result, not a demonstrated test improvement.")
    print(f"Saved results: {output}")
    print("Test data was not evaluated in this experiment.")


if __name__ == "__main__":
    main()
