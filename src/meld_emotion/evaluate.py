"""Score a finished experiment. The test split needs --final and is logged in a ledger."""

import argparse
import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .data import LABEL_TO_ID, LABELS, build_examples, read_split
from .metrics import (
    bootstrap_interval,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    scores,
    softmax,
)
from .modeling import DialogueDataset, collect_logits, configure_tokenizer, pick_device

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "reports" / "test_ledger.json"


def seed_logits(folder, examples, device):
    config = json.loads((folder / "config.json").read_text())
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(folder / "model"))
    model = AutoModelForSequenceClassification.from_pretrained(folder / "model").to(device)
    dataset = DialogueDataset(examples, tokenizer, config["use_speakers"], config["max_length"])
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(dataset, batch_size=16, collate_fn=collator)
    return collect_logits(model, loader, device)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="Folder holding seed_* runs")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--final", action="store_true", help="Required to read the test split")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final:
        parser.error("Test evaluation is final: choose the model on validation, then add --final")

    folders = sorted(path for path in args.experiment.glob("seed_*") if (path / "summary.json").exists())
    if not folders:
        parser.error(f"No finished seeds in {args.experiment}")
    configs = [json.loads((folder / "config.json").read_text()) for folder in folders]
    if len({config["context_size"] for config in configs}) != 1:
        raise ValueError("Seeds in one experiment must share a context size")
    examples = build_examples(read_split(args.split), configs[0]["context_size"])
    labels = np.array([LABEL_TO_ID[item.utterance.emotion] for item in examples])
    dialogues = np.array([item.utterance.dialogue for item in examples])
    device = pick_device(args.device)

    per_seed, logits_by_seed, temperatures = [], [], []
    for folder in folders:
        logits = seed_logits(folder, examples, device)
        logits_by_seed.append(logits)
        with np.load(folder / "validation_logits.npz", allow_pickle=False) as saved:
            temperature = fit_temperature(saved["logits"], saved["labels"])
        temperatures.append(temperature)
        probabilities = softmax(logits / temperature)
        per_seed.append({
            "seed": int(folder.name.removeprefix("seed_")), **scores(labels, logits.argmax(axis=1)),
            "temperature": temperature, "nll_raw": negative_log_likelihood(logits, labels),
            "nll_calibrated": negative_log_likelihood(logits / temperature, labels),
            "ece_raw": expected_calibration_error(softmax(logits), labels),
            "ece_calibrated": expected_calibration_error(probabilities, labels),
        })

    # Average calibrated probabilities across seeds for the ensemble.
    ensemble_probabilities = np.mean([softmax(logits / temperature) for logits, temperature
                                      in zip(logits_by_seed, temperatures, strict=True)], axis=0)
    ensemble = ensemble_probabilities.argmax(axis=1)
    ensemble_metrics = scores(labels, ensemble)
    ensemble_metrics["macro_f1_95ci"] = bootstrap_interval(labels, ensemble, dialogues, "macro_f1")
    ensemble_metrics["weighted_f1_95ci"] = bootstrap_interval(labels, ensemble, dialogues, "weighted_f1")
    ensemble_metrics["ece"] = expected_calibration_error(ensemble_probabilities, labels)

    summary = {
        "split": args.split, "experiment": args.experiment.name, "examples": len(labels),
        "seed_mean": {key: mean(row[key] for row in per_seed) for key in per_seed[0] if key != "seed"},
        "seed_sd": {key: stdev(row[key] for row in per_seed) if len(per_seed) > 1 else 0.0
                    for key in per_seed[0] if key != "seed"},
        "per_seed": per_seed, "ensemble": ensemble_metrics,
    }
    output = args.output or ROOT / "reports" / args.split / args.experiment.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "ensemble_report.txt").write_text(classification_report(
        labels, ensemble, labels=list(range(len(LABELS))), target_names=LABELS, digits=3, zero_division=0))
    matrix = confusion_matrix(labels, ensemble, labels=list(range(len(LABELS))))
    with (output / "ensemble_confusion.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_emotion", *LABELS])
        writer.writerows([[name, *row] for name, row in zip(LABELS, matrix.tolist(), strict=True)])
    np.savez_compressed(ROOT / "artifacts" / f"{args.split}_{args.experiment.name}.npz",
                        probabilities=ensemble_probabilities, labels=labels,
                        keys=np.array([item.utterance.key for item in examples]))

    if args.split == "test":
        ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
        ledger.append({
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "experiment": args.experiment.name, "commit": git_commit(),
            "seeds": [row["seed"] for row in per_seed],
            "macro_f1_seed_mean": summary["seed_mean"]["macro_f1"],
            "weighted_f1_seed_mean": summary["seed_mean"]["weighted_f1"],
        })
        LEDGER.write_text(json.dumps(ledger, indent=2))

    print(f"{args.split}: {args.experiment.name} across {len(per_seed)} seeds")
    print(f"Macro-F1 {summary['seed_mean']['macro_f1']:.3f} ± {summary['seed_sd']['macro_f1']:.3f} | "
          f"weighted-F1 {summary['seed_mean']['weighted_f1']:.3f} ± {summary['seed_sd']['weighted_f1']:.3f}")
    print(f"Ensemble macro-F1 {ensemble_metrics['macro_f1']:.3f} | "
          f"weighted-F1 {ensemble_metrics['weighted_f1']:.3f}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
