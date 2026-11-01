"""Summarize finished experiments on validation data. This command never reads the test split."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 7, 123)


def collect(root):
    rows = []
    for folder in sorted(Path(root).glob("*/seed_*")):
        if folder.parent.name == "smoke" or not (folder / "summary.json").exists():
            continue
        summary = json.loads((folder / "summary.json").read_text())
        rows.append({"experiment": folder.parent.name, "seed": int(folder.name.removeprefix("seed_")),
                     "best_epoch": summary["best_epoch"], "minutes": summary["seconds"] / 60,
                     **summary["validation"]})
    return rows


def aggregate(rows):
    results = []
    for name in sorted({row["experiment"] for row in rows}):
        group = [row for row in rows if row["experiment"] == name]
        entry = {"experiment": name, "seeds": sorted(row["seed"] for row in group)}
        for metric in ("macro_f1", "weighted_f1", "accuracy"):
            values = [row[metric] for row in group]
            entry[metric] = mean(values)
            entry[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        results.append(entry)
    return sorted(results, key=lambda entry: -entry["macro_f1"])


def to_markdown(results):
    lines = ["| Experiment | Seeds | Macro-F1 | Weighted-F1 | Accuracy |", "|---|---:|---:|---:|---:|"]
    for entry in results:
        lines.append(f"| {entry['experiment']} | {len(entry['seeds'])} "
                     f"| {entry['macro_f1']:.3f} ± {entry['macro_f1_sd']:.3f} "
                     f"| {entry['weighted_f1']:.3f} ± {entry['weighted_f1_sd']:.3f} "
                     f"| {entry['accuracy']:.1%} ± {entry['accuracy_sd']:.1%} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "validation")
    args = parser.parse_args(argv)
    rows = collect(args.artifacts)
    if not rows:
        parser.error(f"No finished runs in {args.artifacts}")
    results = aggregate(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "runs.json").write_text(json.dumps(rows, indent=2))
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2))
    table = to_markdown(results)
    (args.output / "comparison.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
