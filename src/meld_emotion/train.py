"""Fine-tune a transformer on MELD; select the checkpoint with validation macro-F1 only."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from .data import LABEL_TO_ID, LABELS, build_examples, class_weights, read_split
from .metrics import scores
from .modeling import DialogueDataset, collect_logits, configure_tokenizer, pick_device

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "revision": "main", "context_size": 0, "use_speakers": False, "epochs": 4, "batch_size": 16,
    "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_fraction": 0.1, "max_length": 128,
    "patience": 2, "max_grad_norm": 1.0, "accumulation_steps": 1, "eval_batch_size": 16,
}
PROGRESS_EVERY = 100
MAX_BAD_BATCHES = 5


def load_config(path):
    values = {**DEFAULTS, **json.loads(Path(path).read_text())}
    if not values.get("name") or not values.get("model"):
        raise ValueError("A config needs both name and model")
    return values


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_revision(model, revision):
    """Record an exact commit so a run can be repeated after the model repository changes."""
    from huggingface_hub import HfApi

    return HfApi(token=False).model_info(model, revision=revision).sha


def labels_of(examples):
    return np.array([LABEL_TO_ID[example.utterance.emotion] for example in examples], dtype=np.int64)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, max_grad_norm, accumulation_steps=1):
    """Update once per accumulation_steps batches, so long inputs fit in memory."""
    model.train()
    losses = []
    bad = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        labels = batch.pop("labels").to(device)
        logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
        loss = criterion(logits, labels)
        # MPS can return a non-finite loss; skip the batch rather than corrupt the weights.
        if torch.isfinite(loss).item():
            (loss / accumulation_steps).backward()
            losses.append(loss.detach())
        else:
            bad += 1
        if step % accumulation_steps == 0 or step == len(loader):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if torch.isfinite(norm).item():
                optimizer.step()
            else:
                bad += 1
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if bad > MAX_BAD_BATCHES:
            raise RuntimeError("Training loss became non-finite")
        if step % PROGRESS_EVERY == 0:
            if bad:
                print(f"  skipped {bad} invalid batches so far this epoch", flush=True)
            # Cached MPS memory otherwise accumulates and slows the epoch.
            if device.type == "mps":
                torch.mps.empty_cache()
            speed = (time.time() - started) / PROGRESS_EVERY
            print(f"  step {step}/{len(loader)} | {speed:.2f}s per step", flush=True)
            started = time.time()
    return torch.stack(losses).mean().item()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--limit", type=int, help="Use only this many train/validation lines (smoke test)")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.epochs:
        config["epochs"] = args.epochs
    root = args.output / "smoke" if args.limit else args.output
    folder = root / config["name"] / f"seed_{args.seed}"
    if (folder / "summary.json").exists() and not args.limit:
        print(f"Already finished, keeping it: {folder}", flush=True)
        return
    # An interrupted run has no summary, so it resumes from its last finished epoch.
    folder.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = pick_device(args.device)
    train_examples = build_examples(read_split("train"), config["context_size"])
    validation_examples = build_examples(read_split("validation"), config["context_size"])
    if args.limit:
        train_examples, validation_examples = train_examples[:args.limit], validation_examples[:args.limit]

    revision = resolve_revision(config["model"], config["revision"])
    cache = ROOT / "data" / "hf_cache"
    tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(config["model"], revision=revision, cache_dir=cache))
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"], revision=revision, cache_dir=cache, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID, use_safetensors=True,
    ).to(device)

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    def make_dataset(items):
        return DialogueDataset(items, tokenizer, config["use_speakers"], config["max_length"])
    generator = torch.Generator()
    train_loader = DataLoader(make_dataset(train_examples), batch_size=config["batch_size"], shuffle=True,
                              collate_fn=collator, generator=generator)
    validation_loader = DataLoader(make_dataset(validation_examples), batch_size=config["eval_batch_size"],
                                   collate_fn=collator)

    y_train, y_validation = labels_of(train_examples), labels_of(validation_examples)
    weights = torch.tensor(class_weights(y_train.tolist()), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config["learning_rate"])
    updates_per_epoch = math.ceil(len(train_loader) / config["accumulation_steps"])
    total_steps = config["epochs"] * updates_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config["warmup_fraction"]), total_steps,
    )

    record = {
        **config, "seed": args.seed, "resolved_revision": revision, "device": str(device),
        "torch_version": torch.__version__, "train_examples": len(y_train),
        "validation_examples": len(y_validation), "test_evaluated": False, "smoke_test": bool(args.limit),
        "class_weights": weights.tolist(),
    }
    (folder / "config.json").write_text(json.dumps(record, indent=2))
    print(f"{config['name']} | seed {args.seed} | {device} | {len(y_train):,} train lines", flush=True)

    best, best_epoch, stale, history, best_metrics = -1.0, 0, 0, [], None
    resume_path = folder / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best, best_epoch, stale = saved["best"], saved["best_epoch"], saved["stale"]
        history, best_metrics = saved["history"], saved["best_metrics"]
        print(f"Resuming after finished epoch {len(history)}", flush=True)
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        generator.manual_seed(args.seed + epoch)
        epoch_started = time.time()
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device,
                           config["max_grad_norm"], config["accumulation_steps"])
        logits = collect_logits(model, validation_loader, device)
        metrics = scores(y_validation, logits.argmax(axis=1))
        improved = metrics["macro_f1"] > best
        seconds = time.time() - epoch_started
        history.append({"epoch": epoch, "train_loss": loss, "seconds": seconds, **metrics})
        (folder / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch} | loss {loss:.3f} | macro-F1 {metrics['macro_f1']:.3f} | "
              f"weighted-F1 {metrics['weighted_f1']:.3f} | {history[-1]['seconds']:.0f}s"
              f"{'  <- best' if improved else ''}", flush=True)
        if improved:
            best, best_epoch, stale = metrics["macro_f1"], epoch, 0
            model.save_pretrained(folder / "model")
            tokenizer.save_pretrained(folder / "model")
            np.savez_compressed(folder / "validation_logits.npz", logits=logits, labels=y_validation,
                                keys=np.array([item.utterance.key for item in validation_examples]))
            best_metrics = metrics
        else:
            stale += 1
        if device.type == "mps":
            torch.mps.empty_cache()
        if stale >= config["patience"]:
            print(f"Stopping: no improvement for {stale} epochs", flush=True)
            break
        # Written to a temporary name first, so an interrupted save keeps the last good checkpoint.
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "best": best, "best_epoch": best_epoch, "stale": stale,
            "history": history, "best_metrics": best_metrics,
        }, folder / "resume.tmp")
        (folder / "resume.tmp").replace(resume_path)

    (folder / "summary.json").write_text(json.dumps({
        "best_epoch": best_epoch, "validation": best_metrics,
        "seconds": sum(row["seconds"] for row in history), "test_evaluated": False,
    }, indent=2))
    resume_path.unlink(missing_ok=True)
    print(f"Best epoch {best_epoch}: validation macro-F1 {best:.3f}. Saved {folder}", flush=True)


if __name__ == "__main__":
    main()
