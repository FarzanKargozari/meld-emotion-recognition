"""Create frozen MiniLM sentence features for MELD train/validation only."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np

from inspect_data import DATA, load


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SPLITS = (("train", 9989), ("val", 1109))


def gather_rows(rows, split, expected, label_index):
    subset = [row for row in rows if row["split"] == split]
    keys = [f"{row['dialog']}_{row['utterance']}" for row in subset]
    if len(subset) != expected or len(keys) != len(set(keys)):
        raise ValueError(f"Unexpected count or duplicate IDs for {split}")
    texts = [row["text"] for row in subset]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError(f"Missing text in {split}")
    labels = np.array([label_index[row["y"]] for row in subset], dtype=np.int64)
    return keys, texts, labels


def check_features(features, count):
    if features.shape != (count, 384) or not np.isfinite(features).all():
        raise ValueError("Sentence features have unexpected dimensions or non-finite values")
    if not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-4):
        raise ValueError("Expected unit-normalized sentence features")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--revision", default="main", help="Model revision, resolved to a fixed commit before downloading")
    parser.add_argument("--output", type=Path, default=DATA / "minilm_features")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}. Keep those features or choose a different --output folder.")

    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple GPU unavailable in this session; use --device cpu")
    print(f"Device: {device}", flush=True)
    print("Reading the same supplied text used in our earlier experiment...", flush=True)
    rows, _, _, _, _, label_index = load("data_emotion.p")
    prepared = {split: gather_rows(rows, split, count, label_index) for split, count in SPLITS}

    print("Resolving the MiniLM version and downloading it if needed...", flush=True)
    revision = HfApi(token=False).model_info(MODEL_ID, revision=args.revision).sha
    if not revision:
        raise RuntimeError("Could not resolve the model revision")
    model = SentenceTransformer(
        MODEL_ID, revision=revision, device=device, trust_remote_code=False,
        cache_folder=str(DATA / "minilm_model_cache"), token=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.get_sentence_embedding_dimension() != 384:
        raise ValueError("Unexpected model embedding dimension")
    print(f"Model loaded. Frozen encoder; batch size {args.batch_size}; 384 features per sentence.", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the feature folder only after both splits pass validation.
    with tempfile.TemporaryDirectory(prefix=".minilm-build-", dir=output.parent) as temporary:
        staged = Path(temporary) / "features"
        staged.mkdir()
        for split, count in SPLITS:
            keys, texts, labels = prepared[split]
            print(f"\nEncoding {split}: {count:,} sentences...", flush=True)
            with torch.inference_mode():
                features = model.encode(
                    texts, batch_size=args.batch_size, show_progress_bar=True,
                    convert_to_numpy=True, normalize_embeddings=True,
                ).astype(np.float32)
            check_features(features, count)
            np.savez_compressed(staged / f"{split}.npz", features=features,
                                labels=labels, keys=np.array(keys, dtype=str))
            print(f"Saved {split}: {features.shape[0]:,} rows x {features.shape[1]} features", flush=True)

        metadata = {
            "model_id": MODEL_ID, "model_revision": revision,
            "requested_revision": args.revision, "device": device,
            "batch_size": args.batch_size, "dimension": 384, "dtype": "float32",
            "normalize_embeddings": True, "max_sequence_length": int(model.max_seq_length),
            "encoder_frozen": True, "dialogue_context": False,
            "text_source": "Provided text field in data_emotion.p, already normalized by the dataset authors",
            "source_sha256": hashlib.sha256((DATA / "data_emotion.p").read_bytes()).hexdigest(),
            "splits": dict(SPLITS), "label_index": label_index,
            "test_encoded": False, "classifier_trained": False,
            "versions": {name: importlib.metadata.version(name) for name in
                         ["sentence-transformers", "transformers", "torch", "numpy"]},
        }
        (staged / "metadata.json").write_text(json.dumps(metadata, indent=2))
        if output.exists():
            raise FileExistsError(f"Output appeared during extraction: {output}")
        staged.rename(output)

    print("\nFinished extracting MiniLM features.")
    print("Training: 9,989 sentences x 384 features")
    print("Validation: 1,109 sentences x 384 features")
    print(f"Saved features: {output}")
    print("No classifier was trained. Test sentences were not encoded.")


if __name__ == "__main__":
    main()
