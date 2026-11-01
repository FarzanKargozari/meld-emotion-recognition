"""Read and validate the official MELD feature files without changing them."""

from collections import Counter
from pathlib import Path
import pickle

import numpy as np


DATA = Path(__file__).resolve().parents[2] / "data"


def load(name):
    # These local files came from the official MELD archive; latin1 supports
    # the older Python pickle format used by that release.
    with (DATA / name).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def main():
    rows, _, _, _, _, label_index = load("data_emotion.p")
    text = load("text_glove_average_emotion.pkl")
    audio = load("audio_embeddings_feature_selection_emotion.pkl")

    print("MELD data check\n")
    for index, (split, title, expected) in enumerate([
        ("train", "Training", 9989),
        ("val", "Validation", 1109),
        ("test", "Test", 2610),
    ]):
        split_rows = [row for row in rows if row["split"] == split]
        keys = [f"{row['dialog']}_{row['utterance']}" for row in split_rows]
        if len(keys) != expected or len(set(keys)) != len(keys):
            raise ValueError(f"Unexpected count or duplicate IDs in {title}")
        # Match modalities by ID, never by dictionary iteration order.
        if set(keys) != set(text[index]) or set(keys) != set(audio[index]):
            raise ValueError(f"Text, audio, and label IDs do not match in {title}")
        for row, key in zip(split_rows, keys):
            if row["y"] not in label_index:
                raise ValueError(f"Unknown emotion in {title}: {key}")
            for name, vector, dimension in [
                ("text", text[index][key], 300),
                ("audio", audio[index][key], 1611),
            ]:
                vector = np.asarray(vector)
                if vector.shape != (dimension,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid {name} features in {title}: {key}")
        print(f"{title}: {len(keys):,} examples; text + audio + labels match")

    print("\nEach example has 300 text features and 1,611 audio features.")
    print("Together, that is 1,911 input numbers for our model.")
    print("\nTraining examples by emotion:")
    counts = Counter(row["y"] for row in rows if row["split"] == "train")
    for emotion, count in counts.most_common():
        print(f"  {emotion:<10} {count:>5,}")
    example = next(row for row in rows if row["split"] == "train")
    print(f"\nExample text: {example['text']}")
    print(f"Correct emotion: {example['y']}")
    print("\nPASS: All three splits are aligned and contain finite feature values.")
    print("No model has been trained yet.")


if __name__ == "__main__":
    main()
