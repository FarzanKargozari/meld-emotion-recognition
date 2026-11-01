"""Read official MELD transcripts and build past-only dialogue inputs."""

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

LABELS = ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)
SPLITS = {
    "train": ("train_sent_emo.csv", 9989),
    "validation": ("dev_sent_emo.csv", 1109),
    "test": ("test_sent_emo.csv", 2610),
}
SOURCE_SHA256 = {
    "train_sent_emo.csv": "d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db",
    "dev_sent_emo.csv": "2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c",
    "test_sent_emo.csv": "8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888",
}
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
_TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "…": "...", "—": "-", "–": "-"})


@dataclass(frozen=True)
class Utterance:
    dialogue: int
    index: int
    speaker: str
    text: str
    emotion: str

    @property
    def key(self):
        return f"{self.dialogue}_{self.index}"


@dataclass(frozen=True)
class Example:
    utterance: Utterance
    context: tuple[Utterance, ...]


def clean_text(text):
    """Normalize typography and spacing; keep case and punctuation, which carry emotion."""
    return " ".join(text.translate(_TYPOGRAPHY).split())


def read_split(split, directory=RAW_DIR, verify=True):
    filename, expected = SPLITS[split]
    path = Path(directory) / filename
    if verify and hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256[filename]:
        raise ValueError(f"{filename} differs from the recorded official file")
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        utterances = [
            Utterance(int(row["Dialogue_ID"]), int(row["Utterance_ID"]), clean_text(row["Speaker"]),
                      clean_text(row["Utterance"]), row["Emotion"])
            for row in csv.DictReader(handle)
        ]
    keys = {item.key for item in utterances}
    if verify and (len(utterances) != expected or len(keys) != expected):
        raise ValueError(f"Unexpected count or duplicate IDs in {split}")
    for item in utterances:
        if item.emotion not in LABEL_TO_ID or not item.text:
            raise ValueError(f"Invalid row in {split}: {item.key}")
    return utterances


def build_examples(utterances, context_size):
    """Attach up to `context_size` earlier lines from the same dialogue.

    Dialogue IDs restart in every split, so call this once per split. Some utterance IDs are
    missing from the source; sorting uses the real order rather than assuming consecutive IDs.
    Context holds earlier lines only, so a live system could produce the same input.
    """
    if context_size < 0:
        raise ValueError("context_size must not be negative")
    dialogues = defaultdict(list)
    for item in utterances:
        dialogues[item.dialogue].append(item)
    examples = []
    for dialogue in sorted(dialogues):
        ordered = sorted(dialogues[dialogue], key=lambda item: item.index)
        for position, item in enumerate(ordered):
            start = max(0, position - context_size)
            examples.append(Example(item, tuple(ordered[start:position])))
    return examples


def render_line(speaker, text, use_speakers):
    return f"{speaker}: {text}" if use_speakers else text


def render(example, use_speakers):
    """Return (context, target) text; a model sees context as a separate segment."""
    context = " ".join(render_line(item.speaker, item.text, use_speakers) for item in example.context)
    return context, render_line(example.utterance.speaker, example.utterance.text, use_speakers)


def class_weights(labels, num_classes=NUM_CLASSES):
    """Inverse square root of class frequency, normalized to mean 1. Use training labels only."""
    counts = [0] * num_classes
    for label in labels:
        counts[label] += 1
    raw = [1.0 / max(count, 1) ** 0.5 for count in counts]
    mean = sum(raw) / num_classes
    return [value / mean for value in raw]
