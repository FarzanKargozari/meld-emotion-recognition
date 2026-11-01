"""Shared tokenization, batching and inference helpers."""

import torch
from torch.utils.data import Dataset

from .data import LABEL_TO_ID, render


def pick_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def configure_tokenizer(tokenizer):
    # When a long context is cut, drop its oldest words; the nearest line matters most.
    tokenizer.truncation_side = "left"
    return tokenizer


def encode(tokenizer, context, target, max_length):
    """Tokenize as (context, target). Only context is truncated, so the target line is kept whole."""
    return tokenizer(context, target, truncation="only_first", max_length=max_length)


class DialogueDataset(Dataset):
    def __init__(self, examples, tokenizer, use_speakers, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.use_speakers = use_speakers
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        context, target = render(example, self.use_speakers)
        item = dict(encode(self.tokenizer, context, target, self.max_length))
        item["labels"] = LABEL_TO_ID[example.utterance.emotion]
        return item


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    outputs = []
    for batch in loader:
        batch.pop("labels", None)
        outputs.append(model(**{key: value.to(device) for key, value in batch.items()}).logits.cpu())
    return torch.cat(outputs).float().numpy()
