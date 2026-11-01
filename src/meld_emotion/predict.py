"""Predict the emotion of one line, using earlier lines from the conversation as context."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import LABELS, clean_text, render_line
from .metrics import fit_temperature, softmax
from .modeling import configure_tokenizer, encode, pick_device


@dataclass(frozen=True)
class Line:
    text: str
    speaker: str = "Speaker"


class EmotionPredictor:
    def __init__(self, run, device="cpu"):
        run = Path(run)
        self.config = json.loads((run / "config.json").read_text())
        self.device = pick_device(device)
        self.tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(run / "model"))
        self.model = AutoModelForSequenceClassification.from_pretrained(run / "model").to(self.device).eval()
        with np.load(run / "validation_logits.npz", allow_pickle=False) as saved:
            self.temperature = fit_temperature(saved["logits"], saved["labels"])

    def render(self, line, context):
        """Mirror the training input: only the most recent configured lines are used."""
        limit = self.config["context_size"]
        recent = context[-limit:] if limit else []
        use_speakers = self.config["use_speakers"]
        history = " ".join(render_line(clean_text(item.speaker), clean_text(item.text), use_speakers)
                           for item in recent)
        return history, render_line(clean_text(line.speaker), clean_text(line.text), use_speakers)

    @torch.no_grad()
    def predict(self, line, context=()):
        history, target = self.render(line, list(context))
        encoded = encode(self.tokenizer, history, target, self.config["max_length"])
        inputs = {key: torch.tensor([value], device=self.device) for key, value in encoded.items()}
        logits = self.model(**inputs).logits.cpu().numpy()
        probabilities = softmax(logits / self.temperature)[0]
        ranked = np.argsort(-probabilities)
        return {
            "emotion": LABELS[int(ranked[0])], "confidence": float(probabilities[ranked[0]]),
            "probabilities": {LABELS[int(index)]: float(probabilities[index]) for index in ranked},
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Saved run folder, such as artifacts/NAME/seed_42")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default="Speaker")
    parser.add_argument("--context", action="append", default=[], metavar="[SPEAKER:]TEXT",
                        help="An earlier line, oldest first; repeat as needed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    args = parser.parse_args(argv)

    context = []
    for entry in args.context:
        speaker, separator, text = entry.partition(":")
        context.append(Line(text, speaker) if separator else Line(entry))
    predictor = EmotionPredictor(args.run, args.device)
    print(json.dumps(predictor.predict(Line(args.text, args.speaker), context), indent=2))


if __name__ == "__main__":
    main()
