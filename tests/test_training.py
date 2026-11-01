"""Exercise the real training and inference code on a tiny random model, offline."""

import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

from meld_emotion.data import LABELS
from meld_emotion.modeling import configure_tokenizer, encode
from meld_emotion.predict import EmotionPredictor, Line
from meld_emotion.train import train_epoch

WORDS = ["<s>", "<pad>", "</s>", "<unk>", "one", "two", "three", "four", "five", "six", "Ross", ":", "Hello"]


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(WORDS)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    return configure_tokenizer(PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>",
    ))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return RobertaForSequenceClassification(RobertaConfig(
        vocab_size=len(WORDS), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64, num_labels=len(LABELS), pad_token_id=1,
    ))


def test_truncation_keeps_the_target_and_newest_context(tokenizer):
    encoded = encode(tokenizer, "one two three four five six", "Hello", max_length=8)
    words = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    assert len(words) == 8
    assert words[-2] == "Hello"
    assert "six" in words
    assert "one" not in words


def test_training_lowers_the_loss(model):
    inputs = torch.tensor([[0, 4, 5, 2], [0, 6, 7, 2], [0, 8, 9, 2], [0, 4, 6, 2]])
    labels = torch.tensor([0, 1, 2, 3])
    loader = [{"input_ids": inputs, "attention_mask": torch.ones_like(inputs), "labels": labels}
              for _ in range(4)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    criterion = torch.nn.CrossEntropyLoss()
    def run():
        return train_epoch(model, [dict(batch) for batch in loader], optimizer, scheduler,
                                  criterion, torch.device("cpu"), 1.0)
    first = run()
    for _ in range(10):
        last = run()
    assert last < first


def test_accumulation_updates_once_per_group(model):
    loader = tiny_loader(5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    train_epoch(model, loader, optimizer, scheduler, torch.nn.CrossEntropyLoss(), torch.device("cpu"), 1.0, 2)
    assert scheduler.last_epoch == 3


def tiny_loader(batches):
    inputs = torch.tensor([[0, 4, 5, 2], [0, 6, 7, 2]])
    labels = torch.tensor([0, 1])
    return [{"input_ids": inputs, "attention_mask": torch.ones_like(inputs), "labels": labels}
            for _ in range(batches)]


def test_invalid_batch_is_skipped_and_weights_stay_finite(model):
    calls = []
    def criterion(logits, labels):
        calls.append(1)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss * float("nan") if len(calls) == 2 else loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loss = train_epoch(model, tiny_loader(4), optimizer, scheduler, criterion, torch.device("cpu"), 1.0)
    assert np.isfinite(loss)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_repeated_invalid_batches_stop_training(model):
    criterion = lambda logits, labels: logits.sum() * float("nan")  # noqa: E731
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    with pytest.raises(RuntimeError, match="non-finite"):
        train_epoch(model, tiny_loader(8), optimizer, scheduler, criterion, torch.device("cpu"), 1.0)


def test_saved_run_predicts_end_to_end(tmp_path, tokenizer, model):
    model.config.id2label = dict(enumerate(LABELS))
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    settings = {"context_size": 2, "use_speakers": True, "max_length": 32}
    (tmp_path / "config.json").write_text(json.dumps(settings))
    np.savez(tmp_path / "validation_logits.npz", logits=np.random.default_rng(0).normal(size=(40, 7)),
             labels=np.arange(40) % 7)

    predictor = EmotionPredictor(tmp_path, "cpu")
    history, target = predictor.render(Line("Hello", "Ross"), [Line("one"), Line("two"), Line("three")])
    assert history == "Speaker: two Speaker: three"
    assert target == "Ross: Hello"

    result = predictor.predict(Line("Hello", "Ross"), [Line("one")])
    assert result["emotion"] in LABELS
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert result["confidence"] == max(result["probabilities"].values())
