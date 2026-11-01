"""Classification, uncertainty and calibration metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def scores(labels, predictions, num_classes=7):
    classes = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=classes, average="weighted",
                                      zero_division=0)),
    }


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def bootstrap_interval(labels, predictions, groups, metric="macro_f1", samples=2000, seed=0, num_classes=7):
    """95% interval resampling whole dialogues, since lines within one dialogue are correlated."""
    labels, predictions, groups = np.asarray(labels), np.asarray(predictions), np.asarray(groups)
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        chosen = generator.integers(len(members), size=len(members))
        rows = np.concatenate([members[index] for index in chosen])
        values.append(scores(labels[rows], predictions[rows], num_classes)[metric])
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def expected_calibration_error(probabilities, labels, bins=15):
    confidence, predicted = probabilities.max(axis=1), probabilities.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            accuracy = (predicted[inside] == labels[inside]).mean()
            error += inside.mean() * abs(accuracy - confidence[inside].mean())
    return float(error)


def negative_log_likelihood(logits, labels):
    probabilities = softmax(logits)[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(probabilities, 1e-12, 1)).mean())


def fit_temperature(logits, labels):
    """Choose one temperature on validation data. Scaling never changes the predicted class."""
    candidates = np.exp(np.linspace(np.log(0.25), np.log(8.0), 141))
    losses = [negative_log_likelihood(logits / value, labels) for value in candidates]
    return float(candidates[int(np.argmin(losses))])
