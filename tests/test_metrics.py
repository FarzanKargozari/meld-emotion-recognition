import numpy as np
import pytest

from meld_emotion.metrics import (
    bootstrap_interval,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    scores,
    softmax,
)


def test_perfect_predictions_score_one():
    labels = np.array([0, 1, 2, 3, 4, 5, 6])
    assert scores(labels, labels) == {"accuracy": 1.0, "macro_f1": 1.0, "weighted_f1": 1.0}


def test_macro_f1_penalizes_majority_guessing():
    labels = np.array([0] * 90 + [1] * 10)
    result = scores(labels, np.zeros_like(labels))
    assert result["accuracy"] == pytest.approx(0.9)
    assert result["macro_f1"] < 0.2


def test_softmax_sums_to_one():
    assert softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]])).sum(axis=1) == pytest.approx([1, 1])


def test_interval_contains_the_point_estimate():
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 7, 600)
    predictions = np.where(generator.random(600) < 0.7, labels, generator.integers(0, 7, 600))
    groups = np.arange(600) // 6
    low, high = bootstrap_interval(labels, predictions, groups, samples=200)
    assert low < scores(labels, predictions)["macro_f1"] < high


def test_temperature_cools_an_overconfident_model():
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 3, 5000)
    wrong = np.where(generator.random(5000) < 0.6, labels, generator.integers(0, 3, 5000))
    logits = np.eye(3)[wrong] * 8.0
    temperature = fit_temperature(logits, labels)
    assert temperature > 1
    assert negative_log_likelihood(logits / temperature, labels) < negative_log_likelihood(logits, labels)


def test_temperature_preserves_predictions():
    logits = np.random.default_rng(1).normal(size=(50, 7))
    assert np.array_equal((logits / 3.0).argmax(axis=1), logits.argmax(axis=1))


def test_calibration_error_is_zero_when_confidence_matches():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probabilities, np.array([0, 1])) == pytest.approx(0.0)
