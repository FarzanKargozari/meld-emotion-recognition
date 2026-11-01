import pytest
from fastapi.testclient import TestClient

from meld_emotion.api import create_app


class StubPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, line, context=()):
        self.calls.append((line, list(context)))
        return {"emotion": "joy", "confidence": 0.8, "probabilities": {"joy": 0.8, "neutral": 0.2}}


@pytest.fixture
def stub():
    return StubPredictor()


@pytest.fixture
def client(stub):
    with TestClient(create_app(stub)) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_health_reports_a_missing_model(monkeypatch):
    monkeypatch.delenv("MELD_RUN", raising=False)
    with TestClient(create_app()) as test_client:
        assert test_client.get("/health").json() == {"status": "model_not_loaded"}


def test_predict_needs_a_model(monkeypatch):
    monkeypatch.delenv("MELD_RUN", raising=False)
    with TestClient(create_app()) as test_client:
        assert test_client.post("/predict", json={"line": {"text": "Hello"}}).status_code == 503


def test_predict_returns_the_contract(client):
    body = client.post("/predict", json={"line": {"text": "We won!"}}).json()
    assert body["emotion"] == "joy"
    assert body["confidence"] == 0.8
    assert body["latency_ms"] >= 0


def test_context_reaches_the_model_in_order(client, stub):
    client.post("/predict", json={"line": {"text": "Yes"}, "context": [
        {"text": "first", "speaker": "A"}, {"text": "second", "speaker": "B"}]})
    assert [item.text for item in stub.calls[0][1]] == ["first", "second"]


@pytest.mark.parametrize("payload", [
    {},
    {"line": {"text": ""}},
    {"line": {"text": "   "}},
    {"line": {"text": "x" * 1001}},
    {"line": {"text": "ok"}, "context": [{"text": "x"}] * 21},
])
def test_invalid_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422
