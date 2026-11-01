"""Small HTTP service around a saved model. Set MELD_RUN to a run folder before starting."""

import os
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .predict import EmotionPredictor, Line


class LineIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    speaker: str = Field(default="Speaker", min_length=1, max_length=80)

    @field_validator("text")
    @classmethod
    def has_content(cls, value):
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class PredictIn(BaseModel):
    line: LineIn
    context: list[LineIn] = Field(default_factory=list, max_length=20,
                                  description="Earlier lines, oldest first")


class PredictOut(BaseModel):
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


def create_app(predictor=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = predictor
        if predictor is None and os.environ.get("MELD_RUN"):
            device = os.environ.get("MELD_DEVICE", "cpu")
            app.state.predictor = EmotionPredictor(os.environ["MELD_RUN"], device)
        yield

    app = FastAPI(title="MELD emotion service", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        ready = request.app.state.predictor is not None
        return {"status": "ok" if ready else "model_not_loaded"}

    @app.post("/predict", response_model=PredictOut)
    def predict(body: PredictIn, request: Request):
        loaded = request.app.state.predictor
        if loaded is None:
            raise HTTPException(status_code=503, detail="Set MELD_RUN to a saved run folder")
        started = perf_counter()
        result = loaded.predict(Line(body.line.text, body.line.speaker),
                                [Line(item.text, item.speaker) for item in body.context])
        return {**result, "latency_ms": (perf_counter() - started) * 1000}

    return app


app = create_app()
