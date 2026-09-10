from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .model import CausalLMScorer


app = FastAPI(title="Rime Context LM Scorer", version="0.1.0")


class ScoreRequest(BaseModel):
    context: str = ""
    pinyin: str = ""
    candidates: list[str] = Field(min_length=1, max_length=50)
    mode: str = "context_gain"


@lru_cache(maxsize=1)
def get_scorer() -> CausalLMScorer:
    return CausalLMScorer()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/score")
def score(request: ScoreRequest) -> dict:
    scorer = get_scorer()
    ranked = scorer.rank(request.context, request.candidates, request.mode)
    return {
        "model": scorer.model_name,
        "context": request.context,
        "pinyin": request.pinyin,
        "mode": request.mode,
        "ranking": [item.to_dict() for item in ranked],
    }
