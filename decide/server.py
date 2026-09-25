"""HTTP service exposing the decision endpoint.

Sits beside the gateway rather than inside it: the gateway owns model loading
and swapping on port 8000, this owns the decision catalogue on port 9200 and
talks to the gateway as an ordinary OpenAI-compatible client.

    uvicorn decide.server:app --host 127.0.0.1 --port 9200
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .decider import DEFAULT_BASE_URL, DEFAULT_MODEL, Decider, GatewayError
from .schema import Decision, DecisionCatalog, DecisionError

CATALOG_PATH = Path(
    os.environ.get(
        "NEMOTRON_DECISIONS", str(Path(__file__).with_name("decisions.yaml"))
    )
)


class DecideRequest(BaseModel):
    decision: str | None = Field(
        default=None, description="Name of a decision in the catalogue."
    )
    inline: dict[str, Any] | None = Field(
        default=None,
        description="Ad-hoc decision definition, same shape as a catalogue entry.",
    )
    variables: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    min_confidence: float | None = None
    min_coverage: float | None = None


class BatchRequest(BaseModel):
    items: list[DecideRequest]
    concurrency: int = 8


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.catalog = DecisionCatalog.from_yaml(CATALOG_PATH)
    app.state.decider = Decider(
        base_url=os.environ.get("NEMOTRON_GATEWAY_URL", DEFAULT_BASE_URL),
        model=os.environ.get("NEMOTRON_DECIDE_MODEL", DEFAULT_MODEL),
    )
    try:
        yield
    finally:
        await app.state.decider.aclose()


app = FastAPI(
    title="Nemotron Decide",
    description="Closed-set decisions with a confidence score, served locally.",
    version="1.0.0",
    lifespan=lifespan,
)


def _resolve(app_: FastAPI, request: DecideRequest) -> Decision:
    if bool(request.decision) == bool(request.inline):
        raise HTTPException(400, "provide exactly one of 'decision' or 'inline'")
    try:
        if request.decision:
            return app_.state.catalog.get(request.decision)
        payload = dict(request.inline or {})
        payload.setdefault("name", "inline")
        return Decision.from_dict(payload)
    except DecisionError as exc:
        raise HTTPException(400, str(exc)) from exc


async def _run(app_: FastAPI, request: DecideRequest) -> dict[str, Any]:
    decision = _resolve(app_, request)
    try:
        result = await app_.state.decider.decide(
            decision,
            request.variables,
            model=request.model,
            min_confidence=request.min_confidence,
            min_coverage=request.min_coverage,
        )
    except DecisionError as exc:
        raise HTTPException(400, str(exc)) from exc
    except GatewayError as exc:
        raise HTTPException(502, str(exc)) from exc
    return result.to_dict()


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "catalog": str(CATALOG_PATH),
        "decisions": len(app.state.catalog),
        "gateway": app.state.decider.base_url,
        "model": app.state.decider.model,
    }


@app.get("/v1/decisions")
async def list_decisions() -> dict[str, Any]:
    return {
        "decisions": [
            {
                "name": decision.name,
                "description": decision.description,
                "mode": decision.mode,
                "options": decision.keys,
                "variables": sorted(decision.required_variables()),
                "min_confidence": decision.min_confidence,
            }
            for decision in app.state.catalog
        ]
    }


@app.post("/v1/decide")
async def decide(request: DecideRequest) -> dict[str, Any]:
    return await _run(app, request)


@app.post("/v1/decide/batch")
async def decide_batch(request: BatchRequest) -> dict[str, Any]:
    import asyncio

    gate = asyncio.Semaphore(max(1, request.concurrency))

    async def one(item: DecideRequest) -> dict[str, Any]:
        async with gate:
            try:
                return await _run(app, item)
            except HTTPException as exc:
                return {"error": exc.detail, "status": exc.status_code}
            except Exception as exc:  # one bad item must not sink the batch
                return {"error": f"{type(exc).__name__}: {exc}", "status": 500}

    return {"results": await asyncio.gather(*(one(item) for item in request.items))}
