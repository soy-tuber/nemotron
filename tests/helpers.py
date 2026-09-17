"""Fake gateway responses so the decision logic is testable without a GPU."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import httpx

Position = Sequence[tuple[str, float]]


def completion(positions: Sequence[Position], content: str | None = None) -> dict[str, Any]:
    """Build an OpenAI chat completion carrying ``top_logprobs`` per position."""
    entries = []
    for candidates in positions:
        ranked = sorted(candidates, key=lambda tp: -tp[1])
        top = [
            # JSON has no -inf, so a vanishingly small logprob stands in for zero.
            {"token": token, "logprob": math.log(prob) if prob > 0 else -1e30}
            for token, prob in ranked
        ]
        entries.append({**top[0], "top_logprobs": top})
    text = content if content is not None else "".join(e["token"] for e in entries)
    return {
        "id": "cmpl-test",
        "model": "nemotron-9b-japanese",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": {"content": entries},
                "finish_reason": "length",
            }
        ],
    }


def transport(
    handler: Callable[[dict[str, Any], int], httpx.Response | dict[str, Any]],
) -> httpx.MockTransport:
    """Wrap a ``(request_json, call_index) -> response`` handler."""
    state = {"calls": 0}

    def respond(request: httpx.Request) -> httpx.Response:
        import json

        index = state["calls"]
        state["calls"] += 1
        result = handler(json.loads(request.content), index)
        if isinstance(result, httpx.Response):
            return result
        return httpx.Response(200, json=result)

    mock = httpx.MockTransport(respond)
    mock.state = state  # type: ignore[attr-defined]
    return mock


def always(positions: Sequence[Position], content: str | None = None) -> httpx.MockTransport:
    return transport(lambda body, index: completion(positions, content))
