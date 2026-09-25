"""Constrained decision inference against the local Nemotron gateway.

The model is never asked to write an answer. It is asked to emit one label
token, and what we keep is the probability distribution over the label set
read out of ``top_logprobs``. The returned choice is therefore *derived from
the distribution*, not parsed from text -- it cannot be off-schema, and it
comes with a confidence number and an explicit abstain path.

Two paths exist:

``logprob`` (default)
    One short, unconstrained generation. We look at the top-k logprobs of the
    first informative position, keep only the label tokens, and renormalise.
    Confidence is meaningful because the unchosen labels are still in the
    distribution.

``constrained`` (fallback)
    If no label token shows up in the top-k at all, we re-ask with vLLM guided
    decoding so the format is still guaranteed. The confidence reported by that
    path is computed over a masked distribution and is therefore inflated --
    it is flagged in ``warnings`` and should not be thresholded on.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import httpx

from .schema import Decision, DecisionError

DEFAULT_BASE_URL = os.environ.get("NEMOTRON_GATEWAY_URL", "http://localhost:8000/v1")
# The full id: vLLM runs without --served-model-name, so a short alias such as
# "nemotron-9b-japanese" makes the gateway load the model and vLLM answer 404.
DEFAULT_MODEL = os.environ.get(
    "NEMOTRON_DECIDE_MODEL", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"
)
DEFAULT_TOP_LOGPROBS = 20  # vLLM's default --max-logprobs ceiling
DEFAULT_SYSTEM = (
    "You are a decision function, not a chat assistant. "
    "Reply with exactly one label character from the given set. "
    "No explanation, no punctuation, no other text."
)

# Sub-word markers various tokenizers put in front of a token.
_TOKEN_PREFIXES = ("Ġ", "▁")  # "Ġ" (byte-BPE space), "▁" (sentencepiece)


def normalize_token(token: str) -> str:
    """Strip tokenizer decoration so ``" A"``/``"Ġ A"``/``"A"`` all compare equal."""
    text = token
    for prefix in _TOKEN_PREFIXES:
        text = text.replace(prefix, " ")
    return text.strip()


@dataclass
class DecisionResult:
    """The outcome of one decision. Always schema-valid, possibly abstaining."""

    decision: str
    choice: str | None
    confidence: float
    coverage: float
    distribution: dict[str, float]
    abstain: bool
    method: str
    model: str
    latency_ms: float
    reason: str | None = None
    expected_value: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def masked(self) -> bool:
        """True when ``confidence`` came from a constrained (masked) distribution.

        Unmasked is not the same as calibrated: even then ``confidence`` is the
        model's own, and thresholds have to be set from real data.
        """
        return self.method != "logprob"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "choice": self.choice,
            "confidence": round(self.confidence, 6),
            "coverage": round(self.coverage, 6),
            "distribution": {k: round(v, 6) for k, v in self.distribution.items()},
            "abstain": self.abstain,
            "method": self.method,
            "masked": self.masked,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "reason": self.reason,
            "expected_value": (
                None if self.expected_value is None else round(self.expected_value, 4)
            ),
            "warnings": list(self.warnings),
        }


class GatewayError(RuntimeError):
    """The gateway/vLLM returned something we cannot decide from."""


class Decider:
    """Client for one-token decisions against an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        top_logprobs: int = DEFAULT_TOP_LOGPROBS,
        scan_tokens: int = 4,
        thinking: bool = False,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.top_logprobs = top_logprobs
        self.scan_tokens = max(1, scan_tokens)
        self.thinking = thinking
        self._owns_client = client is None
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
            transport=transport,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "Decider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- prompt ----------------------------------------------------------

    def build_messages(
        self, decision: Decision, variables: dict[str, Any] | None = None
    ) -> list[dict[str, str]]:
        system = decision.system or DEFAULT_SYSTEM
        options = "\n".join(option.render() for option in decision.options)
        header = (
            "選択肢 / Options"
            if decision.mode == "choice"
            else "尺度 / Scale"
        )
        user = (
            f"{decision.render_question(variables)}\n\n"
            f"{header}:\n{options}\n\n"
            "ラベルを1文字だけ答えよ / Answer with a single label character only:"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # -- transport -------------------------------------------------------

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            # Connection refused, timeouts (a cold start or a failed model load
            # on the gateway) -- report them like any other gateway failure.
            raise GatewayError(
                f"cannot reach {self.base_url}/chat/completions: {exc!r}"
            ) from exc
        if response.status_code >= 400:
            raise GatewayError(
                f"{response.status_code} from {self.base_url}/chat/completions: "
                f"{response.text[:500]}"
            )
        return response.json()

    def _structured_extra(self, labels: Sequence[str]) -> dict[str, Any]:
        # vLLM 0.15.1 only understands ``structured_outputs``. The old
        # ``guided_choice`` is not rejected but silently ignored (requests allow
        # extra fields), so there is no error to detect a fallback from.
        return {"structured_outputs": {"choice": list(labels)}}

    def _chat_template_kwargs(self) -> dict[str, Any]:
        # Nemotron Nano v2's chat template only honours ``enable_thinking``; a
        # "/no_think" in the prompt is ignored, the first tokens become
        # reasoning, and no label mass is left to read.
        return {"chat_template_kwargs": {"enable_thinking": self.thinking}}

    # -- core ------------------------------------------------------------

    async def decide(
        self,
        decision: Decision,
        variables: dict[str, Any] | None = None,
        *,
        model: str | None = None,
        min_confidence: float | None = None,
        min_coverage: float | None = None,
    ) -> DecisionResult:
        """Run one decision and return its distribution."""
        model = model or self.model
        floor_conf = (
            decision.min_confidence if min_confidence is None else min_confidence
        )
        floor_cov = decision.min_coverage if min_coverage is None else min_coverage
        messages = self.build_messages(decision, variables)
        started = time.perf_counter()

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self.scan_tokens,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            **self._chat_template_kwargs(),
        }
        data = await self._post(payload)
        positions = _logprob_positions(data)
        distribution, coverage, raw_token = _best_position(
            positions, decision, floor_cov
        )

        warnings: list[str] = []
        method = "logprob"
        if not distribution:
            # Nothing usable in the top-k: fall back to guaranteed-format decoding.
            distribution, coverage, warnings, method = await self._constrained(
                decision, messages, model
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        return _assemble(
            decision=decision,
            distribution=distribution,
            coverage=coverage,
            method=method,
            model=model,
            latency_ms=latency_ms,
            floor_conf=floor_conf,
            floor_cov=floor_cov,
            warnings=warnings,
        )

    async def _constrained(
        self, decision: Decision, messages: list[dict[str, str]], model: str
    ) -> tuple[dict[str, float], float, list[str], str]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 2,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            **self._chat_template_kwargs(),
            **self._structured_extra(decision.labels),
        }
        data = await self._post(payload)

        positions = _logprob_positions(data)
        distribution, coverage, _ = _best_position(positions, decision, 0.0)
        if not distribution:
            # Last resort: read the emitted text, which guided decoding pinned
            # to the label set.
            text = normalize_token(
                (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            )
            option = decision.by_label(text) or decision.by_label(text.upper())
            if option is None:
                raise GatewayError(
                    f"decision {decision.name!r}: no label token in logprobs and "
                    f"completion {text!r} is not one of {decision.labels}"
                )
            distribution, coverage = {option.key: 1.0}, 1.0
        return (
            distribution,
            coverage,
            [
                "confidence came from a constrained (masked) distribution and is "
                "inflated; do not threshold on it"
            ],
            "constrained",
        )

    # -- batch -----------------------------------------------------------

    async def decide_many(
        self,
        items: Iterable[tuple[Decision, dict[str, Any] | None]],
        *,
        concurrency: int = 8,
        model: str | None = None,
    ) -> list[DecisionResult | Exception]:
        """Run decisions in parallel. Failures are returned, not raised."""
        gate = asyncio.Semaphore(max(1, concurrency))

        async def one(decision: Decision, variables: dict[str, Any] | None):
            async with gate:
                return await self.decide(decision, variables, model=model)

        return await asyncio.gather(
            *(one(d, v) for d, v in items), return_exceptions=True
        )


# -- response parsing ----------------------------------------------------


def _logprob_positions(data: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Return per-position ``top_logprobs`` lists from a chat completion."""
    choices = data.get("choices") or []
    if not choices:
        raise GatewayError("gateway returned no choices")
    logprobs = choices[0].get("logprobs") or {}
    content = logprobs.get("content") or []
    positions: list[list[dict[str, Any]]] = []
    for entry in content:
        candidates = list(entry.get("top_logprobs") or [])
        if not any(c.get("token") == entry.get("token") for c in candidates):
            candidates.append(
                {"token": entry.get("token", ""), "logprob": entry.get("logprob", 0.0)}
            )
        positions.append(candidates)
    return positions


def _score_position(
    candidates: list[dict[str, Any]], decision: Decision
) -> tuple[dict[str, float], float, str | None]:
    """Collapse one token position onto the label set.

    Returns ``(renormalised distribution keyed by option key, raw label mass,
    winning raw token)``.
    """
    mass: dict[str, float] = {option.key: 0.0 for option in decision.options}
    by_label = {option.label: option for option in decision.options}
    by_label_ci = {option.label.upper(): option for option in decision.options}
    best_token: str | None = None
    best_prob = -1.0

    for candidate in candidates:
        token = candidate.get("token")
        if token is None:
            continue
        logprob = candidate.get("logprob")
        if logprob is None or logprob == float("-inf") or math.isnan(logprob):
            continue
        text = normalize_token(str(token))
        if not text:
            continue
        option = by_label.get(text) or by_label_ci.get(text.upper())
        if option is None:
            continue
        probability = math.exp(logprob)
        mass[option.key] += probability
        if probability > best_prob:
            best_prob, best_token = probability, str(token)

    coverage = sum(mass.values())
    if coverage <= 0.0:
        return {}, 0.0, None
    return {k: v / coverage for k, v in mass.items()}, min(coverage, 1.0), best_token


def _best_position(
    positions: list[list[dict[str, Any]]], decision: Decision, floor_coverage: float
) -> tuple[dict[str, float], float, str | None]:
    """Pick the first position that actually carries label mass.

    A stray leading token (a stripped ``<think>``, a quote, a newline) should
    not cost us the decision, so we scan forward and keep the best position.
    """
    best: tuple[dict[str, float], float, str | None] = ({}, 0.0, None)
    for candidates in positions:
        distribution, coverage, token = _score_position(candidates, decision)
        if coverage > best[1]:
            best = (distribution, coverage, token)
        if coverage >= max(floor_coverage, 1e-9) and distribution:
            return distribution, coverage, token
    return best


def _assemble(
    *,
    decision: Decision,
    distribution: dict[str, float],
    coverage: float,
    method: str,
    model: str,
    latency_ms: float,
    floor_conf: float,
    floor_cov: float,
    warnings: list[str],
) -> DecisionResult:
    if not distribution:
        raise GatewayError(f"decision {decision.name!r}: no usable distribution")

    choice, confidence = max(distribution.items(), key=lambda kv: kv[1])
    reason: str | None = None
    abstain = False
    if method == "logprob" and coverage < floor_cov:
        abstain, reason = True, (
            f"coverage {coverage:.3f} < min_coverage {floor_cov:.3f}: the model did "
            "not answer in label form"
        )
    elif confidence < floor_conf:
        abstain, reason = True, (
            f"confidence {confidence:.3f} < min_confidence {floor_conf:.3f}"
        )

    expected_value = None
    if decision.mode == "scale":
        expected_value = sum(float(k) * v for k, v in distribution.items())

    return DecisionResult(
        decision=decision.name,
        choice=None if abstain else choice,
        confidence=confidence,
        coverage=coverage,
        distribution=dict(sorted(distribution.items(), key=lambda kv: -kv[1])),
        abstain=abstain,
        method=method,
        model=model,
        latency_ms=latency_ms,
        reason=reason,
        expected_value=expected_value,
        warnings=warnings,
    )


def decide_sync(
    decision: Decision,
    variables: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DecisionResult:
    """Blocking convenience wrapper. Not for use inside a running event loop."""

    async def run() -> DecisionResult:
        async with Decider(**kwargs) as decider:
            return await decider.decide(decision, variables)

    return asyncio.run(run())
