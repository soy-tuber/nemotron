import asyncio
import math

import httpx
import pytest

from decide.decider import Decider, GatewayError, normalize_token
from decide.schema import Decision
from tests.helpers import always, completion, transport

YES_NO = Decision.from_dict(
    {"name": "d", "question": "q", "options": ["yes", "no"], "min_confidence": 0.0}
)


def run(coro):
    return asyncio.run(coro)


async def decide(decision, mock, **kwargs):
    async with Decider(transport=mock, **kwargs) as decider:
        return await decider.decide(decision)


# -- token normalisation -------------------------------------------------


@pytest.mark.parametrize("token", ["A", " A", "ĠA", "▁A", "A\n"])
def test_normalize_token_strips_tokenizer_decoration(token):
    assert normalize_token(token) == "A"


# -- distribution --------------------------------------------------------


def test_distribution_is_renormalised_over_the_label_set():
    # 40% A, 20% B, 40% mass on tokens that are not labels at all.
    mock = always([[("A", 0.4), ("B", 0.2), ("The", 0.4)]])
    result = run(decide(YES_NO, mock))

    assert result.choice == "yes"
    assert result.method == "logprob"
    assert not result.masked
    assert result.confidence == pytest.approx(0.4 / 0.6)
    assert result.distribution["no"] == pytest.approx(0.2 / 0.6)
    assert result.coverage == pytest.approx(0.6)
    assert not result.abstain


def test_decorated_label_variants_are_summed():
    mock = always([[("A", 0.3), (" A", 0.2), ("B", 0.5)]])
    result = run(decide(YES_NO, mock))
    assert result.distribution["yes"] == pytest.approx(0.5)
    assert result.coverage == pytest.approx(1.0)


def test_lowercase_labels_still_match():
    mock = always([[("a", 0.7), ("b", 0.3)]])
    result = run(decide(YES_NO, mock))
    assert result.choice == "yes"
    assert result.confidence == pytest.approx(0.7)


def test_scan_skips_a_leading_non_label_position():
    mock = always([[("\n", 0.99)], [("A", 0.8), ("B", 0.2)]])
    result = run(decide(YES_NO, mock))
    assert result.choice == "yes"
    assert result.confidence == pytest.approx(0.8)


def test_sampled_token_is_included_when_absent_from_top_logprobs():
    body = completion([[("A", 0.6), ("B", 0.4)]])
    entry = body["choices"][0]["logprobs"]["content"][0]
    entry["top_logprobs"] = [t for t in entry["top_logprobs"] if t["token"] != "A"]
    result = run(decide(YES_NO, httpx.MockTransport(lambda request: httpx.Response(200, json=body))))
    assert result.distribution["yes"] == pytest.approx(0.6)


def test_masked_and_malformed_candidates_are_ignored():
    from decide.decider import _score_position

    distribution, coverage, token = _score_position(
        [
            {"token": "A", "logprob": math.log(0.5)},
            {"token": "B", "logprob": float("-inf")},  # masked by guided decoding
            {"token": "B", "logprob": float("nan")},
            {"token": "B"},  # no logprob at all
            {"token": None, "logprob": -0.1},
        ],
        YES_NO,
    )
    assert distribution == {"yes": pytest.approx(1.0), "no": pytest.approx(0.0)}
    assert coverage == pytest.approx(0.5)
    assert token == "A"


# -- abstain -------------------------------------------------------------


def test_abstains_below_confidence_floor():
    decision = Decision.from_dict(
        {"name": "d", "question": "q", "options": ["yes", "no"], "min_confidence": 0.9}
    )
    mock = always([[("A", 0.55), ("B", 0.45)]])
    result = run(decide(decision, mock))

    assert result.abstain
    assert result.choice is None
    assert "min_confidence" in result.reason
    # The distribution survives abstention: the caller can still inspect it.
    assert result.distribution["yes"] == pytest.approx(0.55)


def test_abstains_when_label_mass_is_negligible():
    decision = Decision.from_dict(
        {
            "name": "d",
            "question": "q",
            "options": ["yes", "no"],
            "min_coverage": 0.2,
            "min_confidence": 0.0,
        }
    )
    mock = always([[("A", 0.01), ("Sure", 0.99)]])
    result = run(decide(decision, mock))

    assert result.abstain
    assert "coverage" in result.reason


def test_per_call_thresholds_override_the_definition():
    decision = Decision.from_dict(
        {"name": "d", "question": "q", "options": ["yes", "no"], "min_confidence": 0.9}
    )
    mock = always([[("A", 0.6), ("B", 0.4)]])

    async def go():
        async with Decider(transport=mock) as decider:
            return await decider.decide(decision, min_confidence=0.5)

    assert not run(go()).abstain


# -- scale ---------------------------------------------------------------


def test_scale_reports_expected_value():
    decision = Decision.from_dict(
        {
            "name": "urgency",
            "question": "q",
            "mode": "scale",
            "scale": {"min": 1, "max": 3},
            "min_confidence": 0.0,
        }
    )
    mock = always([[("1", 0.2), ("2", 0.3), ("3", 0.5)]])
    result = run(decide(decision, mock))

    assert result.choice == "3"
    assert result.expected_value == pytest.approx(1 * 0.2 + 2 * 0.3 + 3 * 0.5)


def test_choice_mode_has_no_expected_value():
    assert run(decide(YES_NO, always([[("A", 1.0)]]))).expected_value is None


# -- constrained fallback ------------------------------------------------


def test_falls_back_to_constrained_decoding_when_no_label_appears():
    def handler(body, index):
        if index == 0:
            return completion([[("Sure", 0.9), (",", 0.1)]])
        assert body["structured_outputs"] == {"choice": ["A", "B"]}
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        return completion([[("B", 0.95), ("A", 0.05)]])

    mock = transport(handler)
    result = run(decide(YES_NO, mock))

    assert mock.state["calls"] == 2
    assert result.method == "constrained"
    assert result.masked
    assert result.choice == "no"
    assert any("inflated" in w for w in result.warnings)


def test_constrained_fallback_reads_text_when_logprobs_are_empty():
    def handler(body, index):
        if index == 0:
            return completion([[("Sure", 1.0)]])
        return {"choices": [{"message": {"content": " B"}, "logprobs": {"content": []}}]}

    result = run(decide(YES_NO, transport(handler)))
    assert result.choice == "no"
    assert result.confidence == pytest.approx(1.0)


def test_constrained_fallback_error_is_raised_not_retried_as_guided_choice():
    # vLLM 0.15.1 silently ignores guided_choice, so a retry with it would come
    # back unconstrained while still being reported as "constrained".
    seen: list[dict] = []

    def handler(body, index):
        seen.append(body)
        if index == 0:
            return completion([[("Sure", 1.0)]])
        return httpx.Response(503, text="model busy")

    with pytest.raises(GatewayError, match="503"):
        run(decide(YES_NO, transport(handler)))
    assert len(seen) == 2
    assert seen[1]["structured_outputs"] == {"choice": ["A", "B"]}
    assert "guided_choice" not in seen[1]


def test_unparseable_constrained_answer_raises():
    def handler(body, index):
        if index == 0:
            return completion([[("Sure", 1.0)]])
        return {"choices": [{"message": {"content": "maybe"}, "logprobs": {"content": []}}]}

    with pytest.raises(GatewayError, match="not one of"):
        run(decide(YES_NO, transport(handler)))


# -- transport behaviour -------------------------------------------------


def test_http_error_becomes_gateway_error():
    mock = httpx.MockTransport(lambda request: httpx.Response(503, text="model loading"))
    with pytest.raises(GatewayError, match="503"):
        run(decide(YES_NO, mock))


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("connection refused"), httpx.ReadTimeout("timed out")]
)
def test_transport_error_becomes_gateway_error(error):
    def refuse(request):
        raise error

    with pytest.raises(GatewayError, match="cannot reach"):
        run(decide(YES_NO, httpx.MockTransport(refuse)))


def test_empty_choices_becomes_gateway_error():
    mock = httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []}))
    with pytest.raises(GatewayError, match="no choices"):
        run(decide(YES_NO, mock))


def test_request_asks_for_logprobs_and_greedy_decoding():
    captured: list[dict] = []

    def handler(body, index):
        captured.append(body)
        return completion([[("A", 1.0)]])

    run(decide(YES_NO, transport(handler), model="override-model"))
    body = captured[0]
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 4
    assert body["model"] == "override-model"


def test_default_model_is_the_full_id_vllm_serves():
    captured: list[dict] = []

    def handler(body, index):
        captured.append(body)
        return completion([[("A", 1.0)]])

    run(decide(YES_NO, transport(handler)))
    assert captured[0]["model"] == "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"


def test_thinking_is_disabled_through_the_chat_template():
    captured: list[dict] = []

    def handler(body, index):
        captured.append(body)
        return completion([[("A", 1.0)]])

    run(decide(YES_NO, transport(handler)))
    system, user = captured[0]["messages"]
    # The template ignores "/no_think"; only enable_thinking reaches it.
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "/no_think" not in system["content"]
    assert "A. yes" in user["content"] and "B. no" in user["content"]


def test_thinking_mode_is_forwarded_to_the_chat_template():
    captured: list[dict] = []

    def handler(body, index):
        captured.append(body)
        return completion([[("A", 1.0)]])

    run(decide(YES_NO, transport(handler), thinking=True))
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": True}


# -- batch ---------------------------------------------------------------


def test_decide_many_runs_in_parallel_and_surfaces_failures():
    def handler(body, index):
        if index == 1:
            return httpx.Response(500, text="boom")
        return completion([[("A", 1.0)]])

    async def go():
        async with Decider(transport=transport(handler)) as decider:
            return await decider.decide_many(
                [(YES_NO, None), (YES_NO, None), (YES_NO, None)], concurrency=1
            )

    results = run(go())
    assert len(results) == 3
    assert isinstance(results[1], GatewayError)
    assert [r.choice for r in results if not isinstance(r, Exception)] == ["yes", "yes"]


def test_result_to_dict_is_json_serialisable():
    import json

    result = run(decide(YES_NO, always([[("A", 0.75), ("B", 0.25)]])))
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["choice"] == "yes"
    assert payload["masked"] is False
    assert payload["distribution"] == {"yes": 0.75, "no": 0.25}
