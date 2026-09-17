import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from decide.decider import Decider
from decide.server import app
from tests.helpers import always, completion, transport


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def use(mock, client):
    """Point the running app at a fake gateway."""
    client.app.state.decider = Decider(transport=mock)


def test_healthz_reports_catalogue_and_gateway(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["decisions"] >= 4
    assert body["gateway"].startswith("http")


def test_lists_catalogue_entries_with_their_variables(client):
    decisions = {d["name"]: d for d in client.get("/v1/decisions").json()["decisions"]}
    assert decisions["route_inquiry"]["variables"] == ["input"]
    assert decisions["urgency"]["mode"] == "scale"
    assert "billing" in decisions["route_inquiry"]["options"]


def test_decide_by_name_returns_the_distribution(client):
    use(always([[("A", 0.9), ("B", 0.1)]]), client)
    body = client.post(
        "/v1/decide",
        json={"decision": "route_inquiry", "variables": {"input": "二重請求です"}},
    ).json()

    assert body["choice"] == "billing"
    assert body["confidence"] == pytest.approx(0.9)
    assert body["calibrated"] is True
    assert set(body["distribution"]) == {
        "billing",
        "technical",
        "account",
        "sales",
        "other",
    }


def test_inline_decision_needs_no_catalogue_entry(client):
    use(always([[("B", 1.0)]]), client)
    body = client.post(
        "/v1/decide",
        json={"inline": {"question": "苦情か", "options": ["yes", "no"]}},
    ).json()
    assert body["choice"] == "no"


def test_missing_variables_is_a_client_error(client):
    use(always([[("A", 1.0)]]), client)
    response = client.post("/v1/decide", json={"decision": "route_inquiry"})
    assert response.status_code == 400
    assert "missing variables" in response.json()["detail"]


def test_requires_exactly_one_of_decision_or_inline(client):
    for payload in ({}, {"decision": "urgency", "inline": {"question": "q"}}):
        response = client.post("/v1/decide", json=payload)
        assert response.status_code == 400
        assert "exactly one" in response.json()["detail"]


def test_unknown_decision_is_a_client_error(client):
    response = client.post("/v1/decide", json={"decision": "nope"})
    assert response.status_code == 400
    assert "unknown decision" in response.json()["detail"]


def test_gateway_failure_is_reported_as_bad_gateway(client):
    use(httpx.MockTransport(lambda request: httpx.Response(503, text="loading")), client)
    response = client.post(
        "/v1/decide", json={"decision": "route_inquiry", "variables": {"input": "x"}}
    )
    assert response.status_code == 502


def test_batch_keeps_position_and_isolates_failures(client):
    def handler(body, index):
        if index == 1:
            return httpx.Response(500, text="boom")
        return completion([[("A", 1.0)]])

    use(transport(handler), client)
    results = client.post(
        "/v1/decide/batch",
        json={
            "concurrency": 1,
            "items": [
                {"decision": "needs_human", "variables": {"input": "a"}},
                {"decision": "needs_human", "variables": {"input": "b"}},
                {"decision": "needs_human", "variables": {"input": "c"}},
            ],
        },
    ).json()["results"]

    assert [r.get("choice") for r in results] == ["escalate", None, "escalate"]
    assert results[1]["status"] == 502
