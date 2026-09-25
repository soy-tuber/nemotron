import pytest

from decide.cli import _parse_vars, _render, build_parser
from decide.decider import DecisionResult


def test_parse_vars_splits_on_the_first_equals():
    assert _parse_vars(["input=a=b", "k=v"]) == {"input": "a=b", "k": "v"}


def test_parse_vars_rejects_a_bare_key():
    with pytest.raises(SystemExit, match="key=value"):
        _parse_vars(["input"])


def test_parse_vars_reads_stdin_for_a_dash(monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    assert _parse_vars(["input=-"]) == {"input": "from stdin"}


def test_list_flag_parses_without_a_decision_name():
    args = build_parser().parse_args(["--list"])
    assert args.list and args.name is None


def test_inline_collects_repeated_options():
    args = build_parser().parse_args(["--inline", "q", "-o", "yes", "-o", "no"])
    assert args.option == ["yes", "no"]


def result(**overrides):
    base = dict(
        decision="d",
        choice="yes",
        confidence=0.8,
        coverage=0.95,
        distribution={"yes": 0.8, "no": 0.2},
        abstain=False,
        method="logprob",
        model="m",
        latency_ms=120.0,
    )
    return DecisionResult(**{**base, **overrides})


def test_render_shows_the_choice_and_the_bars():
    text = _render(result())
    assert "d: yes" in text
    assert "confidence 0.800" in text
    assert "█" in text


def test_render_marks_an_abstention_and_its_reason():
    text = _render(result(choice=None, abstain=True, reason="confidence 0.4 < 0.7"))
    assert "ABSTAIN" in text
    assert "reason: confidence 0.4 < 0.7" in text


def test_render_shows_expected_value_and_warnings():
    text = _render(result(expected_value=3.4, warnings=["inflated"]))
    assert "expected value 3.40" in text
    assert "warning: inflated" in text
