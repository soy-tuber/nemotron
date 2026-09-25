import pytest

from decide.schema import Decision, DecisionCatalog, DecisionError, Option


def make(**overrides):
    base = {"name": "d", "question": "q", "options": ["yes", "no"]}
    return Decision.from_dict({**base, **overrides})


def test_options_get_sequential_single_token_labels():
    decision = make(options=["a", "b", "c"])
    assert decision.labels == ["A", "B", "C"]
    assert decision.keys == ["a", "b", "c"]


def test_option_mapping_keeps_description():
    decision = make(options=[{"key": "billing", "text": "課金"}])
    assert decision.options[0] == Option(key="billing", label="A", text="課金")
    assert "billing" in decision.options[0].render()


def test_rejects_duplicate_keys():
    with pytest.raises(DecisionError, match="duplicate option keys"):
        make(options=["yes", "yes"])


def test_rejects_more_options_than_labels():
    with pytest.raises(DecisionError, match="more than 26"):
        make(options=[f"o{i}" for i in range(27)])


def test_rejects_unknown_mode():
    with pytest.raises(DecisionError, match="unknown mode"):
        make(mode="freeform")


def test_scale_builds_digit_labels_and_descriptions():
    decision = Decision.from_dict(
        {
            "name": "urgency",
            "question": "q",
            "mode": "scale",
            "scale": {"min": 1, "max": 5},
            "labels": {1: "low", 5: "high"},
        }
    )
    assert decision.labels == ["1", "2", "3", "4", "5"]
    assert decision.by_label("1").text == "low"
    assert decision.by_label("3").text == ""


def test_scale_bounds_are_single_digit():
    with pytest.raises(DecisionError, match="single-digit"):
        Decision.from_dict(
            {"name": "s", "question": "q", "mode": "scale", "scale": {"min": 1, "max": 10}}
        )


def test_render_question_substitutes_variables():
    decision = make(question="judge: {input}")
    assert decision.required_variables() == {"input"}
    assert decision.render_question({"input": "text"}) == "judge: text"


def test_render_question_reports_missing_variables():
    decision = make(question="judge: {input} in {locale}")
    with pytest.raises(DecisionError, match=r"missing variables \['input', 'locale'\]"):
        decision.render_question({})


def test_catalog_applies_inheritable_defaults_only():
    catalog = DecisionCatalog.from_dict(
        {
            "defaults": {"min_confidence": 0.8, "model": "ignored-here"},
            "decisions": [
                {"name": "a", "question": "q", "options": ["x", "y"]},
                {"name": "b", "question": "q", "options": ["x", "y"], "min_confidence": 0.2},
            ],
        }
    )
    assert catalog.get("a").min_confidence == 0.8
    assert catalog.get("b").min_confidence == 0.2
    assert catalog.defaults["model"] == "ignored-here"


def test_catalog_rejects_duplicate_names():
    with pytest.raises(DecisionError, match="duplicate decision name"):
        DecisionCatalog.from_dict(
            {"decisions": [{"name": "a", "question": "q", "options": ["x", "y"]}] * 2}
        )


def test_catalog_unknown_name_lists_known_ones():
    catalog = DecisionCatalog.from_dict(
        {"decisions": [{"name": "a", "question": "q", "options": ["x", "y"]}]}
    )
    with pytest.raises(DecisionError, match="known: a"):
        catalog.get("nope")


def test_shipped_catalogue_loads():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "decide" / "decisions.yaml"
    catalog = DecisionCatalog.from_yaml(path)
    assert {"route_inquiry", "needs_human", "urgency", "diff_risk"} <= set(
        catalog.decisions
    )
    assert catalog.get("urgency").mode == "scale"
