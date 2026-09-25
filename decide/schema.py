"""Decision definitions: the catalogue of things the model is allowed to decide.

A decision never asks the model to write prose. It asks it to pick one of a
closed set of options, and the answer we keep is the *probability distribution*
over that set -- not the generated text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Labels are chosen to be a single token in practically every tokenizer.
LETTER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OPTIONS = len(LETTER_LABELS)

_TEMPLATE_VAR = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class DecisionError(ValueError):
    """Raised when a decision definition or its variables are malformed."""


@dataclass(frozen=True)
class Option:
    """One allowed answer.

    ``key`` is what the caller gets back, ``label`` is the single token the
    model actually emits, ``text`` is the description shown in the prompt.
    """

    key: str
    label: str
    text: str = ""

    def render(self) -> str:
        return f"{self.label}. {self.key}" + (f" — {self.text}" if self.text else "")


@dataclass
class Decision:
    """A closed-form question with a fixed answer set."""

    name: str
    question: str
    options: list[Option]
    mode: str = "choice"  # "choice" | "scale"
    system: str | None = None
    min_confidence: float = 0.0
    min_coverage: float = 0.10
    description: str = ""
    scale: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.options:
            raise DecisionError(f"decision {self.name!r}: needs at least one option")
        if len(self.options) > MAX_OPTIONS:
            raise DecisionError(
                f"decision {self.name!r}: {len(self.options)} options exceeds the "
                f"limit of {MAX_OPTIONS} (top_logprobs cannot cover more)"
            )
        keys = [o.key for o in self.options]
        if len(set(keys)) != len(keys):
            raise DecisionError(f"decision {self.name!r}: duplicate option keys {keys}")
        labels = [o.label for o in self.options]
        if len(set(labels)) != len(labels):
            raise DecisionError(f"decision {self.name!r}: duplicate labels {labels}")
        if self.mode not in ("choice", "scale"):
            raise DecisionError(f"decision {self.name!r}: unknown mode {self.mode!r}")

    @property
    def labels(self) -> list[str]:
        return [o.label for o in self.options]

    @property
    def keys(self) -> list[str]:
        return [o.key for o in self.options]

    def by_label(self, label: str) -> Option | None:
        for option in self.options:
            if option.label == label:
                return option
        return None

    def required_variables(self) -> set[str]:
        return set(_TEMPLATE_VAR.findall(self.question))

    def render_question(self, variables: dict[str, Any] | None = None) -> str:
        variables = variables or {}
        missing = self.required_variables() - set(variables)
        if missing:
            raise DecisionError(
                f"decision {self.name!r}: missing variables {sorted(missing)}"
            )
        return self.question.format_map({k: str(v) for k, v in variables.items()})

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any], defaults: dict[str, Any] | None = None) -> "Decision":
        defaults = defaults or {}
        data = {**defaults, **raw}
        name = data.get("name")
        if not name:
            raise DecisionError("decision: 'name' is required")
        question = data.get("question")
        if not question:
            raise DecisionError(f"decision {name!r}: 'question' is required")
        mode = data.get("mode", "choice")

        scale: tuple[int, int] | None = None
        if mode == "scale":
            spec = data.get("scale", {"min": 1, "max": 5})
            lo, hi = int(spec.get("min", 1)), int(spec.get("max", 5))
            if not 0 <= lo < hi <= 9:
                raise DecisionError(
                    f"decision {name!r}: scale must satisfy 0 <= min < max <= 9 "
                    "(single-digit labels only)"
                )
            scale = (lo, hi)
            options = [Option(key=str(n), label=str(n)) for n in range(lo, hi + 1)]
            labelled = {str(k): v for k, v in (data.get("labels") or {}).items()}
            options = [
                Option(key=o.key, label=o.label, text=labelled.get(o.key, ""))
                for o in options
            ]
        else:
            options = cls._parse_options(name, data.get("options"))

        return cls(
            name=name,
            question=str(question).strip(),
            options=options,
            mode=mode,
            system=data.get("system"),
            min_confidence=float(data.get("min_confidence", 0.0)),
            min_coverage=float(data.get("min_coverage", 0.10)),
            description=str(data.get("description", "")),
            scale=scale,
        )

    @staticmethod
    def _parse_options(name: str, raw: Any) -> list[Option]:
        if not raw:
            raise DecisionError(f"decision {name!r}: 'options' is required")
        options: list[Option] = []
        for index, item in enumerate(raw):
            if index >= MAX_OPTIONS:
                raise DecisionError(
                    f"decision {name!r}: more than {MAX_OPTIONS} options"
                )
            label = LETTER_LABELS[index]
            if isinstance(item, str):
                options.append(Option(key=item, label=label))
            elif isinstance(item, dict):
                key = item.get("key")
                if not key:
                    raise DecisionError(f"decision {name!r}: option {index} has no 'key'")
                options.append(
                    Option(
                        key=str(key),
                        label=str(item.get("label", label)),
                        text=str(item.get("text", "")),
                    )
                )
            else:
                raise DecisionError(
                    f"decision {name!r}: option {index} must be a string or mapping"
                )
        return options


@dataclass
class DecisionCatalog:
    """All decisions loaded from a YAML file, plus file-level defaults."""

    decisions: dict[str, Decision] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.decisions)

    def __contains__(self, name: object) -> bool:
        return name in self.decisions

    def __iter__(self):
        return iter(self.decisions.values())

    def get(self, name: str) -> Decision:
        try:
            return self.decisions[name]
        except KeyError:
            known = ", ".join(sorted(self.decisions)) or "(none)"
            raise DecisionError(f"unknown decision {name!r}; known: {known}") from None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DecisionCatalog":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DecisionCatalog":
        defaults = dict(raw.get("defaults") or {})
        inheritable = {
            k: v
            for k, v in defaults.items()
            if k in {"system", "min_confidence", "min_coverage"}
        }
        decisions: dict[str, Decision] = {}
        for item in raw.get("decisions") or []:
            decision = Decision.from_dict(item, inheritable)
            if decision.name in decisions:
                raise DecisionError(f"duplicate decision name {decision.name!r}")
            decisions[decision.name] = decision
        return cls(decisions=decisions, defaults=defaults)
