"""Command line front end.

    python -m decide.cli --list
    python -m decide.cli route_inquiry --var input="請求書が二重に届いた"
    python -m decide.cli --inline "この文章は苦情か" -o yes -o no --var input=...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .decider import DEFAULT_BASE_URL, DEFAULT_MODEL, Decider, GatewayError
from .schema import Decision, DecisionCatalog, DecisionError

DEFAULT_CATALOG = Path(
    os.environ.get(
        "NEMOTRON_DECISIONS", str(Path(__file__).with_name("decisions.yaml"))
    )
)


def _parse_vars(pairs: list[str]) -> dict[str, str]:
    variables: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--var expects key=value, got {pair!r}")
        if value == "-":
            value = sys.stdin.read()
        variables[key] = value
    return variables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decide", description=__doc__)
    parser.add_argument("name", nargs="?", help="decision name from the catalogue")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--list", action="store_true", help="list known decisions")
    parser.add_argument(
        "--inline", metavar="QUESTION", help="ad-hoc question instead of a catalogue entry"
    )
    parser.add_argument(
        "-o", "--option", action="append", default=[], help="option for --inline (repeatable)"
    )
    parser.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    return parser


def _render(result) -> str:
    lines = []
    verdict = result.choice if not result.abstain else "ABSTAIN"
    lines.append(f"{result.decision}: {verdict}")
    lines.append(
        f"  confidence {result.confidence:.3f}"
        f"   coverage {result.coverage:.3f}"
        f"   {result.latency_ms:.0f}ms   [{result.method}]"
    )
    if result.expected_value is not None:
        lines.append(f"  expected value {result.expected_value:.2f}")
    for key, probability in result.distribution.items():
        bar = "█" * int(round(probability * 30))
        lines.append(f"  {key:<12} {probability:6.3f} {bar}")
    if result.reason:
        lines.append(f"  reason: {result.reason}")
    for warning in result.warnings:
        lines.append(f"  warning: {warning}")
    return "\n".join(lines)


async def _main(args: argparse.Namespace) -> int:
    try:
        catalog = DecisionCatalog.from_yaml(args.catalog)
    except (OSError, DecisionError) as exc:
        print(f"catalog error: {exc}", file=sys.stderr)
        return 2

    if args.list:
        for decision in catalog:
            variables = ", ".join(sorted(decision.required_variables())) or "-"
            print(f"{decision.name:<18} {decision.mode:<6} vars={variables}")
            if decision.description:
                print(f"{'':<18} {decision.description}")
        return 0

    try:
        if args.inline:
            if len(args.option) < 2:
                raise DecisionError("--inline needs at least two -o/--option values")
            decision = Decision.from_dict(
                {"name": "inline", "question": args.inline, "options": args.option}
            )
        elif args.name:
            decision = catalog.get(args.name)
        else:
            build_parser().print_help()
            return 2
        variables = _parse_vars(args.var)
    except DecisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    async with Decider(base_url=args.base_url, model=args.model) as decider:
        try:
            result = await decider.decide(
                decision, variables, min_confidence=args.min_confidence
            )
        except (DecisionError, GatewayError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) if args.json
          else _render(result))
    return 0


def main() -> int:
    return asyncio.run(_main(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
