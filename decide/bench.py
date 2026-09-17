"""Latency and throughput benchmark for the decision path.

    python -m decide.bench --decision route_inquiry --repeat 50 --concurrency 1 8 32

Speed here comes from two places, not from a faster model: the generation is
a handful of tokens instead of a paragraph, and independent decisions are
batched by vLLM when issued concurrently.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from pathlib import Path

from .decider import DEFAULT_BASE_URL, DEFAULT_MODEL, Decider
from .schema import DecisionCatalog

DEFAULT_CATALOG = Path(
    os.environ.get(
        "NEMOTRON_DECISIONS", str(Path(__file__).with_name("decisions.yaml"))
    )
)

SAMPLE = (
    "先月の請求が二重に引き落とされています。至急確認して返金してください。"
    "先週も同じ問い合わせをしましたが返信がありません。"
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


async def _run_level(
    decider: Decider, decision, variables: dict[str, str], repeat: int, concurrency: int
) -> dict[str, float]:
    gate = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    failures = 0

    async def one() -> None:
        nonlocal failures
        async with gate:
            try:
                result = await decider.decide(decision, variables)
            except Exception:
                failures += 1
                return
            latencies.append(result.latency_ms)

    started = time.perf_counter()
    await asyncio.gather(*(one() for _ in range(repeat)))
    wall = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "ok": len(latencies),
        "failed": failures,
        "wall_s": wall,
        "rps": (len(latencies) / wall) if wall > 0 else float("nan"),
        "mean_ms": statistics.fmean(latencies) if latencies else float("nan"),
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
    }


async def _main(args: argparse.Namespace) -> int:
    catalog = DecisionCatalog.from_yaml(args.catalog)
    decision = catalog.get(args.decision)
    variables = {name: args.input for name in decision.required_variables()}

    async with Decider(base_url=args.base_url, model=args.model) as decider:
        # Warm up: the gateway may still need to load or swap the model.
        await decider.decide(decision, variables)
        print(f"{'conc':>5} {'ok':>5} {'fail':>5} {'rps':>8} {'mean':>8} {'p50':>8} {'p95':>8}")
        for concurrency in args.concurrency:
            row = await _run_level(decider, decision, variables, args.repeat, concurrency)
            print(
                f"{row['concurrency']:>5} {row['ok']:>5} {row['failed']:>5} "
                f"{row['rps']:>8.2f} {row['mean_ms']:>8.1f} "
                f"{row['p50_ms']:>8.1f} {row['p95_ms']:>8.1f}"
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="decide.bench", description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--decision", default="route_inquiry")
    parser.add_argument("--input", default=SAMPLE)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
