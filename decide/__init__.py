"""Local reproduction of the "decide, don't write" inference pattern.

Instead of generating prose, the model picks from a closed option set and we
keep the probability distribution over that set. Format errors are impossible
by construction; semantic errors are not, which is what ``confidence``,
``coverage`` and the abstain path are for.
"""

from .decider import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    Decider,
    DecisionResult,
    GatewayError,
    decide_sync,
)
from .schema import Decision, DecisionCatalog, DecisionError, Option

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "Decider",
    "Decision",
    "DecisionCatalog",
    "DecisionError",
    "DecisionResult",
    "GatewayError",
    "Option",
    "decide_sync",
]
