"""Seeded random-number derivation.

Reproducibility rule for the whole project: *no stage ever draws from another
stage's stream*.  Each generation stage gets an independent substream derived
from the master seed, so changing the number of draws in the floor-plan stage
cannot silently perturb the collapse, and a stage can be re-run in isolation
and produce bit-identical output.

Stage names are registered explicitly rather than hashed, so a substream index
is stable across Python versions and reviewable in a diff.
"""

from __future__ import annotations

import numpy as np

#: Registered generation stages, in declaration order.  A stage's index in this
#: tuple *is* its substream key -- appending is safe, reordering is not.
STAGES: tuple[str, ...] = (
    "floorplan",
    "structure",
    "collapse",
    "extract",
    "populate",
    "hazard",
    "survivors",
    "search",
    "radio_loss",
    "contact",
)

_STAGE_INDEX: dict[str, int] = {name: i for i, name in enumerate(STAGES)}


def stage_rng(seed: int, stage: str) -> np.random.Generator:
    """Return the independent generator for `stage` under master `seed`.

    Raises:
        KeyError: if `stage` is not registered in `STAGES`.  Silently accepting
            an unknown stage would hand out a stream that collides with another
            stage's, which is exactly the bug this module exists to prevent.
    """
    if stage not in _STAGE_INDEX:
        raise KeyError(
            f"unregistered generation stage {stage!r}; "
            f"add it to cascade.rng.STAGES (known: {', '.join(STAGES)})"
        )
    # SeedSequence mixes the pair, so adjacent seeds do not yield correlated
    # substreams the way a bare `seed + index` would.
    return np.random.default_rng(np.random.SeedSequence([seed, _STAGE_INDEX[stage]]))


def substream(rng: np.random.Generator, label: int) -> np.random.Generator:
    """Derive a child generator for a nested, order-independent sub-task.

    Used where a stage processes a collection whose iteration order should not
    affect any individual element's draws -- e.g. per-storey floor plans.
    """
    return np.random.default_rng(rng.bit_generator.seed_seq.spawn(label + 1)[label])
