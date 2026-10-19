"""Complete-rescue exposure bounds for finite hazard scenarios and escort lag.

One scenario remains fixed for the whole rescue. Integer segment durations can
vary independently inside their declared intervals. The max-plus recurrence
optimizes over every such timing, rather than selecting a different scenario
at each tick. This is an exact finite optimization in real arithmetic; the
implementation evaluates it with NumPy float64 arithmetic. Its guarantee concerns
the supplied scenarios and timing bounds, not clinical safety or unmodeled hazards.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np


@dataclass(frozen=True, slots=True)
class ExposureWitness:
    scenario_index: int
    escort_ticks: tuple[int, ...]
    completion_tick: int
    exposure: float


@dataclass(frozen=True, slots=True)
class RescueCertificate:
    robot_bound: float
    survivor_bound: float
    feasible: bool
    robot_witness: ExposureWitness
    survivor_witness: ExposureWitness
    pickup_tick: int
    earliest_completion_tick: int
    latest_completion_tick: int
    scenario_count: int
    schedule_count: int
    dp_cells: int
    exposure_samples: int
    numeric_scratch_bytes: int


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _path(nodes, node_count, name):
    try:
        result = tuple(_integer(node, name) for node in nodes)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty node sequence") from exc
    if not result or max(result) >= node_count:
        raise ValueError(f"{name} must be nonempty with nodes inside the scenario")
    return result


def _durations(values, count, name):
    if isinstance(values, (int, np.integer)):
        return (_integer(values, name, minimum=1),) * count
    try:
        result = tuple(_integer(value, name, minimum=1) for value in values)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a positive integer or segment sequence"
        ) from exc
    if len(result) != count:
        raise ValueError(f"{name} must contain one duration per path segment")
    return result


def _budget(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be nonnegative or positive infinity") from exc
    if np.isnan(result) or result < 0:
        raise ValueError(f"{name} must be nonnegative or positive infinity")
    return result


def _prefix_doses(hazard, approach, durations, delay, pickup, dt):
    robot = float(np.sum(hazard[:delay, approach[0]], dtype=np.float64))
    time = delay
    for node, duration in zip(approach[:-1], durations, strict=True):
        robot += float(np.sum(hazard[time : time + duration, node], dtype=np.float64))
        time += duration
    survivor = float(np.sum(hazard[:pickup, approach[-1]], dtype=np.float64))
    return robot * dt, survivor * dt


def _escort_maximum(hazard, escort, lower, upper, pickup, horizon, dt):
    """Sliding max of F(u)-P(u) for u in [t-upper,t-lower].

    Rolling score arrays use compressed reachable-time windows. Retained parent
    arrays reconstruct a timing witness. Queue storage is numeric and accounted
    for; no Python deque or heap hides an unreported label collection.
    """
    if not lower:
        return 0.0, (), pickup, 0
    nodes = sorted(set(escort[:-1]))
    columns = {node: index for index, node in enumerate(nodes)}
    prefix = np.zeros((horizon + 1, len(nodes)), dtype=np.float64)
    for column, node in enumerate(nodes):
        np.cumsum(hazard[:horizon, node], out=prefix[1:, column])
    prefix *= dt
    scores = np.zeros(1, dtype=np.float64)
    low_time = high_time = pickup
    parents, next_lows = [], []
    parent_bytes = 0
    peak_bytes = prefix.nbytes + scores.nbytes
    for node, minimum, maximum in zip(escort[:-1], lower, upper, strict=True):
        next_low, next_high = low_time + minimum, high_time + maximum
        next_scores = np.full(next_high - next_low + 1, -np.inf)
        parent = np.full(len(next_scores), -1, dtype=np.int64)
        queue_time = np.empty(len(scores), dtype=np.int64)
        queue_value = np.empty(len(scores), dtype=np.float64)
        parent_bytes += parent.nbytes
        peak_bytes = max(
            peak_bytes,
            prefix.nbytes
            + scores.nbytes
            + next_scores.nbytes
            + parent_bytes
            + queue_time.nbytes
            + queue_value.nbytes,
        )
        head = tail = 0
        column = columns[node]
        for time in range(next_low, next_high + 1):
            earliest = time - maximum
            while head < tail and queue_time[head] < earliest:
                head += 1
            candidate = time - minimum
            if candidate <= high_time:
                value = scores[candidate - low_time] - prefix[candidate, column]
                while head < tail and queue_value[tail - 1] < value:
                    tail -= 1
                queue_time[tail], queue_value[tail] = candidate, value
                tail += 1
            if head == tail:
                raise RuntimeError("reachable escort time lost its predecessor")
            offset = time - next_low
            next_scores[offset] = prefix[time, column] + queue_value[head]
            parent[offset] = queue_time[head]
        parents.append(parent)
        next_lows.append(next_low)
        scores, low_time, high_time = next_scores, next_low, next_high
        del queue_time, queue_value
    endpoint = int(np.argmax(scores))
    exposure = float(scores[endpoint])
    completion = time = low_time + endpoint
    witness = []
    for parent, low in reversed(list(zip(parents, next_lows, strict=True))):
        previous = int(parent[time - low])
        witness.append(time - previous)
        time = previous
    if time != pickup:
        raise RuntimeError("escort witness does not begin at pickup")
    return exposure, tuple(reversed(witness)), completion, peak_bytes


def certify_rescue(
    scenarios,
    approach,
    escort,
    *,
    approach_ticks=1,
    escort_min_ticks=1,
    escort_max_ticks=None,
    departure_delay=0,
    robot_budget=float("inf"),
    survivor_budget=float("inf"),
    dt=1.0,
    max_dp_cells=2_000_000,
    max_numeric_bytes=128 * 1024 * 1024,
):
    """Bound both actors' complete dose across every scenario and escort timing.

    ``scenarios[s,t,node]`` is a finite nonnegative exposure rate. Paths include
    their endpoints. Each segment exposes its actor at the segment's origin for
    its integer duration; the final refuge has no post-completion exposure.
    Robot and survivor follow identical escort occupancy. Before pickup, the
    robot waits at its start for ``departure_delay`` and follows ``approach``;
    the survivor waits at the victim throughout that entire interval.

    Scalar durations broadcast over segments. Min=max gives exact nominal
    timings. Positive infinity budgets explicitly disable an actor's threshold.
    A horizon shorter than the latest possible completion raises ValueError;
    the algorithm never certifies only a conveniently truncated schedule set.

    ``dp_cells`` counts reachable (scenario, segment-index, time) states,
    including the initial pickup states. ``exposure_samples`` counts input-rate
    entries read for temporal prefixes and pre-pickup doses. Numeric scratch
    includes prefix sums, both frontiers, queue buffers, and every witness parent;
    it excludes supplied/normalized scenario inputs and Python object overhead.
    Resource guards bound DP work and numeric scratch before allocation.
    """
    try:
        raw = np.asarray(scenarios)
        if np.iscomplexobj(raw):
            raise ValueError("scenarios must be real")
        hazards = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scenarios must be a real numeric array") from exc
    if hazards.ndim != 3 or hazards.shape[0] < 1 or hazards.shape[2] < 1:
        raise ValueError("scenarios must have shape (nonempty scenarios, ticks, nodes)")
    if not np.all(np.isfinite(hazards)) or np.any(hazards < 0):
        raise ValueError("scenario exposure rates must be finite and nonnegative")
    approach = _path(approach, hazards.shape[2], "approach")
    escort = _path(escort, hazards.shape[2], "escort")
    if approach[-1] != escort[0]:
        raise ValueError("approach must end at the escort's victim node")
    durations = _durations(approach_ticks, len(approach) - 1, "approach_ticks")
    lower = _durations(escort_min_ticks, len(escort) - 1, "escort_min_ticks")
    upper = (
        lower
        if escort_max_ticks is None
        else _durations(escort_max_ticks, len(escort) - 1, "escort_max_ticks")
    )
    if any(low > high for low, high in zip(lower, upper, strict=True)):
        raise ValueError("escort minimum durations must not exceed maximum durations")
    delay = _integer(departure_delay, "departure_delay")
    max_dp_cells = _integer(max_dp_cells, "max_dp_cells", minimum=1)
    max_numeric_bytes = _integer(max_numeric_bytes, "max_numeric_bytes", minimum=1)
    robot_budget = _budget(robot_budget, "robot_budget")
    survivor_budget = _budget(survivor_budget, "survivor_budget")
    try:
        dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be positive and finite") from exc
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    pickup = delay + sum(durations)
    earliest, latest = pickup + sum(lower), pickup + sum(upper)
    if latest > hazards.shape[1]:
        raise ValueError("scenario horizon must cover every admissible complete rescue")
    widths = [1]
    for low, high in zip(lower, upper, strict=True):
        widths.append(widths[-1] + high - low)
    cells = hazards.shape[0] * sum(widths)
    if cells > max_dp_cells:
        raise ValueError("rescue timing DP exceeds max_dp_cells")
    prefix_bytes = (latest + 1) * len(set(escort[:-1])) * 8
    parent_bytes = sum(widths[1:]) * 8
    # Safe upper bound; actual rolling-frontier peak is reported below.
    numeric_limit = prefix_bytes + parent_bytes + 4 * max(widths) * 8 if lower else 0
    if numeric_limit > max_numeric_bytes:
        raise ValueError("rescue timing scratch exceeds max_numeric_bytes")
    robot_best = survivor_best = None
    peak = 0
    try:
        with np.errstate(over="raise", invalid="raise"):
            for index, hazard in enumerate(hazards):
                robot_prefix, survivor_prefix = _prefix_doses(
                    hazard, approach, durations, delay, pickup, dt
                )
                escort_dose, timings, completion, scratch = _escort_maximum(
                    hazard, escort, lower, upper, pickup, latest, dt
                )
                robot = robot_prefix + escort_dose
                survivor = survivor_prefix + escort_dose
                if not np.isfinite(robot) or not np.isfinite(survivor):
                    raise ValueError("scenario exposure exceeds float64 range")
                if robot_best is None or robot > robot_best.exposure:
                    robot_best = ExposureWitness(index, timings, completion, robot)
                if survivor_best is None or survivor > survivor_best.exposure:
                    survivor_best = ExposureWitness(
                        index, timings, completion, survivor
                    )
                peak = max(peak, scratch)
    except FloatingPointError as exc:
        raise ValueError("scenario exposure exceeds float64 range") from exc
    assert robot_best is not None and survivor_best is not None
    samples = hazards.shape[0] * (2 * pickup + latest * len(set(escort[:-1])))
    return RescueCertificate(
        robot_best.exposure,
        survivor_best.exposure,
        robot_best.exposure <= robot_budget
        and survivor_best.exposure <= survivor_budget,
        robot_best,
        survivor_best,
        pickup,
        earliest,
        latest,
        hazards.shape[0],
        prod(high - low + 1 for low, high in zip(lower, upper, strict=True)),
        cells,
        samples,
        peak,
    )


def certify_integer_rescue(scenarios, approach, escort, **options):
    """Evaluate integral exposure units with exact float64 integer arithmetic.

    Rates must have signed/unsigned integer dtype, be nonnegative, and satisfy
    ``max_rate * supplied_horizon <= 2**52``. Every temporal prefix, partial
    rescue dose, and actor total is then an integer in [0, 2**52]. Subtracted
    max-plus keys lie in [-2**52, 2**52]; all are exactly representable in
    float64. Addition, subtraction, comparison, and multiplication by dt=1
    therefore introduce no rounding. Returned bounds and witness exposures
    are integer-valued floats, exact for the supplied integral-unit model.

    Budgets must be nonnegative integers or positive infinity. Integer budgets
    above 2**52 can be replaced by infinity because every possible actor dose
    is already <=2**52. Float/object/bool rate arrays and fractional budgets
    are rejected. dt may only be omitted or explicitly equal to one.

    This does not make real hazards integral or exact. An external upward
    quantization must separately justify its scaling and conservative error.
    """
    try:
        rates = np.asarray(scenarios)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "integer scenarios must be a signed/unsigned integer array"
        ) from exc
    if rates.dtype.kind not in "iu":
        raise ValueError("integer scenarios require signed/unsigned integer dtype")
    if rates.ndim != 3 or rates.shape[0] < 1 or rates.shape[2] < 1:
        raise ValueError("integer scenarios must have shape (scenarios, ticks, nodes)")
    if np.any(rates < 0):
        raise ValueError("integer scenario exposure rates must be nonnegative")
    maximum = int(rates.max()) if rates.size else 0
    exact_limit = 2**52
    if maximum * rates.shape[1] > exact_limit:
        raise ValueError("integer exposure bound max_rate*horizon must be <=2**52")
    dt = options.pop("dt", 1)
    if (
        isinstance(dt, (bool, np.bool_))
        or not isinstance(dt, (int, float, np.integer, np.floating))
        or dt != 1
    ):
        raise ValueError("integer rescue certificates require dt=1")
    for name in ("robot_budget", "survivor_budget"):
        value = options.pop(name, float("inf"))
        if isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            if value < 0:
                raise ValueError("integer exposure budgets must be nonnegative")
            options[name] = int(value) if value <= exact_limit else float("inf")
        elif isinstance(value, (float, np.floating)) and np.isposinf(value):
            options[name] = float("inf")
        else:
            raise ValueError(
                "integer exposure budgets must be integers or positive infinity"
            )
    return certify_rescue(rates, approach, escort, dt=1.0, **options)
