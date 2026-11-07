"""Week 2 Task 1: leakage-safe feature extraction from stored syndromes.

Input is column-indexable data (a dict of columns or a Hugging Face split), the
same kind of object sample_split() returns. Only the allowlisted source fields
below are ever read. label, pymatching, correlated_matching, decoder
correctness, y_router and experiment are never touched, so they cannot end up
in X.

How a row is read: syndrome is flat, so reshape it to (time_steps, width). At
time step t only the first round_widths[t] positions are real; the rest is
padding and is masked out even if it holds a 1.

Features (output column order is FEATURE_COLUMNS):
- detection_event_count: number of 1s in real positions.
- event_density: event count / number of real positions (sum of round_widths).
  If there are zero real positions the density is defined as 0.0.
- activity_round_00 ... activity_round_25: events at each time step. Fixed
  length MAX_TIME_STEPS = 26, because rounds goes up to 25 and
  time_steps = rounds + 1 (see docs/DATA_DICTIONARY.md). Shorter rows are
  padded with 0 at the end. A row with more time steps than that raises an
  error, nothing is truncated. The last slot used is the final measurement
  step.
- active_rounds: number of time steps with at least one real event. The final
  measurement time step counts like any other.
- maximum_round_activity: largest event count in a single time step.
- distance, rounds: copied from the row.
- basis: "X" or "Z", kept as a string category. It is encoded later in the
  model pipeline, not here.
"""

import numpy as np

# Only these source fields are read.
SOURCE_FIELDS = (
    "syndrome",
    "round_widths",
    "time_steps",
    "width",
    "distance",
    "rounds",
    "basis",
)

# Largest rounds in the data is 25, so time_steps is at most 26.
MAX_TIME_STEPS = 26
VALID_BASES = ("X", "Z")

ACTIVITY_COLUMNS = tuple(f"activity_round_{t:02d}" for t in range(MAX_TIME_STEPS))
FEATURE_COLUMNS = (
    ("detection_event_count", "event_density")
    + ACTIVITY_COLUMNS
    + ("active_rounds", "maximum_round_activity", "distance", "rounds", "basis")
)
NUMERIC_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c != "basis")


def _as_int(value, name):
    """Return value as a python int, rejecting bools and non-integer floats."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if as_int != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return as_int


def row_features(syndrome, round_widths, time_steps, width, distance, rounds, basis):
    """Features for one shot. Returns a dict in FEATURE_COLUMNS order."""
    time_steps = _as_int(time_steps, "time_steps")
    width = _as_int(width, "width")
    distance = _as_int(distance, "distance")
    rounds = _as_int(rounds, "rounds")
    if time_steps < 1 or width < 1:
        raise ValueError("time_steps and width must be positive")
    if time_steps > MAX_TIME_STEPS:
        raise ValueError(
            f"time_steps={time_steps} is longer than the fixed activity length "
            f"{MAX_TIME_STEPS}"
        )
    if basis not in VALID_BASES:
        raise ValueError(f"basis must be one of {VALID_BASES}, got {basis!r}")

    syn = np.asarray(syndrome)
    widths = np.asarray(round_widths)
    if syn.ndim != 1 or widths.ndim != 1:
        raise ValueError("syndrome and round_widths must be 1-D")
    if syn.size != time_steps * width:
        raise ValueError(
            f"syndrome has {syn.size} values, expected time_steps*width = "
            f"{time_steps * width}"
        )
    if widths.size != time_steps:
        raise ValueError(
            f"round_widths has {widths.size} values, expected time_steps = {time_steps}"
        )
    if not np.isin(syn, [0, 1]).all():
        raise ValueError("syndrome must contain only 0/1 values")
    if not np.issubdtype(widths.dtype, np.integer):
        raise ValueError("round_widths must be integers")
    if (widths < 0).any() or (widths > width).any():
        raise ValueError("each round width must be between 0 and width")

    grid = syn.reshape(time_steps, width).astype(np.int64)
    real = np.arange(width)[None, :] < widths[:, None]  # True for real slots
    per_step = (grid * real).sum(axis=1)

    events = int(per_step.sum())
    real_positions = int(widths.sum())
    density = events / real_positions if real_positions > 0 else 0.0

    activity = np.zeros(MAX_TIME_STEPS, dtype=np.int64)
    activity[:time_steps] = per_step

    out = {"detection_event_count": events, "event_density": density}
    for name, value in zip(ACTIVITY_COLUMNS, activity):
        out[name] = int(value)
    out["active_rounds"] = int((per_step > 0).sum())
    out["maximum_round_activity"] = int(per_step.max())
    out["distance"] = distance
    out["rounds"] = rounds
    out["basis"] = basis
    return out


def extract_features(rows):
    """Build X from column-indexable data. Returns {column: numpy array}.

    Only SOURCE_FIELDS are read from rows, so extra columns (label, decoder
    predictions, experiment, ...) are ignored. Every column in FEATURE_COLUMNS
    is returned, in that order; numeric ones are numeric arrays and basis is an
    array of strings. pandas.DataFrame(result) gives a table.
    """
    columns = {}
    for name in SOURCE_FIELDS:
        try:
            columns[name] = rows[name]
        except (KeyError, IndexError):
            raise ValueError(f"missing required field: {name}") from None

    n = len(columns["syndrome"])
    for name, values in columns.items():
        if len(values) != n:
            raise ValueError(f"{name} has {len(values)} rows, expected {n}")

    per_row = [
        row_features(*(columns[name][i] for name in SOURCE_FIELDS)) for i in range(n)
    ]
    result = {}
    for name in FEATURE_COLUMNS:
        values = [r[name] for r in per_row]
        if name == "basis":
            result[name] = np.array(values, dtype=str)
        elif name == "event_density":
            result[name] = np.array(values, dtype=float)
        else:
            result[name] = np.array(values, dtype=np.int64)
    return result
