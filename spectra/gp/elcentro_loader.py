"""Parses the El Centro accelerogram from data/ELCENTRO (time-accel pair format)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

_DATA_PATH = Path(__file__).parent / "data" / "ELCENTRO"

# Target uniform time step after resampling
_DT_RESAMPLE = 0.01  # seconds


def _parse_time_accel_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a file of whitespace-separated (time_s, accel_g) pairs.
    Multiple pairs may appear on a single line.
    Returns (times, accels) as 1-D arrays.
    """
    text = path.read_bytes().decode("ascii", errors="ignore")
    tokens = [t for t in text.split() if t.lstrip("-").replace(".", "", 1).isdigit()]
    values = np.array(tokens, dtype=float)
    if values.size % 2 != 0:
        values = values[:-1]  # drop trailing odd value if any
    pairs = values.reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def _resample_uniform(
    times: np.ndarray,
    accels: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, float]:
    """
    Linearly interpolate a non-uniform (time, accel) record onto a uniform grid.
    Duplicate time entries (velocity-break markers) are deduplicated before
    interpolation to avoid zero-length intervals.
    """
    # Remove duplicate / near-duplicate times (keep last of each pair)
    _, unique_idx = np.unique(times.round(6), return_index=True)
    t = times[unique_idx]
    a = accels[unique_idx]

    t_uniform = np.arange(t[0], t[-1] + dt * 0.5, dt)
    interp = interp1d(t, a, kind="linear", bounds_error=False, fill_value=0.0)
    ag_uniform = interp(t_uniform)
    return ag_uniform, dt


def load_elcentro(
    data_path: Path = _DATA_PATH,
    dt: float = _DT_RESAMPLE,
) -> tuple[np.ndarray, float]:
    """
    Load and resample the El Centro record.

    Returns
    -------
    ag_g  : acceleration array [g], uniform time step
    dt    : time step [s]
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"El Centro data file not found: {data_path}\n"
            "Place the ELCENTRO file in the data/ directory."
        )
    times, accels = _parse_time_accel_pairs(data_path)
    return _resample_uniform(times, accels, dt)
