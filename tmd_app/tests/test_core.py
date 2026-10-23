"""Έλεγχοι της ανάγνωσης αρχείων (χωρίς Streamlit UI)."""

import numpy as np
import pytest

import core


def test_two_columns_dt_from_data():
    raw = b"0.00 0.1\n0.02 -0.2\n0.04 0.3\n"
    dt, a = core.parse_accelerogram(raw, 0, "t, a", None)
    assert dt == pytest.approx(0.02)
    assert np.allclose(a, [0.1, -0.2, 0.3])


def test_elcentro_file():
    raw = (core.PARAM_DIR.parent / "aug26" / "elCentro.txt").read_bytes()
    dt, a = core.parse_accelerogram(raw, 0, "t, a", None)
    assert dt == pytest.approx(0.02)
    assert len(a) == 1559


def test_one_column_needs_dt():
    with pytest.raises(ValueError):
        core.parse_accelerogram(b"0.1\n0.2\n", 0, "a", None)
    dt, a = core.parse_accelerogram(b"0.1\n0.2\n", 0, "a", 0.01)
    assert dt == 0.01 and len(a) == 2


def test_many_per_line_with_header():
    raw = b"PEER header\nline 2\n0.1 0.2 0.3\n0.4 0.5\n"
    dt, a = core.parse_accelerogram(raw, 2, "πολλές", 0.005)
    assert np.allclose(a, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_variable_dt_rejected():
    with pytest.raises(ValueError):
        core.parse_accelerogram(b"0 0.1\n0.02 0.2\n0.05 0.3\n", 0, "t, a", None)


@pytest.mark.parametrize("raw", [b"0.0 0.1\n", b"", b"# only a comment\n"])
def test_too_short_file_gives_valueerror(raw):
    # Πρέπει να δίνει ValueError (μήνυμα στον χρήστη), όχι άλλη εξαίρεση
    with pytest.raises(ValueError):
        core.parse_accelerogram(raw, 0, "t, a", None)
