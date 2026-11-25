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


# --------------------------------------------------------------------------
# PEER .AT2 (επικεφαλίδα στην αρχή ή στο τέλος)
# --------------------------------------------------------------------------

PEER_TOP_NEW = (b"PEER NGA STRONG MOTION DATABASE RECORD\nCHI-CHI 09/20/99, CHY006, N\n"
                b"ACCELERATION TIME SERIES IN UNITS OF G\nNPTS=    7, DT=   .0050 SEC\n"
                b" 1.0E-03 2.0E-03 3.0E-03 4.0E-03 5.0E-03\n 6.0E-03 7.0E-03 0.0 0.0\n")
PEER_TOP_OLD = (b"PEER NGA STRONG MOTION DATABASE RECORD\nNORTHRIDGE 01/17/94, SYLMAR, 090\n"
                b"ACCELERATION TIME HISTORY IN UNITS OF G\n7    0.0100    NPTS, DT\n"
                b" 1.0E-03 2.0E-03 3.0E-03 4.0E-03 5.0E-03\n 6.0E-03 7.0E-03\n")


@pytest.mark.parametrize("raw, dt_expected", [(PEER_TOP_NEW, 0.005), (PEER_TOP_OLD, 0.01)])
def test_peer_header_on_top(raw, dt_expected):
    dt, a = core.parse_accelerogram(raw, 0, "PEER .at2", None)
    assert dt == dt_expected
    assert np.allclose(a, np.arange(1, 8) * 1e-3)  # τα μηδενικά μετά το NPTS κόβονται


def test_peer_header_at_end():
    # Τα αρχεία του aug26/GM_X έχουν την επικεφαλίδα στο τέλος (37500 σημεία, dt = 0.004 s)
    raw = (core.PARAM_DIR.parent / "aug26" / "GM_X" / "CHY006-N.at2").read_bytes()
    dt, a = core.parse_accelerogram(raw, 0, "PEER .at2", None)
    assert dt == 0.004
    assert len(a) == 37500


def test_peer_without_header_rejected():
    with pytest.raises(ValueError):
        core.parse_accelerogram(b"0.1 0.2 0.3\n0.4 0.5\n", 0, "PEER .at2", None)
