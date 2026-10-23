"""
Έλεγχοι της εφαρμογής με st.testing.v1.AppTest (χωρίς browser).

Οι τιμές αναφοράς είναι αυτές του CLAUDE.md: ίδιες με του tmd_parametric/
και των _gp scripts. Αν αλλάξουν, κάτι άλλαξε στη φυσική.

    uv run pytest tmd_app/tests
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
PAGES = ["app_pages/excitation.py", "app_pages/model.py",
         "app_pages/analysis.py", "app_pages/parametric.py"]
TIMEOUT = 60

HARMONIC = {"kind": "harmonic", "rho": 1.0, "duration": 60.0, "dt": 0.01}


def open_page(page, **state):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    at.switch_page(page).run()
    assert not at.exception, at.exception
    return at


def metrics(at):
    return {m.label: (m.value, m.delta) for m in at.metric}


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = open_page(page)
    assert not at.error
    # Η σύνοψη στην πλαϊνή στήλη γράφεται σε κάθε σελίδα
    assert "Τρέχουσες επιλογές" in [s.value for s in at.sidebar.subheader]


def test_elcentro_reference():
    m = metrics(open_page("app_pages/analysis.py"))
    assert m["u_max χωρίς TMD"][0] == "0.0641 m"
    assert m["u_max με TMD"] == ("0.0612 m", "-4.48 %")
    assert m["RMS με TMD"][1] == "-23.54 %"
    assert m["Μέγιστη διαδρομή TMD"][0] == "0.2840 m"


def test_harmonic_reference():
    m = metrics(open_page("app_pages/analysis.py", excitation_spec=HARMONIC))
    assert m["u_max χωρίς TMD"][0] == "0.9963 m"
    assert m["u_max με TMD"] == ("0.3806 m", "-61.80 %")
    assert m["RMS με TMD"][1] == "-63.39 %"
    assert m["Μέγιστη διαδρομή TMD"][0] == "2.7925 m"


def test_select_harmonic_in_ui():
    at = open_page("app_pages/excitation.py")
    at.segmented_control(key="exc_source").set_value("Αρμονική").run()
    assert not at.exception
    assert at.session_state["excitation_spec"]["kind"] == "harmonic"
    assert metrics(at)["Σημεία"][0] == "6001"


def test_structure_change_persists_across_pages():
    at = open_page("app_pages/model.py")
    at.number_input(key="T").set_value(0.60).run()
    at.switch_page("app_pages/analysis.py").run()
    assert not at.exception
    assert at.session_state["T"] == pytest.approx(0.60)
    assert "T = 0.6 s" in at.sidebar.markdown[0].value


def test_mass_ratio_sweep():
    at = open_page("app_pages/parametric.py")
    at.number_input[0].set_value(0.01)
    at.number_input[1].set_value(0.05)
    at.number_input[2].set_value(3)
    at.form_submit_button[0].click().run()
    assert not at.exception
    df = at.dataframe[0].value
    assert len(df) == 3
    # m̂ = 0.01 ξανά δίνει την τιμή αναφοράς του El Centro
    assert df["μείωση_u_max_%"].iloc[0] == pytest.approx(4.48, abs=0.01)
    # η διαδρομή μειώνεται καθώς αυξάνει το m̂
    assert df["διαδρομή_TMD"].is_monotonic_decreasing


def test_reversed_range_shows_error_and_keeps_sidebar():
    at = open_page("app_pages/parametric.py")
    at.number_input[0].set_value(0.10)
    at.number_input[1].set_value(0.05)
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert at.error
    assert at.sidebar.subheader, "η σύνοψη χάθηκε (st.stop() στη σελίδα)"
