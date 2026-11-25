"""
Έλεγχοι της βάσης αποτελεσμάτων (store.py) και της αναζήτησης πλέγματος (optimum.py).

    uv run pytest tmd_parametric/tests
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import analysis
import excitation as exc
import optimum
import results
import store as rstore
from structure import Structure, design_tmd, tsai_lin_f, tsai_lin_xi

S = Structure()
M, F, X = (rstore.grid_values(0.01, 0.05, 3), rstore.grid_values(0.9, 1.0, 3),
           rstore.grid_values(0.05, 0.10, 3))


@pytest.fixture(scope="module")
def elcentro():
    return exc.elcentro()


@pytest.fixture
def db(tmp_path):
    s = rstore.ResultStore(tmp_path)
    yield s
    s.close()


def test_stored_equals_fresh_run(db, elcentro):
    df = optimum.grid_search(S, elcentro, M, F, X, store=db)
    tmd = design_tmd(S, 0.03, f=0.95, ξd=0.075)
    fresh = results.compare(analysis.run_analysis(S, elcentro),
                            analysis.run_analysis(S, elcentro, tmd))
    row = df[(df.m_hat == 0.03) & (df.f == 0.95) & (df.ξd == 0.075)].iloc[0]
    for col in ("u_max_με", "RMS_με", "διαδρομή_TMD", "μείωση_u_max_%"):
        assert row[col] == pytest.approx(fresh[col], rel=1e-12)


def test_repeat_computes_nothing_and_gives_same_table(db, elcentro):
    first = optimum.grid_search(S, elcentro, M, F, X, store=db)
    second = optimum.grid_search(S, elcentro, M, F, X, store=db)
    assert first.attrs["n_computed"] == 27
    assert second.attrs["n_computed"] == 0
    assert first.equals(second)
    assert list(db.searches()["n_computed"]) == [0, 27]  # πιο πρόσφατη πρώτη


def test_refined_grid_computes_only_new_points(db, elcentro):
    optimum.grid_search(S, elcentro, M, F, X, store=db)
    finer = optimum.grid_search(S, elcentro, M, rstore.grid_values(0.9, 1.0, 5), X, store=db,
                                chunk=5)
    assert finer.attrs["n_total"] == 45
    assert finer.attrs["n_computed"] == 18  # f = 0.925 και 0.975, για 3 m̂ × 3 ξd


def test_stored_history_equals_fresh(db, elcentro):
    df = optimum.grid_search(S, elcentro, M, F, X, store=db, histories=True)
    t, u, stroke = db.find_history(df.attrs["exc_id"], S, 0.05, 1.0, 0.1)
    r = analysis.run_analysis(S, elcentro, design_tmd(S, 0.05, f=1.0, ξd=0.1))
    assert np.allclose(t, r.t)
    assert np.allclose(u, r.u, atol=1e-7)          # float32
    assert np.allclose(stroke, r.stroke, atol=1e-6)
    base = db.find_history(df.attrs["exc_id"], S)
    assert np.allclose(base[1], analysis.run_analysis(S, elcentro).u, atol=1e-7)


def test_no_history_when_not_requested(db, elcentro):
    df = optimum.grid_search(S, elcentro, M, F, X, store=db)
    assert db.find_history(df.attrs["exc_id"], S, 0.05, 1.0, 0.1) is None


def test_elcentro_reference_values(db, elcentro):
    f, xd = float(tsai_lin_f(0.01, S.ξ)), float(tsai_lin_xi(0.01, S.ξ))
    row = optimum.grid_search(S, elcentro, [0.01], [f], [xd], store=db).iloc[0]
    assert row["u_max_χωρίς"] == pytest.approx(0.0641, abs=5e-5)
    assert row["u_max_με"] == pytest.approx(0.0612, abs=5e-5)
    assert row["μείωση_u_max_%"] == pytest.approx(4.48, abs=0.01)
    assert row["μείωση_RMS_%"] == pytest.approx(23.54, abs=0.01)
    assert row["διαδρομή_TMD"] == pytest.approx(0.2840, abs=5e-5)


def test_excitation_id_depends_on_values(elcentro):
    scaled = exc.Excitation(name=elcentro.name, dt=elcentro.dt, a=elcentro.a * 1.5)
    renamed = exc.Excitation(name="άλλο όνομα", dt=elcentro.dt, a=elcentro.a)
    assert rstore.excitation_id(scaled) != rstore.excitation_id(elcentro)
    assert rstore.excitation_id(renamed) == rstore.excitation_id(elcentro)


def test_excitation_round_trip(db, elcentro):
    exc_id = db.register_excitation(elcentro)
    back = db.excitation(exc_id)
    assert back.dt == elcentro.dt and np.array_equal(back.a, elcentro.a)


def test_batch_script_parallel_and_resumable(tmp_path):
    """Το run_optimum_batch.py σε δική του διεργασία (το AppTest των άλλων ελέγχων
    αλλάζει το __main__ και το multiprocessing δεν δουλεύει μέσα στο ίδιο pytest)."""
    script = Path(rstore.__file__).with_name("run_optimum_batch.py")
    args = [sys.executable, "-X", "utf8", str(script), "elcentro", "--db", str(tmp_path),
            "--m-range", "0.01", "0.05", "3", "--f-range", "0.9", "1.0", "3",
            "--xd-range", "0.05", "0.10", "3", "--workers", "2"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for expected in (27, 0):  # η δεύτερη φορά τα βρίσκει όλα στη βάση
        out = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                             env=env, timeout=300, cwd=script.parent)
        assert out.returncode == 0, out.stderr
        assert f"έτοιμο: {expected} νέες" in out.stdout, out.stdout
    db = rstore.ResultStore(tmp_path)
    assert len(db.runs(db.searches()["exc_id"].iloc[0], S)) == 27
    db.close()
