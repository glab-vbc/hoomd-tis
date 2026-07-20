"""Finite-difference validation of the two TIS many-body custom forces.

This is the correctness proof for tis/custom_forces.py: for BOTH the stacking
term (distance + 2 dihedrals) and the hydrogen-bonding term (distance + 2 angles
+ 3 dihedrals) we finite-difference the TOTAL energy w.r.t. every particle
coordinate and compare to the analytic net force. The error-prone piece is the
dihedral gradient (Blondel-Karplus); the tests below catch any sign / coefficient
mistake in it.

We also:
  * assert Newton's third law (sum of forces ~ 0 for an isolated interaction),
  * check the HOOMD-attached force reports the same total energy as the pure
    numpy evaluator (validates the rtag scatter + potential_energy wiring),
  * run the FULL 6-term model for ~1000 Langevin steps and assert stability.

Run:  PYTHONPATH=<repo-root> pytest pytest/test_custom_fd.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hoomd
import hoomd.md as md

from tis import io, forces, custom_forces as cf

KT_300 = forces.KB * 300.0


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _fd_forces(force, pos, box, h=1e-5):
    """Central-difference the TOTAL energy -> force array (N, 3)."""
    FD = np.zeros_like(pos)
    for a in range(pos.shape[0]):
        for d in range(3):
            pp = pos.copy(); pp[a, d] += h
            pm = pos.copy(); pm[a, d] -= h
            FD[a, d] = -(force.energy_of(pp, box) - force.energy_of(pm, box)) / (2 * h)
    return FD


def _nontrivial_positions(seq, seed=42, amp=1.5):
    """Initial TIS geometry perturbed into a non-degenerate configuration."""
    snap = io.build_strand_snapshot(seq)
    box = np.array(snap.configuration.box[:3], dtype=float)
    pos0 = np.array(snap.particles.position, dtype=float)
    rng = np.random.default_rng(seed)
    pos = pos0 + rng.uniform(-amp, amp, size=pos0.shape)
    return pos0, pos, box


# --------------------------------------------------------------------------- #
# stacking                                                                    #
# --------------------------------------------------------------------------- #
def test_stacking_fd_and_newton():
    seq = "GCAUGC"
    pos0, pos, box = _nontrivial_positions(seq, seed=1)
    st = cf.TISStacking(len(seq), ref_positions=pos0, box=box)
    # push references away from the actual values so every coord is in the
    # varying region (non-zero deviations -> non-trivial gradients).
    st.set_references([(4.0, 0.5, -0.7)] * len(st._pairs))

    U, F = st.compute_energy_forces(pos, box)
    assert np.isfinite(U)
    FD = _fd_forces(st, pos, box)

    fmax = np.abs(F).max()
    assert fmax > 1e-3, "test config is degenerate (forces ~ 0)"
    tol = 1e-4 * (1.0 + fmax)
    assert np.abs(F - FD).max() < tol, (
        f"stacking force != FD: maxerr={np.abs(F - FD).max():.2e} tol={tol:.2e}")

    # Newton's third law: isolated interactions -> net force ~ 0
    assert np.abs(F.sum(axis=0)).max() < 1e-8


# --------------------------------------------------------------------------- #
# hydrogen bonding                                                            #
# --------------------------------------------------------------------------- #
def test_hbond_fd_and_newton():
    seq = "GCAUGC"
    pos0, pos, box = _nontrivial_positions(seq, seed=2)
    hb = cf.TISHydrogenBonding(sequence=seq)

    # bring the two bases of the first complementary candidate into contact
    # (well inside the distance gate so no FD step crosses the cutoff).
    i, j, _ = hb.candidate_pairs()[0]
    pos[3 * j + 2] = pos[3 * i + 2] + np.array([5.0, 1.0, 1.0])
    active = hb.interactions(pos, box)
    assert len(active) >= 1, "no active HB interaction in the test config"

    U, F = hb.compute_energy_forces(pos, box)
    assert np.isfinite(U) and U < 0.0
    FD = _fd_forces(hb, pos, box)

    fmax = np.abs(F).max()
    assert fmax > 1e-3, "test config is degenerate (forces ~ 0)"
    tol = 1e-4 * (1.0 + fmax)
    assert np.abs(F - FD).max() < tol, (
        f"hbond force != FD: maxerr={np.abs(F - FD).max():.2e} tol={tol:.2e}")

    assert np.abs(F.sum(axis=0)).max() < 1e-8


def test_hbond_multiple_active_pairs_fd():
    """FD still holds when several complementary pairs are simultaneously active
    (many-body superposition + shared-particle force accumulation)."""
    seq = "GCAUGC"
    pos0, pos, box = _nontrivial_positions(seq, seed=3)
    hb = cf.TISHydrogenBonding(sequence=seq)
    # cluster all candidate base sites so every pair is within the gate
    center = pos[2].copy()
    rng = np.random.default_rng(9)
    for (i, j, _) in hb.candidate_pairs():
        pos[3 * i + 2] = center + rng.uniform(-2.0, 2.0, 3)
        pos[3 * j + 2] = center + rng.uniform(-2.0, 2.0, 3)
    n_active = len(hb.interactions(pos, box))
    assert n_active >= 2, f"wanted >=2 active pairs, got {n_active}"

    U, F = hb.compute_energy_forces(pos, box)
    FD = _fd_forces(hb, pos, box)
    fmax = np.abs(F).max()
    tol = 1e-4 * (1.0 + fmax)
    assert np.abs(F - FD).max() < tol
    assert np.abs(F.sum(axis=0)).max() < 1e-8


# --------------------------------------------------------------------------- #
# HOOMD wiring: attached force energy == numpy evaluator energy               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("which", ["stacking", "hbond"])
def test_hoomd_energy_matches_numpy(which):
    seq = "GCAUGC"
    snap = io.build_strand_snapshot(seq)
    box = np.array(snap.configuration.box[:3], dtype=float)
    pos = np.array(snap.particles.position, dtype=float)

    if which == "stacking":
        force = forces.stacking_force(snap)
    else:
        force = forces.hydrogen_bonding_force(snap)

    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = md.Integrator(
        dt=0.0, forces=[force],
        methods=[md.methods.Langevin(filter=hoomd.filter.All(), kT=KT_300)])
    sim.run(0)

    u_numpy = force.energy_of(pos, box)
    # HOOMD stores positions / energies in single precision in this build, so
    # match only to float32 precision (the numpy evaluator is the reference).
    assert np.isclose(force.energy, u_numpy, rtol=1e-5, atol=1e-4), (
        f"HOOMD energy {force.energy} != numpy {u_numpy}")


# --------------------------------------------------------------------------- #
# full 6-term model stability                                                 #
# --------------------------------------------------------------------------- #
def test_full_six_term_model_stable():
    seq = "GGGGAAAACCCC"          # 12-mer
    snap = io.build_strand_snapshot(seq)
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=7)
    sim.create_state_from_snapshot(snap)

    force_list, _ = forces.build_forces(snap, temperature=300.0,
                                        ionic_strength_M=0.15,
                                        include_custom=True)
    kinds = [type(f).__name__ for f in force_list]
    assert kinds == ["Harmonic", "Harmonic", "LJ", "Yukawa",
                     "TISStacking", "TISHydrogenBonding"]

    sim.operations.integrator = md.Integrator(
        dt=0.005, forces=force_list,
        methods=[md.methods.Langevin(filter=hoomd.filter.All(), kT=KT_300,
                                     default_gamma=1.0)])
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=KT_300)

    sim.run(0)
    pe0 = sum(f.energy for f in force_list)
    assert np.isfinite(pe0)

    for _ in range(5):
        sim.run(200)
        pos = np.array(sim.state.get_snapshot().particles.position)
        assert np.all(np.isfinite(pos)), "NaN/Inf in positions"
        pe = sum(f.energy for f in force_list)
        assert np.isfinite(pe), "non-finite potential energy"

    final_pe = sum(f.energy for f in force_list)
    assert np.isfinite(final_pe)
    assert final_pe / snap.particles.N < 50.0, "runaway energy per particle"
