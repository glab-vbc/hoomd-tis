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
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hoomd
import hoomd.md as md

from tis import io, forces, params, custom_forces as cf

KT_300 = forces.KB * 300.0


def _place_nerf(A, B, C, L, theta, chi):
    """Place atom D given A, B, C so that |C-D| = L, angle(B,C,D) = theta and
    dihedral(A,B,C,D) = chi, matching the Blondel-Karplus sign convention used
    by custom_forces._dihedral_value_grad (verified in the sanity tests below by
    re-measuring the constructed coordinates)."""
    A, B, C = map(lambda v: np.asarray(v, float), (A, B, C))
    bc = C - B
    bc /= np.linalg.norm(bc)
    n = np.cross(B - A, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array([-L * math.cos(theta),
                   L * math.sin(theta) * math.cos(chi),
                   -L * math.sin(theta) * math.sin(chi)])
    return C + np.column_stack((bc, m, n)) @ d2


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
# physical-sanity tests: attractive where it should be, U == U0 at reference   #
# --------------------------------------------------------------------------- #
def test_stacking_reference_energy_is_well_depth():
    """A dimer placed AT its A-form stacking reference (l0, phi1_0, phi2_0) has
    D = 1, so the stacking energy equals the (negative) per-dimer well depth
    params.stacking_U0(XY, T). Guards against a sign / reference wiring error."""
    box = np.array([1000.0, 1000.0, 1000.0])
    dimer = "GC"
    l0 = params.STACK_R0[dimer]
    phi1_0, phi2_0 = params.STACK_PHI1_0, params.STACK_PHI2_0

    # 3 nucleotides: pair (0,1) is dimer GC and carries phi2 (needs P_2).
    pos = np.zeros((9, 3))
    Pi, Si, Bi = 0, 1, 2
    Pj, Sj, Bj = 3, 4, 5
    Pk = 6
    pos[Si] = [0.0, 0.0, 0.0]
    pos[Pj] = [3.9, 0.0, 0.0]
    pos[Pi] = [-2.0, 3.0, 0.5]
    pos[Sj] = _place_nerf(pos[Pi], pos[Si], pos[Pj], 4.0, math.radians(100), phi1_0)
    # phi2 = dihedral(P_{i+2}, S_j, P_j, S_i) == dihedral(S_i, P_j, S_j, P_{i+2})
    pos[Pk] = _place_nerf(pos[Si], pos[Pj], pos[Sj], 3.7, math.radians(100), phi2_0)
    pos[Bi] = [0.5, -2.0, 1.0]
    pos[Bj] = pos[Bi] + np.array([l0, 0.0, 0.0])
    pos[7] = pos[Pk] + np.array([1.0, 1.0, 1.0])   # S_2, B_2 (unused by pair 0)
    pos[8] = pos[Pk] + np.array([2.0, -1.0, 0.5])

    # self-check: the constructed geometry realizes the references.
    l = cf._dist_value_grad(pos[Bi], pos[Bj], box)[0]
    phi1 = cf._dihedral_value_grad(pos[Pi], pos[Si], pos[Pj], pos[Sj], box)[0]
    phi2 = cf._dihedral_value_grad(pos[Pk], pos[Sj], pos[Pj], pos[Si], box)[0]
    assert abs(l - l0) < 1e-6
    assert abs(cf._wrap(phi1 - phi1_0)) < 1e-6
    assert abs(cf._wrap(phi2 - phi2_0)) < 1e-6

    st = cf.TISStacking(3, sequence="GCA", temperature=300.0, strands=[[0, 1, 2]])
    inter0 = st.interactions(pos, box)[0]            # pair (0,1) = dimer GC
    U, _ = cf.evaluate_interaction(pos, box, inter0)

    U0 = params.stacking_U0(dimer, 300.0)
    assert U0 < 0.0, "stacking well depth must be attractive (negative)"
    assert abs(U - U0) < 1e-6, f"U at reference {U} != well depth {U0}"


def test_hbond_reference_energy_is_well_depth():
    """A complementary G-C pair placed AT the H-bond reference geometry has
    D = 1, so U == params.HB_RNA_U0 * multiplicity (negative, x3 for G-C). Also
    checks a nearby geometry stays attractive. Guards a sign / reference error."""
    box = np.array([1000.0, 1000.0, 1000.0])
    d0 = params.HB_R0["CG"]
    th1_0, th2_0 = params.HB_TH1_0, params.HB_TH2_0
    ps1_0, ps2_0, ps3_0 = params.HB_PSI_0, params.HB_PSI1_0, params.HB_PSI2_0

    # sequence GAAAC -> single complementary candidate (0,4) = G-C (sep 4 >= 3).
    P = np.zeros((15, 3))
    Pi, Si, Bi = 0, 1, 2            # nucleotide 0 (G)
    Pj, Sj, Bj = 12, 13, 14        # nucleotide 4 (C)
    P[Bi] = [0.0, 0.0, 0.0]
    P[Bj] = P[Bi] + np.array([d0, 0.0, 0.0])
    P[Si] = P[Bi] + 4.9 * np.array([math.cos(th1_0), math.sin(th1_0), 0.0])
    P[Sj] = _place_nerf(P[Si], P[Bi], P[Bj], 4.9, th2_0, ps1_0)
    # ps2 = dihedral(P_i,S_i,B_i,B_j) == dihedral(B_j,B_i,S_i,P_i)
    P[Pi] = _place_nerf(P[Bj], P[Bi], P[Si], 3.7, math.radians(100), ps2_0)
    # ps3 = dihedral(P_j,S_j,B_j,B_i) == dihedral(B_i,B_j,S_j,P_j)
    P[Pj] = _place_nerf(P[Bi], P[Bj], P[Sj], 3.7, math.radians(100), ps3_0)
    for k in (3, 4, 5, 6, 7, 8, 9, 10, 11):        # spectator nucleotides, far off
        P[k] = [200.0 + k, 3.0 * k, -2.0 * k]

    # self-check: geometry realizes the six references.
    assert abs(cf._dist_value_grad(P[Bi], P[Bj], box)[0] - d0) < 1e-6
    assert abs(cf._wrap(cf._angle_value_grad(P[Si], P[Bi], P[Bj], box)[0] - th1_0)) < 1e-6
    assert abs(cf._wrap(cf._angle_value_grad(P[Sj], P[Bj], P[Bi], box)[0] - th2_0)) < 1e-6
    assert abs(cf._wrap(cf._dihedral_value_grad(P[Si], P[Bi], P[Bj], P[Sj], box)[0] - ps1_0)) < 1e-6

    hb = cf.TISHydrogenBonding(sequence="GAAAC")
    assert hb.candidate_pairs() == [(0, 4, params.HB_RNA_U0 * params.HB_MULT[("G", "C")])]
    U, _ = hb.compute_energy_forces(P, box)

    U0 = params.HB_RNA_U0 * params.HB_MULT[("G", "C")]
    assert U0 < 0.0, "H-bond well depth must be attractive (negative)"
    assert abs(U - U0) < 1e-6, f"U at reference {U} != well depth {U0}"

    # a nudged geometry (still within the gate) must remain attractive.
    Pn = P.copy()
    Pn[Bj] = P[Bj] + np.array([0.4, -0.3, 0.2])
    Un, _ = hb.compute_energy_forces(Pn, box)
    assert U0 < Un < 0.0, "H-bond energy must stay negative near the reference"


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

    for _ in range(10):                              # ~2000 Langevin steps @ 300 K
        sim.run(200)
        pos = np.array(sim.state.get_snapshot().particles.position)
        assert np.all(np.isfinite(pos)), "NaN/Inf in positions"
        pe = sum(f.energy for f in force_list)
        assert np.isfinite(pe), "non-finite potential energy"

    final_pe = sum(f.energy for f in force_list)
    assert np.isfinite(final_pe)
    assert final_pe / snap.particles.N < 50.0, "runaway energy per particle"
