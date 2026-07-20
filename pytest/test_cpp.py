"""Validation of the compiled (C++) TIS many-body backend against the Python
reference.

The pure-Python analytic forces are themselves finite-difference verified in
``test_custom_fd.py``; here we check that the compiled ``TISManyBodyForceCompute``
reproduces them:

  1. C++ ``getEnergy()`` (double) matches the Python ``compute_energy_forces()[0]``
     for BOTH stacking and hydrogen bonding to < 1e-6 (both double-accumulated).
  2. The attached HOOMD ``Force.forces`` (single precision / ForceReal on this
     build) match the Python analytic forces to < 1e-3 relative.
  3. The full 6-term model with ``backend="cpp"`` runs ~1000 Langevin steps on
     the hairpin GGGGAAAACCCC and stays stable.
  4. A steps/s benchmark, python vs cpp backend, on the 12-mer and a 76-mer.

Run:  PYTHONPATH=<repo-root> pytest pytest/test_cpp.py -v -s
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hoomd
import hoomd.md as md

from tis import io, forces, custom_forces as cf, cpp_forces as cff

KT_300 = forces.KB * 300.0

pytestmark = pytest.mark.skipif(
    not cff.available(),
    reason="compiled tis._engine not built/importable")


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _attach(force, snap):
    """Create a CPU sim for ``snap`` with ``force`` attached, run(0), return sim."""
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_snapshot(snap)
    sim.operations.integrator = md.Integrator(
        dt=0.0, forces=[force],
        methods=[md.methods.Langevin(filter=hoomd.filter.All(), kT=KT_300)])
    sim.run(0)
    return sim


def _tag_ordered(sim):
    """(positions, box) tag-ordered, as the Python evaluator expects them."""
    snap = sim.state.get_snapshot()
    pos = np.array(snap.particles.position, dtype=float)
    box = np.array(snap.configuration.box[:3], dtype=float)
    return pos, box


def _forces_tag_ordered(sim, force):
    """The C++ per-particle forces, reordered from local to tag order."""
    f_local = np.array(force.forces, dtype=float)
    with sim.state.cpu_local_snapshot as s:
        tag = np.array(s.particles.tag)
    f_tag = np.empty_like(f_local)
    f_tag[tag] = f_local
    return f_tag


def _perturbed_snapshot(seq, seed, amp=1.5, cluster_hb=False):
    """A non-degenerate config kept safely inside the (large) box."""
    snap = io.build_strand_snapshot(seq)
    box = np.array(snap.configuration.box[:3], dtype=float)
    pos = np.array(snap.particles.position, dtype=float)
    rng = np.random.default_rng(seed)
    pos = pos + rng.uniform(-amp, amp, size=pos.shape)
    if cluster_hb:
        # pull the two bases of the first complementary candidate into contact
        hb = cf.TISHydrogenBonding(sequence=seq)
        i, j, _ = hb.candidate_pairs()[0]
        pos[3 * j + 2] = pos[3 * i + 2] + np.array([5.0, 1.0, 1.0])
    snap.particles.position[:] = pos
    return snap


# --------------------------------------------------------------------------- #
# 1 + 2. energy (double, tight) and force (float) agreement                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("which", ["stacking", "hbond"])
def test_cpp_matches_python_energy_and_forces(which):
    seq = "GCAUGCAU"
    if which == "stacking":
        snap = _perturbed_snapshot(seq, seed=11)
        force = cff.stacking_force(snap)
    else:
        snap = _perturbed_snapshot(seq, seed=12, cluster_hb=True)
        force = cff.hydrogen_bonding_force(snap)

    sim = _attach(force, snap)
    pos, box = _tag_ordered(sim)

    # Python reference (double) on the SAME positions HOOMD used.
    U_py, F_py = force._builder.compute_energy_forces(pos, box)
    assert np.isfinite(U_py)
    if which == "hbond":
        assert force._cpp_obj.getNumInteractions() >= 1
        # at least one gated interaction must actually be active
        assert U_py < 0.0, "no active HB interaction in the test config"

    # 1. energy: both double-accumulated -> tight
    U_cpp = force.energy
    assert abs(U_cpp - U_py) < 1e-6, (
        f"{which}: C++ energy {U_cpp!r} != Python {U_py!r} "
        f"(diff {abs(U_cpp - U_py):.3e})")

    # 2. forces: C++ writes single precision -> relative tol 1e-3
    F_cpp = _forces_tag_ordered(sim, force)
    fmax = np.abs(F_py).max()
    assert fmax > 1e-3, "degenerate test config (forces ~ 0)"
    rel = np.abs(F_cpp - F_py).max() / (1.0 + fmax)
    assert rel < 1e-3, (
        f"{which}: C++ forces differ from Python by rel {rel:.3e} "
        f"(maxerr {np.abs(F_cpp - F_py).max():.3e}, fmax {fmax:.3e})")


def test_cpp_hbond_multiple_active_pairs():
    """Energy/force agreement with several HB pairs simultaneously active."""
    seq = "GCAUGC"
    snap = io.build_strand_snapshot(seq)
    box = np.array(snap.configuration.box[:3], dtype=float)
    pos = np.array(snap.particles.position, dtype=float)
    hb = cf.TISHydrogenBonding(sequence=seq)
    center = pos[2].copy()
    rng = np.random.default_rng(9)
    for (i, j, _) in hb.candidate_pairs():
        pos[3 * i + 2] = center + rng.uniform(-2.0, 2.0, 3)
        pos[3 * j + 2] = center + rng.uniform(-2.0, 2.0, 3)
    snap.particles.position[:] = pos

    force = cff.hydrogen_bonding_force(snap)
    sim = _attach(force, snap)
    pos, box = _tag_ordered(sim)
    U_py, F_py = force._builder.compute_energy_forces(pos, box)
    n_active = len(force._builder.interactions(pos, box))
    assert n_active >= 2, f"wanted >=2 active pairs, got {n_active}"

    assert abs(force.energy - U_py) < 1e-6
    F_cpp = _forces_tag_ordered(sim, force)
    rel = np.abs(F_cpp - F_py).max() / (1.0 + np.abs(F_py).max())
    assert rel < 1e-3


# --------------------------------------------------------------------------- #
# 3. full 6-term model with backend="cpp" is stable                           #
# --------------------------------------------------------------------------- #
def test_full_six_term_cpp_stable():
    seq = "GGGGAAAACCCC"
    snap = io.build_strand_snapshot(seq)
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=7)
    sim.create_state_from_snapshot(snap)

    force_list, _ = forces.build_forces(snap, temperature=300.0,
                                        ionic_strength_M=0.15,
                                        include_custom=True, backend="cpp")
    kinds = [type(f).__name__ for f in force_list]
    assert kinds == ["Harmonic", "Harmonic", "LJ", "Yukawa",
                     "TISManyBody", "TISManyBody"], kinds

    sim.operations.integrator = md.Integrator(
        dt=0.005, forces=force_list,
        methods=[md.methods.Langevin(filter=hoomd.filter.All(), kT=KT_300,
                                     default_gamma=1.0)])
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=KT_300)

    sim.run(0)
    pe0 = sum(f.energy for f in force_list)
    assert np.isfinite(pe0)

    for _ in range(5):                               # 1000 Langevin steps @ 300 K
        sim.run(200)
        pos = np.array(sim.state.get_snapshot().particles.position)
        assert np.all(np.isfinite(pos)), "NaN/Inf in positions"
        pe = sum(f.energy for f in force_list)
        assert np.isfinite(pe), "non-finite potential energy"

    final_pe = sum(f.energy for f in force_list)
    assert np.isfinite(final_pe)
    assert final_pe / snap.particles.N < 50.0, "runaway energy per particle"


# --------------------------------------------------------------------------- #
# 4. speed benchmark: python vs cpp backend                                   #
# --------------------------------------------------------------------------- #
def _recenter(snap):
    """Center a snapshot on the midpoint of its extent (io centers on the mean,
    which for a long drifting zig-zag can push atoms outside the span-sized box)."""
    pos = np.array(snap.particles.position, dtype=float)
    snap.particles.position[:] = pos - 0.5 * (pos.max(0) + pos.min(0))
    return snap


def _bench(seq, backend, n_steps=2000, warmup=100):
    snap = _recenter(io.build_strand_snapshot(seq))
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=3)
    sim.create_state_from_snapshot(snap)
    force_list, _ = forces.build_forces(snap, temperature=300.0,
                                        ionic_strength_M=0.15,
                                        include_custom=True, backend=backend)
    sim.operations.integrator = md.Integrator(
        dt=0.005, forces=force_list,
        methods=[md.methods.Langevin(filter=hoomd.filter.All(), kT=KT_300,
                                     default_gamma=1.0)])
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=KT_300)
    sim.run(warmup)                                  # JIT / neighbour-list warmup
    t0 = time.perf_counter()
    sim.run(n_steps)
    dt = time.perf_counter() - t0
    return n_steps / dt


@pytest.mark.parametrize("label,seq", [
    ("12-mer", "GGGGAAAACCCC"),
    ("76-mer", ("GGGGAAAACCCC" * 7)[:76]),
])
def test_speed_benchmark(label, seq):
    sps_py = _bench(seq, "python")
    sps_cpp = _bench(seq, "cpp")
    speedup = sps_cpp / sps_py
    print(f"\n[bench {label}] python={sps_py:8.1f} steps/s   "
          f"cpp={sps_cpp:8.1f} steps/s   speedup={speedup:5.1f}x")
    # the whole point: the compiled backend must be substantially faster
    assert sps_cpp > sps_py
