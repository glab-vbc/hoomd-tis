"""Smoke test for the TIS Python layer.

Builds a short RNA strand with tis.io, attaches the FOUR native TIS forces
(harmonic bonds, harmonic angles, WCA excluded volume, Debye-Huckel), runs a
couple thousand Langevin MD steps at 300 K on the CPU, and asserts the topology
integrates stably: positions stay finite, the radius of gyration stays bounded
(no explosion, no collapse), and the potential energy stays finite.

Only the four native terms are active -- stacking and hydrogen bonding are not
yet implemented (custom C++). This validates topology + native-term wiring.

Run:  PYTHONPATH=<repo-root> pytest pytest/test_smoke.py -v
   or just: pytest pytest/test_smoke.py   (sys.path is patched below)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hoomd
import hoomd.md as md

from tis import io, forces

KT_300 = forces.KB * 300.0  # ~0.596 kcal/mol


def _radius_of_gyration(positions: np.ndarray) -> float:
    com = positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1))))


def _make_sim(sequence: str, seed: int = 7):
    snap = io.build_strand_snapshot(sequence)
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=seed)
    sim.create_state_from_snapshot(snap)
    force_list, _ = forces.build_forces(snap, temperature=300.0,
                                        ionic_strength_M=0.15)
    sim.operations.integrator = md.Integrator(
        dt=0.005,
        forces=force_list,
        methods=[md.methods.Langevin(filter=hoomd.filter.All(), kT=KT_300,
                                     default_gamma=1.0)],
    )
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=KT_300)
    return sim, snap, force_list


@pytest.mark.parametrize("sequence", ["GGGGAAAACCCC", "AAAAAAAAAA"])
def test_smoke_stable_md(sequence):
    sim, snap, force_list = _make_sim(sequence)

    # initial state must be valid & non-clashing (WCA energy ~ 0)
    sim.run(0)
    pos0 = np.array(sim.state.get_snapshot().particles.position)
    assert np.all(np.isfinite(pos0))
    rg0 = _radius_of_gyration(pos0)
    wca = force_list[2]
    assert wca.energy < 1.0, f"initial excluded-volume clash: U_EV={wca.energy}"

    # integrate ~2000 steps, checking stability periodically
    n_chunks, chunk = 10, 200
    for _ in range(n_chunks):
        sim.run(chunk)
        pos = np.array(sim.state.get_snapshot().particles.position)
        assert np.all(np.isfinite(pos)), "NaN/Inf in positions"
        pe = sum(f.energy for f in force_list)
        assert np.isfinite(pe), "non-finite potential energy"
        rg = _radius_of_gyration(pos)
        # bounded: neither exploded nor collapsed onto a point
        assert 0.3 * rg0 < rg < 5.0 * rg0 + 20.0, (
            f"Rg blew up/collapsed: rg0={rg0:.2f} rg={rg:.2f}")

    # final sanity: energies finite and per-particle PE modest
    final_pe = sum(f.energy for f in force_list)
    assert np.isfinite(final_pe)
    assert final_pe / snap.particles.N < 50.0, "runaway energy per particle"


def test_forces_are_the_four_native_terms():
    _, _, force_list = _make_sim("GCGCGC")
    kinds = [type(f).__name__ for f in force_list]
    assert kinds == ["Harmonic", "Harmonic", "LJ", "Yukawa"]


def test_custom_terms_are_stubbed():
    with pytest.raises(NotImplementedError):
        forces.stacking_force()
    with pytest.raises(NotImplementedError):
        forces.hydrogen_bonding_force()
