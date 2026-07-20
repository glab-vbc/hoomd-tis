"""Compiled (C++) backend for the two TIS many-body terms.

The pure-Python evaluator in :mod:`tis.custom_forces` recomputes the rational
many-body energy and its analytic Cartesian gradient on the host every step,
which dominates the wall-clock time of a TIS run. This module wires the SAME
interactions onto the compiled ``TISManyBodyForceCompute`` (src/, built into
``tis._engine``), which evaluates them natively for a large speedup.

Crucially the coordinate topology and all parameters still come from the single
validated source, :class:`tis.custom_forces.TISStacking` /
:class:`~tis.custom_forces.TISHydrogenBonding`: those "builder" objects expose
``gated_interactions()`` (U0, coords, optional distance gate), which we flatten
into the nested-list form the C++ ``setInteractions`` accepts. Nothing about the
physics is duplicated here.

Requires the compiled engine; :func:`available` reports whether it imported.
"""
from __future__ import annotations

from typing import List, Tuple

from hoomd.md.force import Force

try:  # compiled engine is optional (native-only builds work without it)
    from . import _engine
    _ENGINE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the build
    _engine = None
    _ENGINE_IMPORT_ERROR = exc


def available() -> bool:
    """True if the compiled ``tis._engine`` imported successfully."""
    return _engine is not None


# kind string -> C++ enum int (TISCoordKind in src/TISForces.h)
_KIND = {"dist": 0, "angle": 1, "dihedral": 2}


def flatten_interactions(gated) -> List[tuple]:
    """Flatten ``gated_interactions()`` into the nested list ``setInteractions``
    consumes. Each interaction becomes

        (U0, has_gate, gate_i, gate_j, gate_cutoff, [ (kind_int, idx_tuple, k, q0), ... ])

    with tags as plain Python ints (they are global particle TAGS; the C++ side
    maps them to local indices via rtag every step).
    """
    out: List[tuple] = []
    for (U0, coords, gate) in gated:
        if gate is None:
            has_gate, gi, gj, gc = False, 0, 0, 0.0
        else:
            gi, gj, gc = gate
            has_gate = True
        cs = [(_KIND[kind], tuple(int(x) for x in idx), float(k), float(q0))
              for (kind, idx, k, q0) in coords]
        out.append((float(U0), bool(has_gate), int(gi), int(gj), float(gc), cs))
    return out


class TISManyBody(Force):
    """Compiled generic TIS many-body force (stacking OR hydrogen bonding).

    Wraps the C++ ``TISManyBodyForceCompute``. Construct from a *builder* that
    exposes ``gated_interactions()`` (a :class:`tis.custom_forces.TISStacking` or
    :class:`~tis.custom_forces.TISHydrogenBonding` object), so the entire
    coordinate/parameter definition stays in the validated Python reference.

    ``energy`` returns the C++ side's DOUBLE accumulator (tight), unlike the
    generic float ``calcEnergySum`` over the single-precision per-particle
    energies.
    """

    _ext_module = _engine
    _cpp_class_name = "TISManyBodyForceCompute"

    def __init__(self, builder):
        if _engine is None:  # pragma: no cover - depends on the build
            raise ImportError(
                "tis._engine (compiled backend) is not available: "
                f"{_ENGINE_IMPORT_ERROR!r}")
        super().__init__()
        self._builder = builder
        self._interactions = flatten_interactions(builder.gated_interactions())

    def _attach_hook(self):
        self._cpp_obj = self._ext_module.TISManyBodyForceCompute(
            self._simulation.state._cpp_sys_def)
        self._cpp_obj.setInteractions(self._interactions)

    @property
    def energy(self):
        """Total potential energy U (double accumulator, computed on read)."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getEnergy()


# --------------------------------------------------------------------------- #
# Factories mirroring tis.forces.stacking_force / hydrogen_bonding_force       #
# --------------------------------------------------------------------------- #
def stacking_force(snapshot, temperature: float = 300.0, **kwargs) -> TISManyBody:
    """C++-backed single-stranded stacking force (see tis.forces.stacking_force)."""
    from . import custom_forces as cf
    from .forces import _sequence_from_snapshot
    n = snapshot.particles.N // 3
    seq = _sequence_from_snapshot(snapshot)
    builder = cf.TISStacking(n, sequence=seq, temperature=temperature, **kwargs)
    return TISManyBody(builder)


def hydrogen_bonding_force(snapshot, **kwargs) -> TISManyBody:
    """C++-backed hydrogen-bonding force (see tis.forces.hydrogen_bonding_force)."""
    from . import custom_forces as cf
    from .forces import _sequence_from_snapshot
    seq = _sequence_from_snapshot(snapshot)
    builder = cf.TISHydrogenBonding(sequence=seq, **kwargs)
    return TISManyBody(builder)
