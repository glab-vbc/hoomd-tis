"""TIS topology / snapshot builder.

Lays down the Three-Interaction-Site (TIS) coarse-grained representation of a
nucleic-acid strand as **point particles** (no orientation / rigid bodies) in a
``hoomd.Snapshot``:

    each nucleotide i  ->  Phosphate P_i, Sugar S_i, Base B_i   (3 point masses)

Backbone connectivity (see ../MODEL.md and tis/params.py):

    5' ... P_i -- S_i -- P_{i+1} -- S_{i+1} ... 3'      (sugar-phosphate backbone)
                  |
                  B_i                                   (base hangs off the sugar)

Bonds
    "PS"  P_i -- S_i           (r0 = 3.74 A)   intra-nucleotide
    "SP"  S_i -- P_{i+1}       (r0 = 3.75 A)   to the next nucleotide (3' side)
    "S?"  S_i -- B_i           base bond, ? in {A,G,C,U}

Angles (only the ones with parameters in tis/params.ANGLE_DNA are emitted; the
rest are a documented TODO until the full angle table lands):
    "PSP"  P_i -- S_i -- P_{i+1}       (backbone, centred on the sugar)
    "SPS"  S_{i-1} -- P_i -- S_i       (backbone, centred on the phosphate)
    "PS?"  P_i -- S_i -- B_i           (base swing, only "PSA" available so far)

Particle types are ["P", "S", "A", "G", "C", "U"] (RNA). Masses are 1.0 for a
first pass (NOTE: the paper uses site-specific masses; equalising them changes
kinetics but not equilibrium thermodynamics -- fine for the native-term smoke
test). Units follow the paper: Angstrom / kcal/mol.

The initial geometry is a gentle planar zig-zag backbone whose vertex angles
approximate the equilibrium PSP / SPS angles, with every base pushed out of the
backbone plane by its equilibrium base-bond length. It is *not* A-form accurate
-- it only needs to be valid and non-overlapping (all non-bonded site pairs stay
well beyond the D0 = 3.2 A excluded-volume diameter) so that MD is stable.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import hoomd

from . import params

# Particle types: phosphate, sugar, then the four RNA bases.
PARTICLE_TYPES = ["P", "S", "A", "G", "C", "U"]
_PTYPEID = {t: i for i, t in enumerate(PARTICLE_TYPES)}
BASES = ("A", "G", "C", "U")

# Base-bond equilibrium lengths (A). RNA uracil (SU) borrows the DNA thymine (ST)
# value as a placeholder until RNA bonds are transcribed -- see params.BOND_DNA.
_BASE_BOND_R0 = {
    "A": params.BOND_DNA["SA"][1],
    "G": params.BOND_DNA["SG"][1],
    "C": params.BOND_DNA["SC"][1],
    "U": params.BOND_DNA["ST"][1],   # placeholder: thymine value for uracil
}

_R0_PS = params.BOND_DNA["PS"][1]    # 3.74 A  (P_i -- S_i)
_R0_SP = params.BOND_DNA["SP"][1]    # 3.75 A  (S_i -- P_{i+1})

# Backbone reference angles (rad) used only to shape the initial zig-zag.
_A_PSP = params.ANGLE_DNA["PSP"][1]  # ~123.30 deg  (centred on sugar)
_A_SPS = params.ANGLE_DNA["SPS"][1]  # ~ 94.60 deg  (centred on phosphate)


def _rot_y(v: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 3-vector about the y-axis (keeps the backbone planar in x-z)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


def _backbone_positions(n: int) -> np.ndarray:
    """Planar zig-zag positions for the 2n backbone beads in the order
    [P_0, S_0, P_1, S_1, ..., P_{n-1}, S_{n-1}].

    Bond lengths alternate PS / SP; the turn at each interior vertex is set so
    the vertex angle approximates its equilibrium value (SPS at phosphates,
    PSP at sugars). Alternating the turn sign gives a compact but non-clashing
    chain.
    """
    m = 2 * n
    pos = np.zeros((m, 3))
    direction = np.array([0.0, 0.0, 1.0])   # start walking along +z
    sign = 1.0
    for k in range(1, m):
        # bond leading into bead k: odd k -> P_i--S_i (PS); even k -> S--P (SP)
        length = _R0_PS if (k % 2 == 1) else _R0_SP
        pos[k] = pos[k - 1] + length * direction
        if k < m - 1:
            # interior vertex angle at bead k: even -> phosphate (SPS), odd -> sugar (PSP)
            interior = _A_SPS if (k % 2 == 0) else _A_PSP
            turn = math.pi - interior           # exterior turn angle
            direction = _rot_y(direction, sign * turn)
            sign = -sign
    return pos


def build_strand_snapshot(
    sequence: str,
    box: Optional[float] = None,
    padding: float = 40.0,
    min_box: float = 120.0,
    base_offset_axis: str = "y",
) -> hoomd.Snapshot:
    """Build a ``hoomd.Snapshot`` for a single TIS RNA strand.

    Parameters
    ----------
    sequence : str
        RNA sequence 5'->3' over the alphabet {A, G, C, U} (case-insensitive;
        'T' is accepted and read as 'U').
    box : float, optional
        Edge length (A) of the cubic simulation box. If ``None`` a generous box
        is chosen from the chain extent (``span + padding``, at least ``min_box``)
        -- large enough that the default excluded-volume / Debye-Huckel cutoffs
        fit without minimum-image artefacts.
    padding : float
        Extra room added around the chain when ``box`` is auto-sized (A).
    min_box : float
        Lower bound on the auto-sized box edge (A).
    base_offset_axis : {"y", "x"}
        Axis the bases are pushed along, out of the x-z backbone plane. "y"
        (default) guarantees no base-base or base-backbone overlap.

    Returns
    -------
    hoomd.Snapshot
        Populated with P/S/B particles (types, positions, masses), backbone
        bonds (PS/SP + base bonds) and the available backbone/base angles.

    Notes
    -----
    * A 5'-terminal phosphate P_0 is included so every nucleotide has all three
      sites; it simply carries no SPS angle.
    * Base *angles* are emitted only for base types whose "PS?" key exists in
      params.ANGLE_DNA (currently only "PSA"); base *bonds* are always emitted
      (uracil borrows the thymine "ST" parameters). See build notes in forces.py.
    * Single strand only. Double strand is a TODO (build a second antiparallel
      strand offset in x and concatenate particles/bonds/angles).
    """
    seq = sequence.strip().upper().replace("T", "U")
    if len(seq) == 0:
        raise ValueError("sequence is empty")
    bad = set(seq) - set(BASES)
    if bad:
        raise ValueError(f"sequence has non-RNA letters: {sorted(bad)}")
    n = len(seq)

    # --- particle positions ------------------------------------------------
    bb = _backbone_positions(n)                # (2n, 3): P_0,S_0,P_1,S_1,...
    n_part = 3 * n
    pos = np.zeros((n_part, 3))
    typeid = np.zeros(n_part, dtype=np.uint32)

    offset = {"x": np.array([1.0, 0.0, 0.0]),
              "y": np.array([0.0, 1.0, 0.0])}[base_offset_axis]

    for i, base in enumerate(seq):
        p_idx, s_idx, b_idx = 3 * i, 3 * i + 1, 3 * i + 2
        p_pos = bb[2 * i]
        s_pos = bb[2 * i + 1]
        pos[p_idx] = p_pos
        pos[s_idx] = s_pos
        pos[b_idx] = s_pos + _BASE_BOND_R0[base] * offset   # base out of plane
        typeid[p_idx] = _PTYPEID["P"]
        typeid[s_idx] = _PTYPEID["S"]
        typeid[b_idx] = _PTYPEID[base]

    # centre the molecule at the origin
    pos -= pos.mean(axis=0)

    # --- bonds -------------------------------------------------------------
    bond_groups: list[tuple[int, int]] = []
    bond_typenames: list[str] = []
    for i, base in enumerate(seq):
        p_idx, s_idx, b_idx = 3 * i, 3 * i + 1, 3 * i + 2
        bond_groups.append((p_idx, s_idx));  bond_typenames.append("PS")     # P_i--S_i
        bond_groups.append((s_idx, b_idx));  bond_typenames.append("S" + base)  # base
        if i < n - 1:
            p_next = 3 * (i + 1)
            bond_groups.append((s_idx, p_next)); bond_typenames.append("SP")  # S_i--P_{i+1}

    # --- angles (only those with parameters available) ---------------------
    available_angles = set(params.ANGLE_DNA.keys())   # {"PSP","SPS","PSA"} so far
    angle_groups: list[tuple[int, int, int]] = []
    angle_typenames: list[str] = []
    for i, base in enumerate(seq):
        p_idx, s_idx, b_idx = 3 * i, 3 * i + 1, 3 * i + 2
        # PSP centred on sugar S_i : P_i -- S_i -- P_{i+1}
        if i < n - 1 and "PSP" in available_angles:
            angle_groups.append((p_idx, s_idx, 3 * (i + 1)))
            angle_typenames.append("PSP")
        # SPS centred on phosphate P_i : S_{i-1} -- P_i -- S_i
        if i >= 1 and "SPS" in available_angles:
            angle_groups.append((3 * (i - 1) + 1, p_idx, s_idx))
            angle_typenames.append("SPS")
        # base swing P_i -- S_i -- B_i (only "PSA" is available so far)
        base_angle = "PS" + base
        if base_angle in available_angles:
            angle_groups.append((p_idx, s_idx, b_idx))
            angle_typenames.append(base_angle)

    # --- assemble snapshot -------------------------------------------------
    if box is None:
        span = float(np.ptp(pos, axis=0).max())
        box = max(span + padding, min_box)

    snap = hoomd.Snapshot()
    snap.configuration.box = [box, box, box, 0, 0, 0]

    snap.particles.N = n_part
    snap.particles.types = list(PARTICLE_TYPES)
    snap.particles.position[:] = pos
    snap.particles.typeid[:] = typeid
    snap.particles.mass[:] = 1.0

    # distinct bond types actually used, with stable ids
    bond_type_list = sorted(set(bond_typenames))
    bond_tid = {t: i for i, t in enumerate(bond_type_list)}
    snap.bonds.N = len(bond_groups)
    snap.bonds.types = bond_type_list
    if bond_groups:
        snap.bonds.group[:] = np.array(bond_groups, dtype=np.uint32)
        snap.bonds.typeid[:] = np.array([bond_tid[t] for t in bond_typenames],
                                        dtype=np.uint32)

    angle_type_list = sorted(set(angle_typenames))
    angle_tid = {t: i for i, t in enumerate(angle_type_list)}
    snap.angles.N = len(angle_groups)
    snap.angles.types = angle_type_list
    if angle_groups:
        snap.angles.group[:] = np.array(angle_groups, dtype=np.uint32)
        snap.angles.typeid[:] = np.array([angle_tid[t] for t in angle_typenames],
                                         dtype=np.uint32)

    return snap
