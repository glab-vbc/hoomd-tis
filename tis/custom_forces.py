"""TIS many-body custom forces: single-stranded stacking and hydrogen bonding.

These are the two TIS energy terms that do NOT map onto a stock HOOMD force
(MODEL.md sections 3 and 4). Both share the same rational "many-body" form

    U = U0 / ( 1 + Sum_a k_a (q_a - q_a0)^2 )                         [Eq. 9 / 13]

where each geometric coordinate q_a is a distance, bond angle, or dihedral among
a handful of point particles (the P/S/B sites of one or two nucleotides). The
directional / chirality information that a rigid-body model (oxDNA) would carry
in orientations is instead encoded here in the angle/dihedral coordinates, so
the model still integrates as ordinary translational MD.

This module implements them as pure-Python `hoomd.md.force.Custom` forces with
**analytic** forces (verified against finite difference in
`pytest/test_custom_fd.py`). A generic engine evaluates the energy and the exact
Cartesian gradient of any interaction of the above form; the two force classes
just supply the list of interactions (their coordinate topology, force
constants, reference values and well depth U0).

=============================================================================
Coordinate topology  (exact paper definitions -- Denesyuk-Thirumalai 2013)
=============================================================================
The site tuples entering each dihedral/angle are now the paper's exact
definitions (MODEL.md, Figs 2 & 4):

  Stacking (consecutive nucleotides i, i+1 along a strand):
    l    = dist(B_i, B_{i+1})
    phi1 = dihedral(P_i,     S_i,     P_{i+1}, S_{i+1})     [Fig 2, phi1]
    phi2 = dihedral(P_{i+2}, S_{i+1}, P_{i+1}, S_i)         [Fig 2, phi2]
  phi1 needs P_i (always present -- io.py includes a 5'-terminal P_0). phi2 needs
  P_{i+2}; at the 3'-terminal step (i = n_nucleotides-2) that phosphate does not
  exist, so the phi2 coordinate is OMITTED for that step (2 coords instead of 3).

  Hydrogen bonding (complementary bases i, j):
    d   = dist(B_i, B_j)
    th1 = angle(S_i, B_i, B_j)
    th2 = angle(S_j, B_j, B_i)
    ps1 = dihedral(S_i, B_i, B_j, S_j)
    ps2 = dihedral(P_i, S_i, B_i, B_j)
    ps3 = dihedral(P_j, S_j, B_j, B_i)

=============================================================================
Parameters
=============================================================================
Force constants, per-dimer stacking well depths, per-pair H-bond depths and the
A-form reference geometry are the FINAL RNA values from params.py:

  * Stacking:  k_l = STACK_RNA_KR, k_phi = STACK_RNA_KPHI; per-dimer well depth
    U0 = params.stacking_U0(XY, T); references l0 = STACK_R0[XY],
    phi1_0 = STACK_PHI1_0, phi2_0 = STACK_PHI2_0 (used when a ``sequence`` is
    given). Without a sequence the class falls back to a scalar placeholder U0
    and references measured from ``ref_positions`` / set via ``set_references``
    (this path is exercised by the finite-difference test).
  * Hydrogen bonding:  k_d = HB_RNA_KR, k_theta = HB_RNA_KTHETA,
    k_psi = HB_RNA_KPSI; per-pair d0 = HB_R0[sorted pair]; distinct references
    HB_TH1_0/HB_TH2_0/HB_PSI_0/HB_PSI1_0/HB_PSI2_0; well depth
    U0 = HB_RNA_U0 * HB_MULT[(base_i, base_j)] (x2 A-U, x3 G-C, x2 G-U wobble).

STACK_U0_PLACEHOLDER remains only as the no-sequence fallback well depth.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import hoomd
import hoomd.md as md

from . import params

# --------------------------------------------------------------------------- #
# Modelling inputs / no-sequence fallback                                     #
# --------------------------------------------------------------------------- #
# Fallback stacking well depth used ONLY when no sequence is supplied (the
# finite-difference test); with a sequence the per-dimer params.stacking_U0 wins.
STACK_U0_PLACEHOLDER = -4.0          # kcal/mol

# Distance below which a complementary pair is considered "in contact" and the
# HB term is switched on. A hard cutoff (truncation) -- a production term would
# use a smooth switch; kept hard here and only used away from the cutoff.
HB_CONTACT_CUTOFF = 12.0            # A

# Minimum sequence separation for an intra-strand H-bond (paper: >= 3).
HB_MIN_SEQ_SEP = 3


# --------------------------------------------------------------------------- #
# Geometry: minimum-image + coordinate value & Cartesian gradient             #
# --------------------------------------------------------------------------- #
def minimum_image(disp: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    """Minimum-image a displacement vector for an orthorhombic box.

    ``disp`` may be a single (3,) vector or an (n, 3) array. Assumes no box tilt
    (the tis.io boxes are cubic / orthorhombic); tilt factors are ignored.
    """
    L = np.asarray(box_lengths, dtype=float)
    return disp - L * np.round(disp / L)


def _dist_value_grad(p_i, p_j, box):
    """q = |B_i - B_j|; returns (q, [dq/dp_i, dq/dp_j])."""
    d = minimum_image(p_i - p_j, box)
    q = math.sqrt(float(d @ d))
    u = d / q
    return q, [u, -u]


def _angle_value_grad(p0, p1, p2, box):
    """Bond angle at the MIDDLE particle p1, between (p0-p1) and (p2-p1).

    Returns (theta, [dtheta/dp0, dtheta/dp1, dtheta/dp2]).
    """
    v1 = minimum_image(p0 - p1, box)
    v2 = minimum_image(p2 - p1, box)
    r1 = math.sqrt(float(v1 @ v1))
    r2 = math.sqrt(float(v2 @ v2))
    u1 = v1 / r1
    u2 = v2 / r2
    c = float(np.clip(u1 @ u2, -1.0, 1.0))
    s = math.sqrt(max(1.0 - c * c, 1e-12))          # guard near 0 / 180 deg
    theta = math.acos(c)
    g0 = -(u2 - c * u1) / (r1 * s)
    g2 = -(u1 - c * u2) / (r2 * s)
    g1 = -(g0 + g2)                                  # translational invariance
    return theta, [g0, g1, g2]


def _dihedral_value_grad(p0, p1, p2, p3, box):
    """Dihedral phi about the p1-p2 axis (Blondel-Karplus gradient).

    phi = atan2( (n1 x b2hat) . n2 , n1 . n2 ), with
        b1 = p1-p0, b2 = p2-p1, b3 = p3-p2,  n1 = b1 x b2, n2 = b2 x b3.
    Returns (phi, [dphi/dp0, dphi/dp1, dphi/dp2, dphi/dp3]).

    The four gradients sum to zero (translational invariance). The middle-atom
    coefficients were verified element-wise against finite difference (the
    error-prone part of the whole module): the correct form is
        g1 = -(1+a) g0 + c g3 ,   g2 = a g0 - (1+c) g3
    with a = (b1.b2)/|b2|^2 and c = (b3.b2)/|b2|^2.
    """
    b1 = minimum_image(p1 - p0, box)
    b2 = minimum_image(p2 - p1, box)
    b3 = minimum_image(p3 - p2, box)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = math.sqrt(float(b2 @ b2))
    n1sq = float(n1 @ n1)
    n2sq = float(n2 @ n2)
    m1 = np.cross(n1, b2 / b2n)
    phi = math.atan2(float(m1 @ n2), float(n1 @ n2))

    g0 = (b2n / n1sq) * n1
    g3 = -(b2n / n2sq) * n2
    a = float(b1 @ b2) / (b2n * b2n)
    c = float(b3 @ b2) / (b2n * b2n)
    g1 = -(1.0 + a) * g0 + c * g3
    g2 = a * g0 - (1.0 + c) * g3
    return phi, [g0, g1, g2, g3]


_COORD_FUNC = {
    "dist": _dist_value_grad,
    "angle": _angle_value_grad,
    "dihedral": _dihedral_value_grad,
}
_ANGULAR = {"angle", "dihedral"}


def _wrap(delta: float) -> float:
    """Wrap an angular difference into (-pi, pi] (identity derivative)."""
    return math.atan2(math.sin(delta), math.cos(delta))


# --------------------------------------------------------------------------- #
# Generic many-body engine                                                    #
# --------------------------------------------------------------------------- #
# An "interaction" is (U0, coords) where coords is a list of coordinate specs
#   (kind, indices, k, q0)
# with kind in {"dist","angle","dihedral"}, indices the GLOBAL particle indices
# (tags) participating, k the force constant, q0 the reference value.
Coord = Tuple[str, Tuple[int, ...], float, float]
Interaction = Tuple[float, List[Coord]]


def evaluate_interaction(positions: np.ndarray, box: np.ndarray,
                         interaction: Interaction):
    """Energy and analytic force of ONE interaction of the rational form.

    Returns (U, contributions) where contributions is a list of
    (global_index, force_vector) pairs (a particle may appear more than once;
    the caller accumulates). ``positions`` is indexed by global index/tag.
    """
    U0, coords = interaction

    # First pass: coordinate values, gradients, deviations -> denominator D.
    D = 1.0
    per_coord = []
    for kind, idx, k, q0 in coords:
        pts = [positions[i] for i in idx]
        q, grads = _COORD_FUNC[kind](*pts, box)
        dq = _wrap(q - q0) if kind in _ANGULAR else (q - q0)
        D += k * dq * dq
        per_coord.append((idx, k, dq, grads))

    U = U0 / D
    # dU/dq_a = -(U/D) * 2 k_a dq_a ; force on x = -dU/dx = -(dU/dq_a)(dq_a/dx)
    pref = U / D
    contributions: List[Tuple[int, np.ndarray]] = []
    for idx, k, dq, grads in per_coord:
        dU_dq = -pref * 2.0 * k * dq
        for i, g in zip(idx, grads):
            contributions.append((i, -dU_dq * g))
    return U, contributions


def evaluate_interactions(positions: np.ndarray, box: np.ndarray,
                          interactions: Sequence[Interaction]):
    """Total energy and net force array (N, 3) for a list of interactions."""
    positions = np.asarray(positions, dtype=float)
    box = np.asarray(box, dtype=float)
    forces = np.zeros_like(positions)
    U_total = 0.0
    for inter in interactions:
        U, contribs = evaluate_interaction(positions, box, inter)
        U_total += U
        for i, f in contribs:
            forces[i] += f
    return U_total, forces


# --------------------------------------------------------------------------- #
# Base class: read positions via rtag, scatter analytic force + energy         #
# --------------------------------------------------------------------------- #
class _ManyBodyCustomForce(md.force.Custom):
    """Shared HOOMD plumbing for the two rational many-body terms.

    Subclasses implement ``interactions(positions, box)`` returning the list of
    interactions to evaluate for the current configuration (static for stacking,
    distance-gated for hydrogen bonding).

    HOOMD reorders particles internally, so local-array indices are NOT tags:
    we map tag -> local index through ``rtag`` on every step (both to read
    positions and to scatter forces).
    """

    def __init__(self, n_particles: int):
        super().__init__(aniso=False)
        self._n = int(n_particles)

    def interactions(self, positions: np.ndarray, box: np.ndarray
                     ) -> List[Interaction]:
        raise NotImplementedError

    # -- direct numpy evaluation (used by the FD test; HOOMD-independent) --
    def compute_energy_forces(self, positions: np.ndarray, box: np.ndarray):
        """(U_total, force array (N,3)) for a tag-indexed ``positions`` array."""
        positions = np.asarray(positions, dtype=float)
        box = np.asarray(box, dtype=float)
        return evaluate_interactions(positions, box,
                                     self.interactions(positions, box))

    def energy_of(self, positions: np.ndarray, box: np.ndarray) -> float:
        return self.compute_energy_forces(positions, box)[0]

    # -- HOOMD callback --
    def set_forces(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            rtag = np.asarray(snap.particles.rtag)
            local_pos = np.asarray(snap.particles.position)
            # tag-indexed positions
            pos = np.empty((self._n, 3), dtype=float)
            pos[:] = local_pos[rtag[: self._n]]

        b = self._state.box
        box = np.array([b.Lx, b.Ly, b.Lz], dtype=float)

        inters = self.interactions(pos, box)
        forces = np.zeros((self._n, 3), dtype=float)
        pe = np.zeros(self._n, dtype=float)
        for U0, coords in inters:
            U, contribs = evaluate_interaction(pos, box, (U0, coords))
            # distribute the interaction energy over its distinct particles so
            # that Force.energy (sum of potential_energy) equals the total U.
            involved = sorted({i for _, idx, _, _ in coords for i in idx})
            share = U / len(involved)
            for i in involved:
                pe[i] += share
            for i, f in contribs:
                forces[i] += f

        with self.cpu_local_force_arrays as arr:
            with self._state.cpu_local_snapshot as snap:
                rtag = np.asarray(snap.particles.rtag)
            loc = rtag[: self._n]
            fa = np.asarray(arr.force)
            ea = np.asarray(arr.potential_energy)
            fa[loc, :3] = forces
            ea[loc] = pe


# --------------------------------------------------------------------------- #
# 3. Single-stranded stacking  U_S  (MODEL.md sec. 3, Eq. 9)                    #
# --------------------------------------------------------------------------- #
class TISStacking(_ManyBodyCustomForce):
    """Single-stranded stacking between consecutive bases (BONDED-style).

    One interaction per consecutive nucleotide pair (i, i+1) along each strand,
    with the paper's exact coordinates (MODEL.md, Fig 2):
        l    = dist(B_i, B_{i+1})                            k = k_l
        phi1 = dihedral(P_i,     S_i,     P_{i+1}, S_{i+1})  k = k_phi
        phi2 = dihedral(P_{i+2}, S_{i+1}, P_{i+1}, S_i)      k = k_phi

    phi2 needs P_{i+2}; at the 3'-terminal step it does not exist and is OMITTED
    (that interaction then has 2 coordinates, U0 / (1 + k_l(l-l0)^2 +
    k_phi(phi1-phi1_0)^2)).

    Parameters
    ----------
    n_nucleotides : int
        Number of nucleotides in the (single) strand; sites are P=3k, S=3k+1,
        B=3k+2.
    sequence : str, optional
        RNA sequence (A/G/C/U). When given, each pair's well depth and
        references are the FINAL params values: per 5'->3' dimer XY,
        U0 = params.stacking_U0(XY, temperature), l0 = params.STACK_R0[XY],
        phi1_0 = params.STACK_PHI1_0, phi2_0 = params.STACK_PHI2_0.
    temperature : float
        Temperature (K) for the sequence-dependent well depth. Default 300.
    ref_positions : array (N,3), optional
        No-sequence path only: geometry from which references l0/phi1_0/phi2_0
        are MEASURED (so U ~ U0 at t=0). Ignored when ``sequence`` is given.
    box : array (3,), optional
        Box lengths for measuring references (needed if ref_positions given).
    U0, k_l, k_phi : float
        No-sequence fallback well depth (STACK_U0_PLACEHOLDER) and force
        constants (default params.STACK_RNA_KR / STACK_RNA_KPHI).
    strands : list[range], optional
        Explicit per-strand nucleotide-index ranges (for multi-strand systems).
        Defaults to a single strand 0..n_nucleotides-1.
    """

    def __init__(self, n_nucleotides: int,
                 sequence: Optional[str] = None,
                 temperature: float = 300.0,
                 ref_positions: Optional[np.ndarray] = None,
                 box: Optional[np.ndarray] = None,
                 U0: float = STACK_U0_PLACEHOLDER,
                 k_l: float = params.STACK_RNA_KR,
                 k_phi: float = params.STACK_RNA_KPHI,
                 strands: Optional[Sequence[Sequence[int]]] = None):
        super().__init__(3 * n_nucleotides)
        self.n_nucleotides = int(n_nucleotides)
        self.temperature = float(temperature)
        self.k_l = float(k_l)
        self.k_phi = float(k_phi)

        seq = None
        if sequence is not None:
            seq = sequence.strip().upper().replace("T", "U")
            if len(seq) != self.n_nucleotides:
                raise ValueError("sequence length != n_nucleotides")

        if strands is None:
            strands = [range(self.n_nucleotides)]
        # consecutive pairs, each tagged with the i+2 nucleotide index used for
        # phi2 (or None at the 3'-terminal step, where P_{i+2} does not exist).
        self._pairs: List[Tuple[int, int]] = []
        self._kk: List[Optional[int]] = []
        for strand in strands:
            s = list(strand)
            for m in range(len(s) - 1):
                self._pairs.append((s[m], s[m + 1]))
                self._kk.append(s[m + 2] if (m + 2) < len(s) else None)

        # per-pair well depth and (l0, phi1_0, phi2_0) references. phi2_0 is None
        # for a terminal step. With a sequence -> FINAL params values; otherwise
        # -> scalar fallback U0 and measured / zero references.
        self._U0s: List[float] = []
        self._refs: List[Tuple[float, float, Optional[float]]] = []
        for (i, j), kk in zip(self._pairs, self._kk):
            if seq is not None:
                dimer = seq[i] + seq[j]
                self._U0s.append(params.stacking_U0(dimer, self.temperature))
                phi2_0 = None if kk is None else params.STACK_PHI2_0
                self._refs.append((params.STACK_R0[dimer],
                                   params.STACK_PHI1_0, phi2_0))
            else:
                self._U0s.append(float(U0))
                if ref_positions is not None:
                    if box is None:
                        raise ValueError("box required to measure references")
                    self._refs.append(self._measure(
                        np.asarray(ref_positions, float),
                        np.asarray(box, float), i, j, kk))
                else:
                    self._refs.append((0.0, 0.0, None if kk is None else 0.0))

    def _coords_for(self, i: int, j: int, kk: Optional[int]):
        """Site index tuples for pair (i, j=i+1); phi2 is None at a terminal step.

        l    = (B_i, B_j)
        phi1 = (P_i, S_i, P_j, S_j)
        phi2 = (P_{i+2}, S_j, P_j, S_i)   or None if kk is None
        """
        Pi, Si, Bi = 3 * i, 3 * i + 1, 3 * i + 2
        Pj, Sj, Bj = 3 * j, 3 * j + 1, 3 * j + 2
        l_idx = (Bi, Bj)
        phi1_idx = (Pi, Si, Pj, Sj)
        phi2_idx = None if kk is None else (3 * kk, Sj, Pj, Si)
        return l_idx, phi1_idx, phi2_idx

    def _measure(self, pos, box, i, j, kk):
        l_idx, phi1_idx, phi2_idx = self._coords_for(i, j, kk)
        l = _dist_value_grad(pos[l_idx[0]], pos[l_idx[1]], box)[0]
        phi1 = _dihedral_value_grad(*[pos[k] for k in phi1_idx], box)[0]
        phi2 = (None if phi2_idx is None
                else _dihedral_value_grad(*[pos[k] for k in phi2_idx], box)[0])
        return (l, phi1, phi2)

    def set_references(self, refs: Sequence[Sequence[float]]):
        """Override per-interaction references.

        Each entry is (l0, phi1_0) or (l0, phi1_0, phi2_0); the phi2_0 slot is
        ignored for a 3'-terminal step (which carries no phi2 coordinate).
        """
        if len(refs) != len(self._pairs):
            raise ValueError("need one (l0,phi1_0[,phi2_0]) per consecutive pair")
        new: List[Tuple[float, float, Optional[float]]] = []
        for r, kk in zip(refs, self._kk):
            r = list(r)
            l0, phi1_0 = float(r[0]), float(r[1])
            if kk is None:
                phi2_0 = None
            else:
                phi2_0 = float(r[2]) if len(r) > 2 and r[2] is not None else 0.0
            new.append((l0, phi1_0, phi2_0))
        self._refs = new

    def interactions(self, positions=None, box=None) -> List[Interaction]:
        inters: List[Interaction] = []
        for (i, j), kk, U0, (l0, phi1_0, phi2_0) in zip(
                self._pairs, self._kk, self._U0s, self._refs):
            l_idx, phi1_idx, phi2_idx = self._coords_for(i, j, kk)
            coords = [
                ("dist", l_idx, self.k_l, l0),
                ("dihedral", phi1_idx, self.k_phi, phi1_0),
            ]
            if phi2_idx is not None:
                coords.append(("dihedral", phi2_idx, self.k_phi, phi2_0))
            inters.append((U0, coords))
        return inters

    def gated_interactions(self) -> List[Tuple[float, List[Coord], object]]:
        """(U0, coords, gate) for every interaction. Stacking is never
        distance-gated so ``gate`` is always ``None``. This is the position-
        independent superset consumed by the compiled C++ engine (which
        re-evaluates gates, here trivially always-on, every step)."""
        return [(U0, coords, None) for (U0, coords) in self.interactions()]


# --------------------------------------------------------------------------- #
# 4. Hydrogen bonding  U_HB  (MODEL.md sec. 4, Eq. 13)                          #
# --------------------------------------------------------------------------- #
class TISHydrogenBonding(_ManyBodyCustomForce):
    """Hydrogen bonding between complementary bases (NONBONDED, distance-gated).

    Candidate pairs are complementary bases (A-U, G-C, G-U wobble) with sequence
    separation >= ``min_seq_sep``; an interaction is active only while the
    base-base distance is below ``contact_cutoff`` (a hard truncation -- a
    production term needs a smooth switch). Coordinates (MODEL.md, Fig 4):
        d   = dist(B_i, B_j)                          k = k_d
        th1 = angle(S_i, B_i, B_j)                    k = k_theta
        th2 = angle(S_j, B_j, B_i)                    k = k_theta
        ps1 = dihedral(S_i, B_i, B_j, S_j)            k = k_psi
        ps2 = dihedral(P_i, S_i, B_i, B_j)            k = k_psi
        ps3 = dihedral(P_j, S_j, B_j, B_i)            k = k_psi

    All parameters are the FINAL RNA values (params.py): per-pair
    U0 = params.HB_RNA_U0 * HB_MULT[(base_i, base_j)] (x2 A-U, x3 G-C, x2 G-U);
    per-pair d0 = params.HB_R0[sorted(base_i, base_j)]; distinct angle/dihedral
    references HB_TH1_0/HB_TH2_0/HB_PSI_0/HB_PSI1_0/HB_PSI2_0; force constants
    HB_RNA_KR/HB_RNA_KTHETA/HB_RNA_KPSI.

    Parameters
    ----------
    sequence : str
        RNA sequence (A/G/C/U) of the single strand, used to find complementary
        candidate pairs and their base identities (for d0 / multiplicity). For
        multi-strand, pass ``pairs`` explicitly (with ``sequence`` for bases).
    k_d, k_theta, k_psi : float
        Force constants (params.HB_RNA_KR / HB_RNA_KTHETA / HB_RNA_KPSI).
    u0_single : float
        Single-H-bond well depth (params.HB_RNA_U0); scaled by multiplicity.
    contact_cutoff : float
        Distance gate (A).
    pairs : list[(i,j)], optional
        Explicit candidate nucleotide-index pairs (overrides the sequence scan);
        base identities for d0 / multiplicity are read from ``sequence`` if given.
    pair_mult : dict[(i,j)->int], optional
        Multiplicity override for explicit ``pairs``.
    """

    def __init__(self, sequence: Optional[str] = None,
                 n_nucleotides: Optional[int] = None,
                 k_d: float = params.HB_RNA_KR,
                 k_theta: float = params.HB_RNA_KTHETA,
                 k_psi: float = params.HB_RNA_KPSI,
                 contact_cutoff: float = HB_CONTACT_CUTOFF,
                 min_seq_sep: int = HB_MIN_SEQ_SEP,
                 u0_single: float = params.HB_RNA_U0,
                 pairs: Optional[Sequence[Tuple[int, int]]] = None,
                 pair_mult: Optional[dict] = None):
        seq = None
        if sequence is not None:
            seq = sequence.strip().upper().replace("T", "U")
            n_nucleotides = len(seq)
        elif n_nucleotides is None:
            raise ValueError("provide either sequence or n_nucleotides")
        super().__init__(3 * int(n_nucleotides))
        self.n_nucleotides = int(n_nucleotides)
        self.k_d = float(k_d)
        self.k_theta = float(k_theta)
        self.k_psi = float(k_psi)
        self.contact_cutoff = float(contact_cutoff)
        self.u0_single = float(u0_single)

        # distinct A-form reference angles / dihedrals (shared by all pairs).
        self.th1_0 = params.HB_TH1_0
        self.th2_0 = params.HB_TH2_0
        self.ps1_0 = params.HB_PSI_0
        self.ps2_0 = params.HB_PSI1_0
        self.ps3_0 = params.HB_PSI2_0
        self._d0_default = sum(params.HB_R0.values()) / len(params.HB_R0)

        # candidate (i, j, U0, d0) tuples; d0 is per pair type.
        self._candidates: List[Tuple[int, int, float, float]] = []
        if pairs is not None:
            for (i, j) in pairs:
                mult, d0 = self._pair_params(seq, i, j, pair_mult)
                self._candidates.append((i, j, self.u0_single * mult, d0))
        else:
            for a in range(self.n_nucleotides):
                for b in range(a + min_seq_sep, self.n_nucleotides):
                    mult = params.HB_MULT.get((seq[a], seq[b]))
                    if mult is not None:
                        d0 = params.HB_R0["".join(sorted((seq[a], seq[b])))]
                        self._candidates.append(
                            (a, b, self.u0_single * mult, d0))

    def _pair_params(self, seq, i, j, pair_mult):
        """(multiplicity, d0) for an explicit pair, from the sequence if given."""
        if seq is not None:
            key = "".join(sorted((seq[i], seq[j])))
            mult = (pair_mult or {}).get(
                (i, j), params.HB_MULT.get((seq[i], seq[j]), 2))
            d0 = params.HB_R0.get(key, self._d0_default)
        else:
            mult = (pair_mult or {}).get((i, j), 2)
            d0 = self._d0_default
        return mult, d0

    @staticmethod
    def _coords_for(i: int, j: int):
        Pi, Si, Bi = 3 * i, 3 * i + 1, 3 * i + 2
        Pj, Sj, Bj = 3 * j, 3 * j + 1, 3 * j + 2
        return {
            "d": (Bi, Bj),
            "th1": (Si, Bi, Bj),
            "th2": (Sj, Bj, Bi),
            "ps1": (Si, Bi, Bj, Sj),
            "ps2": (Pi, Si, Bi, Bj),
            "ps3": (Pj, Sj, Bj, Bi),
        }

    def _coords_list(self, i: int, j: int, d0: float) -> List[Coord]:
        c = self._coords_for(i, j)
        return [
            ("dist", c["d"], self.k_d, d0),
            ("angle", c["th1"], self.k_theta, self.th1_0),
            ("angle", c["th2"], self.k_theta, self.th2_0),
            ("dihedral", c["ps1"], self.k_psi, self.ps1_0),
            ("dihedral", c["ps2"], self.k_psi, self.ps2_0),
            ("dihedral", c["ps3"], self.k_psi, self.ps3_0),
        ]

    def interactions(self, positions: np.ndarray, box: np.ndarray
                     ) -> List[Interaction]:
        positions = np.asarray(positions, dtype=float)
        box = np.asarray(box, dtype=float)
        inters: List[Interaction] = []
        for (i, j, U0, d0) in self._candidates:
            Bi, Bj = 3 * i + 2, 3 * j + 2
            d = minimum_image(positions[Bi] - positions[Bj], box)
            if float(d @ d) <= self.contact_cutoff * self.contact_cutoff:
                inters.append((U0, self._coords_list(i, j, d0)))
        return inters

    def candidate_pairs(self) -> List[Tuple[int, int, float]]:
        """Complementary candidate (i, j, U0) triples (for tests / inspection)."""
        return [(i, j, U0) for (i, j, U0, _d0) in self._candidates]

    def gated_interactions(self) -> List[Tuple[float, List[Coord], object]]:
        """(U0, coords, gate) for EVERY candidate pair (not just the currently
        active ones). ``gate = (B_i tag, B_j tag, contact_cutoff)``: the C++
        engine keeps the interaction only while |B_i - B_j| <= cutoff, exactly
        reproducing ``interactions()``'s per-step distance gate. This is the
        position-independent superset consumed by the compiled engine."""
        out: List[Tuple[float, List[Coord], object]] = []
        for (i, j, U0, d0) in self._candidates:
            Bi, Bj = 3 * i + 2, 3 * j + 2
            out.append((U0, self._coords_list(i, j, d0),
                        (Bi, Bj, self.contact_cutoff)))
        return out
