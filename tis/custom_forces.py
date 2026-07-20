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
PROVISIONAL geometry choices  (reconcile with the paper's exact definitions)
=============================================================================
The paper (Denesyuk-Thirumalai 2013 for RNA; Chakraborty-Hori-Thirumalai 2018
for DNA) fixes exactly which sites enter each dihedral/angle. Those RNA tables
are not yet in hand, so the coordinate choices below are our best reading of
MODEL.md and are marked PROVISIONAL. They are collected in one place so they are
trivial to swap:

  Stacking (consecutive bases i, i+1 along a strand):
    l    = dist(B_i, B_{i+1})
    phi1 = dihedral(S_i, P_{i+1}, S_{i+1}, B_{i+1})
    phi2 = dihedral(B_i, S_i, P_{i+1}, S_{i+1})

  Hydrogen bonding (complementary bases i, j):
    d   = dist(B_i, B_j)
    th1 = angle(S_i, B_i, B_j)
    th2 = angle(S_j, B_j, B_i)
    ps1 = dihedral(S_i, B_i, B_j, S_j)
    ps2 = dihedral(P_i, S_i, B_i, B_j)
    ps3 = dihedral(P_j, S_j, B_j, B_i)

=============================================================================
PLACEHOLDER parameters  (RNA tables are TODO -- see params.py)
=============================================================================
Force constants k_* are the real DNA/RNA values from params.py (STACK_KL,
STACK_KPHI, HB_KD, HB_KTHETA, HB_KPSI). Everything else below is a clearly
marked PLACEHOLDER until the RNA reference-geometry / well-depth tables arrive:

  * STACK_U0_PLACEHOLDER   nominal stacking well depth (-4 kcal/mol).
  * stacking references    l0, phi1_0, phi2_0 default to the values MEASURED
                           from the initial geometry passed at construction
                           (so U ~ U0 at t=0); override per call if desired.
  * HB_U0_SINGLE           -2.43 kcal/mol per H-bond (RNA), x2 A-U, x3 G-C,
                           x2 G-U (the G-U multiplicity is itself a placeholder).
  * HB_D0/TH0/PS0_*        nominal base-pair reference geometry.

None of these placeholders is trustworthy for thermodynamics; they exist only so
the two terms are wired, differentiable and MD-stable for testing. Folding is NOT
expected from them.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import hoomd
import hoomd.md as md

from . import params

# --------------------------------------------------------------------------- #
# PLACEHOLDER parameters (see module docstring)                               #
# --------------------------------------------------------------------------- #
STACK_U0_PLACEHOLDER = -4.0          # kcal/mol, nominal stacking well depth

HB_U0_SINGLE = -2.43                 # kcal/mol per single RNA H-bond
# multiplicity of the complementary pair; G-U (wobble) value is a PLACEHOLDER.
HB_MULTIPLICITY = {
    ("A", "U"): 2, ("U", "A"): 2,
    ("G", "C"): 3, ("C", "G"): 3,
    ("G", "U"): 2, ("U", "G"): 2,   # PLACEHOLDER: G-U wobble strength
}
# Nominal reference geometry for an "ideal" base pair (all PLACEHOLDER).
HB_D0_PLACEHOLDER = 5.8              # A   base-base distance at contact
HB_TH0_PLACEHOLDER = 2.60           # rad S-B-B angle (~149 deg)
HB_PS0_PLACEHOLDER = 0.0            # rad backbone dihedrals
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
    with coordinates (PROVISIONAL, see module docstring):
        l    = dist(B_i, B_{i+1})                            k = k_l
        phi1 = dihedral(S_i, P_{i+1}, S_{i+1}, B_{i+1})      k = k_phi
        phi2 = dihedral(B_i, S_i, P_{i+1}, S_{i+1})          k = k_phi

    Parameters
    ----------
    n_nucleotides : int
        Number of nucleotides in the (single) strand; sites are P=3k, S=3k+1,
        B=3k+2.
    ref_positions : array (N,3), optional
        Geometry from which the per-interaction references l0/phi1_0/phi2_0 are
        MEASURED (PLACEHOLDER). Defaults to ``None``; if not given and no
        explicit references are supplied, references are 0 (must be set later).
        Pass the initial snapshot positions so U ~ U0 at t=0.
    box : array (3,), optional
        Box lengths for measuring references (needed if ref_positions given).
    U0, k_l, k_phi : float
        Well depth (PLACEHOLDER) and force constants (params.STACK_KL/KPHI).
    strands : list[range], optional
        Explicit per-strand nucleotide-index ranges (for multi-strand systems).
        Defaults to a single strand 0..n_nucleotides-1.
    """

    def __init__(self, n_nucleotides: int,
                 ref_positions: Optional[np.ndarray] = None,
                 box: Optional[np.ndarray] = None,
                 U0: float = STACK_U0_PLACEHOLDER,
                 k_l: float = params.STACK_KL,
                 k_phi: float = params.STACK_KPHI,
                 strands: Optional[Sequence[Sequence[int]]] = None):
        super().__init__(3 * n_nucleotides)
        self.n_nucleotides = int(n_nucleotides)
        self.U0 = float(U0)
        self.k_l = float(k_l)
        self.k_phi = float(k_phi)
        if strands is None:
            strands = [range(self.n_nucleotides)]
        # build the (coord-topology) list of consecutive pairs
        self._pairs: List[Tuple[int, int]] = []
        for strand in strands:
            s = list(strand)
            for a, b in zip(s[:-1], s[1:]):
                self._pairs.append((a, b))

        # references (PLACEHOLDER): measure from ref_positions if provided.
        self._refs: List[Tuple[float, float, float]] = []
        for (i, j) in self._pairs:
            if ref_positions is not None:
                if box is None:
                    raise ValueError("box required to measure references")
                self._refs.append(self._measure(np.asarray(ref_positions,
                                                            float),
                                                 np.asarray(box, float), i, j))
            else:
                self._refs.append((0.0, 0.0, 0.0))

    @staticmethod
    def _coords_for(i: int, j: int):
        """Site indices (Bi,Bj / Si,Pj,Sj,Bj / Bi,Si,Pj,Sj) for pair (i,j=i+1)."""
        Pi, Si, Bi = 3 * i, 3 * i + 1, 3 * i + 2
        Pj, Sj, Bj = 3 * j, 3 * j + 1, 3 * j + 2
        l_idx = (Bi, Bj)
        phi1_idx = (Si, Pj, Sj, Bj)
        phi2_idx = (Bi, Si, Pj, Sj)
        return l_idx, phi1_idx, phi2_idx

    def _measure(self, pos, box, i, j):
        l_idx, phi1_idx, phi2_idx = self._coords_for(i, j)
        l = _dist_value_grad(pos[l_idx[0]], pos[l_idx[1]], box)[0]
        phi1 = _dihedral_value_grad(*[pos[k] for k in phi1_idx], box)[0]
        phi2 = _dihedral_value_grad(*[pos[k] for k in phi2_idx], box)[0]
        return (l, phi1, phi2)

    def set_references(self, refs: Sequence[Tuple[float, float, float]]):
        """Override per-interaction (l0, phi1_0, phi2_0) references."""
        if len(refs) != len(self._pairs):
            raise ValueError("need one (l0,phi1_0,phi2_0) per consecutive pair")
        self._refs = [tuple(map(float, r)) for r in refs]

    def interactions(self, positions=None, box=None) -> List[Interaction]:
        inters: List[Interaction] = []
        for (i, j), (l0, phi1_0, phi2_0) in zip(self._pairs, self._refs):
            l_idx, phi1_idx, phi2_idx = self._coords_for(i, j)
            coords = [
                ("dist", l_idx, self.k_l, l0),
                ("dihedral", phi1_idx, self.k_phi, phi1_0),
                ("dihedral", phi2_idx, self.k_phi, phi2_0),
            ]
            inters.append((self.U0, coords))
        return inters


# --------------------------------------------------------------------------- #
# 4. Hydrogen bonding  U_HB  (MODEL.md sec. 4, Eq. 13)                          #
# --------------------------------------------------------------------------- #
class TISHydrogenBonding(_ManyBodyCustomForce):
    """Hydrogen bonding between complementary bases (NONBONDED, distance-gated).

    Candidate pairs are complementary bases (A-U, G-C, G-U wobble) with sequence
    separation >= ``min_seq_sep``; an interaction is active only while the
    base-base distance is below ``contact_cutoff`` (a hard truncation -- a
    production term needs a smooth switch). Coordinates (PROVISIONAL):
        d   = dist(B_i, B_j)                          k = k_d
        th1 = angle(S_i, B_i, B_j)                    k = k_theta
        th2 = angle(S_j, B_j, B_i)                    k = k_theta
        ps1 = dihedral(S_i, B_i, B_j, S_j)            k = k_psi
        ps2 = dihedral(P_i, S_i, B_i, B_j)            k = k_psi
        ps3 = dihedral(P_j, S_j, B_j, B_i)            k = k_psi

    Well depth U0 = HB_U0_SINGLE x multiplicity (x2 A-U, x3 G-C, x2 G-U*), all
    PLACEHOLDER; reference geometry (d0/theta0/psi0) is nominal PLACEHOLDER.

    Parameters
    ----------
    sequence : str
        RNA sequence (A/G/C/U) of the single strand, used to find complementary
        candidate pairs. For multi-strand, pass ``pairs`` explicitly instead.
    d0, th0, ps0 : float
        Reference geometry (PLACEHOLDER).
    k_d, k_theta, k_psi : float
        Force constants (params.HB_KD/HB_KTHETA/HB_KPSI).
    contact_cutoff : float
        Distance gate (A).
    pairs : list[(i,j)], optional
        Explicit candidate nucleotide-index pairs (overrides sequence scan),
        each tagged with its multiplicity via ``pair_mult``.
    pair_mult : dict[(i,j)->int], optional
        Multiplicity for explicit ``pairs``.
    """

    def __init__(self, sequence: Optional[str] = None,
                 n_nucleotides: Optional[int] = None,
                 d0: float = HB_D0_PLACEHOLDER,
                 th0: float = HB_TH0_PLACEHOLDER,
                 ps0: float = HB_PS0_PLACEHOLDER,
                 k_d: float = params.HB_KD,
                 k_theta: float = params.HB_KTHETA,
                 k_psi: float = params.HB_KPSI,
                 contact_cutoff: float = HB_CONTACT_CUTOFF,
                 min_seq_sep: int = HB_MIN_SEQ_SEP,
                 u0_single: float = HB_U0_SINGLE,
                 pairs: Optional[Sequence[Tuple[int, int]]] = None,
                 pair_mult: Optional[dict] = None):
        if sequence is not None:
            seq = sequence.strip().upper().replace("T", "U")
            n_nucleotides = len(seq)
        elif n_nucleotides is None:
            raise ValueError("provide either sequence or n_nucleotides")
        super().__init__(3 * int(n_nucleotides))
        self.n_nucleotides = int(n_nucleotides)
        self.d0 = float(d0)
        self.th0 = float(th0)
        self.ps0 = float(ps0)
        self.k_d = float(k_d)
        self.k_theta = float(k_theta)
        self.k_psi = float(k_psi)
        self.contact_cutoff = float(contact_cutoff)
        self.u0_single = float(u0_single)

        # candidate (i,j,U0) pairs
        self._candidates: List[Tuple[int, int, float]] = []
        if pairs is not None:
            for (i, j) in pairs:
                mult = (pair_mult or {}).get((i, j), 2)
                self._candidates.append((i, j, self.u0_single * mult))
        else:
            for a in range(self.n_nucleotides):
                for b in range(a + min_seq_sep, self.n_nucleotides):
                    mult = HB_MULTIPLICITY.get((seq[a], seq[b]))
                    if mult is not None:
                        self._candidates.append(
                            (a, b, self.u0_single * mult))

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

    def _coords_list(self, i: int, j: int) -> List[Coord]:
        c = self._coords_for(i, j)
        return [
            ("dist", c["d"], self.k_d, self.d0),
            ("angle", c["th1"], self.k_theta, self.th0),
            ("angle", c["th2"], self.k_theta, self.th0),
            ("dihedral", c["ps1"], self.k_psi, self.ps0),
            ("dihedral", c["ps2"], self.k_psi, self.ps0),
            ("dihedral", c["ps3"], self.k_psi, self.ps0),
        ]

    def interactions(self, positions: np.ndarray, box: np.ndarray
                     ) -> List[Interaction]:
        positions = np.asarray(positions, dtype=float)
        box = np.asarray(box, dtype=float)
        inters: List[Interaction] = []
        for (i, j, U0) in self._candidates:
            Bi, Bj = 3 * i + 2, 3 * j + 2
            d = minimum_image(positions[Bi] - positions[Bj], box)
            if float(d @ d) <= self.contact_cutoff * self.contact_cutoff:
                inters.append((U0, self._coords_list(i, j)))
        return inters

    def candidate_pairs(self) -> List[Tuple[int, int, float]]:
        """Complementary candidate (i, j, U0) triples (for tests / inspection)."""
        return list(self._candidates)
