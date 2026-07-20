"""Wire the TIS energy terms onto stock HOOMD force objects.

Four of the six TIS terms map directly onto native HOOMD forces and are built
here; the two many-body terms (single-stranded **stacking** and **hydrogen
bonding**) require custom C++ ForceComputes and are left as documented stubs
(``stacking_force`` / ``hydrogen_bonding_force``) until those land.

Native terms built (see ../MODEL.md):

  1. Bond stretching  U_B = k_r (r - r0)^2                 -> md.bond.Harmonic
  2. Bond angle       U_A = k_a (a - a0)^2                 -> md.angle.Harmonic
  5. Excluded volume  U_EV = eps0[(D0/r)^12 - 2(D0/r)^6+1] -> md.pair.LJ (WCA)
  6. Electrostatics   U_E  = Q^2 e^2/eps * exp(-r/lD)/r    -> md.pair.Yukawa

Unit / convention decisions
---------------------------
* Energies in kcal/mol, lengths in Angstrom, kB = 0.0019872 kcal/mol/K, so
  kT(300 K) ~ 0.596 kcal/mol.
* HOOMD's Harmonic bond/angle use U = 1/2 k (x - x0)^2, whereas the TIS paper
  uses U = k_r (x - x0)^2 with NO 1/2. We therefore pass k_hoomd = 2 * k_paper
  for both bonds and angles.
* Excluded volume: the paper's shifted form eps0[(D0/r)^12 - 2(D0/r)^6 + 1] for
  r < D0 is EXACTLY a Weeks-Chandler-Andersen (WCA) potential. With
  U_LJ = 4 eps[(s/r)^12 - (s/r)^6] and sigma = D0 / 2^(1/6), one gets
  4 eps[(s/r)^12 - (s/r)^6] + eps = eps[(D0/r)^6 - 1]^2, i.e. the paper form with
  eps = eps0. So we use md.pair.LJ with sigma = D0/2^(1/6), epsilon = eps0,
  r_cut = D0 (= 2^(1/6) sigma, the potential minimum) and mode="shift" (the +eps
  shift). Purely repulsive, zero and smooth at r = D0. Applied to ALL site pairs.
* Debye-Huckel on phosphates only. Yukawa is U = eps_Y exp(-kappa r)/r. Matching
  the paper's per-pair energy Q^2 e^2/eps * exp(-r/lD)/r and using the Bjerrum
  length l_B = e^2/(eps kB T) gives eps_Y = Q^2 * l_B * kB * T (units kcal/mol.A)
  and kappa = 1/lambda_D. Only the (P,P) pair gets a non-zero epsilon; every
  other pair is set to epsilon = 0 (no electrostatics off the phosphates).
* Debye length: params.py has no salt concentration, so the Debye length is
  computed here from an explicit ``ionic_strength_M`` (default 0.15 M monovalent)
  via lambda_D^-2 = 4 pi l_B sum_n q_n^2 rho_n = 8 pi l_B I (Eq. 15). NOTE:
  ionic strength is a modelling input, not (yet) a params.py entry.
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import hoomd
import hoomd.md as md

from . import params

# Boltzmann constant in the paper's unit system.
KB = 0.0019872  # kcal/mol/K

# Avogadro's number, used to convert mol/L -> number density in A^-3.
_NA_PER_L_TO_PER_A3 = 6.02214076e-4  # (1 mol/L) = 6.022e-4 ions / A^3


# --------------------------------------------------------------------------- #
# Debye-Huckel helpers                                                         #
# --------------------------------------------------------------------------- #
def debye_length_A(t_kelvin: float, ionic_strength_M: float) -> float:
    """Debye screening length lambda_D (A) for a monovalent 1:1 salt.

    lambda_D^-2 = 4 pi l_B sum_n q_n^2 rho_n = 8 pi l_B I   (Eq. 15),
    with l_B the Bjerrum length (params.bjerrum_length_A) and I the ionic
    strength expressed as a number density. For 0.15 M at ~300 K this yields
    ~7.9 A, matching the textbook lambda_D = 3.04/sqrt(I[M]) A.
    """
    if ionic_strength_M <= 0:
        return math.inf
    l_b = params.bjerrum_length_A(t_kelvin)               # A
    i_number = ionic_strength_M * _NA_PER_L_TO_PER_A3     # ions / A^3
    kappa2 = 8.0 * math.pi * l_b * i_number               # A^-2
    return 1.0 / math.sqrt(kappa2)


def yukawa_epsilon(t_kelvin: float, manning_b: float = params.DH_MANNING_B_DNA) -> float:
    """Prefactor eps_Y = Q^2 l_B kB T (kcal/mol . A) for the phosphate Yukawa pair."""
    q = params.phosphate_charge(t_kelvin, manning_b=manning_b)  # renormalised charge (e)
    l_b = params.bjerrum_length_A(t_kelvin)                     # A
    return q * q * l_b * KB * t_kelvin


# --------------------------------------------------------------------------- #
# Parameter resolution                                                         #
# --------------------------------------------------------------------------- #
def _resolve_bond(bond_type: str):
    """(k_r, r0) for a bond type, with the RNA placeholder SU -> DNA ST."""
    if bond_type in params.BOND_DNA:
        return params.BOND_DNA[bond_type]
    if bond_type == "SU":
        return params.BOND_DNA["ST"]     # uracil borrows thymine (placeholder)
    raise KeyError(f"no bond parameters for type {bond_type!r} "
                   f"(available: {sorted(params.BOND_DNA)})")


def _resolve_angle(angle_type: str):
    """(k_a, a0[rad]) for an angle type from params.ANGLE_DNA."""
    if angle_type in params.ANGLE_DNA:
        return params.ANGLE_DNA[angle_type]
    raise KeyError(f"no angle parameters for type {angle_type!r} "
                   f"(available: {sorted(params.ANGLE_DNA)})")


# --------------------------------------------------------------------------- #
# Native-term assembler                                                        #
# --------------------------------------------------------------------------- #
def build_forces(
    snapshot: hoomd.Snapshot,
    temperature: float = 300.0,
    ionic_strength_M: float = 0.15,
    dh_rcut_factor: float = 3.0,
    nlist_buffer: float = 0.4,
    exclusions=("bond", "angle"),
    manning_b: float = params.DH_MANNING_B_DNA,
):
    """Assemble the four native TIS forces for the topology in ``snapshot``.

    Parameters
    ----------
    snapshot : hoomd.Snapshot
        Built by tis.io.build_strand_snapshot; read for its particle / bond /
        angle type names.
    temperature : float
        Temperature in K (sets the Debye-Huckel charge, dielectric, screening).
    ionic_strength_M : float
        Monovalent-salt ionic strength (mol/L) for the Debye length. NOTE: a
        modelling input, not a params.py value.
    dh_rcut_factor : float
        Debye-Huckel real-space cutoff = dh_rcut_factor * lambda_D.
    nlist_buffer : float
        Neighbor-list skin (A).
    exclusions : tuple
        Neighbor-list pair exclusions. Default ("bond", "angle") removes 1-2
        (bonded) and 1-3 (angle) pairs from BOTH pair forces.
    manning_b : float
        Oosawa-Manning charge-spacing b (A) for phosphate charge renormalisation.

    Returns
    -------
    (forces, nlist) : (list[hoomd.md.force.Force], hoomd.md.nlist.NeighborList)
        forces = [harmonic_bond, harmonic_angle, wca_lj, debye_yukawa]; nlist is
        the shared cell list (returned so callers can inspect / reuse it).
    """
    ptypes = list(snapshot.particles.types)
    btypes = list(snapshot.bonds.types)
    atypes = list(snapshot.angles.types)

    # 1. Harmonic bonds:  U = k_r (r-r0)^2  ->  HOOMD 1/2 k (r-r0)^2, so k = 2 k_r
    bond = md.bond.Harmonic()
    for bt in btypes:
        k_r, r0 = _resolve_bond(bt)
        bond.params[bt] = dict(k=2.0 * k_r, r0=r0)

    # 2. Harmonic angles: U = k_a (a-a0)^2  ->  k = 2 k_a ; t0 already in radians
    angle = md.angle.Harmonic()
    for at in atypes:
        k_a, a0 = _resolve_angle(at)
        angle.params[at] = dict(k=2.0 * k_a, t0=a0)

    # shared neighbor list
    nlist = md.nlist.Cell(buffer=nlist_buffer, exclusions=tuple(exclusions))

    # 5. Excluded volume as WCA via LJ (see module docstring for the mapping)
    sigma = params.EV_D0 / (2.0 ** (1.0 / 6.0))
    r_cut_ev = params.EV_D0                       # = 2^(1/6) sigma (the minimum)
    wca = md.pair.LJ(nlist, default_r_cut=r_cut_ev, mode="shift")
    for a, b in itertools.combinations_with_replacement(ptypes, 2):
        wca.params[(a, b)] = dict(sigma=sigma, epsilon=params.EV_EPS0)
        wca.r_cut[(a, b)] = r_cut_ev

    # 6. Debye-Huckel on phosphates via Yukawa: U = eps_Y exp(-kappa r)/r
    lambda_d = debye_length_A(temperature, ionic_strength_M)
    r_cut_dh = dh_rcut_factor * lambda_d if math.isfinite(lambda_d) else 0.0
    eps_pp = yukawa_epsilon(temperature, manning_b=manning_b)
    kappa = 1.0 / lambda_d if math.isfinite(lambda_d) else 0.0
    dh = md.pair.Yukawa(nlist, default_r_cut=r_cut_dh)
    for a, b in itertools.combinations_with_replacement(ptypes, 2):
        if a == "P" and b == "P":
            dh.params[(a, b)] = dict(epsilon=eps_pp, kappa=kappa)
            dh.r_cut[(a, b)] = r_cut_dh
        else:
            # electrostatics only between phosphates; everything else is silent
            dh.params[(a, b)] = dict(epsilon=0.0, kappa=kappa)
            dh.r_cut[(a, b)] = 0.0

    return [bond, angle, wca, dh], nlist


# --------------------------------------------------------------------------- #
# TODO: custom many-body terms (need C++ ForceComputes)                        #
# --------------------------------------------------------------------------- #
def stacking_force(*args, **kwargs):
    """TODO -- single-stranded stacking U_S (MODEL.md sec. 3, Eq. 9).

    Many-body term over consecutive bases: one base-base distance + two backbone
    dihedrals, with a temperature/sequence-dependent well depth
    U_S0 = -h + kB (T - Tm) s. Requires a custom C++ ForceCompute (same pattern
    as the oxDNA bonded force) plus the RNA stacking table + reference geometry,
    which are not yet in params.py (STACK_RNA / STACK_GEOM_RNA are empty).
    """
    raise NotImplementedError(
        "stacking (U_S) needs a custom C++ ForceCompute and RNA stacking "
        "parameters (params.STACK_RNA / STACK_GEOM_RNA are TODO)."
    )


def hydrogen_bonding_force(*args, **kwargs):
    """TODO -- hydrogen bonding U_HB (MODEL.md sec. 4, Eq. 13).

    Many-body term over complementary bases: one distance + two angles + three
    dihedrals, x2 for A-U and x3 for G-C (plus the RNA G-U wobble). Requires a
    custom C++ ForceCompute plus the HB reference geometry, which is not yet in
    params.py (HB_GEOM is empty).
    """
    raise NotImplementedError(
        "hydrogen bonding (U_HB) needs a custom C++ ForceCompute and HB "
        "reference geometry (params.HB_GEOM is TODO)."
    )
