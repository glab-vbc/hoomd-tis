"""TIS model parameters.

Functional forms are in ../MODEL.md. Values below are transcribed from the open
TIS-DNA paper (arXiv:1802.01612, Tables 3-4). Units: energy kcal/mol, length A,
angle stored in radians (source tables give degrees; converted on load).

STATUS: DNA parameters are PARTIAL (the summarizing fetch returned representative
rows only) and RNA parameters are NOT yet transcribed. Anything not yet confirmed
from the primary tables is marked TODO and must not be trusted for production.
Convert to HOOMD's energy unit (oxDNA/TIS use kcal/mol; pick a consistent unit
system before running).
"""
import math

_DEG = math.pi / 180.0

# --- 1. Bond stretching:  U = k_r (r - r0)^2   [Eq. 2] ---
# (bond_type): (k_r [kcal/mol/A^2], r0 [A])   -- DNA, Table 4 (complete)
BOND_DNA = {
    "SP": (62.59, 3.75), "PS": (17.63, 3.74),
    "SA": (44.31, 4.85), "SG": (48.98, 4.96),
    "SC": (43.25, 4.30), "ST": (46.56, 4.40),
}
# RNA: replace ST->SU and refit values. TODO from Denesyuk-Thirumalai 2013.
BOND_RNA = {}  # TODO

# --- 2. Bond angle:  U = k_a (a - a0)^2   [Eq. 3] ---
# (angle_type): (k_a [kcal/mol/rad^2], a0 [rad])  -- DNA, Table 4 (PARTIAL: 3 of ~10)
ANGLE_DNA = {
    "PSP": (25.67, 123.30 * _DEG),
    "SPS": (67.50, 94.60 * _DEG),
    "PSA": (29.53, 107.38 * _DEG),
    # TODO: PSG, PSC, PST, ASP, GSP, CSP, TSP, ... (full Table 4)
}
ANGLE_RNA = {}  # TODO

# --- 3. Stacking:  U = US0 / (1 + k_l dl^2 + k_phi dphi1^2 + k_phi dphi2^2)   [Eq. 9]
# with US0 = -h + kB (T - Tm) s   (per dinucleotide step) ---
STACK_KL = 1.45       # A^-2
STACK_KPHI = 3.00     # rad^-2
# dimer step -> (h [kcal/mol], s [dimensionless], Tm [K]); DNA Table 3 (PARTIAL)
STACK_DNA = {
    "GC": (5.13, 2.41, 331.9),  # ΔG0 = 0.26 kcal/mol
    # TODO: remaining 15 of 16 steps (AA, AC, AG, AT, CA, CC, CG, CT, GA, GG, GT,
    #       TA, TC, TG, TT) from Table 3
}
STACK_RNA = {}  # TODO: RNA 16-step table (fit to RNA nearest-neighbor / Turner)

# reference geometry per step: (l0 [A], phi1_0 [rad], phi2_0 [rad]) -- TODO (Table)
STACK_GEOM_DNA = {}  # TODO
STACK_GEOM_RNA = {}  # TODO

# --- 4. Hydrogen bonding:  U = UHB0 / (1 + k_d dd^2 + k_th dth^2*2 + k_ps dps^2*3)  [Eq. 13] ---
HB_KD = 4.00          # A^-2
HB_KTHETA = 1.50      # rad^-2
HB_KPSI = 0.15        # rad^-2
HB_U0 = -1.92         # kcal/mol (calibrated to hairpin melting)
HB_MULTIPLICITY = {"AT": 2, "AU": 2, "GC": 3}   # RNA adds GU wobble -> TODO strength
# reference geometry (d0, theta1_0, theta2_0, psi1_0, psi2_0, psi3_0) per pair -- TODO
HB_GEOM = {}  # TODO (Table 4 geometry rows)

# --- 5. Excluded volume (WCA):  U = eps0 [(D0/r)^12 - 2(D0/r)^6 + 1], r<D0   [Eq. 8] ---
EV_D0 = 3.2           # A
EV_EPS0 = 1.0         # kcal/mol

# --- 6. Electrostatics (Debye-Huckel on phosphates)   [Eq. 14-17] ---
DH_MANNING_B_DNA = 4.4   # A (charge spacing); RNA value TODO
def phosphate_charge(t_kelvin, manning_b=DH_MANNING_B_DNA):
    """Oosawa-Manning renormalized phosphate charge Q = b / l_B(T)  [Eq. 16]."""
    l_b = bjerrum_length_A(t_kelvin)
    return -manning_b / l_b   # ~ -0.6 e at 298 K for DNA

def dielectric(t_kelvin):
    """Temperature-dependent water dielectric, T in Celsius  [Eq. 17]."""
    tc = t_kelvin - 273.15
    return 87.740 - 0.4008*tc + 9.398e-4*tc**2 - 1.410e-6*tc**3

def bjerrum_length_A(t_kelvin):
    """l_B = e^2 / (eps kB T) in Angstrom."""
    # l_B(A) = 167100 / (eps * T)  (standard water constant); refine units on wiring
    return 167100.0 / (dielectric(t_kelvin) * t_kelvin)
