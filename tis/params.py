"""TIS model parameters.

RNA parameters (primary) are transcribed from Denesyuk & Thirumalai, J. Phys. Chem.
B 117 (2013) 4901-4911 (DOI 10.1021/jp401087x) -- Methods + Tables 1-2. DNA values
(from the open TIS-DNA paper, arXiv:1802.01612) are kept for reference. Functional
forms are in ../MODEL.md.

Units: energy kcal/mol, length Angstrom, angle radians. kB = 0.0019872 kcal/mol/K.

COMPLETE from the papers: all force constants, the full 16-dimer stacking table, the
three fitted parameters (dG0, U_HB0, b), electrostatics. The one piece NOT tabulated
in the paper is the reference GEOMETRY (bond rho0, angle alpha0, stacking r0/phi0, HB
r0/theta0/psi0): these are obtained by "coarse-graining an ideal A-form RNA helix" and
must be generated (see MODEL.md, A-form reference geometry TODO).
"""
import math

KB = 0.0019872          # kcal/mol/K   (kT(300K) = 0.596)
_DEG = math.pi / 180.0
_C2K = 273.15


# ======================= RNA  (Denesyuk & Thirumalai 2013) =======================

# --- 1. Bonds:  U = k_rho (rho - rho0)^2   (k in kcal/mol/A^2) ; rho0 from A-form helix ---
# "->" is the downstream (5'->3') direction.
BOND_RNA_K = {"SP": 64.0, "PS": 23.0, "SB": 10.0}   # S->P, P->S, S-B

# --- 2. Angles:  U = k_alpha (alpha - alpha0)^2 ; alpha0 from A-form helix ---
ANGLE_RNA_K_BASE = 5.0        # kcal/mol/rad^2, if the angle involves a base (B)
ANGLE_RNA_K_BACKBONE = 20.0   # otherwise (P/S-only triplets)

# --- 3. Excluded volume (WCA):  U = eps0[(D0/r)^12 - 2(D0/r)^6 + 1], r<D0 ---
EV_D0 = 3.2          # A  (all sites)
EV_EPS0 = 1.0        # kcal/mol

# --- 4. Stacking (eq 3):  U = UST0 / (1 + kr(r-r0)^2 + kphi(phi1-phi1_0)^2 + kphi(phi2-phi2_0)^2)
#   consecutive nucleotides i, i+1:
#     r    = dist(B_i, B_{i+1})
#     phi1 = dihedral(P_i,   S_i,   P_{i+1}, S_{i+1})     [Fig 2: phi1(P1,S1,P2,S2)]
#     phi2 = dihedral(P_{i+2}, S_{i+1}, P_{i+1}, S_i)     [Fig 2: phi2(P3,S2,P2,S1)]
#   (phi1 needs P_i; phi2 needs P_{i+2} -> terminal steps need edge handling.)
STACK_RNA_KR = 1.4       # A^-2
STACK_RNA_KPHI = 4.0     # rad^-2
STACK_DG0 = 0.6          # kcal/mol (fitted; Tables 2 h-values correspond to this)
# well depth is temperature-dependent:  UST0(T) = -h + kB (T - Tm) s
# per ordered 5'->3' dimer XY:  (h [kcal/mol], s [-], Tm [K])   (Tables 1 + 2)
STACK_RNA = {
    "UU": (3.37, -3.56, -21 + _C2K),
    "CC": (4.01, -1.57,  13 + _C2K),
    "CU": (3.99, -1.57,  13 + _C2K), "UC": (3.99, -1.57, 13 + _C2K),
    "AA": (4.35, -0.32,  26 + _C2K),
    "AU": (4.29, -0.32,  26 + _C2K), "UA": (4.31, -0.32, 26 + _C2K),
    "AC": (4.29, -0.32,  26 + _C2K), "CA": (4.31, -0.32, 26 + _C2K),
    "GC": (4.60,  0.77,  42 + _C2K),
    "GU": (5.03,  2.92,  65 + _C2K), "UG": (4.98,  2.92, 65 + _C2K),
    "CG": (5.07,  4.37,  70 + _C2K),
    "GA": (5.12,  5.30,  68 + _C2K), "AG": (5.08,  5.30, 68 + _C2K),
    "GG": (5.56,  7.35,  93 + _C2K),
}

def stacking_U0(dimer, t_kelvin):
    """Temperature-dependent stacking well depth UST0(T) for a 5'->3' dimer (kcal/mol)."""
    h, s, tm = STACK_RNA[dimer]
    return -h + KB * (t_kelvin - tm) * s

# --- 5. Hydrogen bonding (eq 7):
#   U = mult * U_HB0 / (1 + 5(r-r0)^2 + 1.5(th1-)^2 + 1.5(th2-)^2
#                          + 0.15(ps-)^2 + 0.15(ps1-)^2 + 0.15(ps2-)^2)
#   For a Watson-Crick base pair (B_i, B_j) the six coords follow Fig 4 (B-B analogue):
#     r  = dist(B_i, B_j)
#     th1= angle(S_i, B_i, B_j),  th2 = angle(S_j, B_j, B_i)
#     ps = dih(S_i, B_i, B_j, S_j) and two more dihedrals into the downstream backbone.
#   Base model: only H-bonds present in the reference (PDB) structure ("native").
#   Extended model: allow any G-C / A-U / G-U pair.
HB_RNA_U0 = -2.43       # kcal/mol, single H-bond (fitted)
HB_RNA_KR = 5.0         # A^-2
HB_RNA_KTHETA = 1.5     # rad^-2
HB_RNA_KPSI = 0.15      # rad^-2
# multiplicity = number of H-bonds in the pair
HB_MULT = {("A", "U"): 2, ("U", "A"): 2, ("G", "C"): 3, ("C", "G"): 3,
           ("G", "U"): 2, ("U", "G"): 2}   # G-U wobble ~2 H-bonds

# --- 6. Electrostatics (Debye-Huckel on phosphates)  (eq 8-12) ---
#   U_EL = Q^2 e^2/(2 eps) * sum exp(-r/lambda)/r ;  Q = b/lB(T) ;  lB = e^2/(eps kB T)
DH_B_RNA = 4.4          # A  (fitted charge spacing; Q ~ -0.6e at 37 C)

def dielectric(t_kelvin):
    """Water dielectric, eq 12 (T in Celsius): 87.740 -0.4008 T +9.398e-4 T^2 -1.410e-6 T^3."""
    tc = t_kelvin - _C2K
    return 87.740 - 0.4008 * tc + 9.398e-4 * tc**2 - 1.410e-6 * tc**3

def bjerrum_length_A(t_kelvin):
    """Bjerrum length lB = e^2/(eps kB T) in Angstrom."""
    return 167100.0 / (dielectric(t_kelvin) * t_kelvin)

def phosphate_charge(t_kelvin, manning_b=DH_B_RNA):
    """Manning-renormalized phosphate charge Q = -b/lB(T)  (eq 10)."""
    return -manning_b / bjerrum_length_A(t_kelvin)


# ============================ DNA  (arXiv:1802.01612) ============================
# Kept for reference / a possible TIS-DNA build. Bonds full; angles/stacking partial.
BOND_DNA = {"SP": (62.59, 3.75), "PS": (17.63, 3.74), "SA": (44.31, 4.85),
            "SG": (48.98, 4.96), "SC": (43.25, 4.30), "ST": (46.56, 4.40)}
ANGLE_DNA = {"PSP": (25.67, 123.30 * _DEG), "SPS": (67.50, 94.60 * _DEG),
             "PSA": (29.53, 107.38 * _DEG)}  # partial
STACK_DNA_KL, STACK_DNA_KPHI = 1.45, 3.00
HB_DNA_U0, HB_DNA_KD, HB_DNA_KTHETA, HB_DNA_KPSI = -1.92, 4.00, 1.50, 0.15
