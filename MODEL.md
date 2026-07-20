# TIS model — energy function specification

The Three Interaction Site (TIS) coarse-grained nucleic-acid model of Hyeon &
Thirumalai / Denesyuk & Thirumalai. Each nucleotide is **three point particles** —
Phosphate (P), Sugar (S), Base (B) — with **no orientation degree of freedom**
(unlike oxDNA's rigid bodies). Directionality of stacking and base pairing is
carried by *many-body* (multi-site) potentials over the point particles, so the
model integrates as ordinary translational MD.

Total energy (TIS-DNA, Chakraborty–Hori–Thirumalai 2018, Eq. 1; identical form to
the TIS-RNA model, Denesyuk–Thirumalai 2013):

```
U_T = U_B + U_A + U_S + U_HB + U_EV + U_E
```

Six terms: **bond**, **angle**, single-stranded **stacking**, **hydrogen bonding**,
**excluded volume**, **electrostatics**. Note there is *no separate dihedral term* —
the dihedral dependence is folded into the stacking (that is where the helical twist
and handedness live, closer in spirit to oxDNA than to 3SPN).

Sources: forms + DNA parameters extracted from arXiv:1802.01612 (open access, Tables
3–4). **RNA-specific parameters must come from Denesyuk & Thirumalai, *J. Phys. Chem.
B* 117 (2013) 4901** (see "Parameter status" below).

---

## 1. Bond stretching — `U_B` (HOOMD `md.bond.Harmonic`)

```
U_B = k_r (r − r0)^2                         [Eq. 2]
```
Per bonded pair along the backbone (S–P, P–S, S–B). DNA constants (Table 4, units
kcal/mol/Å², Å):

| bond | k_r    | r0   |
|------|--------|------|
| SP   | 62.59  | 3.75 |
| PS   | 17.63  | 3.74 |
| SA   | 44.31  | 4.85 |
| SG   | 48.98  | 4.96 |
| SC   | 43.25  | 4.30 |
| ST   | 46.56  | 4.40 |

## 2. Bond angle — `U_A` (HOOMD `md.angle.Harmonic`)

```
U_A = k_alpha (alpha − alpha0)^2             [Eq. 3]
```
~10 backbone triplet types (PSP, SPS, PSB, …). DNA (partial, Table 4):

| angle | k_alpha | alpha0   |
|-------|---------|----------|
| PSP   | 25.67   | 123.30°  |
| SPS   | 67.50   | 94.60°   |
| PSA   | 29.53   | 107.38°  |

*(Full 10-row table to be transcribed from the paper.)*

## 3. Single-stranded stacking — `U_S` (CUSTOM C++)

Between **consecutive** bases i, i+1:

```
U_S = U_S0 / ( 1 + k_l (l − l0)^2 + k_phi (phi1 − phi1_0)^2 + k_phi (phi2 − phi2_0)^2 )   [Eq. 9]
```
- `l` = distance between the two stacked base sites
- `phi1, phi2` = two backbone dihedral angles (this is the many-body / geometry-and-
  chirality-carrying part)
- `k_l = 1.45` Å⁻², `k_phi = 3.00` rad⁻² (Table 4)
- Temperature- and sequence-dependent well depth:

```
U_S0 = −h + k_B (T − T_m) s                  [per dinucleotide step]
```
Per-dimer `h, s, T_m` fit to nearest-neighbor melting thermodynamics (16 steps,
Table 3). Example (GC): h=5.13 kcal/mol, s=2.41, T_m=331.9 K, ΔG0=0.26 kcal/mol.
*(Full 16-step table to be transcribed.)*

## 4. Hydrogen bonding — `U_HB` (CUSTOM C++)

Between complementary bases on opposite strands (or tertiary pairs):

```
U_HB = U_HB0 / ( 1 + k_d (d−d0)^2 + k_theta (θ1−θ1_0)^2 + k_theta (θ2−θ2_0)^2
                   + k_psi (ψ1−ψ1_0)^2 + k_psi (ψ2−ψ2_0)^2 + k_psi (ψ3−ψ3_0)^2 )   [Eq. 13]
```
Depends on the base–base distance `d`, two angles `θ`, and three dihedrals `ψ`
(a genuine multi-site term over both bases and their neighboring sugars). Constants
(Table 4): `k_d=4.00` Å⁻², `k_theta=1.50` rad⁻², `k_psi=0.15` rad⁻²,
`U_HB0=−1.92` kcal/mol (calibrated to hairpin melting). Multiplicity: **×2 for A–T
(A–U in RNA), ×3 for G–C**. Reference geometry (d0, θ0, ψ0) per pair type from the
paper. RNA additionally needs the **G–U wobble** pair.

## 5. Excluded volume — `U_EV` (HOOMD WCA)

```
U_EV = eps0 [ (D0/r)^12 − 2 (D0/r)^6 + 1 ],  r < D0                              [Eq. 8]
```
`D0 = 3.2` Å (all sites), `eps0 = 1` kcal/mol.

## 6. Electrostatics — `U_E` (HOOMD Debye–Hückel / Yukawa)

Screened Coulomb between **phosphate** sites:

```
U_E = (Q^2 e^2 / 2 eps) Σ_{i,j}  exp(−r_ij / λ_D) / r_ij                          [Eq. 14]
λ_D^-2 = (4π / eps k_B T) Σ_n q_n^2 ρ_n                                          [Eq. 15]
```
Phosphate charge is **renormalized** by Oosawa–Manning counterion condensation:
```
Q = b / l_B(T),   l_B = e^2 / (eps k_B T),   b = 4.4 Å (DNA)  →  Q ≈ −0.6 e at 298 K   [Eq. 16]
eps(T) = 87.740 − 0.4008 T + 9.398e-4 T^2 − 1.410e-6 T^3   (T in °C)              [Eq. 17]
```
Divalent ions (Mg²⁺): **not** in the base DH model — later TIS-RNA variants
(Nguyen/Hori) add explicit Mg²⁺ / RISM. That is a separate, larger extension.

---

## Parameter status

- **Forms:** complete for all six terms (above).
- **DNA parameters:** partially transcribed (bonds full; angles/stacking/HB tables
  only representative rows so far — need the complete Tables 3–4).
- **RNA parameters:** NOT yet obtained. Need Denesyuk–Thirumalai 2013 for: A-form
  reference geometry (l0, angles, dihedral references), RNA nearest-neighbor stacking
  table (16 steps, fit to RNA melting), RNA HB geometry incl. **G–U wobble**, and the
  RNA charge-renormalization `b`. The forms are identical; only the constants differ.

## HOOMD mapping

| term | HOOMD | new code? |
|------|-------|-----------|
| bond | `md.bond.Harmonic` | no |
| angle | `md.angle.Harmonic` | no |
| excluded volume | `md.pair.LJ` (WCA, shifted) | no |
| electrostatics | Debye–Hückel (`md.pair.Yukawa`-style) | no |
| **stacking** | — | **custom ForceCompute** (dist + 2 dihedrals, consecutive bases) |
| **hydrogen bonding** | — | **custom ForceCompute** (dist + 2 angles + 3 dihedrals) |

Only two custom C++ terms — the same multi-body ForceCompute pattern as the oxDNA
plugin's bonded force. Everything else is stock HOOMD.

## Validation plan (open question)

No public TIS reference oracle is available (unlike oxDNA's `split_energy.dat`). The
intended validation target is **thermodynamic**: reproduce hairpin / duplex melting
temperatures against the experimental nearest-neighbor (Turner for RNA) parameters
the model was fit to. Term-by-term numerical validation would require generating TIS
reference output from the authors' code (not public) or CafeMol's TIS-type model.
