# hoomd-tis

A HOOMD-blue implementation of the **TIS (Three Interaction Site)** coarse-grained
nucleic-acid model (Hyeon & Thirumalai; Denesyuk & Thirumalai) — targeting **RNA**
(the model's home ground), with DNA sharing the same functional forms.

This is a **separate project** from [`hoomd-oxDNA`](../hoomd-oxDNA): TIS is a
*point-particle* model (3 beads/nucleotide, no orientation/quaternions), so it shares
essentially none of oxDNA's rigid-body engine — only the build scaffolding and the
custom-ForceCompute pattern carry over.

## Why TIS (vs oxRNA2)

- **Point particles, no rotational integration.** Directionality comes from many-body
  bond/angle/dihedral-in-stacking potentials, not quaternion-tracked patches.
- **Fit to nearest-neighbor thermodynamics** — stacking well depths are temperature-
  and sequence-dependent, matched to melting data.
- **Ion physics.** The lineage adds explicit Mg²⁺ (via RISM) in later variants — the
  physics that drives RNA *tertiary* folding and that oxRNA2 lacks. (Base model here
  uses Debye–Hückel with Manning-renormalized phosphate charge; explicit Mg²⁺ is a
  planned extension.)

## The model

Six energy terms — see [`MODEL.md`](MODEL.md) for exact forms + parameters + sources:

```
U = U_bond + U_angle + U_stacking + U_hbond + U_excluded-volume + U_electrostatics
```

Four map onto stock HOOMD (bonds, angles, WCA, Debye–Hückel); only **stacking** and
**hydrogen bonding** need custom C++ (multi-body ForceComputes, same pattern as the
oxDNA bonded force).

## Layout

```
MODEL.md              extracted energy-function spec (forms + parameters + provenance)
tis/                  Python model layer
  io.py               3-site (P/S/B) topology + snapshot builder      [TODO]
  params.py           parameter tables (bond/angle/stacking/HB/EV/DH)
  forces.py           wire the 6 terms (4 native + 2 custom)          [TODO]
  model/rna.py        RNA assembly                                    [TODO]
src/                  custom C++ ForceComputes (stacking, hydrogen bonding) [TODO]
pytest/               validation (melting-curve / thermodynamic)      [TODO]
```

## Status

Early scaffold. Done:
- Repo skeleton + build layout.
- **Complete energy-function specification** extracted from the open TIS-DNA paper
  (arXiv:1802.01612); functional forms for all six terms.
- Partial DNA parameter tables.

Next:
1. Transcribe the **complete** parameter tables (all angles, 16 stacking dimers, HB
   geometries) — and obtain the **RNA-specific** parameters (Denesyuk–Thirumalai 2013).
2. `tis/io.py`: 3-site topology builder (P/S/B point particles + bonds/angles).
3. Wire the four stock-HOOMD terms.
4. Implement the two custom C++ terms (stacking, hydrogen bonding).
5. Validation: reproduce a hairpin/duplex melting curve.

## Provenance

Model: Hyeon & Thirumalai (2005); Denesyuk & Thirumalai, *J. Phys. Chem. B* 117
(2013) 4901 (RNA); Chakraborty, Hori & Thirumalai, arXiv:1802.01612 (DNA, open —
source of the extracted forms). Not affiliated with or endorsed by those authors;
this is an independent HOOMD re-implementation.
