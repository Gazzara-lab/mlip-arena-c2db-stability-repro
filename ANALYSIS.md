# C2DB dynamical-stability benchmark: a reproduction and root-cause analysis

An independent reproduction and diagnosis of the C2DB dynamical-stability task in
[MLIP Arena](https://github.com/atomind-ai/mlip-arena), following the discussion in
[issue #89](https://github.com/atomind-ai/mlip-arena/issues/89) and building on the
full-set rerun contributed there by @rhirota2001. Focus model: MACE-MP(M) (`mace_mp_0`
medium, float32).

## Context and prior work

Issue #89 (opened by @rhirota2001) reported that the released `benchmarks/c2db` pipeline
does not reproduce the paper's Fig. S10: essentially all truly-stable monolayers are
predicted unstable. The issue already identified the two central mechanisms — that the
−10⁻⁷ THz threshold sits within residual Γ-point acoustic noise, and that the released
pipeline runs phonons without force-constant symmetrization (`symmetry=False`). A follow-up
full-set rerun by @rhirota2001 confirmed, across all 1000 materials, that symmetrization
plus a physical tolerance restores a strong classifier, and raised the open question of
which tolerance and symmetry setting generated the published figure.

This repository takes those results as its starting point and adds two things: a direct
answer to that open question, from the benchmark's own released data; and a per-material
decomposition of the errors that remain once the pipeline is corrected. Where our earlier
notes over-stated a result, we say so explicitly.

## 1. Reproduction setup

For the labelled 1000-material draw (`random.seed(0)`), the C2DB ground truth is 287 stable
and 218 unstable. Our reproduction (`repro_lib.py`) mirrors the released numerics:
`symprec = 1e-5`, displacement 0.01, a (2,2,1) supercell, the same
`symmetrize_force_constants` calls, and Γ-point `get_frequencies`; relaxation follows the
released `OPT` (FIRE, `fmax = 0.05`, `FixSymmetry`, positions-only).

## 2. The published figure corresponds to the released threshold

With the geometry held fixed and the phonon recomputed repeatedly, the symmetrized Γ
minimum frequency scatters over about 10⁻⁷ THz (standard deviation ≈ 3×10⁻⁸;
`control_determinism.py`). At a −10⁻⁷ cut, the verdict for any near-zero material is
therefore governed by numerical noise.

Applying the released analysis to the released `MACE-MP(M).parquet` (which stores the paper
run's raw Γ frequencies), joined to the C2DB labels, reproduces Fig. S10 exactly:

```
              predicted stable   predicted unstable   missing
true stable          43                 244              0
true unstable        21                 196              1
```

Sweeping the threshold on the same data moves the matrix far from the published one (for
example −0.025 gives [[251, 36], [109, 108]]). Three independent sources agree the cut is
−10⁻⁷: the released code (all versions), the released data (above), and the paper text
(Appendix A.10.2: "both values should be greater than −10⁻⁷ to be labeled as stable"). This
resolves the open question in issue #89 — the published figure is the released −10⁻⁷ applied
to the released data, not a different tolerance. The single "missing" entry is the one
material absent from the parquet, which accounts for the 217-versus-218 row sum. Notably,
the published Fig. S10 already shows CHGNet, ORBv2 and ALIGNN identifying no stable
materials, so the pattern is not specific to one model.

## 3. A defensible protocol

Two changes, both consistent with the full-set rerun in issue #89, make the classification
reproducible: enabling `symmetry=True` (which collapses the acoustic residuals to about
10⁻⁷ THz) and raising the tolerance above the noise floor (for example −10⁻² THz). Under
this protocol MACE-MP(M) reaches a Stable F1 of about 0.78 (versus 0.245 published), and the
verdict is stable across −0.01 to −0.1 THz. Sections 4 and 5 diagnose the residual confusion
that remains.

## 4. Stable-side errors: a genuine model signature

Of the 287 true-stable monolayers, 29 remain condemned by deep, threshold-insensitive soft
modes under the corrected protocol (`p3_deep_stables.py`). Characterized against the 258
that pass:

- **Enriched in correlated-3d chemistry:** 14 of 29 (48%) contain V, Cr, Mn, Fe, Co, Ni or
  Cu, versus a 26.4% passed-stable background (a factor of 1.8; z ≈ 2.7, p ≈ 0.004).
- **The most severe cases are magnetic or 3d:** of the 10 with a minimum frequency below
  −2 THz, 7 are magnetic or 3d (for example CrH₂ at −14.3, H₂V₂O₅ at −7.3, NMn₂ at
  −4.1 THz). This is physically reasonable: potentials trained without explicit spin cannot
  capture curvature that depends on magnetic order.
- The C2DB `is_magnetic` flag alone is not enriched (24% versus 20%); the elemental
  (correlated-3d) signal is the better descriptor.

## 5. Unstable-side misses: mostly an observable-coverage effect

Of the 218 true-unstable monolayers, the corrected classifier misses 114
(`p4_missed_unstables.py`). Two observations point away from a purely model-side
explanation:

- The **elastic leg contributes almost no signal**: across all 218 it flags 2 materials on
  its own, and none of the 23 misses whose instability is stiffness-driven in DFT.
- The **miss rate is flat in DFT instability depth** (about 50% even for the most violently
  unstable materials), which is not the pattern expected of near-threshold cases.

This pointed to zone-boundary instabilities invisible to a Γ-only observable. The (2,2,1)
supercell's exact commensurate wavevectors are Γ and the three zone-boundary (M) points, so
the same force constants can be evaluated there at no additional cost (`p5_mpoint.py`). On
the 35 deepest misses (DFT `minhessianeig` < −2 eV/Å²):

| category | count | interpretation |
|---|---|---|
| unstable at the exact zone boundary | 23 of 35 (66%) | recovered by reading M; the model does see the instability |
| genuine model disagreement | 12 of 35 (34%) | clean at Γ, M and K (below) |

An initial curated set of 15 charge-density-wave metals gave 14 of 15; the unbiased 66%
figure corrects that selection effect, and we report it as the honest rate.

Extending to the K point via a (3,3,1) supercell (`p6_kpoint.py`), all 12 remaining
materials stay clean at every commensurate wavevector up to (3,3,1) despite strong DFT
instabilities, so they are genuine model–DFT disagreements rather than hidden zone-boundary
modes. This group is 67% correlated-3d transition metals (Co, Ni, Cu, V, Ti, Sc oxides and
chalcogenides) — the same chemistry as the stable-side errors in Section 4.

## 6. An open question

The unsymmetrized acoustic-residual scale differs by about ten times across environments:
the released parquet and our CPU runs agree (median ≈ −0.009 THz), while the issue-#89
rerun is larger (median ≈ −0.10 THz). This is the clearest argument for symmetrization, but
we did not isolate its cause (relaxation endpoint, floating-point precision, or library
versions); it would benefit from a controlled one-factor-at-a-time study.

## 7. Summary

Once the numerical-noise threshold is set aside, the benchmark's residual errors separate
cleanly: zone-boundary instabilities that a Γ-only observable does not see (recoverable at
no cost from force constants already computed), and genuine model disagreements concentrated
in correlated-3d chemistry. We hope this is useful for a future revision of the protocol.

## Data provenance

`c2db.db` (sha256 `caf58205…`) and `MACE-MP(M).parquet` (sha256 `f722eac6…`) are from the
MLIP Arena Hugging Face Space. The issue-#89 full-set bundles are @rhirota2001's public
attachments (their filenames are swapped between the symmetrized and unsymmetrized runs; the
analysis scripts detect this automatically).

## Caveats

- The environment-dependent residual scale (Section 6) is unexplained.
- The stable-side enrichment is modest (a factor of 1.8; n = 29).
- "Genuine model error" versus a deep incommensurate instability cannot be fully separated
  without denser sampling; the robustly-positive commensurate spectra favor the former.
- The zone-boundary decomposition is measured on the deepest (DFT `minhessianeig` below
  −2 eV/Å²) subset of misses.
