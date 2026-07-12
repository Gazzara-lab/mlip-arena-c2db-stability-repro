# Reproducing and diagnosing the C2DB dynamical-stability benchmark (MLIP Arena)

An independent, inference-only reproduction and root-cause analysis of the
two-dimensional dynamical-stability task in
[MLIP Arena](https://github.com/atomind-ai/mlip-arena) (`benchmarks/c2db`), carried out
in the spirit of the discussion in
[issue #89](https://github.com/atomind-ai/mlip-arena/issues/89). Focus model: MACE-MP(M);
all experiments run on CPU.

## Acknowledgements

This analysis builds directly on prior community work and would not exist without it:

- **@rhirota2001**, who opened issue #89 and contributed the full 1000-material rerun
  (both symmetrized and unsymmetrized, with per-material data). Issue #89 already identified
  the two measurement-side mechanisms below — that the −10⁻⁷ THz threshold sits within the
  residual Γ-point acoustic noise, and that the released pipeline runs phonons without
  force-constant symmetrization — and the rerun both demonstrated the Stable F1 0.245 → 0.78
  recovery and is the foundation for the quantitative decomposition below.
- **The MLIP Arena authors**, whose open benchmark, publicly released per-material data,
  and reproducible pipeline are what let every result here be checked against the original
  source. The points we raise are subtle and, we believe, straightforward to address; they
  do not diminish the value of the benchmark or the effort behind it.
- **The C2DB team**, whose curated stability labels and metadata serve as the ground truth
  throughout.

## What this repository shows

The published C2DB stability results report low stability-classification scores for modern
potentials and read this as a model limitation. Reproducing the pipeline from the released
files, we find that most of this signal has two measurement-side explanations, and that the
remaining real disagreements have a clear chemical signature:

1. **The headline numbers are governed by the measurement, not the model.** The −10⁻⁷ THz
   stability threshold sits below the acoustic-mode residual of the released, unsymmetrized
   pipeline. In that pipeline the residual is a *systematic*, environment-dependent effect
   (a broken acoustic-sum-rule bias of about 10⁻² THz, skewed negative), so near-zero
   materials are effectively decided by a residual far below the physically meaningful
   resolution of the pipeline — and this is what drives Fig. S10's low stable count.
   This repository's contribution here is forensic: using the benchmark's own released data,
   the published confusion matrix is reproduced exactly at that threshold, which answers the
   open question in issue #89 of which tolerance and symmetry setting generated the figure.
   Symmetrizing the force constants collapses that residual to a genuine run-to-run noise
   floor (σ ≈ 3×10⁻⁸ THz), so a −10⁻⁷ cut remains a coin flip even after the fix. Under a
   symmetrized pipeline with a physical tolerance, the same model improves from a Stable F1
   of 0.245 to about 0.78 (from @rhirota2001's full-set rerun).

2. **The remaining errors split cleanly into two causes — by wavevector and by chemistry:**
   - Most unstable-side misses are an observable-coverage effect: the classifier reads
     phonons only at one point of the Brillouin zone (Γ), while many true instabilities
     live at the zone boundary. Reading the force constants the benchmark already computes
     at the zone boundary recovers 23 of the 35 deepest missed materials, at no extra cost.
   - The remaining disagreements are genuine model errors, concentrated in correlated-3d
     transition-metal chemistry — the same chemistry that is enriched in the stable-side
     errors.

A fully worked narrative, with all numbers and caveats, is in [`ANALYSIS.md`](ANALYSIS.md).

## Reproducing

```bash
pip install pandas pyarrow scikit-learn matplotlib scipy   # data-only scripts
pip install -r requirements.txt                            # phonon scripts (MACE + phonopy)
# place c2db.db and the released MACE-MP(M).parquet in the repo root (see ANALYSIS.md)
```

Data-only scripts (no potential needed; `pandas pyarrow scikit-learn matplotlib scipy`):

| script | what it does |
|---|---|
| `forensic_operating_point.py`, `forensic_standalone.py` | reproduce the published confusion matrix from the released data + C2DB labels; sweep the threshold |
| `analyze_fullset.py` | analyze the issue-#89 full-set bundles: residual scaling and environment comparison |
| `p3_deep_stables.py` | characterize the condemned true-stable materials (stable-side errors) |
| `p4_missed_unstables.py` | decompose the true-unstable materials that pass the classifier |

Scripts that require MACE and phonopy (`mace-torch phonopy ase`):

| script | what it does |
|---|---|
| `repro_main.py`, `control_determinism.py`, `control_convergence.py` | original reproduction and numerical controls |
| `p5_mpoint.py [very_deep]` | evaluate the existing (2,2,1) force constants at the zone boundary |
| `p6_kpoint.py [uids...]` | extend the evaluation to the K point via a (3,3,1) supercell |

Per-material outputs are committed under `data/`.

## Suggestions (offered constructively)

- Report a tolerance sweep, or move the stability cut above the numerical noise floor
  (for example −10⁻² THz), and enable force-constant symmetrization (`symmetry=True`) in the
  released `run.py` so the force constants are symmetrized before the frequencies are read.
- Evaluate the minimum frequency over the commensurate wavevectors of the existing supercell
  (Γ and the zone boundary), not Γ alone — this is free, reusing force constants already
  stored.
- Review the elastic leg and the handling of complex-valued results in the analysis notebook
  (details in `ANALYSIS.md`).

## Limitations

We did not identify the exact cause of an environment-dependent difference (about ten times)
in the unsymmetrized acoustic residuals between the released data, our CPU runs, and the
issue-#89 rerun; the zone-boundary versus genuine-model-error split is measured on the
deepest subset of missed materials. See `ANALYSIS.md` for the full list of caveats.

## References

- MLIP Arena code and leaderboard: <https://github.com/atomind-ai/mlip-arena> · [issue #89](https://github.com/atomind-ai/mlip-arena/issues/89)
- Y. Chiang et al., "MLIP Arena: Advancing Fairness and Transparency in Machine Learning
  Interatomic Potentials via an Open, Accessible Benchmark Platform," NeurIPS 2025 (Datasets
  and Benchmarks Track); arXiv:2509.20630.
- S. Haastrup et al., "The Computational 2D Materials Database: high-throughput modeling and
  discovery of atomically thin crystals," 2D Mater. **5**, 042002 (2018),
  doi:10.1088/2053-1583/aacfc1.
- M. N. Gjerding et al., "Recent progress of the Computational 2D Materials Database (C2DB),"
  2D Mater. **8**, 044002 (2021), doi:10.1088/2053-1583/ac1059.

## License

MIT.
