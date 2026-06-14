# mlip-arena · C2DB dynamical-stability reproduction

An independent, inference-only reproduction of the **phonon (Γ-point `freq_min`) leg** of the
C2DB dynamical-stability task in [`atomind-ai/mlip-arena`](https://github.com/atomind-ai/mlip-arena),
prompted by [issue #89](https://github.com/atomind-ai/mlip-arena/issues/89). It quantifies why the
released `benchmarks/c2db/run.py` does not reproduce the paper's Figure S10 and separates a
numerical artifact from genuine model–DFT disagreement.

Everything here runs on CPU with `MACE-MP(M)` (`float32`). It mirrors the numerics of
`mlip_arena/tasks/phonon.py` (`symprec=1e-5`, `distance=0.01`, `(2,2,1)` supercell, the same
`symmetrize_force_constants()` + `symmetrize_force_constants_by_space_group()` calls,
Γ-point `get_frequencies`) and the `OPT` relaxation in `run.py`
(`FIRE`, `fmax=0.05`, `FixSymmetry`, positions-only).

## Scope (read first)

- **Phonon leg only.** The c2db analysis labels a material unstable if **either**
  `eigval_min < tol` **or** `freq_min < tol`. This reproduces only the Γ-point `freq_min` leg
  (the elastic leg was already shown to be healthy in the issue). Counts here are phonon-pass
  counts, not full-classifier labels.
- **Small, stratified subset**, not the random 1000-UID draw: 15 `dyn_stab=Yes` + 10 `dyn_stab=No`
  monolayers, 3/2 per cell size for `natoms ∈ {2,3,4,5,6}`. This is an **existence proof** of the
  mechanism across cell sizes.
- **Γ-only observable** (matching the benchmark); zone-boundary / flexural instabilities are not
  captured.

## Findings

1. **The `-1e-7` THz cut is below the numerical noise floor.** At `-1e-7` the stable/unstable
   verdict is decided by `float32` force noise: with the geometry held fixed, recomputing the
   phonon gives Γ `freq_min` spanning `~1e-7` THz across identical runs (`control_determinism.py`),
   so the `-1e-7` row of the sweep is not reproducible.
2. **The threshold is the primary lever; symmetrization is the complement.** `run.py` calls
   `PHONON(...)` without `symmetry=True`, so it uses the `symmetry=False` default and leaves
   `~1e-3`–`1e-2` THz acoustic residuals. Loosening the tolerance to `-1e-2` (with `symmetry=True`)
   recovers the stable class to its structural ceiling (`11/15` here); at the pathological `-1e-7`
   cut, symmetrization actually lowers the pass count, because it centres the acoustic modes on a
   numerical zero that sits just below the cut.
3. **The deep soft modes are genuine model–DFT disagreements, not artifacts.** For BP/BSb/AsB
   (planar honeycombs) the deep Γ mode is invariant to allowing buckling (`FixSymmetry` off) and to
   enlarging the supercell to `(3,3,1)` (`control_convergence.py`).

See [issue #89](https://github.com/atomind-ai/mlip-arena/issues/89) for the full write-up.

## Suggested fix

1. **Primary:** report a tolerance sweep, or move the stability cut from `-1e-7` to a pragmatic
   tolerance (e.g. `-1e-2` THz).
2. **Complementary:** pass `symmetry=True` to the `PHONON` call in `benchmarks/c2db/run.py` so the
   released script matches `phonon.py`'s symmetrized path.

## Reproduce

```bash
pip install -r requirements.txt
# download c2db.db (~67 MB) from the mlip-arena HF Space into ./c2db.db
#   https://huggingface.co/spaces/atomind/mlip-arena/blob/main/benchmarks/c2db/c2db.db
python repro_main.py            # stratified tolerance sweep      -> results_main.json
python control_determinism.py   # noise-floor control            -> determinism.json
python control_convergence.py   # deep-mode convergence control  -> convergence.json
```

The scripts read `c2db.db` from the repo root by default. Committed `*.json` are reference outputs
from one run (the `-1e-7` counts shift by ±1–2 between runs, by design — see finding 1).

## Files

| file | what it does |
|---|---|
| `repro_lib.py` | shared helpers: MACE-MP(M) loader, OPT-matching relaxation, phonon `freq_min` (no-sym & sym) |
| `repro_main.py` | stratified subset, full tolerance sweep (4 columns) → `results_main.json` |
| `control_determinism.py` | fixed-geometry vs re-relax `freq_min` spread → `determinism.json` |
| `control_convergence.py` | BP/BSb/AsB under {FixSym on/off} × {(2,2,1),(3,3,1)} → `convergence.json` |

## License

MIT.
