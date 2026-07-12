"""
Forensics for atomind-ai/mlip-arena issue #89: which (thres) reproduces Fig. S10
from the paper's OWN released per-material data?

The PyPI sdist of mlip-arena (>=0.1.1) ships `benchmarks/c2db/` with the released
per-model parquets, analysis.ipynb (thres=-1e-7) and run.py (PHONON called without
`symmetry`, i.e. symmetry=False default). The parquets store the RAW Gamma-point
frequency arrays of the paper's run, so the effective operating point of Fig. S10
can be measured directly by re-running the released notebook logic at swept thres.

Inputs (put in repo root; verify hashes below):
  c2db.db              sha256 caf58205692de480e06149ac43a437385f18e14582e7d9a8dab8b3cb5d4bd678 (70762496 B)
  MACE-MP(M).parquet   sha256 f722eac6799bfecaa02188d59475862895a639cc596fa8b7d1e9d2b96cfb415b (293633 B)
    from https://huggingface.co/spaces/atomind/mlip-arena/resolve/main/benchmarks/c2db/MACE-MP(M).parquet
  copy.parquet (optional, 21349 B, only in v0.1.1 sdist / repo history)
    sha256 7fdc16667361b10bfb032862d5d0610c242d75cb88f7f3883c43a406b245e991

Deps: pandas pyarrow ase scikit-learn numpy
Run:  python forensic_operating_point.py
      python forensic_operating_point.py CHGNet.parquet a b c d
        (a..d = that model's Fig. S10 cells, read from the released
         c2db-confusion_matrices.pdf: stable->stable, stable->unstable,
         unstable->stable, unstable->unstable. Other model parquets are in the
         same sdist/HF dir — the cross-model check needs NO new compute.)
"""
from __future__ import annotations
import hashlib, json, os, random, sys, warnings
import numpy as np
import pandas as pd
from ase.db import connect
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "c2db.db")
PQ = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "MACE-MP(M).parquet")
# Fig. S10 confusion matrix for this model (default: MACE-MP(M) per the #89 thread)
# rows: true stable, true unstable; cols: pred stable, pred unstable
PAPER = (np.array([int(x) for x in sys.argv[2:6]]).reshape(2, 2)
         if len(sys.argv) >= 6 else np.array([[43, 244], [21, 196]]))

KNOWN = {"c2db.db": "caf58205", "MACE-MP(M).parquet": "f722eac6"}
for path in (DB, PQ):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    want = KNOWN.get(os.path.basename(path))
    tag = "OK" if (want and h.startswith(want)) else ("MISMATCH — wrong file!" if want else "unverified")
    print(f"{os.path.basename(path)}: sha256 {h[:8]}... ({tag})")

# --- replicate the released analysis.ipynb ground-truth block verbatim ---
db = connect(DB)
n = len(db)
random.seed(0)
idx = set(random.sample(range(1, n + 1), 1000))
uids, stab = [], []
for row in db.select(filter=lambda r: r["id"] in idx):
    s = row.key_value_pairs["dyn_stab"]
    s = None if s.lower() == "unknown" else (s.lower() == "yes")
    uids.append(row.key_value_pairs["uid"])
    stab.append(s)
stab = np.array(stab)
print(f"\ndb len={n}; labeled draw: {sum(s is True for s in stab)} stable / "
      f"{sum(s is False for s in stab)} unstable / {sum(s is None for s in stab)} unknown")

df = pd.read_parquet(PQ)
print(f"parquet rows={len(df)}; in draw={df['uid'].isin(uids).sum()}")

fmin = df["frequencies"].apply(lambda x: np.min(x) if np.isreal(x).all() else np.nan).dropna()
print(f"\nRAW paper freq_min over {len(fmin)}: max={fmin.max():+.5f} THz  median={np.median(fmin):+.5f}"
      f"  >=0: {(fmin >= 0).sum()}  in(-0.1,0): {((fmin > -0.1) & (fmin < 0)).sum()}  <=-0.1: {(fmin <= -0.1).sum()}")
print("(compare vs rhirota2001 full rerun: max=-0.0022, median=-0.101, 494 in (-0.1,0))")

arg = np.argsort(uids)
U, S = np.array(uids)[arg], stab[arg]
mask = ~(S == None)  # noqa: E711  (notebook-verbatim)

def cm3(thres: float) -> np.ndarray:
    d = df.copy()
    d["eigval_min"] = d["eigenvalues"].apply(lambda x: x.min() if np.isreal(x).all() else thres)
    d["freq_min"] = d["frequencies"].apply(lambda x: x.min() if np.isreal(x).all() else thres)
    d["pred"] = ~np.logical_or(d["eigval_min"] < thres, d["freq_min"] < thres)
    sd = d[d["uid"].isin(U)].set_index("uid").reindex(U)
    y_true = S[mask].astype(int)
    y_pred = sd["pred"][mask].fillna(-1).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[1, 0, -1])  # 3rd class = "missing"

c = cm3(-1e-7)
print(f"\nreleased notebook verbatim (thres=-1e-7):\n{c}")
print(f"missing column (materials in draw without parquet row): stable={c[0,2]}, unstable={c[1,2]}"
      "  <- candidate explanation for the 217-vs-218 row sum in the thread")

print("\n=== thres sweep on the PAPER'S OWN raw values vs Fig. S10 ===")
grid = sorted(set([-1e-7, -1e-4, -1e-3] + list(-np.geomspace(5e-3, 1e-1, 25))), reverse=True)
best = (1e9, None, None)
for t in grid:
    c = cm3(t)
    dist = int(np.abs(c[:2, :2] - PAPER).sum())
    if dist < best[0]:
        best = (dist, t, c)
    if dist <= 8:
        print(f"thres={t:+.5f}: [[{c[0,0]:3d},{c[0,1]:3d}],[{c[1,0]:3d},{c[1,1]:3d}]] miss=({c[0,2]},{c[1,2]}) L1={dist}")
print(f"\nBEST thres={best[1]:+.5f}  L1={best[0]}\n{best[2]}")
if best[0] == 0:
    lo = hi = best[1]
    for t in grid:
        if int(np.abs(cm3(t)[:2, :2] - PAPER).sum()) == 0:
            lo, hi = min(lo, t), max(hi, t)
    print(f"exact-match thres range: [{lo:+.5f}, {hi:+.5f}]")

# --- per-material overlap vs the local CPU rerun (environment dependence) ---
loc = json.load(open(os.path.join(HERE, "results_main.json")))["rows"]
dfi = df.set_index("uid")
print("\n=== paper parquet (GPU, Perlmutter) vs local CPU rerun, shared uids ===")
for r in loc:
    if r["uid"] in dfi.index and "freq_min_nosym" in r:
        x = dfi.loc[r["uid"], "frequencies"]
        pf = np.min(x) if np.isreal(x).all() else float("nan")
        print(f"{r['uid']:14s} gt={r['gt']:3s} paper={pf:+.5f}  local_nosym={r['freq_min_nosym']:+.5f}"
              f"  local_sym={r['freq_min_sym']:+.3e}")

cp_path = os.path.join(HERE, "copy.parquet")
if os.path.exists(cp_path):
    cp = pd.read_parquet(cp_path)
    print(f"\ncopy.parquet: rows={len(cp)} cols={list(cp.columns)} "
          f"models={cp['model'].unique().tolist() if 'model' in cp else '?'}")
