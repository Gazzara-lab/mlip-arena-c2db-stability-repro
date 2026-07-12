"""
Self-contained forensic check for atomind-ai/mlip-arena issue #89.

Needs only two files in the SAME folder as this script:
  - c2db.db              (the C2DB database, ~67 MB)
  - MACE-MP(M).parquet   (the paper's released per-material results)
Auto-detects them (any *.db and the MACE parquet). No ASE / sklearn needed.

Deps:  pip install pandas pyarrow numpy
Run:   python forensic_standalone.py
"""
import glob, hashlib, os, random, sqlite3, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
PAPER = np.array([[43, 244], [21, 196]])  # Fig. S10 MACE-MP(M): stable/unstable x stable/unstable
THRES = -1e-7


def find(patterns, label):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(HERE, pat)))
        if hits:
            return hits[0]
    sys.exit(f"ERROR: could not find {label} (looked for {patterns}) in {HERE}")


db_path = find(["c2db.db", "c2db", "*.db"], "the C2DB database")
pq_path = find(["MACE-MP(M).parquet", "*MACE*M*.parquet", "*MACE*.parquet", "*.parquet"],
               "the MACE-MP(M) parquet")
for p in (db_path, pq_path):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]
    print(f"using {os.path.basename(p):22s} sha256 {h}...")
print(f"  (expected: c2db.db=caf58205, MACE-MP(M).parquet=f722eac6)\n")

# --- replicate analysis.ipynb ground-truth block with plain sqlite3 ---
con = sqlite3.connect(db_path)
n = con.execute("SELECT COUNT(*) FROM systems").fetchone()[0]
kv = {}
for sid, key, val in con.execute(
        "SELECT id, key, value FROM text_key_values WHERE key IN ('uid','dyn_stab')"):
    kv.setdefault(sid, {})[key] = val
con.close()

random.seed(0)
idx = set(random.sample(range(1, n + 1), 1000))
uids, stab = [], []
for sid in idx:
    d = kv.get(sid)
    if not d or "dyn_stab" not in d or "uid" not in d:
        continue
    s = d["dyn_stab"].lower()
    if s == "unknown":
        lab = None
    else:
        lab = (s == "yes")
    uids.append(d["uid"])
    stab.append(lab)
uids = np.array(uids)
stab = np.array(stab, dtype=object)
print(f"db has {n} systems; drawn 1000; labeled: "
      f"{sum(s is True for s in stab)} stable / {sum(s is False for s in stab)} unstable / "
      f"{sum(s is None for s in stab)} unknown")

df = pd.read_parquet(pq_path)
print(f"parquet rows={len(df)}; in draw={df['uid'].isin(uids).sum()}\n")


def arr_min(x, thres):
    a = np.asarray(x)
    if np.iscomplexobj(a) and np.any(np.abs(a.imag) > 0):
        return thres  # non-real -> notebook maps to thres (counts as pass)
    return float(np.real(a).min())


def confusion(thres):
    d = df.copy()
    d["fmin"] = d["frequencies"].apply(lambda x: arr_min(x, thres))
    d["emin"] = d["eigenvalues"].apply(lambda x: arr_min(x, thres))
    d["pred"] = ~((d["emin"] < thres) | (d["fmin"] < thres))  # True=stable
    pred = dict(zip(d["uid"], d["pred"]))
    cm = np.zeros((2, 3), dtype=int)  # rows: true stable/unstable; cols: pred stable/unstable/missing
    for u, lab in zip(uids, stab):
        if lab is None:
            continue
        r = 0 if lab else 1
        if u not in pred:
            c = 2
        else:
            c = 0 if pred[u] else 1
        cm[r, c] += 1
    return cm


cm = confusion(THRES)
print(f"=== released threshold {THRES:g} on the paper's own parquet ===")
print("            pred_stable  pred_unstable  missing")
print(f"true stable   {cm[0,0]:8d} {cm[0,1]:12d} {cm[0,2]:9d}")
print(f"true unstable {cm[1,0]:8d} {cm[1,1]:12d} {cm[1,2]:9d}")
print(f"\nPaper Fig. S10 MACE-MP(M): [[43,244,0],[21,196,1]]")
match = np.array_equal(cm[:, :2], PAPER) and cm[0, 2] == 0 and cm[1, 2] == 1
print("EXACT MATCH -> Fig. S10 == released -1e-7 on the released data.\n" if match
      else "(not an exact match; see sweep below)\n")

print("=== threshold sweep vs Fig. S10 ===")
best = (10**9, None, None)
for t in [-1e-7, -1e-4, -1e-3, -5e-3, -1e-2, -1.5e-2, -2e-2, -2.5e-2, -3e-2, -5e-2, -1e-1]:
    c = confusion(t)
    dist = int(np.abs(c[:, :2] - PAPER).sum())
    if dist < best[0]:
        best = (dist, t, c)
    print(f"thres={t:+.5f}: [[{c[0,0]:3d},{c[0,1]:3d}],[{c[1,0]:3d},{c[1,1]:3d}]] "
          f"miss=({c[0,2]},{c[1,2]}) L1_vs_paper={dist}")
print(f"\nBEST thres={best[1]:+g}  (L1 distance {best[0]} from Fig. S10)")

f = df["frequencies"].apply(lambda x: arr_min(x, np.nan)).dropna()
print(f"\nraw paper freq_min: max={f.max():+.5f}  median={np.median(f):+.5f}  "
      f">=0: {(f>=0).sum()}/{len(f)}  (rhirota's rerun: max -0.0022, 0 positive)")
print("\nDone. The two blocks above reproduce Fig. S10 and the threshold sweep.")
