"""
Analysis of rhirota2001's full-set result bundles (issue #89, comment of 2026-07-10):
  MACE-MP(M)_threshould_nosymmetry.zip  (original run, 998 UIDs, symmetry=False)
  MACE-MP(M)_threshould_symmetry.zip    (fresh run,   997 UIDs, symmetry=True)

Answers, from @rhirota2001's raw per-material values:
  Q1  does the unsymmetrized acoustic residual grow with cell size (natoms)?
      -> tests whether the clean gap seen in our natoms<=6 stratified subset is a
         coverage effect (@rhirota2001's correction) rather than a contradiction
  Q2  sign structure + per-material comparison vs our local CPU run (environment)
  Q4a size of the flip zone at thres=-1e-7 on the symmetrized run, and the predicted
      run-to-run sd of the reported 135-stable count using our measured noise floor (determinism.json)
  Q5b which materials make up the plateau's stable->unstable residual (~30):
      deep genuine soft modes (<-0.1 THz) or intermediate?
Also writes the summary figure -> fullset_freqmin.png

Usage:  python analyze_fullset.py <dir-or-zip nosym> <dir-or-zip sym>
Deps: pandas pyarrow matplotlib scipy numpy (+ c2db.db in repo root for GT labels)
"""
from __future__ import annotations
import glob, json, os, sys, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

def load_freqmin(src: str) -> pd.DataFrame:
    """Return DataFrame(uid, freq_min, natoms) from a result dir or zip."""
    if src.endswith(".zip"):
        out = src[:-4]
        if not os.path.isdir(out):
            zipfile.ZipFile(src).extractall(out)
        src = out
    pqs = glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True)
    assert pqs, f"no parquet under {src}"
    frames = []
    for p in pqs:
        d = pd.read_parquet(p)
        if "frequencies" in d.columns:            # raw arrays (paper schema)
            d["freq_min"] = d["frequencies"].apply(
                lambda x: np.min(x) if np.isreal(x).all() else np.nan)
            d["natoms"] = d["frequencies"].apply(len) // 3
            frames.append(d[["uid", "freq_min", "natoms"]])
        elif "freq_min" in d.columns:             # pre-reduced scalar
            d["natoms"] = np.nan
            frames.append(d[["uid", "freq_min", "natoms"]])
    df = pd.concat(frames).drop_duplicates("uid").dropna(subset=["freq_min"])
    print(f"{os.path.basename(src)}: {len(df)} materials from {len(pqs)} parquet(s)")
    return df

def gt_labels() -> dict:
    dbp = os.path.join(HERE, "c2db.db")
    if not os.path.exists(dbp):
        print("(c2db.db not found -> skipping label-dependent sections)")
        return {}
    from ase.db import connect
    lab = {}
    with connect(dbp) as db:
        for row in db.select("dyn_stab"):
            v = row.key_value_pairs.get("dyn_stab", "").lower()
            if v in ("yes", "no"):
                lab[row.key_value_pairs.get("uid")] = (v == "yes")
    return lab

nosym = load_freqmin(sys.argv[1])
sym = load_freqmin(sys.argv[2])
# rhirota2001's attachment filenames are swapped (nosymmetry.zip holds the
# symmetrized run and vice versa) — detect by the symmetrized signature
# (acoustic residuals collapsed to ~1e-7) and fix silently:
if np.median(nosym["freq_min"]) > -1e-4 > np.median(sym["freq_min"]):
    print("NOTE: inputs look swapped (arg1 is the symmetrized run) — swapping.")
    nosym, sym = sym, nosym
lab = gt_labels()

# --- Q2: sign structure & distribution ---
for name, d in [("nosym", nosym), ("sym", sym)]:
    f = d["freq_min"]
    print(f"\n[{name}] max={f.max():+.5f} median={np.median(f):+.3e}  >=0: {(f>=0).sum()}/{len(f)}"
          f"  in(-0.1,0): {((f>-0.1)&(f<0)).sum()}  <=-0.1: {(f<=-0.1).sum()}")

loc = {r["uid"]: r for r in json.load(open(os.path.join(HERE, "results_main.json")))["rows"]
       if "freq_min_nosym" in r}
shared = [u for u in loc if u in set(nosym["uid"])]
if shared:
    ni, si = nosym.set_index("uid"), sym.set_index("uid")
    print(f"\n[Q2] shared uids with local CPU run: {len(shared)}")
    for u in shared:
        r = loc[u]
        sv = f"{si.loc[u,'freq_min']:+.3e}" if u in si.index else "   n/a"
        print(f"  {u:14s} his_nosym={ni.loc[u,'freq_min']:+.5f} local_nosym={r['freq_min_nosym']:+.5f}"
              f"  his_sym={sv} local_sym={r['freq_min_sym']:+.3e}")

# --- Q1: residual magnitude vs natoms (nosym run, residual population only) ---
res = nosym[(nosym["freq_min"] > -0.1) & (nosym["freq_min"] < 0)].dropna(subset=["natoms"])
if len(res):
    print("\n[Q1] |unsymmetrized residual| vs natoms (population in (-0.1, 0)):")
    for lo, hi in [(1, 4), (5, 6), (7, 9), (10, 14), (15, 99)]:
        g = res[(res["natoms"] >= lo) & (res["natoms"] <= hi)]
        if len(g):
            a = -g["freq_min"]
            print(f"  natoms {lo:2d}-{hi:2d}: n={len(g):3d}  median={np.median(a):.4f}"
                  f"  p90={np.percentile(a,90):.4f}  max={a.max():.4f}")

# --- Q4a: flip zone at -1e-7 on the symmetrized run ---
det = json.load(open(os.path.join(HERE, "determinism.json")))
noise = np.concatenate([np.array(v["fixed_geometry_sym"]) - np.mean(v["fixed_geometry_sym"])
                        for v in det.values()])
sigma = float(np.std(noise, ddof=1))
print(f"\n[Q4a] measured noise floor sigma (fixed-geometry, from determinism.json): {sigma:.2e} THz")
try:
    from scipy.stats import norm
    cdf = lambda z: norm.cdf(z)
except ImportError:
    from math import erf
    cdf = lambda z: 0.5 * (1 + erf(z / np.sqrt(2)))
pool = sym if not lab else sym[sym["uid"].map(lab) == True]  # true-stables if labels available
mu = pool["freq_min"].values
p_pass = 1 - cdf((-1e-7 - mu) / sigma)
exp, sd = p_pass.sum(), float(np.sqrt((p_pass * (1 - p_pass)).sum()))
inzone = int(((p_pass > 0.02) & (p_pass < 0.98)).sum())
print(f"  over {len(pool)} {'true-stable' if lab else 'ALL'} materials: expected pass at -1e-7 = "
      f"{exp:.0f} +/- {sd:.1f} (1sd); {inzone} materials are genuine coin-flips"
      f"\n  -> the reported 135-stable count should wander by ~{2*sd:.0f} (2sd) between reruns; the plateau should not.")

# --- Q5b: composition of the plateau residual (stable->unstable at -0.01) ---
if lab:
    st = sym[sym["uid"].map(lab) == True]
    fail = st[st["freq_min"] < -0.01]
    deep = (fail["freq_min"] <= -0.1).sum()
    print(f"\n[Q5b] true-stable failing at plateau (-0.01): {len(fail)}"
          f" -> deep (<=-0.1): {deep}, intermediate (-0.1..-0.01): {len(fail)-deep}")
    print("  deep list:", ", ".join(f"{r.uid}({r.freq_min:+.2f})" for r in
                                    fail[fail['freq_min'] <= -0.1].itertuples()))

# --- figure ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True, layout="constrained")
for ax, (name, d) in zip(axs, [("symmetry=False", nosym), ("symmetry=True", sym)]):
    f = np.abs(d["freq_min"][d["freq_min"] < 0])
    ax.hist(np.log10(np.clip(f, 1e-9, None)), bins=60, color="#4878b0")
    for t, c, l in [(1e-7, "crimson", "-1e-7 (released)"), (1e-2, "darkorange", "-1e-2"),
                    (2.5e-2, "green", "-0.025"), (1e-1, "gray", "-0.1")]:
        ax.axvline(np.log10(t), color=c, ls="--", lw=1, label=l)
    ax.set_title(f"{name} (n={len(d)})"); ax.set_xlabel("log10 |freq_min| (THz), negatives only")
axs[0].set_ylabel("materials"); axs[0].legend(fontsize=7)
fig.suptitle("Gamma freq_min: acoustic residuals vs genuine soft modes (MACE-MP(M), full set)")
fig.savefig(os.path.join(HERE, "fullset_freqmin.png"), dpi=160)
print("\nwrote fullset_freqmin.png")
