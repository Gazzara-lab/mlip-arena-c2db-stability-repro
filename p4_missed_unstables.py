"""
P4 — Decompose the true-unstable monolayers that PASS the corrected protocol
(~115/218): how many were invisible by construction (instability living off-Gamma
or in the elastic channel) vs genuine MLIP misses?

Levers available without new DFT:
  - c2db.db `minhessianeig`: min eigenvalue of C2DB's DFT Hessian computed on a
    2x2 supercell -> sees Gamma AND the commensurate zone-boundary points. Its
    distribution by dyn_stab class self-calibrates the effective C2DB cut.
  - rhirota's symmetrized-run parquet: MLIP Gamma freq_min (fm) + MLIP elastic
    eigenvalues (emin) per material.
Decomposition of passing true-unstables:
  S  (stiffness-driven):  minhessianeig ~ 0 (phonon-side fine in DFT) -> C2DB
     "No" came from the stiffness tensor; check whether the MLIP elastic leg
     caught it (if not: elastic-leg model error, not invisibility).
  P  (phonon-driven): minhessianeig deep -> DFT sees an unstable mode in the
     2x2 Hessian. The MLIP Gamma check saw nothing: either a genuine model error
     at Gamma or a zone-boundary mode invisible to the Gamma-only observable.
     Depth-conditional miss-rate (caught vs missed vs |minhessianeig|) bounds
     how much is near-threshold physics vs hard failure.
Also emits the passed-stable background (ehull/gap/3d/magnetic) that closes P3.

Run on a machine with the data:  python p4_missed_unstables.py <nosym.zip> <sym.zip>
(c2db.db in the same dir as the script; zips auto-detected as in analyze_fullset)
Deps: pandas pyarrow numpy
"""
from __future__ import annotations
import glob, json, os, re, sqlite3, sys, tempfile, zipfile
from collections import Counter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = -0.01
D3 = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"}

def load_runs(zpaths):
    T = tempfile.mkdtemp()
    runs = {}
    for z in zpaths:
        with zipfile.ZipFile(z) as f:
            pq = [n for n in f.namelist() if n.endswith(".parquet")]
            if not pq:
                continue
            f.extract(pq[0], T)
            d = pd.read_parquet(os.path.join(T, pq[0]))
            d["fm"] = d["frequencies"].apply(lambda x: float(np.real(np.asarray(x)).min()))
            d["emin"] = d["eigenvalues"].apply(lambda x: float(np.real(np.asarray(x)).min()))
            d["na"] = d["frequencies"].apply(len) // 3
            key = "sym" if float(np.median(d["fm"])) > -1e-4 else "nosym"
            runs[key] = d[["uid", "fm", "emin", "na"]].drop_duplicates("uid")
    return runs

def elems(uid):
    return set(re.findall(r"[A-Z][a-z]?", re.sub(r"^\d+", "", uid.split("-")[0])))

def main(zpaths):
    sym = load_runs(zpaths)["sym"]
    con = sqlite3.connect(os.path.join(HERE, "c2db.db"))
    tk, nk = {}, {}
    for i, k, v in con.execute("SELECT id,key,value FROM text_key_values"):
        tk.setdefault(i, {})[k] = v
    for i, k, v in con.execute("SELECT id,key,value FROM number_key_values"):
        nk.setdefault(i, {})[k] = v
    con.close()
    meta = {d["uid"]: {**d, **nk.get(i, {})} for i, d in tk.items() if "uid" in d}

    # self-calibrate C2DB's effective minhessianeig cut from the whole db
    mh_yes = [m["minhessianeig"] for m in meta.values()
              if str(m.get("dyn_stab", "")).lower() == "yes" and "minhessianeig" in m]
    mh_no = [m["minhessianeig"] for m in meta.values()
             if str(m.get("dyn_stab", "")).lower() == "no" and "minhessianeig" in m]
    q = lambda a: np.percentile(a, [1, 10, 50, 90, 99]).round(4).tolist()
    print(f"minhessianeig db-wide: yes n={len(mh_yes)} q={q(mh_yes)} | no n={len(mh_no)} q={q(mh_no)}")

    sym = sym[sym["uid"].isin(meta)].copy()
    lab = sym["uid"].map(lambda u: str(meta[u].get("dyn_stab", "?")).lower())
    un = sym[lab == "no"].copy()
    st = sym[lab == "yes"].copy()
    un["passed"] = un["fm"] >= TOL
    print(f"true-unstable: {len(un)} | passed (missed): {int(un['passed'].sum())} "
          f"| caught: {int((~un['passed']).sum())}")

    # P3 closure: passed-stable background
    ps = st[st["fm"] >= TOL]
    bg = {
        "n": len(ps),
        "ehull_q": np.percentile([meta[u].get("ehull", np.nan) for u in ps["uid"]],
                                 [10, 25, 50, 75, 90]).round(3).tolist(),
        "frac_ehull_gt0.2": float(np.mean([meta[u].get("ehull", 0) > 0.2 for u in ps["uid"]])),
        "frac_metal": float(np.mean([meta[u].get("gap", 1) == 0 for u in ps["uid"]])),
        "frac_3d": float(np.mean([bool(elems(u) & D3) for u in ps["uid"]])),
        "frac_magnetic": float(np.mean([meta[u].get("is_magnetic", 0) == 1 for u in ps["uid"]])),
    }
    print("P3 background (passed stables):", json.dumps(bg))

    rows = []
    for r in un.itertuples():
        m = meta[r.uid]
        rows.append({"uid": r.uid, "fm": round(r.fm, 4), "emin": round(r.emin, 3),
                     "na": int(r.na), "passed": bool(r.passed),
                     "mh": (round(m["minhessianeig"], 4) if "minhessianeig" in m else None),
                     "ehull": round(m.get("ehull", float("nan")), 3),
                     "gap": round(m.get("gap", float("nan")), 2),
                     "mag": int(m.get("is_magnetic", 0))})
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "data", "p4_missed_unstables.json"), "w"), indent=2)
    print(f"\nWrote {len(rows)} true-unstable records to data/p4_missed_unstables.json")

if __name__ == "__main__":
    zp = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "*threshould*.zip")))
    main(zp)
