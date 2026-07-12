"""
P3 — Characterize the true-stable monolayers that the CORRECTED protocol still
condemns (deep soft modes under symmetry=True), looking for a chemical /
structural pattern. Continues @rhirota2001's point 5 in issue #89: @rhirota2001's
Table B leaves ~30 stable->unstable at the plateau, uncharacterized.

Inputs:
  data/rhirota_sym_freqmin.csv  (in this repo; derived from rhirota2001's
                                 issue-#89 attachment, symmetrized run, 997 uids)
  c2db.db                       (repo root; sha256 caf58205...)

Method: label each material with C2DB dyn_stab; among true-stables, split
condemned (freq_min < -0.01 THz, threshold-insensitive region) vs passed;
report per-material metadata and enrichment of elements / C2DB categories in
condemned vs passed.

Deps: pandas numpy.  Run: python p3_deep_stables.py
"""
from __future__ import annotations
import json, os, re, sqlite3
from collections import Counter
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = -0.01  # plateau tolerance (verdicts insensitive across -0.01..-0.1)

sym = pd.read_csv(os.path.join(HERE, "data", "rhirota_sym_freqmin.csv"))

con = sqlite3.connect(os.path.join(HERE, "c2db.db"))
tk, nk = {}, {}
for i, k, v in con.execute("SELECT id,key,value FROM text_key_values"):
    tk.setdefault(i, {})[k] = v
for i, k, v in con.execute("SELECT id,key,value FROM number_key_values"):
    nk.setdefault(i, {})[k] = v
con.close()
meta = {}
for i, d in tk.items():
    if "uid" in d:
        meta[d["uid"]] = {**d, **nk.get(i, {})}

ckeys = Counter(k for d in meta.values() for k in d)
print("available metadata keys:", sorted(ckeys))

sym = sym[sym["uid"].isin(meta)].copy()
sym["stab"] = sym["uid"].map(lambda u: str(meta[u].get("dyn_stab", "?")).lower())
st = sym[sym["stab"] == "yes"].copy()
st["cond"] = st["freq_min"] < TOL
nC, nP = int(st["cond"].sum()), int((~st["cond"]).sum())
print(f"\ntrue-stable in sym run: {len(st)} | condemned (freq_min<{TOL}): {nC} | passed: {nP}")

COLS = [k for k in ["magstate", "crystal_type", "class", "spacegroup", "layergroup",
                    "gap", "gap_hse", "hform", "ehull", "thermodynamic_stability_level",
                    "is_magnetic", "nspecies"] if k in ckeys]

def elems(uid: str) -> list[str]:
    return re.findall(r"[A-Z][a-z]?", re.sub(r"^\d+", "", uid.split("-")[0]))

print("\n=== condemned true-stables (deepest first) ===")
rows = []
for r in st[st["cond"]].sort_values("freq_min").itertuples():
    m = meta[r.uid]
    rec = {"uid": r.uid, "freq_min": round(r.freq_min, 3), "natoms": int(r.natoms),
           **{c: m.get(c) for c in COLS}}
    rows.append(rec)
    print(rec)

print("\n=== element enrichment (condemned vs passed true-stables) ===")
ce, pe = Counter(), Counter()
for r in st.itertuples():
    for e in set(elems(r.uid)):
        (ce if r.cond else pe)[e] += 1
enr = []
for e in set(ce) | set(pe):
    a, b = ce.get(e, 0), pe.get(e, 0)
    if a + b >= 3:
        enr.append((e, a, b, (a / max(nC, 1)) / ((b + 0.5) / max(nP, 1))))
for e, a, b, r in sorted(enr, key=lambda x: -x[3])[:18]:
    print(f"  {e:3s} condemned {a:2d}/{nC}  passed {b:3d}/{nP}  enrichment x{r:.1f}")

for c in COLS:
    vc, vp = Counter(), Counter()
    for r in st.itertuples():
        v = meta[r.uid].get(c)
        if v is not None:
            (vc if r.cond else vp)[str(v)[:14]] += 1
    if 0 < len(vc) <= 14:
        print(f"\n{c}: condemned={dict(vc.most_common())} | passed(top)={dict(vp.most_common(6))}")

os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
json.dump(rows, open(os.path.join(HERE, "data", "p3_condemned_stables.json"), "w"), indent=2)
print(f"\nWrote {len(rows)} condemned true-stable records to data/p3_condemned_stables.json")
