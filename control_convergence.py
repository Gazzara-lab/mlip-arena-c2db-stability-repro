"""
Control: are the deep soft modes of BP / BSb / AsB genuine MACE-MP(M) predictions,
or artifacts of the benchmark's constrained relaxation + minimal supercell?

For each, compute Gamma freq_min under 4 settings:
  1. FixSym + (2,2,1)   = benchmark default (what the main run uses)
  2. noFixSym + (2,2,1) = allow out-of-plane buckling (planar honeycomb confound)
  3. FixSym + (3,3,1)   = larger supercell (FC range / q-sampling confound)
  4. noFixSym + (3,3,1) = both relaxed
If the deep mode survives all four -> genuine model prediction. If it vanishes when
buckling is allowed or the supercell grows -> constrained-relaxation/supercell artifact.
Run:  python control_convergence.py
"""
from __future__ import annotations
import os, json
from ase.db import connect
from repro_lib import get_mace_mp_medium, relax, phonon_freq_min

DB = os.path.join(os.path.dirname(__file__), "c2db.db")  # download into the repo root (see README)
DEEP = [(10004, "1BP-1"), (10138, "1BSb-1"), (10725, "1AsB-1")]   # ids resolved from c2db.db


def main():
    calc = get_mace_mp_medium()
    out = {}
    with connect(DB) as db:
        for sid, uid in DEEP:
            atoms0 = db.get(sid).toatoms()
            cfgs = {
                "FixSym_221":   (relax(atoms0, calc, fix_symmetry=True),  (2, 2, 1)),
                "noFixSym_221": (relax(atoms0, calc, fix_symmetry=False), (2, 2, 1)),
                "FixSym_331":   (relax(atoms0, calc, fix_symmetry=True),  (3, 3, 1)),
                "noFixSym_331": (relax(atoms0, calc, fix_symmetry=False), (3, 3, 1)),
            }
            res = {}
            for name, (a, sc) in cfgs.items():
                _, fs, nd = phonon_freq_min(a, calc, supercell=sc)
                res[name] = round(fs, 4)
            out[uid] = res
            print(f"{uid:8s} " + "  ".join(f"{k}={v:+.3f}" for k, v in res.items()), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "convergence.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
