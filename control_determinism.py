"""
Control: where does the run-to-run variation at -1e-7 come from?

For a few borderline stable diatomics, compare:
  (A) relax ONCE, then recompute phonon+symmetrize K times on the FIXED geometry
      -> isolates force-evaluation / threading non-determinism
  (B) relax from scratch K times (each its own FIRE endpoint)
      -> isolates relaxation-endpoint sensitivity on a near-flat PES
Run:  python control_determinism.py
"""
from __future__ import annotations
import os, json
import numpy as np
from ase.db import connect
from repro_lib import get_mace_mp_medium, relax, phonon_freq_min

DB = os.path.join(os.path.dirname(__file__), "c2db.db")  # download into the repo root (see README)
BORDERLINE = [(9980, "1AsIn-1"), (10046, "1HgTe-1"), (10063, "1InP-1"), (10065, "1SnTe-1")]
K = 5


def spread(vals):
    return f"min={min(vals):+.3e} max={max(vals):+.3e} span={max(vals)-min(vals):.2e}"


def main():
    calc = get_mace_mp_medium()
    out = {}
    with connect(DB) as db:
        for sid, uid in BORDERLINE:
            atoms0 = db.get(sid).toatoms()
            # (A) fixed geometry: relax once, recompute phonons K times
            relaxed = relax(atoms0, calc)
            A = [phonon_freq_min(relaxed, calc)[1] for _ in range(K)]   # sym freq_min
            # (B) relax from scratch K times
            B = [phonon_freq_min(relax(atoms0, calc), calc)[1] for _ in range(K)]
            out[uid] = {"fixed_geometry_sym": A, "from_scratch_sym": B}
            print(f"{uid:10s} (A) fixed-geom phonon x{K}: {spread(A)}")
            print(f"{uid:10s} (B) re-relax    phonon x{K}: {spread(B)}")
            print(f"{'':10s}     -> variation is dominated by {'RELAXATION' if (max(B)-min(B))>10*(max(A)-min(A)+1e-30) else 'FORCES/threading'}\n")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "determinism.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
