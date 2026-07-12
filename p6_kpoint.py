"""
P6 - The K-point escalation: settle P5's one exception (2CoS2-1) and quantify a
"commensurate-complete" observable.

P5 showed 14/15 Gamma-passed true-unstables are unstable at the (2,2,1) zone
boundary (M). The lone exception, 2CoS2-1, is clean at Gamma AND at M yet DFT is
violently unstable (minhessianeig -4.7). A (2,2,1) supercell cannot represent the
K point (1/3,1/3,0): 1/3 is not a multiple of 1/2. A (3,3,1) supercell can.

This evaluates MLIP freq_min at the EXACT commensurate q-points of BOTH supercells
(no interpolation):
  - (2,2,1) force constants -> commensurate set {(i/2,j/2,0)} = Gamma + M-type
  - (3,3,1) force constants -> commensurate set {(i/3,j/3,0)} = Gamma + K-type + 1/3-line
The union (Gamma union M union K) is a high-symmetry-complete observable built only
from supercell force constants the benchmark-style pipeline already knows how to make.

Verdict per material:
  - if any commensurate q gives freq_min < -0.05 -> RESCUED at that q (which supercell
    would have caught it); the Gamma-only observable was blind by construction.
  - else -> still blind (instability is incommensurate, or a genuine MLIP-vs-DFT
    disagreement) -> not fixable by denser commensurate sampling alone.

Needs MACE + phonopy + ase + c2db.db (user's machine). Deps: mace-torch phonopy ase numpy.
Run: python p6_kpoint.py                 # 2CoS2-1 (the P5 exception)
     python p6_kpoint.py 2CoS2-1 1TaI2-1 # any uids; 1TaI2-1 is a hexagonal positive control
"""
from __future__ import annotations
import os, sqlite3, json, time, warnings, urllib.request
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
TOL = -0.05


def get_calc():
    from mace.calculators import MACECalculator
    url = ("https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
           "2023-12-03-mace-128-L1_epoch-199.model")
    cp = os.path.join(os.path.expanduser("~"), ".cache", "mace"); os.makedirs(cp, exist_ok=True)
    mp = os.path.join(cp, "".join(c for c in os.path.basename(url) if c.isalnum() or c == "_"))
    if not os.path.isfile(mp):
        print("downloading MACE model..."); urllib.request.urlretrieve(url, mp)
    return MACECalculator(model_paths=mp, device="cpu", default_dtype="float32")


def relax(atoms, calc):
    from ase.optimize import FIRE
    from ase.constraints import FixSymmetry
    a = atoms.copy(); a.calc = calc; a.set_constraint(FixSymmetry(a))
    FIRE(a, logfile=None).run(fmax=0.05, steps=500); a.set_constraint(); return a


def commensurate_fmin(atoms, calc, n):
    """Build (n,n,1) symmetrized FCs; return {q_tuple: freq_min} over the exact
    commensurate q-points (i/n, j/n, 0) and the overall min (excluding Gamma)."""
    from ase import Atoms
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    ph = Phonopy(PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.get_cell(),
                              scaled_positions=atoms.get_scaled_positions(wrap=True),
                              masses=atoms.get_masses()),
                 symprec=1e-5, supercell_matrix=(n, n, 1))
    ph.generate_displacements(distance=0.01)
    ds = [s for s in ph.supercells_with_displacements if s is not None]
    F = []
    for s in ds:
        b = Atoms(symbols=s.symbols, cell=s.cell, scaled_positions=s.scaled_positions, pbc=True)
        b.calc = calc; F.append(b.get_forces())
    ph.forces = F; ph.produce_force_constants()
    ph.symmetrize_force_constants(); ph.symmetrize_force_constants_by_space_group()
    res = {}
    for i in range(n):
        for j in range(n):
            q = (i / n, j / n, 0.0)
            res[q] = float(np.min(ph.get_frequencies(q=q)))
    nong = {q: v for q, v in res.items() if q != (0.0, 0.0, 0.0)}
    qmin = min(nong, key=nong.get)
    return res, qmin, nong[qmin], len(ds)


def main():
    import sys
    from ase.db import connect
    targets = sys.argv[1:] or ["2CoS2-1", "1TaI2-1"]  # 2CoS2-1 = the exception; 1TaI2-1 = hexagonal M-point positive control
    dbp = next((os.path.join(HERE, x) for x in ["c2db.db", "c2db"] if os.path.exists(os.path.join(HERE, x))), None)
    assert dbp, "c2db.db not found in this folder"
    con = sqlite3.connect(dbp)
    u2i = {v: i for i, k, v in con.execute("SELECT id,key,value FROM text_key_values WHERE key='uid'")}
    i2mh = {i: v for i, k, v in con.execute("SELECT id,key,value FROM number_key_values WHERE key='minhessianeig'")}
    i2lg = {i: v for i, k, v in con.execute("SELECT id,key,value FROM text_key_values WHERE key='layergroup'")}
    con.close()
    calc = get_calc(); adb = connect(dbp)
    out = []
    for uid in targets:
        if uid not in u2i:
            print(uid, "not in db"); continue
        sid = u2i[uid]; a0 = relax(adb.get(sid).toatoms(), calc)
        t0 = time.time()
        r2, q2, f2, nd2 = commensurate_fmin(a0, calc, 2)
        r3, q3, f3, nd3 = commensurate_fmin(a0, calc, 3)
        g = float(np.min([r2[(0.0, 0.0, 0.0)], r3[(0.0, 0.0, 0.0)]]))
        overall_q, overall = min([(q2, f2), (q3, f3)], key=lambda t: t[1])
        which = "M(2x2)" if overall_q == q2 else "K/1third(3x3)"
        isK = abs(overall_q[0] - 1 / 3) < 1e-6 and abs(overall_q[1] - 1 / 3) < 1e-6
        mh = i2mh.get(sid, float("nan")); lg = i2lg.get(sid, "?")
        if overall < TOL:
            verdict = f"RESCUED at q={tuple(round(x,3) for x in overall_q)} [{which}{' =K' if isK else ''}]"
        else:
            verdict = "still blind (incommensurate or genuine model error)"
        rec = {"uid": uid, "layergroup": lg, "mh_DFT": round(mh, 3), "fmin_gamma": round(g, 4),
               "fmin_M_2x2": round(f2, 4), "qM": tuple(round(x, 3) for x in q2),
               "fmin_K_3x3": round(f3, 4), "qK": tuple(round(x, 3) for x in q3),
               "overall_min": round(overall, 4), "verdict": verdict,
               "all_2x2": {str(tuple(round(x, 2) for x in q)): round(v, 3) for q, v in r2.items()},
               "all_3x3": {str(tuple(round(x, 2) for x in q)): round(v, 3) for q, v in r3.items()},
               "sec": round(time.time() - t0, 1)}
        out.append(rec)
        print(f"\n{uid}  (layergroup {lg}, DFT minhessianeig {mh:.2f})")
        print(f"  fmin  Gamma={g:+.4f}   M-set(2x2)={f2:+.4f} @ {rec['qM']}   K-set(3x3)={f3:+.4f} @ {rec['qK']}")
        print(f"  => {verdict}", flush=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "data", "p6_kpoint_results.json"), "w"), indent=2)
    print("\nWrote results to data/p6_kpoint_results.json")


if __name__ == "__main__":
    main()
