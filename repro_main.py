"""
Main reproduction for atomind-ai/mlip-arena issue #89 (phonon / freq_min leg only).

Stratified subset across cell sizes (natoms 2-6), single deterministic run, full
tolerance sweep. Every number printed here is regenerable from this one script + c2db.db.
Run:  python repro_main.py
"""
from __future__ import annotations
import os, sqlite3, json, time
from ase.db import connect
from repro_lib import get_mace_mp_medium, relax, phonon_freq_min

DB = os.path.join(os.path.dirname(__file__), "c2db.db")  # download into the repo root (see README)
TOLS = [-1e-7, -1e-4, -1e-3, -1e-2, -1e-1]      # -1e-7 is the benchmark's current cut
NATOMS = [2, 3, 4, 5, 6]
PER_BUCKET_STABLE = 3
PER_BUCKET_UNSTABLE = 2


def pick_stratified(db_path, value, per_bucket):
    con = sqlite3.connect(db_path)
    out = []
    for na in NATOMS:
        q = """SELECT s.id, tk.value FROM systems s
               JOIN text_key_values t ON t.id=s.id AND t.key='dyn_stab' AND t.value=?
               LEFT JOIN text_key_values tk ON tk.id=s.id AND tk.key='uid'
               WHERE s.natoms=? ORDER BY s.id ASC LIMIT ?"""
        out += [(i, u, value, na) for i, u in con.execute(q, (value, na, per_bucket)).fetchall()]
    con.close()
    return out


def main():
    stable = pick_stratified(DB, "Yes", PER_BUCKET_STABLE)
    unstable = pick_stratified(DB, "No", PER_BUCKET_UNSTABLE)
    sample = stable + unstable
    assert sample, "empty sample — check c2db.db schema (dyn_stab/uid keys)"
    print(f"DB={os.path.abspath(DB)}")
    print(f"stratified sample: {len(stable)} stable + {len(unstable)} unstable, natoms in {NATOMS}\n")
    calc = get_mace_mp_medium()
    rows = []
    with connect(DB) as db:
        for sid, uid, gt, na in sample:
            t0 = time.time()
            try:
                atoms = relax(db.get(sid).toatoms(), calc)          # OPT mirror: FixSym, positions-only
                fn, fs, nd = phonon_freq_min(atoms, calc)
                r = {"uid": uid, "gt": gt, "natoms": na, "ndisp": nd,
                     "freq_min_nosym": fn, "freq_min_sym": fs, "sec": round(time.time()-t0, 1)}
            except Exception as e:
                r = {"uid": uid, "gt": gt, "natoms": na, "error": repr(e)}
            rows.append(r)
            msg = (f"  {uid:12s} N={na} gt={gt:3s} | nosym {r.get('freq_min_nosym',0):+.4f} "
                   f"| sym {r.get('freq_min_sym',0):+.4f}" if "error" not in r
                   else f"  {uid:12s} ERROR {r['error']}")
            print(msg, flush=True)

    ok = [r for r in rows if "error" not in r]
    stab = [r for r in ok if r["gt"] == "Yes"]
    unst = [r for r in ok if r["gt"] == "No"]
    deep = [r for r in stab if r["freq_min_sym"] < -1e-1]

    def cnt(rs, key, T): return sum(r[key] >= T for r in rs)
    print("\n=== TOLERANCE SWEEP (phonon freq_min leg only; Gamma point) ===")
    print(f"{'tol(THz)':>9} | {'stable sym':>10} {'stable nosym':>12} | {'unstable sym':>12} {'unstable nosym':>14}")
    print("-"*70)
    for T in TOLS:
        print(f"{T:>9.0e} | {cnt(stab,'freq_min_sym',T):>4}/{len(stab):<5} {cnt(stab,'freq_min_nosym',T):>5}/{len(stab):<6}"
              f"| {cnt(unst,'freq_min_sym',T):>5}/{len(unst):<6} {cnt(unst,'freq_min_nosym',T):>6}/{len(unst):<7}")
    print(f"\nstable structural ceiling: {len(stab)-len(deep)}/{len(stab)} (the {len(deep)} deep-mode "
          f"stable monolayers {[r['uid'] for r in deep]} can never pass any tol)")
    print(f"by cell size: " + ", ".join(f"N{na}:{sum(1 for r in stab if r['natoms']==na)}stab/"
          f"{sum(1 for r in unst if r['natoms']==na)}unstab" for na in NATOMS))

    json.dump({"sample_n": {"stable": len(stab), "unstable": len(unst)}, "tols": TOLS, "rows": rows},
              open(os.path.join(os.path.dirname(__file__), "results_main.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
