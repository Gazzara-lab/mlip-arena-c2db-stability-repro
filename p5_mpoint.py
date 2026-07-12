"""
P5 - The zone-boundary (M-point) test for issue #89.

Decisive experiment for P4's central claim: are the true-unstable monolayers that
MACE-MP(M) passes on the Gamma-only check "invisible by construction" (instability
lives at the zone boundary) or genuine model errors (model blind everywhere)?

Key physics: the benchmark builds force constants on a (2,2,1) supercell. The
EXACT commensurate q-points of that supercell are Gamma=(0,0,0) and the three
zone-boundary points (1/2,0,0),(0,1/2,0),(1/2,1/2,0). Evaluating the SAME force
constants at those zone-boundary points is exact (no interpolation) and detects
2x2 cell-doubling / CDW instabilities -- at ZERO extra force evaluations beyond
what the benchmark already computes. The benchmark simply discards them by only
reading q=Gamma.

So this reruns the corrected protocol (relax with FixSymmetry, (2,2,1) phonons,
symmetrized force constants -- same numerics as repro_lib) but reports freq_min
at Gamma AND at the zone-boundary points. Prediction (falsifiable): the CDW-family
targets show freq_min(Gamma) ~ 0 but freq_min(zone-boundary) strongly negative ->
the model DOES see the instability; the Gamma-only observable was blind.

A dense interpolated mesh min is also reported (indicative only -- interpolated
from the 2x2 FCs, so it approximately probes K and incommensurate points a 2x2
cell cannot represent exactly; a real K-point test needs a (3,3,1) rerun).

Needs: MACE + phonopy + ase environment (same one that ran repro_main.py) and
c2db.db in the same folder. Deps: mace-torch phonopy ase numpy.
Run: python p5_mpoint.py            # smoking-gun set (~15 materials, fast)
     python p5_mpoint.py very_deep  # all 35 mh<-2 misses
"""
from __future__ import annotations
import os, sqlite3, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
SYMPREC, DISTANCE = 1e-5, 0.01
ZB_TOL = -0.05   # a zone-boundary mode below this (THz) counts as a real instability

# curated 1T-MX2 / MX CDW-family metals with violent DFT instability but clean MLIP Gamma
SMOKING_GUN = ["2CoS2-1", "1TaI2-1", "1NbI2-3", "1WI2-3", "1ReBr2-2", "1ReSe2-1",
               "1RuS2-2", "1RuSe2-2", "1RhS2-3", "1IrO2-2", "1IrO2-3", "1ClINb-2",
               "1BrITa-1", "2RuSe-4", "2SeW-3"]
VERY_DEEP = ["1IrO2-3", "1F2N2V3-1", "1N2O2Mo3-1", "1TaI2-1", "1WI2-3", "1F2N2Nb3-1",
             "1BrITa-1", "1ReBr2-2", "1SSbSeSnCl2-1", "2RuSe-4", "1NbI2-3", "1ClINb-2",
             "1IrO2-2", "1RuS2-2", "1ReSe2-1", "1RhS2-3", "1RuSe2-2", "2AgBr2-1",
             "1OsBr2-2", "1ClITa-2", "2CoS2-1", "1AuBrF2-1", "1C2F2V3-1", "1TiZrI2S2-1",
             "2SeW-3", "4CuTeO3-1", "2CuSiO3-1", "1HfSeI2S2Sc2-1", "2PVS3-1", "1TlH2O2-1",
             "1AuO2-1", "2NbSe-3", "1BrClW-1", "2RuTaTe4-1", "4NiSe-1"]

ZB_QPOINTS = {"M1": (0.5, 0.0, 0.0), "M2": (0.0, 0.5, 0.0), "M3": (0.5, 0.5, 0.0)}


def get_mace_mp_medium():
    from mace.calculators import MACECalculator
    import urllib.request
    url = ("https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
           "2023-12-03-mace-128-L1_epoch-199.model")
    cache = os.path.join(os.path.expanduser("~"), ".cache", "mace")
    os.makedirs(cache, exist_ok=True)
    name = "".join(c for c in os.path.basename(url) if c.isalnum() or c in "_")
    path = os.path.join(cache, name)
    if not os.path.isfile(path):
        urllib.request.urlretrieve(url, path)
    return MACECalculator(model_paths=path, device="cpu", default_dtype="float32")


def relax(atoms, calc, fmax=0.05, steps=500):
    from ase.optimize import FIRE
    from ase.constraints import FixSymmetry
    atoms = atoms.copy()
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms))       # positions-only, symmetry-preserved (run.py OPT)
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)
    atoms.set_constraint()
    return atoms


def freq_min_at_qpoints(atoms, calc, supercell=(2, 2, 1)):
    """Corrected protocol (symmetrized (2,2,1) FCs); return freq_min at Gamma, at each
    commensurate zone-boundary point, and an interpolated dense-mesh min (indicative)."""
    from ase import Atoms
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    ph = Phonopy(
        PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.get_cell(),
                     scaled_positions=atoms.get_scaled_positions(wrap=True),
                     masses=atoms.get_masses()),
        symprec=SYMPREC, supercell_matrix=supercell)
    ph.generate_displacements(distance=DISTANCE)
    disps = [sc for sc in ph.supercells_with_displacements if sc is not None]
    forces = []
    for sc in disps:
        a = Atoms(symbols=sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    ph.forces = forces
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    ph.symmetrize_force_constants_by_space_group()
    g = float(np.min(ph.get_frequencies(q=(0, 0, 0))))
    zb = {k: float(np.min(ph.get_frequencies(q=q))) for k, q in ZB_QPOINTS.items()}
    try:
        ph.run_mesh([13, 13, 1], with_eigenvectors=False)
        mesh = float(np.min(ph.get_mesh_dict()["frequencies"]))
    except Exception:
        mesh = float("nan")
    return g, zb, mesh, len(disps)


def main():
    from ase.db import connect
    targets = VERY_DEEP if (len(sys.argv) > 1 and sys.argv[1] == "very_deep") else SMOKING_GUN
    dbp = next((os.path.join(HERE, n) for n in ["c2db.db", "c2db"] if os.path.exists(os.path.join(HERE, n))), None)
    assert dbp, "c2db.db not found in this folder"

    con = sqlite3.connect(dbp)
    uid2id = {v: i for i, k, v in con.execute("SELECT id,key,value FROM text_key_values WHERE key='uid'")}
    id2mh = {i: v for i, k, v in con.execute("SELECT id,key,value FROM number_key_values WHERE key='minhessianeig'")}
    con.close()

    calc = get_mace_mp_medium()
    adb = connect(dbp)
    print(f"{'uid':16s} {'na':>3} {'mh_DFT':>8} {'fmin_G':>8} {'fmin_ZB':>8} {'mesh~':>8}  verdict")
    print("-" * 78)
    out = []
    for uid in targets:
        if uid not in uid2id:
            print(f"{uid:16s}  (uid not in c2db.db)"); continue
        sid = uid2id[uid]
        atoms = adb.get(sid).toatoms()
        t0 = time.time()
        try:
            g, zb, mesh, nd = freq_min_at_qpoints(relax(atoms, calc), calc)
        except Exception as e:
            print(f"{uid:16s} ERROR {e!r}"); continue
        zbmin = min(zb.values()); qmin = min(zb, key=zb.get)
        mh = id2mh.get(sid, float("nan"))
        if g < ZB_TOL:
            verdict = "caught@Gamma(sanity)"
        elif zbmin < ZB_TOL:
            verdict = f"RESCUED @ {qmin} (Gamma-blind)"
        elif not np.isnan(mesh) and mesh < ZB_TOL:
            verdict = "rescued off-symmetry (interp; needs 3x3)"
        else:
            verdict = "model-blind everywhere"
        rec = {"uid": uid, "na": len(atoms), "mh": round(mh, 3), "fmin_gamma": round(g, 4),
               "fmin_zb": round(zbmin, 4), "zb_q": qmin, "zb_all": {k: round(v, 4) for k, v in zb.items()},
               "mesh_min": round(mesh, 4), "ndisp": nd, "verdict": verdict, "sec": round(time.time() - t0, 1)}
        out.append(rec)
        print(f"{uid:16s} {len(atoms):>3} {mh:>8.2f} {g:>8.4f} {zbmin:>8.4f} {mesh:>8.4f}  {verdict}", flush=True)

    n = len(out)
    resc = sum("RESCUED" in r["verdict"] for r in out)
    resc_off = sum("off-symmetry" in r["verdict"] for r in out)
    blind = sum(r["verdict"] == "model-blind everywhere" for r in out)
    print("-" * 78)
    print(f"n={n} | RESCUED at exact zone boundary: {resc} | rescued off-symmetry (interp): {resc_off} "
          f"| model-blind: {blind}")
    print(f"=> {resc}/{n} of Gamma-passed true-unstables are unstable at a commensurate zone-boundary")
    print(f"   point the (2,2,1) FCs ALREADY contain -- invisible to the Gamma-only observable by construction.")
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "data", "p5_mpoint_results.json"), "w"), indent=2)
    print("\nWrote results to data/p5_mpoint_results.json")


if __name__ == "__main__":
    main()
