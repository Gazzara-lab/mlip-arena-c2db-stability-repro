"""
Shared helpers for the atomind-ai/mlip-arena issue #89 reproduction.

Mirrors the numerics of mlip_arena/tasks/phonon.py (symprec=1e-5, distance=0.01,
the same symmetrize_force_constants() + symmetrize_force_constants_by_space_group()
calls, Gamma-point get_frequencies) and the MACE_MP_Medium wrapper (registry
checkpoint, default_dtype="float32"). The phonon (freq_min) leg is ONE half of the
benchmark's OR criterion `~(eigval_min < tol OR freq_min < tol)`; the elastic leg is
not recomputed here; a separate elastic-leg check found it healthy at ~97% pass
(see ANALYSIS.md).
"""
from __future__ import annotations
import os, urllib.request, warnings
import numpy as np
from ase import Atoms
from ase.optimize import FIRE
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

warnings.filterwarnings("ignore")

SYMPREC = 1e-5
DISTANCE = 0.01


def get_mace_mp_medium():
    """The standard mace_mp_0 medium checkpoint that the MACE-MP(M) registry entry points to."""
    from mace.calculators import MACECalculator
    url = ("https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
           "2023-12-03-mace-128-L1_epoch-199.model")
    cache = os.path.join(os.path.expanduser("~"), ".cache", "mace")
    os.makedirs(cache, exist_ok=True)
    name = "".join(c for c in os.path.basename(url) if c.isalnum() or c in "_")
    path = os.path.join(cache, name)
    if not os.path.isfile(path):
        urllib.request.urlretrieve(url, path)
    return MACECalculator(model_paths=path, device="cpu", default_dtype="float32")


def relax(atoms, calc, fix_symmetry=True, relax_cell=False, fmax=0.05, steps=500):
    """Default mirrors benchmarks/c2db/run.py OPT (FIRE, fmax=0.05, FixSymmetry, positions-only).
    fix_symmetry=False allows out-of-plane buckling; relax_cell=True relaxes the in-plane cell."""
    atoms = atoms.copy()
    atoms.calc = calc
    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))
    target = atoms
    if relax_cell:
        # relax in-plane cell only (mask keeps the c/vacuum axis and its shears fixed)
        target = FrechetCellFilter(atoms, mask=[True, True, False, False, False, True])
    FIRE(target, logfile=None).run(fmax=fmax, steps=steps)
    atoms.set_constraint()
    return atoms


def gamma_freq_min(phonon):
    f = phonon.get_frequencies(q=(0, 0, 0))
    return float(np.min(f))


def phonon_freq_min(atoms, calc, supercell=(2, 2, 1)):
    """Compute Gamma freq_min WITHOUT and WITH force-constant symmetrization on one
    set of forces (the symmetry flag only changes the post-hoc symmetrization)."""
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
    fmin_nosym = gamma_freq_min(ph)                 # symmetry=False (current benchmark default)
    ph.symmetrize_force_constants()                 # symmetry=True path
    ph.symmetrize_force_constants_by_space_group()
    fmin_sym = gamma_freq_min(ph)
    return fmin_nosym, fmin_sym, len(disps)
