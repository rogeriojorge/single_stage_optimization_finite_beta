"""Run free-bdry DESC on a finite beta eq given from a VMEC .nc file
First cmd line arg: path to the VMEC wout file
Second cmd line arg: path to the MAKEGRID coils file for this equilibrium
Third cmd line arg : path to the VMEC input file (so DESC can get the pressure profile)
"""

import os
import pathlib
import pickle
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))
from desc import set_device

set_device("gpu")

import desc.examples
from desc.continuation import solve_continuation_automatic
from desc.coils import CoilSet, MixedCoilSet
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.magnetic_fields import FourierCurrentPotentialField, SplineMagneticField
from desc.objectives import (
    BoundaryError,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixPsi,
    ForceBalance,
    ObjectiveFunction,
)
from desc.profiles import PowerSeriesProfile
from desc.vmec import VMECIO
from desc.input_reader import InputReader
from desc.equilibrium.utils import parse_profile
from desc.io import load

optimizer = "proximal-lsq-exact"
nn = 7
maxiter = 100

path_with_K = "./free_bdry_DESC_K_with_re_opt_coils.h5"




def get_constraints(eq, i):

    R_modes = eq.surface.R_basis.modes[
        np.max(np.abs(eq.surface.R_basis.modes), 1) > int(i / nn * eq.M), :
    ]
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > int(i / nn * eq.M), :
    ]

    return (
        ForceBalance(eq=eq),
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )


def get_objective(ext_field, eq):
    return ObjectiveFunction(BoundaryError(eq, ext_field, field_fixed=True))


os.getcwd()

ext_field = load("re_optimized_coilset_n11.h5")
eq = load("desc_initial_fixed_bdry_solve_final.h5")


###### Then solve with sheet current included ######

eq.surface = FourierCurrentPotentialField.from_surface(eq.surface)
eqf = EquilibriaFamily(eq)

for i in range(1, nn + 1):

    jax.clear_caches()
    print("\n==================================")
    print("Optimizing boundary modes M,N <= {}".format(int(i / nn * eq.M)))
    print("====================================")
    eq.surface.change_Phi_resolution(int(i / nn * eq.M), int(i / nn * eq.N))
    eqf[-1].solve(ftol=1e-2, verbose=3)

    eq_new, out = eqf[-1].optimize(
        get_objective(ext_field, eqf[-1]),
        get_constraints(eqf[-1], i),
        optimizer=optimizer,
        maxiter=maxiter,
        ftol=1e-4,
        xtol=0,
        verbose=3,
        copy=True,
        options={},
    )
    eqf.append(eq_new)
    eqf.save(path_with_K)

