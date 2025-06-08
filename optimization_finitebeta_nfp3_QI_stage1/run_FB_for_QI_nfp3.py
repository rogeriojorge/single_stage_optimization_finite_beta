#!/usr/bin/env python
# coding: utf-8

"""Run free-bdry DESC on a finite beta eq given from a VMEC .nc file
"""
from desc import set_device

set_device("gpu")

import os
import pathlib
import pickle
import sys

import jax
import numpy as np
from desc.grid import LinearGrid

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))


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
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.vmec import VMECIO
from desc.input_reader import InputReader
from desc.equilibrium.utils import parse_profile
from desc.io import load

optimizer = "proximal-lsq-exact"
nn = 7
maxiter = 100
base_dirname = "."
wout_filename = base_dirname + "/wout_final.nc"
subfoldername = wout_filename.split("/")[0]
name = wout_filename.split("/")[1]
name = name.strip("wout_").strip(".nc")

path_init_fixed_solve = base_dirname + "/" + f"desc_initial_fixed_bdry_solve_{name}_updated_integrals.h5"
path_no_K = base_dirname + "/desc_fb_no_sheet_current_updated_integrals.h5"
path_with_K = base_dirname + "/desc_fb_with_sheet_current_updated_integrals.h5"

print("SAVING NO SHEET CURRENT SOLVE TO ", path_no_K)
print("SAVING  SHEET CURRENT SOLVE TO ", path_with_K)


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

ext_field = CoilSet.from_makegrid_coilfile(
    base_dirname + "/coils/coils.biot_savart_opt_nfp3.txt",
).to_FourierXYZ(N=40)
if isinstance(ext_field, MixedCoilSet):
    # make a CoilSet if it is a MixedCoilSet
    ext_field = CoilSet(*ext_field)

eq = VMECIO.load(wout_filename, profile="current")

# get the pressure, which is a power series, from the input file
input_info = InputReader.parse_vmec_inputs("./input.final")
pressure = parse_profile(input_info[0]["pressure"])
pressure.change_resolution(eq.L)
eq.pressure = pressure
current = PowerSeriesProfile()
current.change_resolution(eq.L)
eq.current = current


# In[2]:


import scipy

# In[3]:


import matplotlib.pyplot as plt


# In[4]:

eq.change_resolution(10, 10, 10, 20, 20, 20)
if not os.path.exists(path_init_fixed_solve):
    eq0 = solve_continuation_automatic(eq)[-1]
    eq0.save(path_init_fixed_solve)
else:
    eq0= load(path_init_fixed_solve)
eqf = EquilibriaFamily(eq0)


# calc field Psi
grid = LinearGrid(L=50, M=50, zeta=np.array(0.0))
data = eq0.compute(["R", "phi", "Z", "|e_rho x e_theta|"], grid=grid)
field_B = ext_field.compute_magnetic_field(
    np.vstack([data["R"], data["phi"], data["Z"]]).T, source_grid=LinearGrid(N=200)
)
psi_from_field = np.sum(
    grid.spacing[:, 0] * grid.spacing[:, 1] * data["|e_rho x e_theta|"] * field_B[:, 1]
)
print(f"eq psi = {eq0.Psi}")

print(f"field psi = {psi_from_field}")

# just make sure sign matches in psi from coils and eq psi
ratio = np.sign(eq0.Psi / psi_from_field)
for i in range(len(ext_field)):
    ext_field[i].current = ext_field[i].current * ratio


###### First solve with no sheet current ######
# because these coils were made for these equilibria, sheet
# current should be small, so it is better to initially solve without
if not os.path.exists(path_no_K):
    for i in range(1, nn + 1):

        jax.clear_caches()
        print("\n==================================")
        print("Optimizing boundary modes M,N <= {}".format(int(i / nn * eq.M)))
        print("====================================")
        # eqf[-1].solve(ftol=1e-3, verbose=3)

        eq_new, out = eqf[-1].optimize(
            get_objective(ext_field, eqf[-1]),
            get_constraints(eqf[-1], i),
            optimizer=optimizer,
            maxiter=maxiter,
            ftol=1e-6,
            xtol=0,
            verbose=3,
            copy=True,
            options={"solve_options": {"ftol": 1e-6, "xtol": 5e-7}},
        )
        eqf.append(eq_new)
        eqf.save(path_no_K)
else:
    eqf = load(path_no_K)


###### Then solve with sheet current included ######
eqf = load(path_no_K)
eq = eqf[-1]
eq.surface = FourierCurrentPotentialField.from_surface(eq.surface)
eqf = EquilibriaFamily(eq)
for i in range(1, nn + 1):

    jax.clear_caches()
    print("\n==================================")
    print("Optimizing boundary modes M,N <= {}".format(int(i / nn * eq.M)))
    print("====================================")
    eq.surface.change_Phi_resolution(
        int(np.ceil(i / nn * eq.M)), int(np.ceil(i / nn * eq.N))
    )
    # eqf[-1].solve(ftol=1e-3, verbose=3)

    eq_new, out = eqf[-1].optimize(
        get_objective(ext_field, eqf[-1]),
        get_constraints(eqf[-1], i),
        optimizer=optimizer,
        maxiter=maxiter,
        ftol=1e-6,
        xtol=1e-7,
        verbose=3,
        copy=True,
        options={
            "solve_options": {"ftol": 1e-5, "xtol": 5e-7},
            "initial_trust_ratio": 1e-3,
            "perturb_options": {"order": 1, "verbose": 0},
        },
    )
    eqf.append(eq_new)
    eqf.save(path_with_K)
