#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Run free-bdry DESC on a finite beta eq given from a VMEC .nc file
First cmd line arg: path to the VMEC wout file
Second cmd line arg: path to the MAKEGRID coils file for this equilibrium
Third cmd line arg : path to the VMEC input file (so DESC can get the pressure profile)
"""
# from desc import set_device

# set_device("gpu")

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
maxiter = 10
wout_filename ="optimization_finitebeta_nfp2_QA_stage1/wout_final.nc"
subfoldername = wout_filename.split("/")[0]
name = wout_filename.split("/")[1]
name = name.strip("wout_").strip(".nc")

path_init_fixed_solve = "optimization_finitebeta_nfp2_QA_stage1" + "/" + f"desc_initial_fixed_bdry_solve_{name}.h5"
path_no_K = "optimization_finitebeta_nfp2_QA_stage1/desc_fb_no_sheet_current.h5"
path_with_K = "optimization_finitebeta_nfp2_QA_stage1/desc_fb_with_sheet_current.h5"

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
    "optimization_finitebeta_nfp2_QA_stage1/coils/coils_makegrid_format.txt",
).to_FourierXYZ(N=60)
if isinstance(ext_field, MixedCoilSet):
    # make a CoilSet if it is a MixedCoilSet
    ext_field = CoilSet(*ext_field)

eq = VMECIO.load(wout_filename,profile="current")

# get the pressure, which is a power series, from the input file
input_info = InputReader.parse_vmec_inputs("optimization_finitebeta_nfp2_QA_stage1/input.final")
pressure = parse_profile(input_info[0]["pressure"])
pressure.change_resolution(eq.L)
eq.pressure = pressure
curr_vals_at_s_knots = np.fromstring("-4018351.7373274835 -4509947.555348026 -4923533.108593054 -5267268.908277436 -5549315.465616044 -5777833.291823739 -5960982.8981153965 -6106924.795705884 -6223819.495810065 -6319582.869577966 -6400504.4210085 -6472210.190244721 -6540324.260309157 -6610147.413192208 -6684166.514493454 -6763419.562224023 -6848932.580282736 -6941573.616269168 -7040374.2433041455 -7143181.212264138 -7247821.526988209 -7352123.738179599 -7453941.14185851 -7551146.935828236 -7641614.881618079 -7723246.450259706 -7794580.43133927 -7854792.932997399 -7903087.77287709 -7938584.198583662 -7957415.797112135 -7952003.86908348 -7914537.654935272 -7837225.133491413 -7713398.586755748 -7538132.966661016 -7306653.132230618 -7014204.767016738 -6658553.322554472 -6242358.014643251 -5768840.321359683 -5241225.707008204 -4664090.967126065 -4045325.4525734275 -3393316.792688641 -2716452.616810041 -2023120.5542759728 -1321708.2344247787 -620603.2865948015 71806.65987561649",
                                    sep=" ", dtype=float)
s_knots = np.fromstring("0.0 0.02040816326530612 0.04081632653061224 0.061224489795918366 0.08163265306122448 0.1020408163265306 0.12244897959183673 0.14285714285714285 0.16326530612244897 0.18367346938775508 0.2040816326530612 0.22448979591836732 0.24489795918367346 0.26530612244897955 0.2857142857142857 0.3061224489795918 0.32653061224489793 0.3469387755102041 0.36734693877551017 0.3877551020408163 0.4081632653061224 0.42857142857142855 0.44897959183673464 0.4693877551020408 0.4897959183673469 0.5102040816326531 0.5306122448979591 0.5510204081632653 0.5714285714285714 0.5918367346938775 0.6122448979591836 0.6326530612244897 0.6530612244897959 0.673469387755102 0.6938775510204082 0.7142857142857142 0.7346938775510203 0.7551020408163265 0.7755102040816326 0.7959183673469387 0.8163265306122448 0.836734693877551 0.8571428571428571 0.8775510204081632 0.8979591836734693 0.9183673469387754 0.9387755102040816 0.9591836734693877 0.9795918367346939 1.0",
                                    sep=" ", dtype=float)


# In[2]:


import scipy
current_prime = SplineProfile(knots=s_knots, values=curr_vals_at_s_knots)
current = scipy.integrate.cumulative_trapezoid(x=s_knots, y=current_prime(s_knots),initial=0.0)
CURTOR = -6133652.330056256
current *= CURTOR/current[-1]


# In[3]:


import matplotlib.pyplot as plt
rho = np.linspace(0,1,100)
plt.plot(np.sqrt(s_knots), current)
current_poly = PowerSeriesProfile.from_values(x=np.sqrt(s_knots), y=current,sym=True,order=14)
plt.plot(rho, current_poly(rho),"--")
plt.scatter(rho,eq.current(rho))


# In[4]:


eq.current = current_poly
eq.current.params[0]=0 # ensure 0 on-axis current
eq.change_resolution(10, 10, 10, 20, 20, 20)
eq.solve(ftol=1e-4,xtol=1e-7)
eq.save(path_init_fixed_solve)
eqf = EquilibriaFamily(eq)



# calc field Psi
grid = LinearGrid(L=50, M=50, zeta=np.array(0.0))
data = eq.compute(["R", "phi", "Z", "|e_rho x e_theta|"], grid=grid)
field_B = ext_field.compute_magnetic_field(np.vstack([data["R"], data["phi"], data["Z"]]).T,
                                      source_grid = LinearGrid(N=200))
psi_from_field = np.sum(
    grid.spacing[:, 0] * grid.spacing[:, 1] * data["|e_rho x e_theta|"] * field_B[:, 1]
)
print(f"eq psi = {eq.Psi}")

print(f"field psi = {psi_from_field}")

# just make sure sign matches in psi from coils and eq psi
ratio = np.sign(eq.Psi / psi_from_field)
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
            ftol=1e-4,
            xtol=0,
            verbose=3,
            copy=True,
            options={"solve_options":{"ftol":1e-4,"xtol":5e-7}},
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
    eq.surface.change_Phi_resolution(int(np.ceil(i / nn * eq.M)), int(np.ceil(i / nn * eq.N)))
    # eqf[-1].solve(ftol=1e-3, verbose=3)

    eq_new, out = eqf[-1].optimize(
        get_objective(ext_field, eqf[-1]),
        get_constraints(eqf[-1], i),
        optimizer=optimizer,
        maxiter=maxiter,
        ftol=1e-4,
        xtol=1e-7,
        verbose=3,
        copy=True,
        options={"solve_options":{"ftol":1e-4,"xtol":5e-7},
                "initial_trust_ratio":1e-3,
                "perturb_options": {"order": 1, "verbose": 0}},
    )
    eqf.append(eq_new)
    eqf.save(path_with_K)




