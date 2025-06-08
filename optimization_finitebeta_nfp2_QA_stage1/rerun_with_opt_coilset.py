# uncomment these two lines if using a gpu for a speedup
from desc import set_device
set_device("gpu")

from desc.magnetic_fields import FourierCurrentPotentialField

import numpy as np
from desc.coils import CoilSet, FourierPlanarCoil
import desc.examples
from desc.equilibrium import Equilibrium
from desc.plotting import plot_surfaces, plot_2d, plot_3d, plot_coils
from desc.grid import LinearGrid
from desc.coils import MixedCoilSet
from desc.objectives import (
    ObjectiveFunction,
    CoilCurvature,
    CoilLength,
    CoilTorsion,
    CoilSetMinDistance,
    PlasmaCoilSetMinDistance,
    QuadraticFlux,
    ToroidalFlux,
    FixCoilCurrent,
    FixParameters,
    QuasisymmetryTwoTerm,
    AspectRatio,
    ForceBalance,
    VacuumBoundaryError,
    BoundaryError,
    FixPsi,
    FixPressure,
    FixCurrent,
    FixBoundaryR,
    FixBoundaryZ
)
from desc.optimize import Optimizer
from desc.magnetic_fields import field_line_integrate
from desc.integrals.singularities import compute_B_plasma
import time
import plotly.express as px
import plotly.io as pio
from desc.io import load
# This ensures Plotly output works in multiple places:
# plotly_mimetype: VS Code notebook UI
# notebook: "Jupyter: Export to HTML" command in VS Code
# See https://plotly.com/python/renderers/#multiple-renderers
pio.renderers.default = "png"
import jax


eq = load("desc_initial_fixed_bdry_solve_final.h5")
optimized_coilset = load("re_optimized_coilset_n11.h5")

# number of points used to discretize coils. This could be different for each objective
# (eg if you want higher resolution for some calculations), but we'll use the same for all of them
coil_grid = LinearGrid(N=150)
# similarly define a grid on the plasma surface where B*n errors will be evaluated
plasma_grid = LinearGrid(M=35, N=35, NFP=eq.NFP, sym=False)


# just do sheet current opt first, to get bdry field jump lower
eq.surface = FourierCurrentPotentialField.from_surface(eq.surface,M_Phi=2,N_Phi=2)

# define our objective function
obj = ObjectiveFunction(
    (
        BoundaryError(
            eq,
            field=optimized_coilset,
            # grid of points on plasma surface to evaluate normal field error
            eval_grid=plasma_grid,
            field_grid=coil_grid,
            field_fixed=True
        ),
    )
)

constraints = (FixPsi(eq),
               FixCurrent(eq),
               FixPressure(eq),
               ForceBalance(eq),
              FixBoundaryR(eq),
              FixBoundaryZ(eq)) # force balance in the constraints bc the equilibrium is there


optimizer = Optimizer("proximal-lsq-exact")
for k in np.arange(2,11,2):
    eq.surface.change_Phi_resolution(M=k,N=k)
    obj = ObjectiveFunction(
    (
        BoundaryError(
            eq,
            field=optimized_coilset,
            # grid of points on plasma surface to evaluate normal field error
            eval_grid=plasma_grid,
            field_grid=coil_grid,
            field_fixed=True
        ),
    )
)

    constraints = (FixPsi(eq),
                   FixCurrent(eq),
                   FixPressure(eq),
                   ForceBalance(eq),
                  ) # force balance in the constraints bc the equilibrium is there
    # get modes where |m|, |n| > k
    R_modes = eq.surface.R_basis.modes[
        np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
    ]
    Z_modes = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]

    # fix those modes
    bdry_constraints = (
        FixBoundaryR(eq=eq, modes=R_modes),
        FixBoundaryZ(eq=eq, modes=Z_modes),
    )
    (eq,), _ = optimizer.optimize(
        (eq,),
        objective=obj,
        constraints=constraints+bdry_constraints,
        maxiter=100,
        ftol=1e-3,
        verbose=3,
        # copy=True,
    )
    eq.save("re_solved_free_bdry_with_re_opt_coils_K_overnight.h5")
    Bn,_ = optimized_coilset.compute_Bnormal(eq)
    print(np.max(Bn))