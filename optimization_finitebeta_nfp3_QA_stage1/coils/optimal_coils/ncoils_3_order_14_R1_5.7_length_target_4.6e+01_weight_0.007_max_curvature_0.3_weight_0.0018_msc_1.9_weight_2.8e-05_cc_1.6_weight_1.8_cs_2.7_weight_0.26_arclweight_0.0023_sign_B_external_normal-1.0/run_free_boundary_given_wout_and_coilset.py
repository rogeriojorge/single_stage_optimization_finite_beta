#!/usr/bin/env python
# coding: utf-8


"""Run free-bdry DESC on a finite beta eq given from a VMEC .nc file
First cmd line arg: path to the VMEC wout file
Second cmd line arg: path to the DESC coils .h5 for this equilibrium
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
from desc import set_device

set_device("gpu")


import os
import sys
import shutil
import subprocess
# this is to read in certain inputs from the vmec input file
import f90nml

import jax
import scipy
import numpy as np
import matplotlib.pyplot as plt
from desc.grid import LinearGrid
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))


from desc.equilibrium import EquilibriaFamily
from desc.magnetic_fields import FourierCurrentPotentialField
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
import desc.io
from desc.plotting import *
from matplotlib.backends.backend_pdf import PdfPages


def run_free_boundary(wout_filename, path_to_coilset, nn=7, maxiter=50):
    """
    Runs DESC free boundary based off of wout given and the coilset given.

    wout_filename: str, the path to the VMEC .nc file
    
    path_to_coilset: str, path to the coilset to use for the free boundary solve

    nn: int, the number of steps to use in the free boundary solve

    maxiter: the max number of iterations to run at each step in the free boundary solve
    """

    # assuming all vmec input files are names "input.final"
    # and all wout files are names "wout_final.nc"
    optimizer = "proximal-lsq-exact"

    wout_filename = wout_filename
    subfoldername = wout_filename.split("/")[-2]

    subfolderpath = "/" + os.path.join(*wout_filename.split("/")[0:-1]) # path all the way to the folder above the wout file

    coilfolder_path = "/" +  os.path.join(*path_to_coilset.split("/")[0:-1]) # path to coilset's folder (which we will save stuff at, since the coilset names are too long to include with the other information in saving filenames)


    path_init_fixed_solve = subfolderpath + "/" + f"desc_initial_fixed_bdry_solve.h5"
    path_no_K = os.path.join(coilfolder_path, "desc_fb_no_sheet_current.h5")
    path_with_K = os.path.join(coilfolder_path, "desc_fb_with_sheet_current.h5")
    path_no_K_final = os.path.join(coilfolder_path, "desc_fb_no_sheet_current_final.h5")
    path_with_K_final = os.path.join(coilfolder_path, "desc_fb_with_sheet_current_final.h5")

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


    # get the coilset to use for free boundary
    ext_field = desc.io.load(path_to_coilset)

    # construct the DESC equilibrium
    # easiest to first load in the VMEC fixed bdry eq
    eq = VMECIO.load(wout_filename,profile="current")

    # then get the pressure, which is a power series, from the VMEC input file
    input_info = InputReader.parse_vmec_inputs(os.path.join(subfolderpath,"input.final"))
    nml = f90nml.read(os.path.join(subfolderpath,"input.final")) # use f90nml to get the current profile data since DESC does not have that I/O functionality with VMEC input files implemented yet
    pressure = parse_profile(input_info[0]["pressure"])
    eq.pressure = pressure

    # DESC cannot read spline profiles from vmec input files,
    # the current profile is typically a spline profile of the derivative of current w.r.t. s
    # To ensure that the profile is an even function of rho near-axis, we will integrate (as DESC needs I, not I')
    # then fit with a powerseries 
    curr_prime_vals_at_s_knots =  nml["indata"]["AC_AUX_F"]
    s_knots = nml["indata"]["AC_AUX_S"]
    CURTOR = nml["indata"]["curtor"]
    
    if np.isclose(CURTOR,0.0):
        # just use a zero power series
        current_poly = PowerSeriesProfile()
    else:
        s_knots = np.asarray(s_knots)
        s_knots = s_knots[0:len(curr_prime_vals_at_s_knots)] # only take up to the s=1.0 knot, which matches length of the I' values
        
        current_prime = SplineProfile(knots=s_knots, values=curr_prime_vals_at_s_knots)
        current = scipy.integrate.cumulative_trapezoid(x=s_knots, y=current_prime(s_knots),initial=0.0)
        # scale to correct curtor at edge
        current *= CURTOR/current[-1]

        rho = np.linspace(0,1,100)
        plt.plot(np.sqrt(s_knots), current)
        current_poly = PowerSeriesProfile.from_values(x=np.sqrt(s_knots), y=current,sym=True,order=14)
        plt.plot(rho, current_poly(rho),"--",label="Power Series")
        plt.scatter(rho,eq.current(rho),label="Spline")
        plt.savefig(os.path.join(subfolderpath, "compare_current_spline_to_current_power_series.png"))

    eq.current = current_poly
    eq.current.params[0]=0 # ensure 0 on-axis net toroidal current
    
    eq.change_resolution(max(10,eq.L), max(10,eq.M), max(10,eq.N),
                         max(20,eq.L_grid), max(20,eq.M_grid), max(20,eq.N_grid))

    # re-solve the initial fixed bdry eq in DESC and save
    if not os.path.exists(path_init_fixed_solve):
        eq.solve(ftol=1e-4,xtol=1e-7)
        eq.save(path_init_fixed_solve)
    else:
        eq = desc.io.load(path_init_fixed_solve)
    eqf = EquilibriaFamily(eq)

    # Here we will double check that the field the coils make is in the correct
    # direction for the equilibrium (as dictated by the eq.Psi).
    # NOTE: Some coilsets seem to have opposite Psi as they should, meaning
    # the toroidal field they make is opposite that of what the plasma
    # equilibrium dictates.
    grid = LinearGrid(L=50, M=50, zeta=np.array(0.0))
    data = eq.compute(["R", "phi", "Z", "|e_rho x e_theta|"], grid=grid)
    field_B = ext_field.compute_magnetic_field(np.vstack([data["R"], data["phi"], data["Z"]]).T,
                                        source_grid = LinearGrid(N=200))
    psi_from_field = np.sum(
        grid.spacing[:, 0] * grid.spacing[:, 1] * data["|e_rho x e_theta|"] * field_B[:, 1]
    )
    print(f"eq psi = {eq.Psi}")
    print(f"field psi = {psi_from_field}")

    # make sure sign matches in psi from coils and eq psi
    ratio = np.sign(eq.Psi / psi_from_field)
    for i in range(len(ext_field)):
        ext_field[i].current = ext_field[i].current * ratio

    ###### First solve with no sheet current ######
    # because these coils were made for these equilibria, sheet
    # current should be small, so it is better to initially solve without
    if not os.path.exists(path_no_K_final):
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
                options={"solve_options":{"ftol":1e-4,"xtol":5e-7},
                },
            )
            eqf.append(eq_new)
            eqf.save(path_no_K)
        eqf.save(path_no_K_final)
    else:
        eqf = desc.io.load(path_no_K_final)

    ###### Then solve with sheet current included ######
    eq = eqf[-1]
    eq.surface = FourierCurrentPotentialField.from_surface(eq.surface)
    eqf = EquilibriaFamily(eq)
    if not os.path.exists(path_with_K_final):
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
                        "perturb_options": {"order": 1, "verbose": 0},
                        },
            )
            eqf.append(eq_new)
            eqf.save(path_with_K)
        eqf.save(path_with_K_final)

    # finally, make plots
    eq = desc.io.load(path_init_fixed_solve)

    eq_fb_no_K = desc.io.load(path_no_K_final)[-1]
    eq_fb_K = desc.io.load(path_with_K_final)[-1]

    figs_dirpath = os.path.join(coilfolder_path, "figs")
    if not os.path.exists(figs_dirpath):
        os.mkdir(figs_dirpath)

    summary_pdf_path = os.path.join(figs_dirpath, "summary.pdf" )

    pp = PdfPages(summary_pdf_path)


    plt.rcParams.update({"font.size":18})
    plot_comparison([eq,eq_fb_no_K],labels=["fixed","fb no K"])
    plt.savefig(os.path.join(figs_dirpath, "surface_comparison_FB_fixed_no_surface_current.png" ))
    pp.savefig()


    plot_comparison([eq,eq_fb_K],labels=["fixed","fb K"])
    plt.savefig(os.path.join(figs_dirpath, "surface_comparison_FB_fixed_with_surface_current.png" ))
    pp.savefig()

    fig,ax=plot_1d(eq_fb_K,"iota",label="fb K",figsize=(5,5))
    fig,ax=plot_1d(eq_fb_no_K,"iota",label="fb no K",figsize=(5,5),ax=ax,linecolor="m")
    fig,ax=plot_1d(eq,"iota",label="fixed bdry",ax=ax,linecolor="r")
    plt.savefig(os.path.join(figs_dirpath, "iota_comparison_plot.png" ))
    pp.savefig()

    fig,ax=plot_1d(eq_fb_K,"D_Mercier",label="fb K",log=True,figsize=(5,5))
    fig,ax=plot_1d(eq_fb_no_K,"D_Mercier",label="fb no K",figsize=(5,5),ax=ax,linecolor="m",log=True)
    fig,ax=plot_1d(eq,"D_Mercier",label="fixed bdry",ax=ax,linecolor="r",log=True)
    plt.savefig(os.path.join(figs_dirpath, "Dmerc_comparison_plot.png" ))
    pp.savefig()
    
    fig,ax=plot_boozer_surface(eq)
    ax.set_title("Fixed bdry |B|")
    plt.savefig(os.path.join(figs_dirpath, "Boozer_surface_fixed_bdry.png" ))
    pp.savefig()

    fig2,ax2=plot_boozer_surface(eq_fb_K)
    ax2.set_title("Free bdry (with sheet current) |B|")
    plt.savefig(os.path.join(figs_dirpath, "Boozer_surface_free_bdry_K.png" ))
    pp.savefig()

    fig2,ax2=plot_boozer_surface(eq_fb_no_K)
    ax2.set_title("Free bdry (no sheet current) |B|")
    plt.savefig(os.path.join(figs_dirpath, "Boozer_surface_free_bdry_no_K.png" ))
    pp.savefig()

    fig,ax = plot_2d(eq, "B*n", field=ext_field, field_grid=LinearGrid(N=150))
    ax.set_title(r"$B_n$ Fixed Bdry")
    plt.savefig(os.path.join(figs_dirpath, "fixed_bdry_Bn_error.png" ))
    pp.savefig()

    fig,ax = plot_2d(eq_fb_K, "B*n", field=ext_field, field_grid=LinearGrid(N=150))
    ax.set_title(r"$B_n$ Free Bdry (with sheet current)")
    plt.savefig(os.path.join(figs_dirpath, "free_bdry_K_Bn_error.png" ))
    pp.savefig()

    fig,ax = plot_2d(eq_fb_no_K, "B*n", field=ext_field, field_grid=LinearGrid(N=150))
    ax.set_title(r"$B_n$ Free Bdry (no sheet current)")
    plt.savefig(os.path.join(figs_dirpath, "free_bdry_no_K_Bn_error.png" ))
    pp.savefig()
    try:
        fig,ax = plot_2d(eq_fb_K.surface, "K")
        plt.savefig(os.path.join(figs_dirpath, "free_bdry_K_sheet_current.png" ))
        pp.savefig()
    except:
        print(f"error in plotting K, likely K=0")
    pp.close()
    plt.close("all")



wout_filepath = sys.argv[1]
path_to_coilset = sys.argv[2]
run_free_boundary(wout_filepath, path_to_coilset)


