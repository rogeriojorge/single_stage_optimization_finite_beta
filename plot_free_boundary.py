#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Run free-bdry DESC on a finite beta eq given from a VMEC .nc file
First cmd line arg: path to the VMEC wout file
Second cmd line arg: path to the MAKEGRID coils file for this equilibrium
Third cmd line arg : path to the VMEC input file (so DESC can get the pressure profile)
"""
from desc import set_device

# set_device("gpu")

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

from scipy.constants import mu_0
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
from netCDF4 import Dataset

def plot_free_boundary(wout_filename, path_to_coilset, nn=7, maxiter=50):
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
    ext_field = desc.io.load(path_to_coilset)

    path_init_fixed_solve = subfoldername + "/" + f"desc_initial_fixed_bdry_solve.h5"
    path_no_K = os.path.join(coilfolder_path, "desc_fb_no_sheet_current.h5")
    path_with_K = os.path.join(coilfolder_path, "desc_fb_with_sheet_current.h5")
    path_no_K_final = os.path.join(coilfolder_path, "desc_fb_no_sheet_current_final.h5")
    path_with_K_final = os.path.join(coilfolder_path, "desc_fb_with_sheet_current_final.h5")

    vmec = Dataset(wout_filename, "r")

    # finally, make plots
    eq = desc.io.load(path_init_fixed_solve)

    eq_fb_no_K = desc.io.load(path_no_K_final)[-1]
    eq_fb_K = desc.io.load(path_with_K_final)[-1]

    figs_dirpath = os.path.join(coilfolder_path, "figs")
    if not os.path.exists(figs_dirpath):
        os.mkdir(figs_dirpath)
    
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

    summary_pdf_path = os.path.join(figs_dirpath, "summary.pdf" )



    fig, axes = plt.subplot_mosaic(
    [
        ["top left", "top center", "top right"],
        ["bottom left", "bottom center", "bottom right"],
    ],
    figsize=(24, 18),
    gridspec_kw={"wspace": 0.425, "hspace": 0.265},
)
    plt.rcParams.update({"font.size":24})

    # pp = PdfPages(summary_pdf_path)

    s_vmec = vmec.variables["phi"][:].filled()
    rho_vmec = np.sqrt(np.abs(s_vmec/s_vmec[-1]))
    plt.rcParams.update({"font.size":18})
    plot_1d(eq, "p", label="DESC")
    plt.plot(rho_vmec, vmec.variables["presf"][:].filled(),"r--",label="VMEC")
    plt.legend()
    plt.savefig(os.path.join(figs_dirpath, "pressure_profile_comparison_DESC_VMEC.png" ))
    # pp.savefig()

    plot_1d(eq, "current", label="DESC")
    plt.plot(rho_vmec, 2 * np.pi / mu_0 * vmec.variables["buco"][:].filled(),"r--",label="VMEC")
    plt.legend()
    plt.savefig(os.path.join(figs_dirpath, "current_profile_comparison_DESC_VMEC.png" ))
    # pp.savefig()

    # plot_comparison([eq,eq_fb_no_K],labels=["fixed","fb no K"])
    # plt.savefig(os.path.join(figs_dirpath, "surface_comparison_FB_fixed_no_surface_current.png" ))
    # pp.savefig()


    # plot_comparison([eq,eq_fb_K],labels=["fixed","fb K"])
    # plt.savefig(os.path.join(figs_dirpath, "surface_comparison_FB_fixed_with_surface_current.png" ))
    # pp.savefig()

    plot_1d(eq_fb_K,"iota",label="fb K",figsize=(5,5),ax=axes["top left"])
    plot_1d(eq_fb_no_K,"iota",label="fb no K",figsize=(5,5),ax=axes["top left"],linecolor="m")
    plot_1d(eq,"iota",label="fixed bdry",ax=axes["top left"],linecolor="r")
    # plt.savefig(os.path.join(figs_dirpath, "iota_comparison_plot.png" ))
    # pp.savefig()

    plot_1d(eq_fb_K,"D_Mercier",label="fb K",log=True,figsize=(5,5),ax=axes["bottom left"])
    plot_1d(eq_fb_no_K,"D_Mercier",label="fb no K",figsize=(5,5),ax=axes["bottom left"],linecolor="m",log=True)
    plot_1d(eq,"D_Mercier",label="fixed bdry",ax=axes["bottom left"],linecolor="r",log=True)
    # plt.savefig(os.path.join(figs_dirpath, "Dmerc_comparison_plot.png" ))
    # pp.savefig()
    
    plot_boozer_surface(eq,ax=axes["top center"])
    axes["top center"].set_title("Fixed bdry |B|")
    # plt.savefig(os.path.join(figs_dirpath, "Boozer_surface_fixed_bdry.png" ))
    # pp.savefig()

    plot_boozer_surface(eq_fb_K,ax=axes["bottom center"])
    axes["bottom center"].set_title("Free bdry (with sheet current) |B|")
    # plt.savefig(os.path.join(figs_dirpath, "Boozer_surface_free_bdry_K.png" ))
    # pp.savefig()

    # fig2,ax2=plot_boozer_surface(eq_fb_no_K)
    # ax2.set_title("Free bdry (no sheet current) |B|")
    # plt.savefig(os.path.join(figs_dirpath, "Boozer_surface_free_bdry_no_K.png" ))
    # pp.savefig()

    plot_2d(eq, "B*n", field=ext_field, field_grid=LinearGrid(N=150),ax=axes["top right"])
    axes["top right"].set_title(r"$B_n$ Fixed Bdry")
    # # plt.savefig(os.path.join(figs_dirpath, "fixed_bdry_Bn_error.png" ))
    # # pp.savefig()

    plot_2d(eq_fb_K, "B*n", field=ext_field, field_grid=LinearGrid(N=150),ax=axes["bottom right"])
    axes["bottom right"].set_title(r"$B_n$ Free Bdry (with sheet current)")

    fig.savefig(os.path.join(figs_dirpath, "results_summary_figure.png" ))
    # # plt.savefig(os.path.join(figs_dirpath, "free_bdry_K_Bn_error.png" ))
    # # pp.savefig()

    # # fig,ax = plot_2d(eq_fb_no_K, "B*n", field=ext_field, field_grid=LinearGrid(N=150))
    # # ax.set_title(r"$B_n$ Free Bdry (no sheet current)")
    # # plt.savefig(os.path.join(figs_dirpath, "free_bdry_no_K_Bn_error.png" ))
    # # pp.savefig()
    # # try:
    # #     fig,ax = plot_2d(eq_fb_K.surface, "K")
    # #     plt.savefig(os.path.join(figs_dirpath, "free_bdry_K_sheet_current.png" ))
    # #     pp.savefig()
    # # except:
    # #     print(f"error in plotting K, likely K=0")
    # # pp.close()
    plt.close("all")
    


# we always know the wout file is "wout_final.nc" and is inside the folder starting with "optimization"
# so let's here just look for the path to the DESC coilset, then from there, find the woutfile path to pass to the above free bdry script
for dirpath, dirnames, filenames in os.walk(os.getcwd()):
    if dirpath.find("optimal_coil") != -1 and filenames and dirpath.find("ipynb") == -1: # we reached one of the folders below optimal_coils that contains coil files (and we are not in a jupyter checkpoint folder)
        if dirpath.find("ipynb"): 
            pass # skip jupyter stuff
        found_a_DESC_coilset = False
        coilset_filename = ""
        for file in filenames:
            if file.find("coilset_desc") !=1 and file.find(".h5") != -1 and file.find("nfp") != -1: # is a coilset_DESC...h5 file
                found_a_DESC_coilset = True
                coilset_filename = file
                
        if found_a_DESC_coilset:
            path_to_coilset = os.path.join(dirpath, coilset_filename)
            # now get path to the folder starting with optimization... where the wout_final.nc lives
            path_split = dirpath.split("/")
            for i in range(len(path_split)):
                if path_split[i].find("optimization_") != -1 and path_split[i].find("single_stage_optimization_finite_beta") == -1: # last part is to jump past the topmost folder
                    break
            wout_filepath = "/" + os.path.join(*path_split[0:i+1], "wout_final.nc")
            print(f"{wout_filepath=}")
            print(f"{path_to_coilset=}")
            try:
                plot_free_boundary(wout_filepath, path_to_coilset)
            except Exception as e:
                print(e)

