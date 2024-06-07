#!/usr/bin/env python3

# This script is run after "stage_2_scan.py" has generated some optimized coils.
# This script reads the results.json files in the subdirectories, plots the
# distribution of results, filters out unacceptable runs, and prints out runs that
# are Pareto-optimal.

import os
import re
import json
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from simsopt import load
from paretoset import paretoset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from simsopt.mhd import VirtualCasing
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves)
parent_path = os.path.dirname(os.path.abspath(__file__))
###########################################
os.chdir(parent_path)
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=1)
args = parser.parse_args()
if   args.type == 1: QA_or_QH = 'nfp2_QA'
elif args.type == 2: QA_or_QH = 'nfp4_QH'
elif args.type == 3: QA_or_QH = 'nfp3_QA'
elif args.type == 4: QA_or_QH = 'nfp3_QH'
elif args.type == 5: QA_or_QH = 'nfp1_QI'
elif args.type == 6: QA_or_QH = 'nfp2_QI'
elif args.type == 7: QA_or_QH = 'nfp3_QI'
elif args.type == 8: QA_or_QH = 'nfp4_QI'
else: raise ValueError('Invalid type')
###########################################
# Directories
directory = f'optimization_finitebeta_{QA_or_QH}_stage1'
this_path = os.path.join(parent_path, directory)
os.chdir(this_path)
out_dir = os.path.join(this_path,"coils","scan")
os.chdir(out_dir)
print_file=os.path.join(this_path,'optimal_coils.txt')
# Initialize an empty DataFrame
df = pd.DataFrame()

results = glob.glob("*/results.json")
# df = None
for results_file in results:
    with open(results_file, "r") as f:
        data = json.load(f)

    # Wrap lists in another list
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = [value]

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
df = df[df["max_max_curvature"] < 0.8]
# df = df[df["ncoils"] == 3]
#########################################################
# Here you can define criteria to filter out the most interesting runs.
#########################################################

succeeded = df["linking_number"] < 0.1
if QA_or_QH == 'nfp2_QA':
    # df = df[df["max_max_curvature"] < 0.8]
    succeeded = np.logical_and(succeeded, df["Jf"]                         < 1.0e-3)
    # succeeded = np.logical_and(succeeded, df["ncoils"]                     < 5)
    succeeded = np.logical_and(succeeded, df["coil_coil_distance"]         > 2.0)
    succeeded = np.logical_and(succeeded, df["max_max_curvature"]          < 0.6)
    succeeded = np.logical_and(succeeded, df["max_mean_squared_curvature"] < 0.10)
    succeeded = np.logical_and(succeeded, df["coil_surface_distance"]      > 3.0)
    succeeded = np.logical_and(succeeded, df["average_length_per_coil"]    < 52)
elif QA_or_QH == 'nfp4_QH':
    succeeded = np.logical_and(succeeded, df["Jf"]                         < 4.0e-3)
    # succeeded = np.logical_and(succeeded, df["coil_coil_distance"]         > 0.5)
    # succeeded = np.logical_and(succeeded, df["max_max_curvature"]          < 1.0)
    # succeeded = np.logical_and(succeeded, df["max_mean_squared_curvature"] < 0.2)
    # succeeded = np.logical_and(succeeded, df["coil_surface_distance"]      > 1.0)
    # succeeded = np.logical_and(succeeded, df["average_length_per_coil"]    < 50)
elif QA_or_QH == 'nfp3_QA':
    succeeded = np.logical_and(succeeded, df["Jf"]                         < 6.0e-4)
    succeeded = np.logical_and(succeeded, df["coil_coil_distance"]         > 1.5)
    succeeded = np.logical_and(succeeded, df["max_max_curvature"]          < 0.4)
    succeeded = np.logical_and(succeeded, df["max_mean_squared_curvature"] < 0.11)
    succeeded = np.logical_and(succeeded, df["coil_surface_distance"]      > 2.5)
    succeeded = np.logical_and(succeeded, df["average_length_per_coil"]    < 50)
elif QA_or_QH == 'nfp3_QH':
    succeeded = np.logical_and(succeeded, df["Jf"]                         < 1.5e-3)
    succeeded = np.logical_and(succeeded, df["coil_coil_distance"]         > 0.6)
    succeeded = np.logical_and(succeeded, df["max_max_curvature"]          < 3.0)
    succeeded = np.logical_and(succeeded, df["max_mean_squared_curvature"] < 0.2)
    succeeded = np.logical_and(succeeded, df["coil_surface_distance"]      > 1.5)
    succeeded = np.logical_and(succeeded, df["average_length_per_coil"]    < 49)
elif QA_or_QH == 'nfp1_QI':
    succeeded = np.logical_and(succeeded, df["Jf"]                         < 4.0e-3)
    succeeded = np.logical_and(succeeded, df["coil_coil_distance"]         > 1.0)
    succeeded = np.logical_and(succeeded, df["max_max_curvature"]          < 1.0)
    succeeded = np.logical_and(succeeded, df["max_mean_squared_curvature"] < 0.3)
    succeeded = np.logical_and(succeeded, df["coil_surface_distance"]      > 1.2)
    succeeded = np.logical_and(succeeded, df["average_length_per_coil"]    < 52)
elif QA_or_QH == 'nfp2_QI':
    succeeded = np.logical_and(succeeded, df["Jf"]                         < 2.0e-3)
    succeeded = np.logical_and(succeeded, df["coil_coil_distance"]         > 1.0)
    succeeded = np.logical_and(succeeded, df["max_max_curvature"]          < 0.8)
    succeeded = np.logical_and(succeeded, df["max_mean_squared_curvature"] < 0.3)
    succeeded = np.logical_and(succeeded, df["coil_surface_distance"]      > 1.7)
    succeeded = np.logical_and(succeeded, df["average_length_per_coil"]    < 52)
elif QA_or_QH == 'nfp3_QI':
    succeeded = np.logical_and(succeeded, df["Jf"]                         < 5.0e-3)
    succeeded = np.logical_and(succeeded, df["coil_coil_distance"]         > 1.0)
    succeeded = np.logical_and(succeeded, df["max_max_curvature"]          < 2.0)
    succeeded = np.logical_and(succeeded, df["max_mean_squared_curvature"] < 0.3)
    succeeded = np.logical_and(succeeded, df["coil_surface_distance"]      > 1.5)
    succeeded = np.logical_and(succeeded, df["average_length_per_coil"]    < 50)
elif QA_or_QH == 'nfp4_QI':
    succeeded = np.logical_and(succeeded, df["Jf"]                         < 3.0e-2)
    succeeded = np.logical_and(succeeded, df["coil_coil_distance"]         > 0.65)
    succeeded = np.logical_and(succeeded, df["max_max_curvature"]          < 0.9)
    succeeded = np.logical_and(succeeded, df["max_mean_squared_curvature"] < 0.3)
    succeeded = np.logical_and(succeeded, df["coil_surface_distance"]      > 0.8)
    succeeded = np.logical_and(succeeded, df["average_length_per_coil"]    < 52)

#########################################################
# End of filtering criteria
#########################################################
with open(print_file, 'w') as f: print('best_from_coil_optimization_scan.py', file=f)
def print_both(*args, **kwargs):
    # Print to console
    print(*args, **kwargs)
    # Print to file
    with open(print_file, 'a') as f:
        print(*args, **kwargs, file=f)

df_filtered = df[succeeded]
print_both(f"Number of runs before filtering: {len(df)}")
print_both(f"Number of runs after filtering: {len(df_filtered)}")

pareto_mask = paretoset(df_filtered[["Jf", "max_max_curvature", "coil_coil_distance", "coil_surface_distance"]], sense=[min, min, max, max])
df_pareto = df_filtered[pareto_mask]

print_both(f"Best Pareto-optimal results (total of {len(df_pareto)}):")
print_both(
    df_pareto[
        [
            "directory",
            "Jf",
            "max_max_curvature",
            "average_length_per_coil",
            "max_mean_squared_curvature",
            "coil_coil_distance",
            "coil_surface_distance",
        ]
    ]
)
print_both("Directory names only:")
for dirname in df_pareto["directory"]:
    print_both(dirname)
#########################################################
# Plotting
#########################################################

os.chdir(this_path)

plt.figure(figsize=(14.5, 8))
plt.rc("font", size=8)
nrows = 4
ncols = 5
markersize = 5

subplot_index = 1
plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(df["Jf"], df["max_max_curvature"], c=df["length"], s=1)
plt.colorbar(label="length")
plt.scatter(
    df_filtered["Jf"],
    df_filtered["max_max_curvature"],
    c=df_filtered["length"],
    s=markersize,
)
plt.scatter(
    df_pareto["Jf"], df_pareto["max_max_curvature"], c=df_pareto["length"], marker="+"
)
plt.xlabel("Bnormal objective")
plt.ylabel("Max curvature")
plt.xscale("log")

plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(
    df_filtered["length_target"],
    df_filtered["length"],
    c=df_filtered["Jf"],
    s=markersize,
    norm=colors.LogNorm(),
)
plt.colorbar(label="Bnormal objective")
plt.xlabel("length_target")
plt.ylabel("length")

plt.subplot(nrows, ncols, subplot_index)
subplot_index += 1
plt.scatter(
    df_filtered["max_curvature_threshold"],
    df_filtered["max_max_curvature"],
    c=df_filtered["Jf"],
    s=markersize,
    norm=colors.LogNorm(),
)
plt.colorbar(label="Bnormal objective")
plt.xlabel("max_curvature_threshold")
plt.ylabel("max_max_curvature")


def plot_2d_hist(field, log=False):
    global subplot_index
    plt.subplot(nrows, ncols, subplot_index)
    subplot_index += 1
    nbins = 20
    if log:
        data = df[field]
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), nbins)
    else:
        bins = nbins
    plt.hist(df[field], bins=bins, label="before filtering")
    plt.hist(df_filtered[field], bins=bins, alpha=1, label="after filtering")
    plt.xlabel(field)
    plt.legend(loc=0, fontsize=6)
    if log:
        plt.xscale("log")


# 2nd entry of each tuple is True if the field should be plotted on a log x-scale.
fields = (
    ("R1", False),
    ("order", False),
    ("length", False),
    ("length_target", False),
    ("length_weight", True),
    ("max_curvature_threshold", False),
    ("max_curvature_weight", True),
    ("max_max_curvature", False),
    ("msc_threshold", False),
    ("msc_weight", True),
    ("coil_coil_distance", False),
    ("cc_threshold", False),
    ("cc_weight", True),
    ("coil_surface_distance", False),
    ("cs_threshold", False),
    ("cs_weight", True),
    ("arclength_weight", True),
)

for field, log in fields:
    plot_2d_hist(field, log)

plt.figtext(0.5, 0.995, os.path.abspath(__file__), ha="center", va="top", fontsize=6)
plt.tight_layout()
# plt.show()
plt.savefig('coil_scan.png', dpi=300)

nfp = int(QA_or_QH.split('_')[0][3:])
nphi = 32
ntheta = 32
nphi_big = nphi * 2 * nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi = np.linspace(0, 1, nphi_big)
filename = os.path.join(this_path,'input.final')
surf = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
surf_big = SurfaceRZFourier(dofs=surf.dofs,nfp=nfp, mpol=surf.mpol,ntor=surf.ntor,
                            quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta)

def process_surface_and_flux(bs, surf, ncoils, R1, surf_big=None, new_OUT_DIR="", prefix="surf_opt", sign_B_external_normal=1.0):
    vc = VirtualCasing.load(os.path.join(this_path,'vcasing_final.nc'))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    initial_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=surf.get_rc(0, 0), R1=R1, order=5, numquadpoints=120)
    curves_to_vtk(initial_curves, os.path.join(new_OUT_DIR,"curves_init"), close=True)
    curves = [c.curve for c in bs.coils]
    curves_to_vtk(curves, os.path.join(new_OUT_DIR,"curves_opt_big"), close=True)
    base_curves = curves[:ncoils]
    curves_to_vtk(base_curves, os.path.join(new_OUT_DIR,"curves_opt"), close=True)
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    # sign_B_external_normal=-1.0
    BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2) - sign_B_external_normal*vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    Bmod = bs.AbsB().reshape((nphi,ntheta,1))
    pointData = {"B.n/B": BdotN[:, :, None], "B": Bmod}
    surf.to_vtk(os.path.join(new_OUT_DIR, prefix), extra_data=pointData)
    if surf_big is not None:
        bs.set_points(surf_big.gamma().reshape((-1, 3)))
        Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
        BdotN = (np.sum(Bbs * surf_big.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
        Bmod = bs.AbsB().reshape((nphi_big,ntheta_big,1))
        pointData = {"B.n/B": BdotN[:, :, None], "B": Bmod}
        surf_big.to_vtk(os.path.join(new_OUT_DIR, prefix + "_big"), extra_data=pointData)

# Copy the best results to a separate directory
optimal_coils_path = os.path.join(out_dir, "..", "optimal_coils")
Path(optimal_coils_path).mkdir(parents=True, exist_ok=True)
if os.path.exists(optimal_coils_path):
    shutil.rmtree(optimal_coils_path)
for dirname in df_pareto["directory"]:
    ncoils = int(  re.search(r"ncoils_(\d+)", dirname).group(1))
    R1     = float(re.search(r"R1_([\d.]+)",  dirname).group(1))
    try:    sign_B_external_normal = float(re.search(r"sign_B_external_normal([\d.]+)",  dirname).group(1))
    except: sign_B_external_normal = -1.0
    source_dir = os.path.join(out_dir, dirname)
    destination_dir = os.path.join(optimal_coils_path, dirname)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
    bs=load(os.path.join(source_dir,"biot_savart.json"))
    process_surface_and_flux(bs, surf, ncoils=ncoils, R1=R1, surf_big=surf_big, new_OUT_DIR=destination_dir, sign_B_external_normal=sign_B_external_normal)
    