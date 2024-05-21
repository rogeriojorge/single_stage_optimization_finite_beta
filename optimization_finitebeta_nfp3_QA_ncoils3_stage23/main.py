#!/usr/bin/env python3
import os
import re
import time
import glob
import shutil
import argparse
import numpy as np
from math import isnan
from simsopt import load
from pathlib import Path
from scipy.optimize import minimize
from simsopt import make_optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.solve import least_squares_mpi_solve
from simsopt.util.constants import ELEMENTARY_CHARGE
from simsopt.util import MpiPartition, proc0_print, comm_world
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.mhd.bootstrap import RedlGeomBoozer, VmecRedlBootstrapMismatch, RedlGeomVmec
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual, VirtualCasing, Boozer
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
from simsopt.mhd.profiles import ProfilePolynomial, ProfilePressure, ProfileScaled, ProfileSpline
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature, SurfaceRZFourier, LinkingNumber,
                         LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves, CurveSurfaceDistance)
mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=1)
# parser.add_argument("--ncoils", type=int, default=4)
parser.add_argument("--stage1_coils", dest="stage1_coils", default=False, action="store_true")
parser.add_argument("--stage1", dest="stage1", default=False, action="store_true")
parser.add_argument("--stage2", dest="stage2", default=False, action="store_true")
parser.add_argument("--stage3", dest="stage3", default=False, action="store_true")
args = parser.parse_args()
if   args.type == 1: QA_or_QH = 'nfp2_QA'
elif args.type == 2: QA_or_QH = 'nfp4_QH'
elif args.type == 3: QA_or_QH = 'nfp3_QA'
elif args.type == 4: QA_or_QH = 'nfp3_QH'
elif args.type == 5: QA_or_QH = 'nfp3_QI'
else: raise ValueError('Invalid type')
# ncoils = args.ncoils
##########################################################################################
############## Input parameters
##########################################################################################
use_original_vmec_inut = False # if true, use nfp2_QA_original or nfp4_QH_original
optimize_stage1 = args.stage1
optimize_stage1_with_coils = args.stage1_coils
optimize_stage2 = args.stage2
optimize_stage3 = args.stage3
MAXITER_stage_1 = 40
MAXITER_stage_2 = 600
tol_coils       = 1e-7
MAXITER_single_stage = 40
MAXFEV_single_stage  = 50

#### INITIAL COILS PROPERTIES BEING OBTAINED FROM OPTIMAL_COILS_FINAL FOLDER
if QA_or_QH == 'nfp2_QA':
    max_mode_array                    = [1] *2 + [2] * 0 + [3] * 0 + [4] * 0 + [5] * 0 + [6] * 0
    # quasisymmetry_weight_mpol_mapping = {1: 1e+1, 2: 1e+2,  3: 4e+2,  4: 7e+2,  5: 8e+2}
    # DMerc_weight_mpol_mapping         = {1: 6e+9, 2: 2e+13, 3: 1e+14, 4: 3e+14, 5: 4e+14}
    # DMerc_fraction_mpol_mapping       = {1: 0.7,  2: 0.15,  3: 0.1,   4: 0.05,  5: 0.05}
    quasisymmetry_weight_mpol_mapping = {1: 2e+1, 2: 1e+2,  3: 4e+2,  4: 7e+2,  5: 8e+2}
    DMerc_weight_mpol_mapping         = {1: 2e+12, 2: 2e+13, 3: 1e+14, 4: 3e+14, 5: 4e+14}
    DMerc_fraction_mpol_mapping       = {1: 0.1,  2: 0.15,  3: 0.1,   4: 0.05,  5: 0.05}
    coils_objective_array = [1e3, 1.1e3, 1.2e3, 1.5e3, 1.8e3, 2.2e3, 2.5e3, 3.3e3, 3.4e3, 3.6e3, 4.0e3]
    JACOBIAN_THRESHOLD_array = [7e3, 5e2, 3e2, 2e2, 1e2]
    aspect_ratio_target = 6.5
    max_iota            = 0.9
    min_iota            = 0.15
    min_average_iota    = 0.41
    ncoils              = 4
    nmodes_coils        = 5
    R0                  = 11.14
    R1                  = 0.59*R0
    LENGTH_THRESHOLD    = (4.2-0.1)*R0
    LENGTH_CON_WEIGHT   = 0.69
    CURVATURE_THRESHOLD = (3.8-0.3)/R0
    CURVATURE_WEIGHT    = 7.8e-5
    MSC_THRESHOLD       = (15.6-2)/R0
    MSC_WEIGHT          = 9.4e-5
    CC_THRESHOLD        = (0.17+0.03)*R0
    CC_WEIGHT           = 5.2e-1
    CS_THRESHOLD        = (0.17+0.03)*R0
    CS_WEIGHT           = 8.2e1
    ARCLENGTH_WEIGHT    = 3.5e-5
    bootstrap_mismatch_weight = 1e1
elif QA_or_QH == 'nfp4_QH':
    max_mode_array                    = [1] * 0 + [2] * 0 + [3] * 1 + [4] * 0 + [5] * 0 + [6] * 0
    quasisymmetry_weight_mpol_mapping = {1: 3e+2,  2: 5e+2,  3: 7e+2,  4: 8e+2,  5: 9e+2}
    DMerc_weight_mpol_mapping         = {1: 2e+13, 2: 5e+13, 3: 1e+14, 4: 3e+14, 5: 4e+14}
    DMerc_fraction_mpol_mapping       = {1: 0.1,   2: 0.05,  3: 0.05,  4: 0.05,  5: 0.05}
    coils_objective_array = [2e2, 5e2, 1e3]
    JACOBIAN_THRESHOLD_array = [3e2, 2e2, 1e2, 9e1, 5e1]
    aspect_ratio_target = 5.0
    max_iota            = 1.9
    min_iota            = 1.02
    min_average_iota    = 1.05
    ncoils              = 4
    nmodes_coils        = 8
    R0                  = 11.5
    R1                  = 0.45*R0
    LENGTH_THRESHOLD    = (3.4-0.0)*R0
    LENGTH_CON_WEIGHT   = 0.012
    CURVATURE_THRESHOLD = (2.5-0.0)/R0
    CURVATURE_WEIGHT    = 1.5e-5
    MSC_THRESHOLD       = (1.7-0.0)/R0
    MSC_WEIGHT          = 2.0e-6
    CC_THRESHOLD        = (0.075-0.0)*R0
    CC_WEIGHT           = 1.4e+2
    CS_THRESHOLD        = (0.07-0.0)*R0
    CS_WEIGHT           = 6.0e-2
    ARCLENGTH_WEIGHT    = (5.1e-6-3.0e-6)
    bootstrap_mismatch_weight = 1e2
elif QA_or_QH == 'nfp3_QA':
    max_mode_array                    = [1] *1 + [2] * 7 + [3] * 0 + [4] * 0 + [5] * 0 + [6] * 0
    # quasisymmetry_weight_mpol_mapping = {1: 1e+1,  2: 1e+2,  3: 6e+2,  4: 7e+2,  5: 8e+2}
    # DMerc_weight_mpol_mapping         = {1: 1e+13, 2: 2e+13, 3: 1e+14, 4: 3e+14, 5: 4e+14}
    quasisymmetry_weight_mpol_mapping = {1: 5e+2,  2: 5e+2,  3: 6e+2,  4: 7e+2,  5: 8e+2}
    DMerc_weight_mpol_mapping         = {1: 1e+14, 2: 1e+14, 3: 1e+14, 4: 3e+14, 5: 4e+14}
    DMerc_fraction_mpol_mapping       = {1: 0.05,  2: 0.05,  3: 0.05,  4: 0.05,  5: 0.05}
    coils_objective_array    = [1e3, 1.1e3, 1.2e3, 1.3e3, 1.4e3, 1.5e3]
    JACOBIAN_THRESHOLD_array = [7e3, 5e2, 3e2, 2e2, 1e2]
    aspect_ratio_target = 6.5
    max_iota            = 0.9
    min_iota            = 0.25
    min_average_iota    = 0.55
    ncoils              = 3
    nmodes_coils        = 5
    R0                  = 11.14
    R1                  = 0.44*R0
    LENGTH_THRESHOLD    = (4.1-0.3)*R0
    LENGTH_CON_WEIGHT   = 0.13
    CURVATURE_THRESHOLD = (2.7-0.5)/R0
    CURVATURE_WEIGHT    = 6.0e-4-1.0e-4
    MSC_THRESHOLD       = (17.8-17.75)/R0
    MSC_WEIGHT          = (7.3e-4+1e-2)
    CC_THRESHOLD        = (0.14+0.06)*R0
    CC_WEIGHT           = 4.9e-1
    CS_THRESHOLD        = (0.215+0.025)*R0
    CS_WEIGHT           = 2.0e-2
    ARCLENGTH_WEIGHT    = (3.7e-4-0.00e-4)
    bootstrap_mismatch_weight = 1e2
elif QA_or_QH == 'nfp3_QH':
    max_mode_array                    = [1] *0 + [2] * 1 + [3] * 0 + [4] * 0 + [5] * 0 + [6] * 0
    quasisymmetry_weight_mpol_mapping = {1: 1e+1, 2: 4e+2,  3: 6e+2,  4: 7e+2,  5: 8e+2}
    DMerc_weight_mpol_mapping         = {1: 1e+7, 2: 7e+13, 3: 1e+14, 4: 3e+14, 5: 4e+14}
    DMerc_fraction_mpol_mapping       = {1: 0.8,  2: 0.05,  3: 0.05,  4: 0.05,  5: 0.05}
    coils_objective_array    = [1e3, 1.1e3, 1.2e3, 1.3e3, 1.4e3, 1.5e3]
    JACOBIAN_THRESHOLD_array = [7e3, 5e2, 3e2, 2e2, 1e2]
    aspect_ratio_target = 6.8
    max_iota            = 0.97
    min_iota            = 0.8
    min_average_iota    = 0.85
    nmodes_coils        = 15
    ncoils              = 4
    R0                  = 11.14
    R1                  = 0.5*R0
    LENGTH_THRESHOLD    = (3.95+0.05)*R0
    LENGTH_CON_WEIGHT   = 0.014
    CURVATURE_THRESHOLD = (2.34+0.06)/R0
    CURVATURE_WEIGHT    = (2.0e-3-1.0e-3)
    MSC_THRESHOLD       = (15.6-0.0)/R0
    MSC_WEIGHT          = 1.0e-6
    CC_THRESHOLD        = (0.14-0.0)*R0
    CC_WEIGHT           = 1.2e-1
    CS_THRESHOLD        = (0.19-0.0)*R0
    CS_WEIGHT           = 1.7e-2
    ARCLENGTH_WEIGHT    = 4.3e-9
    bootstrap_mismatch_weight = 1e1
else:
    raise ValueError('Invalid QA_or_QH (QI not implemented yet)')
# print('Compare the following directory with the one in the optimal_coils folder')
# coils_directory = (
#     f"ncoils_{ncoils}_order_{nmodes_coils}_R1_{R1:.2}_length_target_{LENGTH_THRESHOLD:.2}_weight_{LENGTH_CON_WEIGHT:.2}"
#     + f"_max_curvature_{CURVATURE_THRESHOLD:.2}_weight_{CURVATURE_WEIGHT:.2}"
#     + f"_msc_{MSC_THRESHOLD:.2}_weight_{MSC_WEIGHT:.2}"
#     + f"_cc_{CC_THRESHOLD:.2}_weight_{CC_WEIGHT:.2}"
#     + f"_cs_{CS_THRESHOLD:.2}_weight_{CS_WEIGHT:.2}"
#     + f"_arclweight_{ARCLENGTH_WEIGHT:.2}"
# )
# print('   '+coils_directory)

maxmodes_mpol_mapping = {1: 5,    2: 5,     3: 5,     4: 6,     5: 7, 6: 7}
optimize_DMerc = True
optimize_Well  = False
optimize_aminor = False
optimize_mean_iota = True
# JACOBIAN_THRESHOLD = 100 #30
aspect_ratio_weight = 1e+2
aminor_weight = 5e-2
# quasisymmetry_weight = 1e+1
weight_iota = 1e5
volavgB_weight = 5e+0
well_Weight = 1e2
# DMerc_Weight = 1e+10
betatotal_weight = 1e1
aminor_target = 1.70442622782386
volavgB_target = 5.86461221551616
nphi_VMEC   = 28
ntheta_VMEC = 28
vc_src_nphi = ntheta_VMEC
ftol = 1e-4
nquadpoints = 120
diff_method = "forward"
opt_method = 'trf'#'lm'
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
finite_difference_abs_step = 1e-6
finite_difference_rel_step = 0#1e-4
ftol_stage_1 = 1e-5
rel_step_stage1 = 2e-5
abs_step_stage1 = 2e-7
initial_DMerc_index = 2
## Self-consistent bootstrap current
beta = 2.5 #%
ne0 = 3e20 * (beta/100/0.05)**(1/3)
Te0 = 15e3 * (beta/100/0.05)**(2/3)
# ne = ProfilePolynomial(ne0 * np.array([1, 0, 0, 0, 0, -1.0]))
# Te = ProfilePolynomial(Te0 * np.array([1, -1.0]))
### Look at experimental profiles of ne and Te (W7-X?)
ne = ProfilePolynomial(ne0 * np.array([1, 0, 0, 0, 0, -0.98]))
Te = ProfilePolynomial(Te0 * np.array([1, -0.98]))
Zeff = 1.0
ni = ne
Ti = Te
pressure = ProfilePressure(ne, Te, ni, Ti)
pressure_Pa = ProfileScaled(pressure, ELEMENTARY_CHARGE)
######
vmec_input_filename = os.path.join(parent_path, 'vmec_inputs', 'input.'+ QA_or_QH + ('_original' if use_original_vmec_inut else ''))
directory = f'optimization_finitebeta_{QA_or_QH}'
if optimize_stage2 or optimize_stage3: directory += f'_ncoils{ncoils}'
directory += f'_stage'
if optimize_stage1: directory += '1'
if optimize_stage1_with_coils: directory += '1c'
if optimize_stage2: directory += '2'
if optimize_stage3: directory += '3'
helicity_n=-1 if 'QH' in QA_or_QH  else 0
##########################################################################################
##########################################################################################
vmec_verbose = False
# Create output directories
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
shutil.copyfile(os.path.join(parent_path, 'main.py'), os.path.join(this_path, 'main.py'))
os.chdir(this_path)
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm_world.rank == 0:
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)
##########################################################################################
##########################################################################################
# Stage 1
proc0_print(f' Using vmec input file {vmec_input_filename}')
vmec = Vmec(vmec_input_filename, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
surf = vmec.boundary
R0 = surf.get_rc(0, 0)
nphi_big   = nphi_VMEC * 2 * surf.nfp + 1
ntheta_big = ntheta_VMEC + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi   = np.linspace(0, 1, nphi_big)
surf_big = SurfaceRZFourier(dofs=surf.dofs, nfp=surf.nfp, mpol=surf.mpol, ntor=surf.ntor, quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta, stellsym=surf.stellsym)
## Set pressure and current for self-consistent bootstrap current
vmec.unfix('phiedge')
vmec.pressure_profile = pressure_Pa
vmec.n_pressure = 7
vmec.indata.pcurr_type = 'cubic_spline_ip'
vmec.n_current = 50
## Finite Beta Virtual Casing Principle
vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
total_current_vmec = vmec.external_current() / (2 * surf.nfp)
##########################################################################################
##########################################################################################
# Stage 2
def parse_directory_name(directory_name):
    pattern = r"ncoils_(\d+)_order_(\d+)_R1_([^_]+)_length_target_([^_]+)_weight_([^_]+)_max_curvature_([^_]+)_weight_([^_]+)_msc_([^_]+)_weight_([^_]+)_cc_([^_]+)_weight_([^_]+)_cs_([^_]+)_weight_([^_]+)_arclweight_([^_]+)"
    match = re.match(pattern, directory_name)
    if match:
        return {
            'ncoils': int(match.group(1)), 'nmodes_coils': int(match.group(2)), 'R1': float(match.group(3)),
            'LENGTH_THRESHOLD': float(match.group(4)), 'LENGTH_CON_WEIGHT': float(match.group(5)),
            'CURVATURE_THRESHOLD': float(match.group(6)), 'CURVATURE_WEIGHT': float(match.group(7)),
            'MSC_THRESHOLD': float(match.group(8)), 'MSC_WEIGHT': float(match.group(9)),
            'CC_THRESHOLD': float(match.group(10)), 'CC_WEIGHT': float(match.group(11)),
            'CS_THRESHOLD': float(match.group(12)), 'CS_WEIGHT': float(match.group(13)),
            'ARCLENGTH_WEIGHT': float(match.group(14))
        }
    return None
# Select directory with optimal coils
def select_directory(directories, ncoils, nmodes):
    for dir_name in directories:
        params = parse_directory_name(dir_name)
        if params and params['ncoils'] == ncoils and params['nmodes_coils'] == nmodes:
            return dir_name, params  # Return both the directory name and parameters
    return None, None  # Return None if no matching directory is found
optimal_coils_directory = os.path.join(parent_path, f'optimization_finitebeta_{QA_or_QH}_stage1','coils','optimal_coils_final')
directories = os.listdir(optimal_coils_directory)
selected_directory, coils_params = select_directory(directories, ncoils=ncoils, nmodes=nmodes_coils)
proc0_print(f"Selected directory: {selected_directory}")
proc0_print(f"Coils parameters: {coils_params}")
# Define function for creating coils from scratch
def create_coils_from_scratch(ncoils, R0, R1, nmodes_coils):
    base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=nmodes_coils, numquadpoints=128)
    base_currents = [Current(total_current_vmec / ncoils * 1e-7) * 1e7 for _ in range(ncoils - 1)]
    total_current = Current(total_current_vmec)
    base_currents += [total_current - sum(base_currents)]
    return base_curves, base_currents
# Create base_curve and base_currents
if selected_directory:
    optimal_coils_path = os.path.join(optimal_coils_directory, selected_directory)
    try:
        bs_json_files = [file for file in os.listdir(optimal_coils_path) if 'biot_savart.json' in file]
        bs = load(os.path.join(optimal_coils_path, bs_json_files[0]))  # Assuming you want the first .json file
        proc0_print('Optimal coils found.')
        curves = [c.curve for c in bs.coils]
        currents = [c._current for c in bs.coils]
        base_curves = curves[:ncoils]
        base_currents = currents[:ncoils]
    except Exception as e:
        print(e)
        proc0_print('Error reading coil files, falling back to creating coils from scratch.')
        base_curves, base_curves = create_coils_from_scratch(ncoils, selected_directory['R0'], selected_directory['R1'], selected_directory['nmodes_coils'])
else:
    proc0_print("No matching directory found. Creating coils from scratch.")
    if 'R1' not in globals(): R1 = 0.59 * R0
    coils_params={'R0': R0, 'R1': R1}
    base_curves, base_currents = create_coils_from_scratch(ncoils, R0, R1, nmodes_coils)
## Create coils
# optimal_coils_path = os.path.join(parent_path, f'optimization_finitebeta_{QA_or_QH}_stage1','coils','optimal_coils_final',coils_directory)
# try:
#     bs_json_files = [file for file in os.listdir(optimal_coils_path) if '.json' in file]
#     bs = load(os.path.join(optimal_coils_path, bs_json_files[1]))
#     print(' Optimal coils found.')
#     curves   = [c.curve    for c in bs.coils]
#     currents = [c._current for c in bs.coils]
#     base_curves = curves[:ncoils]
#     base_currents = currents[:ncoils]
# except Exception as e:
#     print(e)
#     print(' No optimal coils found. Creating coils from scratch')
#     base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=nmodes_coils, numquadpoints=128)
#     base_currents = [Current(total_current_vmec / ncoils * 1e-7) * 1e7 for _ in range(ncoils-1)]
#     total_current = Current(total_current_vmec)
#     # total_current.fix_all()
#     base_currents += [total_current - sum(base_currents)]
coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, stellsym=True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)

##########################################################################################
##########################################################################################
# Save initial surface and coil data
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = (np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    pointData = {"B.n/B": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_init"), extra_data=pointData)
bs.set_points(surf_big.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
BdotN_surf = np.sum(Bbs * surf_big.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
if comm_world.rank == 0:
    pointData = {"Bcoils.n/B": BdotN_surf[:, :, None]}
    surf_big.to_vtk(os.path.join(coils_results_path, "surf_init_big"), extra_data=pointData)
bs.set_points(surf.gamma().reshape((-1, 3)))
##########################################################################################
##########################################################################################
Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(base_curves)]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
J_CC = CC_WEIGHT * Jccdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
J_CS = CS_WEIGHT * Jcsdist
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_THRESHOLD*ncoils)
linkNum = LinkingNumber(curves, 2)
JF = Jf + J_CC + J_LENGTH_PENALTY + J_CURVATURE + J_MSC + J_ALS + linkNum + J_CS
##########################################################################################
proc0_print('  Starting optimization')
global previous_J, coils_objective_weight, JACOBIAN_THRESHOLD, previous_previous_J
previous_J = 1e19
previous_previous_J = 1e19
coils_objective_weight = coils_objective_array[0]
JACOBIAN_THRESHOLD = JACOBIAN_THRESHOLD_array[0]
##########################################################################################
# Initial stage 2 optimization
##########################################################################################
def fun_coils(dofss, info):
    info['Nfeval'] += 1
    JF.x = dofss
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        BdotN = np.max(np.abs(((np.sum(Bbs * surf.unitnormal(), axis=2) - Jf.target)) / np.linalg.norm(Bbs, axis=2)))
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, max⟨B·n⟩/B={BdotN:.1e}"
        # outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
        cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{j.J():.2f}" for j in Jmscs)
        outstr += f" L=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}], msc=[{msc_string}]"
        outstr += f", C-S-Sep={Jcsdist.shortest_distance():.2f}"
        print(outstr)
        outstr  = f"fun_coils#{info['Nfeval']} "
        outstr += f"- JF: {J:.3e}"
        outstr += f", Jf: {jf:.3e}"
        outstr += f", J_CC: {J_CC.J():.2e}"
        outstr += f", J_CURVATURE: {J_CURVATURE.J():.2e}"
        outstr += f", J_MSC: {J_MSC.J():.2e}"
        outstr += f", J_CS: {J_CS.J():.2e}"
        outstr += f", J_ALS: {J_ALS.J():.2e}"
        outstr += f", J_LENGTH_PENALTY: {J_LENGTH_PENALTY.J():.2e}"
        # outstr += f", LinkingNumber: {linkNum.J():.2e}"
        print(outstr)
        # print(f"Currents: {[c.current.get_value() for c in coils]}")
    return J, grad
##########################################################################################
# Single stage optimization
##########################################################################################
def fun_J(prob, coils_prob):
    global previous_surf_dofs, coils_objective_weight, JACOBIAN_THRESHOLD, previous_J, previous_previous_J
    J_stage_1 = prob.objective()
    if np.any(previous_surf_dofs != prob.x):  # Only run virtual casing if surface dofs have changed
        previous_surf_dofs = prob.x
        try:
            vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
            Jf.target = vc.B_external_normal
        except ObjectiveFailure:
            pass
    bs.set_points(surf.gamma().reshape((-1, 3)))
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2
    return J
def fun(dofss, prob_jacobian=None, info={'Nfeval': 0}):
    global previous_J, coils_objective_weight, JACOBIAN_THRESHOLD, previous_J, previous_previous_J
    start_time = time.time()
    info['Nfeval'] += 1
    os.chdir(vmec_results_path)
    prob.x = dofss[-number_vmec_dofs:]
    coil_dofs = dofss[:-number_vmec_dofs]
    # Un-fix the desired coil dofs so they can be updated:
    JF.full_unfix(free_coil_dofs)
    JF.x = coil_dofs
    J = fun_J(prob, JF)
    if (info['Nfeval'] > MAXFEV_single_stage or (np.abs(J-previous_J)/previous_J < ftol and np.abs(J-previous_previous_J)/previous_J < ftol)) and J < JACOBIAN_THRESHOLD:
    # if info['Nfeval'] > MAXFEV_single_stage and J < JACOBIAN_THRESHOLD:
        return J, [0] * len(dofs)
    if J > JACOBIAN_THRESHOLD or isnan(J):
        proc0_print(f"fun#{info['Nfeval']}: Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * number_vmec_dofs
        grad_with_respect_to_coils   = [0] * len(coil_dofs)
    else:
        proc0_print(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
        coils_dJ = JF.dJ()
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        JF.fix_all()  # Must re-fix the coil dofs before beginning the finite differencing.
        grad_with_respect_to_surface = prob_jacobian.jac(prob.x)[0]
    JF.fix_all()
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
    previous_previous_J = previous_J
    previous_J = J
    # proc0_print(f"  Time taken = {time.time()-start_time:.2f}s")
    return J, grad
##########################################################################################
#############################################################
## Perform optimization
#############################################################
##########################################################################################
max_mode_previous = 0
free_coil_dofs_all = JF.dofs_free_status
for iteration, max_mode in enumerate(max_mode_array):
    proc0_print(f'###############################################')
    proc0_print(f'  Performing optimization for max_mode={max_mode}')
    proc0_print(f'###############################################')
    vmec.indata.mpol = maxmodes_mpol_mapping[max_mode]
    vmec.indata.ntor = maxmodes_mpol_mapping[max_mode]
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    
    coils_objective_weight = coils_objective_array[np.min((iteration, len(coils_objective_array)-1))]
    JACOBIAN_THRESHOLD  = JACOBIAN_THRESHOLD_array[np.min((iteration, len(JACOBIAN_THRESHOLD_array)-1))]
    
    n_spline = np.min((np.max((iteration * 2 + 7, 9)), 15))
    s_spline = np.linspace(0, 1, n_spline)
    if iteration == 0:
        if use_original_vmec_inut:
            current = ProfileSpline(s_spline, s_spline * (1 - s_spline) * 4)
            factor = -1e6
        else:
            s_spline = vmec.indata.ac_aux_s
            f_spline = vmec.indata.ac_aux_f
            index = np.where(s_spline[1:] <= 0)[0][0] + 1
            s_spline = s_spline[:index]
            f_spline = f_spline[:index]
            factor = -2e6
            current0 = ProfileSpline(s_spline, f_spline / factor)
            s_spline = np.linspace(0, 1, n_spline)
            current = current0.resample(s_spline)
    else:
        current = current.resample(s_spline)
    current.unfix_all()
    vmec.current_profile = ProfileScaled(current, factor)
        
    # Define bootstrap objective:
    booz = Boozer(vmec, mpol=12, ntor=12)
    ns = 50
    s_full = np.linspace(0, 1, ns)
    ds = s_full[1] - s_full[0]
    s_half = s_full[1:] - 0.5 * ds
    s_redl = []
    for s in s_spline:
        index = np.argmin(np.abs(s_half - s))
        s_redl.append(s_half[index])
    assert len(s_redl) == len(set(s_redl))
    assert len(s_redl) == len(s_spline)
    s_redl = np.array(s_redl)
    redl_geom = RedlGeomBoozer(booz, s_redl, helicity_n)
    
    # redl_s = np.linspace(0, 1, 22)
    # redl_geom = RedlGeomVmec(vmec, redl_s[1:-1])  # Drop s=0 and s=1 to avoid problems with epsilon=0 and p=0
    
    logfile = None
    if mpi.proc0_world: logfile = f'jdotB_log_max_mode{max_mode}'
    bootstrap_mismatch = VmecRedlBootstrapMismatch(redl_geom, ne, Te, Ti, Zeff, helicity_n, logfile=logfile)
    
    # Define remaining objective functions
    def aspect_ratio_max_objective(vmec): return np.max((vmec.aspect()-aspect_ratio_target,0))
    def minor_radius_objective(vmec):     return np.min((np.abs(vmec.wout.Aminor_p-aminor_target),0))
    def iota_min_objective(vmec):         return np.min((np.min(np.abs(vmec.wout.iotaf))-min_iota,0))
    def iota_mean_min_objective(vmec):    return np.min((np.abs(vmec.mean_iota())-min_average_iota,0))
    def iota_max_objective(vmec):         return np.max((np.max(np.abs(vmec.wout.iotaf))-max_iota,0))
    def volavgB_objective(vmec):          return vmec.wout.volavgB
    len_DMerc = len(vmec.wout.DMerc[initial_DMerc_index:])
    index_DMerc = int(len_DMerc * DMerc_fraction_mpol_mapping[max_mode])
    middle_index_DMerc = int(len_DMerc * 0.5)
    # def DMerc_min_objective(vmec):        return np.abs(np.min((np.min(vmec.wout.DMerc),0)))
    def DMerc_min_objective(vmec):        return np.abs(np.min((np.min(vmec.wout.DMerc[initial_DMerc_index:][index_DMerc:]),0,vmec.wout.DMerc[initial_DMerc_index:][middle_index_DMerc])))
    def magnetic_well_objective(vmec):    return np.abs(np.min((vmec.vacuum_well(),0)))
    def betatotal_objective(vmec):        return np.abs(vmec.wout.betatotal)
    aspect_ratio_max_optimizable = make_optimizable(aspect_ratio_max_objective, vmec)
    minor_radius_optimizable     = make_optimizable(minor_radius_objective, vmec)
    iota_min_optimizable         = make_optimizable(iota_min_objective, vmec)
    iota_mean_min_optimizable = make_optimizable(iota_mean_min_objective, vmec)
    iota_max_optimizable         = make_optimizable(iota_max_objective, vmec)
    volavgB_optimizable          = make_optimizable(volavgB_objective, vmec)
    DMerc_optimizable            = make_optimizable(DMerc_min_objective, vmec)
    magnetic_well_optimizable    = make_optimizable(magnetic_well_objective, vmec)
    betatotal_optimizable        = make_optimizable(betatotal_objective, vmec)
    objective_tuple = [(aspect_ratio_max_optimizable.J, 0, aspect_ratio_weight)]
    objective_tuple.append((iota_min_optimizable.J, 0, weight_iota))
    if optimize_mean_iota: objective_tuple.append((iota_mean_min_optimizable.J, 0, weight_iota))
    objective_tuple.append((iota_max_optimizable.J, 0, weight_iota*1e5)) # This prevents axisymmetry with high iota on-axis for lower resolutions
    objective_tuple.append((volavgB_optimizable.J, volavgB_target, volavgB_weight))
    objective_tuple.append((betatotal_optimizable.J, beta/100, betatotal_weight))
    if optimize_aminor: objective_tuple = [((minor_radius_optimizable.J, 0.0, aminor_weight))]
    if optimize_DMerc: objective_tuple.append((DMerc_optimizable.J, 0.0, DMerc_weight_mpol_mapping[max_mode]))
    if optimize_Well: objective_tuple.append((magnetic_well_optimizable.J, 0.0, well_Weight))
    ## Self-consistent bootstrap current objective
    objective_tuple.append((bootstrap_mismatch.residuals, 0, bootstrap_mismatch_weight))
    # Quasisymmetry objective
    qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=helicity_n)
    # objective_tuple.append((qs.residuals, 0, quasisymmetry_weight))
    objective_tuple.append((qs.residuals, 0, quasisymmetry_weight_mpol_mapping[max_mode]))
    # Put all together
    prob = LeastSquaresProblem.from_tuples(objective_tuple)
    previous_surf_dofs = prob.x
    number_vmec_dofs = int(len(prob.x))
    vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
    # Jf.target = vc.B_external_normal
    Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
    Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
    JF = Jf + J_CC + J_LENGTH_PENALTY + J_CURVATURE + J_MSC + J_ALS + linkNum + J_CS
    free_coil_dofs = JF.dofs_free_status
    dofs = np.concatenate((JF.x, prob.x))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    
    proc0_print("Initial aspect ratio:", vmec.aspect())
    proc0_print("Initial min iota:", np.min(np.abs(vmec.wout.iotaf)))
    proc0_print("Initial mean iota:", vmec.mean_iota())
    proc0_print("Initial max iota:", np.max(np.abs(vmec.wout.iotaf)))
    proc0_print("Initial mean shear:", vmec.mean_shear())
    proc0_print("Initial magnetic well:", vmec.vacuum_well())
    proc0_print("Initial quasisymmetry:", qs.total())
    proc0_print("Initial volavgB:", vmec.wout.volavgB)
    proc0_print("Initial min DMerc:", np.min(vmec.wout.DMerc[initial_DMerc_index:]))
    # proc0_print("Initial DMerc:", (vmec.wout.DMerc[initial_DMerc_index:]))
    # proc0_print("Initial DMerc objective:", DMerc_optimizable.J())
    proc0_print("Initial Aminor:", vmec.wout.Aminor_p)
    proc0_print("Initial betatotal:", vmec.wout.betatotal)
    proc0_print("Initial bootstrap_mismatch:", bootstrap_mismatch.J())
    proc0_print("Initial squared flux:", Jf.J())
    ### Stage 1 optimization
    if optimize_stage1:
        if optimize_stage3 and max_mode_previous != 0: proc0_print('Not performing stage 1 optimization since stage 3 is enabled')
        else:
            proc0_print(f'  Performing stage 1 optimization with ~{MAXITER_stage_1} iterations')
            abs_step = np.max((abs_step_stage1/(10**max_mode_previous), 1e-5))
            rel_step = np.max((rel_step_stage1/(10**max_mode_previous), 1e-7))
            least_squares_mpi_solve(prob, mpi, grad=True, rel_step=rel_step_stage1, abs_step=abs_step_stage1, max_nfev=MAXITER_stage_1,
                                    ftol=ftol_stage_1, xtol=ftol_stage_1, gtol=ftol_stage_1, method=opt_method)
            if optimize_stage3 and max_mode_previous == 0:
                proc0_print('Performing stage 1 optimization again since stage 3 is enabled')
                least_squares_mpi_solve(prob, mpi, grad=True, rel_step=rel_step_stage1, abs_step=abs_step_stage1, max_nfev=MAXITER_stage_1,
                                        ftol=ftol_stage_1, xtol=ftol_stage_1, gtol=ftol_stage_1, method=opt_method)
            dofs = np.concatenate((JF.x, prob.x))
            bs.set_points(surf.gamma().reshape((-1, 3)))
    vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
    # proc0_print("Initial DMerc:", (vmec.wout.DMerc[initial_DMerc_index:]))
    # Jf.target = vc.B_external_normal
    Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
    Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
    JF = Jf + J_CC + J_LENGTH_PENALTY + J_CURVATURE + J_MSC + J_ALS + linkNum + J_CS
    ### Stage 2 optimization
    if optimize_stage2:
        proc0_print(f'  Performing stage 2 optimization with ~{MAXITER_stage_2} iterations')
        coils_dofs = dofs[:-number_vmec_dofs]
        if comm_world.rank == 0:
            res = minimize(fun_coils, coils_dofs, jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=tol_coils)
            print(res.message)
            dofs[:-number_vmec_dofs] = res.x
            coils_dofs = res.x
        mpi.comm_world.Barrier()
        mpi.comm_world.Bcast(coils_dofs, root=0)
        dofs[:-number_vmec_dofs] = coils_dofs
        JF.x = coils_dofs
        bs.set_points(surf.gamma().reshape((-1, 3)))
    vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
    # Jf.target = vc.B_external_normal
    Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
    Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
    JF = Jf + J_CC + J_LENGTH_PENALTY + J_CURVATURE + J_MSC + J_ALS + linkNum + J_CS
    ## Stage 1 optimization with coils
    if optimize_stage1_with_coils:
        def JF_objective(vmec):
            bs.set_points(vmec.boundary.gamma().reshape((-1, 3)))
            return JF.J()
        JF_objective_optimizable = make_optimizable(JF_objective, vmec)
        Jf_residual = JF_objective_optimizable.J()
        prob_residual = prob.objective()
        new_Jf_weight = coils_objective_weight#(prob_residual/Jf_residual)**2
        objective_tuples_with_coils = tuple(objective_tuple)+tuple([(JF_objective_optimizable.J, 0, new_Jf_weight**2/3)])
        prob_with_coils = LeastSquaresProblem.from_tuples(objective_tuples_with_coils)
        proc0_print(f'  Performing stage 1 optimization with coils with ~{MAXITER_stage_1} iterations')
        JF.fix_all()
        abs_step = np.max((abs_step_stage1/(10**max_mode_previous), 1e-5))
        rel_step = np.max((rel_step_stage1/(10**max_mode_previous), 1e-7))
        least_squares_mpi_solve(prob_with_coils, mpi, grad=True, rel_step=rel_step, abs_step=abs_step, max_nfev=MAXITER_stage_1, ftol=ftol_stage_1, xtol=ftol_stage_1, gtol=ftol_stage_1)
        JF.full_unfix(free_coil_dofs_all)
    vmec.write_input(os.path.join(this_path, f'input.after_stage12_maxmode{max_mode}'))
    ## Broadcast dofs and save surfs/coils
    # mpi.comm_world.Bcast(dofs, root=0)
    # JF.x = dofs[:-number_vmec_dofs]
    #
    vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
    # Jf.target = vc.B_external_normal
    Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
    Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
    JF = Jf + J_CC + J_LENGTH_PENALTY + J_CURVATURE + J_MSC + J_ALS + linkNum + J_CS
    #
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    BdotN_surf = (np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    if comm_world.rank == 0:
        curves_to_vtk(base_curves, os.path.join(coils_results_path, f"base_curves_after_stage12_maxmode{max_mode}"))
        curves_to_vtk(curves, os.path.join(coils_results_path, f"curves_after_stage12_maxmode{max_mode}"))
        pointData = {"B.n/B": BdotN_surf[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path, f"surf_after_stage12_maxmode{max_mode}"), extra_data=pointData)
    bs.set_points(surf_big.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
    BdotN_surf = np.sum(Bbs * surf_big.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
    if comm_world.rank == 0:
        pointData = {"Bcoils.n/B": BdotN_surf[:, :, None]}
        surf_big.to_vtk(os.path.join(coils_results_path, f"surf_big_after_stage12_maxmode{max_mode}"), extra_data=pointData)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    bs.save(os.path.join(coils_results_path, f"biot_savart_after_stage12_maxmode{max_mode}.json"))
    ## Single stage optimization
    if optimize_stage3:
        proc0_print(f'  Performing single stage optimization with ~{MAXFEV_single_stage} iterations')
        dofs = np.concatenate((JF.x, prob.x))
        mpi.comm_world.Bcast(dofs, root=0)
        opt = make_optimizable(fun_J, prob, JF)
        free_coil_dofs = JF.dofs_free_status
        JF.fix_all()
        abs_step = np.max((finite_difference_abs_step/(10**max_mode_previous), 1e-5))
        rel_step = np.max((finite_difference_rel_step/(10**max_mode_previous), 1e-7))
        with MPIFiniteDifference(opt.J, mpi, diff_method=diff_method, abs_step=abs_step, rel_step=rel_step) as prob_jacobian:
            if mpi.proc0_world:
                res = minimize(fun, dofs, args=(prob_jacobian, {'Nfeval': 0}), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage, 'gtol': ftol}, tol=ftol)
                print(res.message)
        JF.full_unfix(free_coil_dofs_all)
    ## Broadcast dofs and save surfs/coils
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    BdotN_surf = (np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    if comm_world.rank == 0:
        curves_to_vtk(base_curves, os.path.join(coils_results_path, f"base_curves_opt_maxmode{max_mode}"))
        curves_to_vtk(curves, os.path.join(coils_results_path, f"curves_opt_maxmode{max_mode}"))
        pointData = {"B.n/B": BdotN_surf[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path, f"surf_opt_maxmode{max_mode}"), extra_data=pointData)
    bs.set_points(surf_big.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
    BdotN_surf = np.sum(Bbs * surf_big.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
    if comm_world.rank == 0:
        pointData = {"Bcoils.n/B": BdotN_surf[:, :, None]}
        surf_big.to_vtk(os.path.join(coils_results_path, f"surf_big_opt_maxmode{max_mode}"), extra_data=pointData)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    bs.save(os.path.join(coils_results_path, f"biot_savart_maxmode{max_mode}.json"))
    vmec.write_input(os.path.join(this_path, f'input.maxmode{max_mode}'))
    
    # Remove spurious files
    try:
        os.chdir(vmec_results_path)
        for jac_file in glob.glob("jac_log_*"): os.remove(jac_file)
        for obj_file in glob.glob("objective_*"): os.remove(obj_file)
    except Exception as e: proc0_print(f'Exception when removing spurious files in {vmec_results_path}: {e}')
    try:
        os.chdir(parent_path)
        for jac_file in glob.glob("jac_log_*"): os.remove(jac_file)
        for obj_file in glob.glob("objective_*"): os.remove(obj_file)
    except Exception as e: proc0_print(f'Exception when removing spurious files in {parent_path}: {e}')
    try:
        os.chdir(this_path)
        for jac_file in glob.glob("jac_log_*"): os.remove(jac_file)
        for obj_file in glob.glob("objective_*"): os.remove(obj_file)
    except Exception as e: proc0_print(f'Exception when removing spurious files in {this_path}: {e}')
    
    max_mode_previous+=1
##########################################################################################
if optimize_stage3 or optimize_stage1:
    vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC, filename=None)
    # Jf.target = vc.B_external_normal
    Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
    Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
    JF = Jf + J_CC + J_LENGTH_PENALTY + J_CURVATURE + J_MSC + J_ALS + linkNum + J_CS
    ### Stage 2 optimization
    if optimize_stage2:
        proc0_print(f'  Performing final stage 2 optimization with ~{MAXITER_stage_2*3} iterations')
        dofs = np.concatenate((JF.x, prob.x))
        coils_dofs = dofs[:-number_vmec_dofs]
        if comm_world.rank == 0:
            res = minimize(fun_coils, coils_dofs, jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2*3, 'maxcor': 300}, tol=tol_coils)
            print(res.message)
            dofs[:-number_vmec_dofs] = res.x
            coils_dofs = res.x
        mpi.comm_world.Barrier()
        mpi.comm_world.Bcast(coils_dofs, root=0)
        dofs[:-number_vmec_dofs] = coils_dofs
        JF.x = coils_dofs
        bs.set_points(surf.gamma().reshape((-1, 3)))
##########################################################################################
############## Save final results
##########################################################################################
bs.save(os.path.join(coils_results_path, "biot_savart_opt.json"))
vmec.write_input(os.path.join(this_path, 'input.final'))

proc0_print("Final aspect ratio:", vmec.aspect())
proc0_print("Final min iota:", np.min(np.abs(vmec.wout.iotaf)))
proc0_print("Final mean iota:", vmec.mean_iota())
proc0_print("Final max iota:", np.max(np.abs(vmec.wout.iotaf)))
proc0_print("Final mean shear:", vmec.mean_shear())
proc0_print("Final magnetic well:", vmec.vacuum_well())
proc0_print("Final quasisymmetry:", qs.total())
proc0_print("Final volavgB:", vmec.wout.volavgB)
proc0_print("Final min DMerc:", np.min(vmec.wout.DMerc[initial_DMerc_index:]))
proc0_print("Final Aminor:", vmec.wout.Aminor_p)
proc0_print("Final betatotal:", vmec.wout.betatotal)
proc0_print("Final bootstrap_mismatch:", bootstrap_mismatch.J())
proc0_print("Final squared flux:", Jf.J())

BdotN_surf = (np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
BdotN = np.mean(np.abs(BdotN_surf))
BdotNmax = np.max(np.abs(BdotN_surf))
outstr = f"Coil parameters: ⟨B·n/B⟩={BdotN:.1e}, B·n/B max={BdotNmax:.1e}"
outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
msc_string = ", ".join(f"{j.J():.2f}" for j in Jmscs)
outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}], msc=[{msc_string}]"
proc0_print(outstr)
if mpi.proc0_world:
    try:
        vmec_final = Vmec(os.path.join(this_path, f'input.final'), mpi=mpi, verbose=False)
        vmec_final.indata.ns_array[:3]    = [  16,    51,    101]
        vmec_final.indata.niter_array[:3] = [ 300,   500,  20000]
        vmec_final.indata.ftol_array[:3]  = [ 1e-9, 1e-10, 1e-14]
        vmec_final.write_input(os.path.join(this_path, 'input.final'))
        # vmec_final.run()
        # shutil.move(os.path.join(this_path, f"wout_final_000_000000.nc"), os.path.join(this_path, f"wout_final.nc"))
        # os.remove(os.path.join(this_path, f'input.final_000_000000'))
    except Exception as e:
        proc0_print('Exception when creating final vmec file:')
        proc0_print(e)