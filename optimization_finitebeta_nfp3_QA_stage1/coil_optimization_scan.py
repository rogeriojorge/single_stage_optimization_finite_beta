#!/usr/bin/env python3
import os
import json
import shutil
import argparse
import numpy as np
from scipy.optimize import minimize
from simsopt.mhd import VirtualCasing, Vmec
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier,
                        LinkingNumber, CurveLength, CurveCurveDistance, ArclengthVariation,
                        MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
parent_path = os.path.dirname(os.path.abspath(__file__))
###########################################
os.chdir(parent_path)
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=1)
parser.add_argument("--ncoils", type=int, default=4)
args = parser.parse_args()
if   args.type == 1: QA_or_QH = 'nfp2_QA'
elif args.type == 2: QA_or_QH = 'nfp4_QH'
elif args.type == 3: QA_or_QH = 'nfp3_QA'
elif args.type == 4: QA_or_QH = 'nfp3_QH'
elif args.type == 5: QA_or_QH = 'nfp3_QI'
else: raise ValueError('Invalid type')
ncoils = args.ncoils
###########################################
create_surface_coils = False
R0_factor = 11
filename = 'wout_final.nc'
MAXITER = 200 if 'QA' in QA_or_QH else 300
R1_mean = 0.44*R0_factor
R1_std = 0.4*R0_factor
min_length_per_coil = 3.5*R0_factor
max_length_per_coil = 4.7*R0_factor
min_curvature = 4/R0_factor
max_curvature = 25/R0_factor/R0_factor
CC_min = 0.05*R0_factor
CC_max = 0.20*R0_factor
order_min = 5
order_max = 16
nphi = 32
ntheta = 32
vc_src_nphi = 80
CS_min = 0.05*R0_factor
CS_max = 0.30*R0_factor
###########################################
# Directories
directory = f'optimization_finitebeta_{QA_or_QH}_stage1'
this_path = os.path.join(parent_path, directory)
shutil.copyfile(os.path.join(parent_path, 'coil_optimization_scan.py'), os.path.join(this_path, 'coil_optimization_scan.py'))
os.chdir(this_path)
coils_results_path = os.path.join(this_path, "coils")
vmec_file = os.path.join(this_path,filename)
vmec = Vmec(vmec_file, verbose=False)
basename = os.path.basename(vmec_file)
# File for the target plasma surface. It can be either a wout or vmec input file.
if basename[:4] == "wout":
    surf = SurfaceRZFourier.from_wout(vmec_file, range="half period", nphi=nphi, ntheta=ntheta)
    
    head, tail = os.path.split(vmec_file)
    vc_filename = os.path.join(head, tail.replace('wout', 'vcasing'))
    if os.path.isfile(vc_filename):
        print('Loading saved virtual casing result')
        vc = VirtualCasing.load(vc_filename)
    else:
        print('Running the virtual casing calculation')
        vc = VirtualCasing.from_vmec(vmec_file, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)
else:
    surf = SurfaceRZFourier.from_vmec_input(vmec_file, range="half period", nphi=nphi, ntheta=ntheta)
    vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)
out_dir = os.path.join(this_path,"coils","scan")
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)
# Load the target plasma surface:
nfp = surf.nfp
R0 = surf.get_rc(0, 0)
print(f'R0 = {R0}')
# exit()
total_current = vmec.external_current() / (2*nfp)
## Create a copy of the surface that is closed in theta and phi, and covers the full torus toroidally. This is nice for visualization.
# nphi_big = nphi * 2 * nfp + 1
# ntheta_big = ntheta + 1
# quadpoints_theta = np.linspace(0, 1, ntheta_big)
# quadpoints_phi = np.linspace(0, 1, nphi_big)
# surf_big = SurfaceRZFourier(dofs=surf.dofs,nfp=nfp,mpol=surf.mpol,ntor=surf.ntor,quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta,)

def run_optimization(
    R1,order,length_target,total_current,
    length_weight,max_curvature_threshold,
    max_curvature_weight,msc_threshold,msc_weight,
    cc_threshold,cc_weight,cs_threshold,
    cs_weight,arclength_weight,index,
):
    directory = (
        f"ncoils_{ncoils}_order_{order}_R1_{R1:.2}_length_target_{length_target:.2}_weight_{length_weight:.2}"
        + f"_max_curvature_{max_curvature_threshold:.2}_weight_{max_curvature_weight:.2}"
        + f"_msc_{msc_threshold:.2}_weight_{msc_weight:.2}"
        + f"_cc_{cc_threshold:.2}_weight_{cc_weight:.2}"
        + f"_cs_{cs_threshold:.2}_weight_{cs_weight:.2}"
        + f"_arclweight_{arclength_weight:.2}"
    )

    print()
    print("***********************************************")
    print(f"Job {index+1}")
    print("Parameters:", directory)
    print("***********************************************")
    print()

    # Directory for output
    new_OUT_DIR = directory + "/"
    os.mkdir(directory)

    # Create the initial coils:
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=order * 16)
    # # base_currents = [Current(1e5) for i in range(ncoils)]
    # base_currents = [Current(1.0) * (1e5) for i in range(ncoils)]
    # base_currents[0].fix_all()
    base_currents = [Current(total_current / ncoils * 1e-7) * 1e7 for _ in range(ncoils-1)]
    total_current = Current(total_current)
    # total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]

    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    
    if create_surface_coils:
        curves_to_vtk(curves, new_OUT_DIR + "curves_init", close=True)
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
        pointData = {"B.n/B": BdotN[:, :, None]}
        surf.to_vtk(new_OUT_DIR + "surf_init", extra_data=pointData)

    # surf_big.to_vtk(new_OUT_DIR + "surf_big")

    # Define the individual terms objective function:
    Jf = SquaredFlux(surf, bs, target=vc.B_external_normal, definition="local")
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, surf, cs_threshold)
    Jcs = [LpCurveCurvature(c, 2, max_curvature_threshold) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jals = [ArclengthVariation(c) for c in base_curves]

    # Form the total objective function. To do this, we can exploit the
    # fact that Optimizable objects with J() and dJ() functions can be
    # multiplied by scalars and added:
    JF = (
        Jf
        + length_weight * QuadraticPenalty(sum(Jls), length_target * ncoils)
        + cc_weight * Jccdist
        + cs_weight * Jcsdist
        + max_curvature_weight * sum(Jcs)
        + msc_weight * sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs)
        + LinkingNumber(curves, 2)
        + arclength_weight * sum(Jals)
    )

    iteration = 0

    def fun(dofs):
        nonlocal iteration
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = np.max(np.abs((np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)))
        outstr = f"{iteration:4}  J={J:.1e}, Jf={jf:.1e}, max⟨B·n⟩/B={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
        outstr += f", C-S-Sep={Jcsdist.shortest_distance():.2f}"
        # outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        outstr += f", max curr={max(c.get_value() for c in base_currents):.1e}"
        print(outstr)
        iteration += 1
        return J, grad

    res = minimize( fun, JF.x, jac=True, method="L-BFGS-B", options={"maxiter": MAXITER, "maxcor": 300}, tol=1e-11)
    JF.x = res.x
    print(res.message)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    
    if create_surface_coils:
        curves_to_vtk(curves, new_OUT_DIR + "curves_opt_big", close=True)
        curves_to_vtk(base_curves, new_OUT_DIR + "curves_opt", close=True)
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
        pointData = {"B.n/B": BdotN[:, :, None]}
        surf.to_vtk(new_OUT_DIR + "surf_opt", extra_data=pointData)

    # bs_big = BiotSavart(coils)
    # bs_big.set_points(surf_big.gamma().reshape((-1, 3)))
    # pointData = {
    #     "B_N": np.sum(
    #         bs_big.B().reshape((nphi_big, ntheta_big, 3)) * surf_big.unitnormal(),
    #         axis=2,
    #     )[:, :, None]
    # }
    # surf_big.to_vtk(new_OUT_DIR + "surf_big_opt")#, extra_data=pointData)

    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    bs.save(new_OUT_DIR + "biot_savart.json")

    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.max(np.abs(np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2))

    results = {
        "nfp": nfp,
        "R0": R0,
        "R1": R1,
        "ncoils": ncoils,
        "order": order,
        "nphi": nphi,
        "ntheta": ntheta,
        "length_target": length_target,
        "length_weight": length_weight,
        "max_curvature_threshold": max_curvature_threshold,
        "max_curvature_weight": max_curvature_weight,
        "msc_threshold": msc_threshold,
        "msc_weight": msc_weight,
        "JF": float(JF.J()),
        "Jf": float(Jf.J()),
        "BdotN": BdotN,
        "lengths": [float(J.J()) for J in Jls],
        "length": float(sum(J.J() for J in Jls)),
        "average_length_per_coil": float(sum(J.J() for J in Jls))/ncoils,
        "max_curvatures": [np.max(c.kappa()) for c in base_curves],
        "max_max_curvature": max(np.max(c.kappa()) for c in base_curves),
        "coil_coil_distance": Jccdist.shortest_distance(),
        "cc_threshold": cc_threshold,
        "cc_weight": cc_weight,
        "cs_threshold": cs_threshold,
        "cs_weight": cs_weight,
        "arclength_weight": arclength_weight,
        "gradient_norm": np.linalg.norm(JF.dJ()),
        "linking_number": LinkingNumber(curves).J(),
        "directory": directory,
        "mean_squared_curvatures": [float(J.J()) for J in Jmscs],
        "max_mean_squared_curvature": float(max(J.J() for J in Jmscs)),
        "message": res.message,
        "success": res.success,
        "iterations": res.nit,
        "function_evaluations": res.nfev,
        "coil_currents": [c.get_value() for c in base_currents],
        "coil_surface_distance":  float(Jcsdist.shortest_distance()),
    }

    with open(new_OUT_DIR + "results.json", "w") as outfile:
        json.dump(results, outfile, indent=2)


#########################################################################
# Carry out the scan. Below you can adjust the ranges for the random weights and
# thresholds.
#########################################################################


def rand(min, max):
    """Generate a random float between min and max."""
    return np.random.rand() * (max - min) + min


for index in range(10000):
    # Initial radius of the coils:
    R1 = np.random.rand() * R1_std + R1_mean

    # Number of Fourier modes describing each Cartesian component of each coil:
    order = int(np.round(rand(order_min, order_max)))

    # Target length (per coil!) and weight for the length term in the objective function:
    length_target = rand(min_length_per_coil, max_length_per_coil)
    length_weight = 10.0 ** rand(-3, 1)

    # Threshold and weight for the curvature penalty in the objective function:
    max_curvature_threshold = rand(min_curvature, max_curvature)
    max_curvature_weight = 10.0 ** rand(-6, -2)

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    msc_threshold = rand(0.1*min_curvature, max_curvature*10)
    msc_weight = 10.0 ** rand(-7, -3)

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    cc_threshold = rand(CC_min, CC_max)
    cc_weight = 10.0 ** rand(-2, 3)
    
    # Threshold and weight for the coil-to-surface penalty in the objective function:
    cs_threshold = rand(CS_min, CS_max)
    cs_weight = 10.0 ** rand(-3, 2)
    
    # Weight for the arclength variation penalty in the objective function:
    arclength_weight = 10.0 ** rand(-9, -2)

    run_optimization(
        R1, order,
        length_target,
        total_current,
        length_weight,
        max_curvature_threshold,
        max_curvature_weight,
        msc_threshold,
        msc_weight,
        cc_threshold,
        cc_weight,
        cs_threshold,
        cs_weight,
        arclength_weight,
        index,
    )
