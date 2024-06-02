#!/usr/bin/env python3
import os
import time
import shutil
import numpy as np
from pathlib import Path
from simsopt import load
from simsopt.mhd import Vmec
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simsopt.util import proc0_print, comm_world
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (curves_to_vtk, create_equally_spaced_curves, LpCurveCurvature,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature, SurfaceRZFourier)
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk,
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data)
######################################################
do_stage_2      = False
do_Poincare     = True
do_FreeBoundary = True
######################################################
filename = os.path.join(str(Path(__file__).parent.resolve()),"input.rotating_ellipse")
nphi = 64
ntheta = 28
vmec = Vmec(filename, verbose=False, range_surface="full torus", nphi=nphi, ntheta=ntheta)
s = vmec.boundary
## Create a wout file to use later
vmec.run()
vmec_original_output = os.path.join(str(Path(__file__).parent.resolve()),"wout.nc")
shutil.move(vmec.output_file, os.path.join(vmec.output_file,vmec_original_output))
## Define surfaces to plot later
nfieldlines = 4
R_max = np.max(s.gamma()[0,:,0])
R_axis = np.sum(vmec.wout.raxis_cc)
R0 = np.linspace(R_axis*1.03, R_max*0.975, nfieldlines, endpoint=True)
################ STAGE 2 OPTIMIZATION ################
if do_stage_2:
    ncoils = 3
    R0 = 5.0
    R1 = 3.0
    order = 12
    LENGTH_CON_WEIGHT = 0.1
    LENGTH_THRESHOLD = 22
    CC_THRESHOLD = 0.70
    CC_WEIGHT = 1000
    CURVATURE_THRESHOLD = 3.0
    CURVATURE_WEIGHT = 1e-2
    MSC_THRESHOLD = 1.0
    MSC_WEIGHT = 1e-2
    MAXITER = 500
    proc0_print(f'Loading VMEC file {filename}')
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    base_currents = [Current(1)*1e5 for i in range(ncoils)]
    total_current_vmec = vmec.external_current() / (2 * s.nfp)
    base_currents = [Current(total_current_vmec / ncoils * 1e-5) * 1e5 for _ in range(ncoils - 1)]
    total_current = Current(total_current_vmec)*1
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    if comm_world is None or comm_world.rank == 0:
        curves_to_vtk(curves, "curves_init")
        pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
        s.to_vtk("surf_init", extra_data=pointData)
    proc0_print(f'Initializing stage 2 objective function')
    Jf = SquaredFlux(s, bs, definition="local")
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jls = [CurveLength(c) for c in base_curves]
    JF = Jf \
        + LENGTH_CON_WEIGHT * sum(QuadraticPenalty(J, LENGTH_THRESHOLD, "max") for J in Jls) \
        + CC_WEIGHT * Jccdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        outstr = f" J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad
    f = fun
    dofs = JF.x
    proc0_print(f'Performing stage 2 optimization')
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    curves_to_vtk(curves, "curves_opt")
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk("surf_opt", extra_data=pointData)
    bs.save(os.path.join(str(Path(__file__).parent.resolve()),"biot_savart_opt.json"))
else:
    bs = load(os.path.join(str(Path(__file__).parent.resolve()),"biot_savart_opt.json"))
################ POINCARE PLOT ################
if do_Poincare:
    proc0_print(f'Defining Poincaré plot functions')
    tmax_fl = 6000
    degree = 3
    sc_fieldline = SurfaceClassifier(s, h=0.1, p=2)
    sc_fieldline.to_vtk('levelset', h=0.1)
    def trace_fieldlines(bfield, label):
        t1 = time.time()
        Z0 = np.zeros(nfieldlines)
        phis = [(i/4)*(2*np.pi/s.nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-15, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        t2 = time.time()
        proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        if comm_world is None or comm_world.rank == 0:
            particles_to_vtk(fieldlines_tys, f'fieldlines_{label}')
            plot_poincare_data(fieldlines_phi_hits, phis, f'poincare_fieldline_{label}.png', dpi=150, surf=s)
        return fieldlines_phi_hits
    n = 20
    rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
    zs = s.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2*np.pi/s.nfp, n*2)
    zrange = (0, np.max(zs), n//2)
    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.3).flatten())
        proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip
    proc0_print('Initializing InterpolatedField')
    bsh = InterpolatedField(bs, degree, rrange, phirange, zrange, True, nfp=s.nfp, stellsym=True, skip=skip)
    proc0_print('Done initializing InterpolatedField.')
    bsh.set_points(s.gamma().reshape((-1, 3)))
    bs.set_points(s.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    proc0_print("Mean(|B|) on plasma surface =", np.mean(bs.AbsB()))
    proc0_print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))
    proc0_print('Beginning field line tracing')
    fieldlines_phi_hits = trace_fieldlines(bsh, 'bsh')
################ FREE BOUNDARY VMEC ################
if do_FreeBoundary:
    nphi_mgrid = 24
    mgrid_file = os.path.join(str(Path(__file__).parent.resolve()),"mgrid.test.nc")
    R_surf = np.sqrt(s.gamma()[:, :, 0] ** 2 + s.gamma()[:, :, 1] ** 2)
    Z_surf = s.gamma()[:, :, 2]
    bs.to_mgrid(
        mgrid_file,
        nr=64,
        nz=65,
        nphi=nphi_mgrid,
        rmin=0.8*np.min(R_surf),
        rmax=1.2*np.max(R_surf),
        zmin=1.2*np.min(Z_surf),
        zmax=1.2*np.max(Z_surf),
        nfp=s.nfp,
    )
    vmec_freeb = Vmec(filename, verbose=True)
    vmec_freeb.indata.lfreeb = True
    vmec_freeb.indata.mgrid_file = mgrid_file
    vmec_freeb.indata.nzeta = nphi_mgrid
    vmec_freeb.indata.extcur[0] = 1.0
    vmec_freeb.indata.mpol = 5
    vmec_freeb.indata.ntor = 5
    vmec_freeb.indata.ns_array[2] = 0
    ftol = 1e-11
    vmec_freeb.indata.ftol_array[1] = ftol
    vmec_freeb.run()
    assert vmec_freeb.wout.fsql < ftol
    assert vmec_freeb.wout.fsqr < ftol
    assert vmec_freeb.wout.fsqz < ftol
    assert vmec_freeb.wout.ier_flag == 0
    
    s_array = np.linspace((R0[0]-R_axis)/R_axis, R0[-1]/R_max, nfieldlines, endpoint=True)**2
    for j, s in enumerate(s_array):
        surf_original = SurfaceRZFourier.from_wout(vmec_original_output, s=s)    
        cross_section_original = surf_original.cross_section(phi=0)
        r_original = np.sqrt(cross_section_original[:, 0] ** 2 + cross_section_original[:, 1] ** 2)
        z_original = cross_section_original[:, 2]
        plt.plot(r_original, z_original, 'r', linewidth=1, label='Fixed Boundary' if s==s_array[0] else '_no_legend_')
        
        surf_freeb = SurfaceRZFourier.from_wout(vmec_freeb.output_file, s=s)    
        cross_section_freeb = surf_freeb.cross_section(phi=0)
        r_freeb = np.sqrt(cross_section_freeb[:, 0] ** 2 + cross_section_freeb[:, 1] ** 2)
        z_freeb = cross_section_freeb[:, 2]
        plt.plot(r_freeb, z_freeb, 'k--', linewidth=1, label='Free Boundary' if s==s_array[0] else '_no_legend_')
        
        if do_Poincare:
            lost = fieldlines_phi_hits[j][-1, 1] < 0
            color = 'r' if lost else 'g'
            data_this_phi = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == 0)[0], :]
            if data_this_phi.size == 0: continue
            r = np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2)
            plt.scatter(r, data_this_phi[:, 4], marker='o', s=2, linewidths=0, c=color)
    
    plt.legend()
    plt.savefig(os.path.join(str(Path(__file__).parent.resolve()),"Fixed_vs_Free_boundary.png"), dpi=150)
    plt.show()