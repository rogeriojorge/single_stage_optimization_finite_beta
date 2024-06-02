#!/usr/bin/env python3
import os
import time
import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simsopt import load
from simsopt.mhd import Vmec, VirtualCasing
from simsopt.util import proc0_print, comm_world
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (curves_to_vtk, create_equally_spaced_curves, LpCurveCurvature, LinkingNumber,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature, SurfaceRZFourier)
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk,
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data)
this_path = str(Path(__file__).parent.resolve())
os.chdir(this_path)
######################################################
do_stage_2      = True
do_Poincare     = True
do_FreeBoundary = True
nfieldlines = 6
tmax_fl = 8000
nphi = 64
ntheta = 34
#################################################
## CHOOSE BETWEEN input.final (finite beta, does not work) and input.rotating_ellipse (vacuum, works)
filename = os.path.join(this_path,"input.final")
# filename = os.path.join(this_path,"input.rotating_ellipse")
#################################################
## CHOOSE TO EXPLICITLY RUN VMEC IN VACUUM OR NOT
run_in_vacuum = True
sign_B_external_normal = 1 if not run_in_vacuum else 0
######################################################
vacuum_filename = os.path.join(this_path,"input.final_vac")
wout_freeb_file = os.path.join(this_path,"wout_freeb.nc")
vmec = Vmec(filename, verbose=False, nphi=nphi, ntheta=ntheta, range_surface='half period')
if run_in_vacuum:
    vmec.indata.curtor*=0
    vmec.indata.am*=0
    vmec.write_input(vacuum_filename)
    vmec = Vmec(vacuum_filename, verbose=False, nphi=nphi, ntheta=ntheta, range_surface='half period')
s = vmec.boundary
## Create a wout file to use later
vmec.run()
vc = VirtualCasing.from_vmec(vmec, src_nphi=nphi, trgt_nphi=nphi, trgt_ntheta=ntheta, filename=None)
vmec_original_output = os.path.join(this_path,"wout.nc" if not run_in_vacuum else "wout_vac.nc")
shutil.move(vmec.output_file, vmec_original_output)
## Define surfaces to plot later
R_theta0_phi0_array = np.sort(np.sum(vmec.wout.rmnc,axis=0))
indices_to_plot = np.array(np.linspace(2,len(R_theta0_phi0_array)-2,nfieldlines),dtype=int)
R0 = R_theta0_phi0_array[indices_to_plot]
s_array = np.abs(np.linspace(-1,0,vmec.wout.ns)) if 'ellipse' in filename else np.linspace(0,1,vmec.wout.ns)
s_array = s_array[indices_to_plot]
################ STAGE 2 OPTIMIZATION ################
if do_stage_2:
    ncoils = 3
    R0_coils = np.sum(vmec.wout.raxis_cc)
    R1_coils = np.min((vmec.wout.Aminor_p*2.6,R0_coils/1.4))
    order = 12
    LENGTH_CON_WEIGHT = 0.1
    LENGTH_THRESHOLD = 26*R0_coils/5
    CC_THRESHOLD = 0.50*R0_coils/5
    CC_WEIGHT = 1000
    CURVATURE_THRESHOLD = 4.0
    CURVATURE_WEIGHT = 1e-2
    MSC_THRESHOLD = 1.0
    MSC_WEIGHT = 1e-2
    MAXITER = 500
    proc0_print(f'Loading VMEC file {filename}')
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0_coils, R1=R1_coils, order=order)
    base_currents = [Current(1)*1e5 for i in range(ncoils)]
    total_current_vmec = vmec.external_current() / (2 * s.nfp)
    base_currents = [Current(total_current_vmec / ncoils * 1e-5) * 1e5 for _ in range(ncoils - 1)]
    total_current = Current(total_current_vmec)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    if comm_world is None or comm_world.rank == 0:
        curves_to_vtk(curves, "curves_init")
        curves_to_vtk(base_curves, "base_curves_init")
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = (np.sum(Bbs * s.unitnormal(), axis=2) - sign_B_external_normal*vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
        Bmod = bs.AbsB().reshape((nphi,ntheta,1))
        s.to_vtk(os.path.join(this_path,"surf_init"), extra_data= {"B.n/B": BdotN[:, :, None], "B": Bmod})
    proc0_print(f'Initializing stage 2 objective function')
    Jf = SquaredFlux(s, bs, target=sign_B_external_normal*vc.B_external_normal, definition="local")
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jls = [CurveLength(c) for c in base_curves]
    JF = Jf \
        + LENGTH_CON_WEIGHT * sum(QuadraticPenalty(J, LENGTH_THRESHOLD, "max") for J in Jls) \
        + CC_WEIGHT * Jccdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
        + LinkingNumber(curves, 2)
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = (np.sum(Bbs * s.unitnormal(), axis=2) - sign_B_external_normal*vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
        outstr = f" J={J:.1e}, Jf={jf:.1e}, max(B·n)/B={np.max(np.abs(BdotN))/np.mean(np.abs(Bbs)):.1e}"
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
    print(res.message)
    curves_to_vtk(curves, "curves_opt")
    curves_to_vtk(base_curves, "base_curves_opt")
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = (np.sum(Bbs * s.unitnormal(), axis=2) - sign_B_external_normal*vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    Bmod = bs.AbsB().reshape((nphi,ntheta,1))
    s.to_vtk(os.path.join(this_path,"surf_opt"), extra_data= {"B.n/B": BdotN[:, :, None], "B": Bmod})
    bs.save(os.path.join(this_path,"biot_savart_opt.json"))
else:
    bs = load(os.path.join(this_path,"biot_savart_opt.json"))
################ POINCARE PLOT ################
if do_Poincare:
    proc0_print(f'Defining Poincaré plot functions')
    degree = 4
    ## BIG SURFACE FOR POINCARE WITH FULL TORUS
    nphi_big = nphi * 2 * s.nfp + 1
    ntheta_big = ntheta + 1
    quadpoints_theta = np.linspace(0, 1, ntheta_big)
    quadpoints_phi = np.linspace(0, 1, nphi_big)
    surf_big = SurfaceRZFourier(dofs=s.dofs,nfp=s.nfp,mpol=s.mpol,ntor=s.ntor,quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta,)
    sc_fieldline = SurfaceClassifier(surf_big, h=0.1, p=2)
    # sc_fieldline.to_vtk('levelset', h=0.1)
    def trace_fieldlines(bfield, label):
        t1 = time.time()
        Z0 = np.zeros(nfieldlines)
        phis = [(i/4)*(2*np.pi/s.nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-14, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        t2 = time.time()
        proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        if comm_world is None or comm_world.rank == 0:
            # particles_to_vtk(fieldlines_tys, f'fieldlines_{label}')
            plot_poincare_data(fieldlines_phi_hits, phis, f'poincare_fieldline_{label}.png', dpi=150, surf=s)
        return fieldlines_phi_hits
    n = 40
    rs = np.linalg.norm(surf_big.gamma()[:, :, 0:2], axis=2)
    zs = surf_big.gamma()[:, :, 2]
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
    bsh.set_points(surf_big.gamma().reshape((-1, 3)))
    bs.set_points(surf_big.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    proc0_print("Mean(|B|) on plasma surface =", np.mean(bs.AbsB()))
    proc0_print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))
    proc0_print('Beginning field line tracing')
    fieldlines_phi_hits = trace_fieldlines(bsh, 'bsh')
################ FREE BOUNDARY VMEC ################
if do_FreeBoundary:
    nphi_mgrid = 46
    mgrid_file = os.path.join(this_path,"mgrid.test.nc")
    R_surf = np.sqrt(s.gamma()[:, :, 0] ** 2 + s.gamma()[:, :, 1] ** 2)
    Z_surf = s.gamma()[:, :, 2]
    bs.to_mgrid(
        mgrid_file,
        nr=64,
        nz=64,
        nphi=nphi_mgrid,
        rmin=0.8*np.min(R_surf),
        rmax=1.2*np.max(R_surf),
        zmin=1.2*np.min(Z_surf),
        zmax=1.2*np.max(Z_surf),
        nfp=s.nfp,
    )
    vmec_freeb = Vmec(filename if not run_in_vacuum else vacuum_filename, verbose=True)
    vmec_freeb.indata.lfreeb = True
    vmec_freeb.indata.mgrid_file = mgrid_file
    vmec_freeb.indata.nzeta = nphi_mgrid
    vmec_freeb.indata.extcur[0] = 1.0
    vmec_freeb.indata.mpol = 6
    vmec_freeb.indata.ntor = 6
    ftol = 2e-11
    vmec.indata.ftol_array[1] = ftol
    vmec.write_input(os.path.join(this_path,"input.freeb"))
    vmec_freeb.run()
    assert vmec_freeb.wout.fsql < ftol
    assert vmec_freeb.wout.fsqr < ftol
    assert vmec_freeb.wout.fsqz < ftol
    assert vmec_freeb.wout.ier_flag == 0
    shutil.move(vmec_freeb.output_file, wout_freeb_file)
    
    for j, s in enumerate(s_array):
        surf_original = SurfaceRZFourier.from_wout(vmec_original_output, s=s, ntheta=ntheta, nphi=nphi)   
        if s == np.max(s_array):
            bs.set_points(surf_original.gamma().reshape((-1, 3)))
            Bbs = bs.B().reshape((nphi, ntheta, 3))
            BdotN = (np.sum(Bbs * surf_original.unitnormal(), axis=2) - sign_B_external_normal*vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
            Bmod = bs.AbsB().reshape((nphi,ntheta,1))
            surf_original.to_vtk(os.path.join(this_path,"surf_original"), extra_data= {"B.n/B": BdotN[:, :, None], "B": Bmod})
        cross_section_original = surf_original.cross_section(phi=0)
        r_original = np.sqrt(cross_section_original[:, 0] ** 2 + cross_section_original[:, 1] ** 2)
        z_original = cross_section_original[:, 2]
        plt.plot(r_original, z_original, 'r', linewidth=1, label='Fixed Boundary' if s==s_array[0] else '_no_legend_')
        
        surf_freeb = SurfaceRZFourier.from_wout(wout_freeb_file, s=s, ntheta=ntheta, nphi=nphi)
        if s == np.max(s_array):
            bs.set_points(surf_freeb.gamma().reshape((-1, 3)))
            Bbs = bs.B().reshape((nphi, ntheta, 3))
            BdotN = (np.sum(Bbs * surf_freeb.unitnormal(), axis=2) - sign_B_external_normal*vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
            Bmod = bs.AbsB().reshape((nphi,ntheta,1))
            surf_freeb.to_vtk(os.path.join(this_path,"surf_freeb"), extra_data= {"B.n/B": BdotN[:, :, None], "B": Bmod})
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
            plt.scatter(r, data_this_phi[:, 4], marker='o', s=3, linewidths=1, c=color, label='Poincaré' if s==s_array[0] else '_no_legend_')
    
    plt.legend()
    plt.savefig(os.path.join(this_path,"Fixed_vs_Free_boundary.png"), dpi=150)
    plt.show()