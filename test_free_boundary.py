#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
from subprocess import run
import matplotlib.pyplot as plt
import booz_xform as bx
from simsopt import load
from simsopt.geo import curves_to_vtk
from simsopt.util import MpiPartition
from simsopt.field.coil import coils_to_makegrid
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual, Boozer, VirtualCasing
###################################################################################
mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
directory = f'optimization_QH_finitebeta'
this_path = os.path.join(parent_path, directory)
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
vmec_input_filename = os.path.join(this_path, 'input.final')
coils_filename = os.path.join(coils_results_path, 'biot_savart_opt.json')
OUT_DIR = os.path.join(this_path, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

bs = load(coils_filename)

nphi_VMEC = 34
ntheta_VMEC = 34
vmec_final = Vmec(vmec_input_filename, mpi=mpi, verbose=False, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
surf = vmec_final.boundary

ncoils = int(len(bs.coils)/surf.nfp/2)
base_curves   = [coil.curve for coil in bs.coils[0:ncoils]]
base_currents = [coil.current for coil in bs.coils[0:ncoils]]

coils_to_makegrid(os.path.join(coils_results_path,'coils.opt_coils'), base_curves, base_currents, nfp=surf.nfp, stellsym=True)

r0 = np.sqrt(surf.gamma()[:, :, 0] ** 2 + surf.gamma()[:, :, 1] ** 2)
z0 = surf.gamma()[:, :, 2]
nzeta = 45
nr = 47
nz = 49
rmin=0.9*np.min(r0)
rmax=1.1*np.max(r0)
zmin=1.1*np.min(z0)
zmax=1.1*np.max(z0)

with open(os.path.join(coils_results_path,'input_xgrid.dat'), 'w') as f:
    f.write('opt_coils\n')
    f.write('S\n')
    f.write('y\n')
    f.write(f'{rmin}\n')
    f.write(f'{rmax}\n')
    f.write(f'{zmin}\n')
    f.write(f'{zmax}\n')
    f.write(f'{nzeta}\n')
    f.write(f'{nr}\n')
    f.write(f'{nz}\n')

print("Running makegrid")
os.chdir(coils_results_path)
mgrid_executable = '/Users/rogeriojorge/bin/xgrid'
run_string = f"{mgrid_executable} < {os.path.join(coils_results_path,'input_xgrid.dat')} > {os.path.join(coils_results_path,'log_xgrid.opt_coils')}"
run(run_string, shell=True, check=True)
os.chdir(this_path)
print(" done")

os.chdir(coils_results_path)
mgrid_file = os.path.join(coils_results_path,'mgrid_opt_coils.nc')

vmec_final.indata.lfreeb = True
vmec_final.indata.mgrid_file = os.path.join(coils_results_path,'mgrid_opt_coils.nc')
vmec_final.indata.extcur[0:len(bs.coils)] = [-c.current.get_value()*1.515*1e-7 for c in bs.coils]
vmec_final.indata.nvacskip = 6
vmec_final.indata.nzeta = nzeta
vmec_final.indata.phiedge = vmec_final.indata.phiedge

vmec_final.indata.ns_array[:4]    = [   9,    29,    49,   101]
vmec_final.indata.niter_array[:4] = [4000,  6000,  6000,  8000]
vmec_final.indata.ftol_array[:4]  = [1e-5,  1e-6, 1e-12, 1e-15]

vmec_final.write_input(os.path.join(this_path,'input.final_freeb'))

# vmec_final.run()

print("Running VMEC")
os.chdir(vmec_results_path)
vmec_executable = '/Users/rogeriojorge/bin/xvmec2000'
run_string = f"{vmec_executable} {os.path.join(this_path,'input.final_freeb')}"
run(run_string, shell=True, check=True)
os.chdir(this_path)
print(" done")

print("Plotting VMEC result")
if os.path.isfile(os.path.join(vmec_results_path, f"wout_final_freeb.nc")):
    print('Found final vmec file')
    print("Plot VMEC result")
    import vmecPlot2
    vmecPlot2.main(file=os.path.join(vmec_results_path, f"wout_final_freeb.nc"), name='free_b', figures_folder=OUT_DIR)
    vmec_freeb = Vmec(os.path.join(vmec_results_path, f"wout_final_freeb.nc"), nphi=nphi_VMEC, ntheta=ntheta_VMEC)
    quasisymmetry_target_surfaces = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    if 'QA' in directory:
        qs = QuasisymmetryRatioResidual(vmec_freeb, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=0)
    else:
        qs = QuasisymmetryRatioResidual(vmec_freeb, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1)

    print('####################################################')
    print('####################################################')
    print('Quasisymmetry objective free boundary =',qs.total())
    print('Mean iota free boundary =',vmec_freeb.mean_iota())
    print('####################################################')
    print('####################################################')

    print('Creating Boozer class for vmec_freeb')
    b1 = Boozer(vmec_freeb, mpol=64, ntor=64)
    print('Defining surfaces where to compute Boozer coordinates')
    boozxform_nsurfaces = 10
    booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
    print(f' booz_surfaces={booz_surfaces}')
    b1.register(booz_surfaces)
    print('Running BOOZ_XFORM')
    try:
        b1.run()
        # b1.bx.write_boozmn(os.path.join(vmec_results_path,'vmec',"boozmn_free_b.nc"))
        print("Plot BOOZ_XFORM")
        fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.symplot(b1.bx, helical_detail = True if 'QH' in directory else False, sqrts=True)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    except Exception as e: print(e)

    s_final = vmec_freeb.boundary
    B_on_surface_final = bs.set_points(s_final.gamma().reshape((-1, 3))).AbsB()
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    vc_src_nphi = nphi_VMEC
    vc_final = VirtualCasing.from_vmec(vmec_freeb, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
    BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc_final.B_external_normal
    curves = [c.curve for c in bs.coils]
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_freeb"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    s_final.to_vtk(os.path.join(coils_results_path, "surf_freeb"), extra_data=pointData)