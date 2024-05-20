#!/usr/bin/env python3
import re
import os
import shutil
import numpy as np
from simsopt import load
from simsopt.mhd import Vmec, VirtualCasing
from simsopt.util import MpiPartition
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.util import MpiPartition, proc0_print, comm_world
this_path = os.path.dirname(os.path.abspath(__file__))
mpi = MpiPartition()

filename_input  = f'input.final'
filename_output = f"wout_final.nc"
results_folder = f'optimization_finitebeta_nfp2_QA_ncoils4_stage12'
coils_file = f'biot_savart_opt.json'
ncoils = int(re.search(r'ncoils(\d+)', results_folder).group(1))

run_original_input = False
run_freeb_input = False
create_freeb_surf = True

out_dir = os.path.join(this_path,results_folder)
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)
OUT_DIR = os.path.join(out_dir,"coils")
vmec_file_input = os.path.join(out_dir,filename_input)
surf = SurfaceRZFourier.from_vmec_input(vmec_file_input, nphi=200, ntheta=64, range="full torus")
r0 = np.sqrt(surf.gamma()[:, :, 0] ** 2 + surf.gamma()[:, :, 1] ** 2)
z0 = surf.gamma()[:, :, 2]

wout_freeb_file = os.path.join(out_dir, "wout_final_freeb.nc")
input_freeb_file = os.path.join(out_dir, "input.final_freeb")

coils_filename = os.path.join(OUT_DIR,coils_file)
bs = load(coils_filename)

coils = bs.coils
base_curves = [coils[i]._curve for i in range(ncoils)]
base_currents = [coils[i]._current for i in range(ncoils)]

if run_freeb_input:
    nphi_mgrid = 24
    if comm_world.rank == 0:
        mgrid_file = os.path.join(OUT_DIR, "mgrid.nc")
        bs.to_mgrid(
            mgrid_file, nr=64, nz=65, nphi=nphi_mgrid,
            rmin=0.9*np.min(r0), rmax=1.1*np.max(r0),
            zmin=1.1*np.min(z0), zmax=1.1*np.max(z0), nfp=surf.nfp,
        )
    mpi.comm_world.Barrier()
    vmec = Vmec(vmec_file_input, mpi=mpi)
    vmec.indata.lfreeb = True
    vmec.indata.mgrid_file = mgrid_file
    vmec.indata.nzeta = nphi_mgrid
    vmec.indata.extcur[0] = 1.0
    vmec.write_input(input_freeb_file)
    vmec.run()
    if comm_world.rank == 0:
        shutil.move(os.path.join(out_dir, f"{filename_output[:-3]}_000_000000.nc"), wout_freeb_file)
        os.remove(os.path.join(out_dir, f'{filename_input}_000_000000'))

if run_original_input:
    vmec = Vmec(vmec_file_input, mpi=mpi)
    vmec.run()
    if comm_world.rank == 0:
        shutil.move(os.path.join(out_dir, f"{filename_output[:-3]}_000_000000.nc"), os.path.join(out_dir, filename_output))
        os.remove(os.path.join(out_dir, f'{filename_input}_000_000000'))

if os.path.isfile(wout_freeb_file):
    nphi_vmec_freeb = 28
    ntheta_vmec_freeb = 28
    vmec_freeb = Vmec(wout_freeb_file, mpi=mpi, verbose=False, nphi=nphi_vmec_freeb, ntheta=ntheta_vmec_freeb, range_surface='half period')
    surf_freeb = vmec_freeb.boundary
    nphi_big   = nphi_vmec_freeb * 2 * surf_freeb.nfp + 1
    ntheta_big = ntheta_vmec_freeb + 1
    surf_freeb_big = SurfaceRZFourier(dofs=surf_freeb.dofs, nfp=surf_freeb.nfp, mpol=surf_freeb.mpol, ntor=surf_freeb.ntor, quadpoints_phi=np.linspace(0, 1, nphi_big), quadpoints_theta=np.linspace(0, 1, ntheta_big), stellsym=surf_freeb.stellsym)
    
    vc = VirtualCasing.from_vmec(vmec_freeb, src_nphi=nphi_vmec_freeb, trgt_nphi=nphi_vmec_freeb, trgt_ntheta=ntheta_vmec_freeb, filename=None)
    
    bs.set_points(surf_freeb.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_vmec_freeb, ntheta_vmec_freeb, 3))
    BdotN_surf = (np.sum(Bbs * surf_freeb.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    if comm_world.rank == 0:
        pointData = {"B.n/B": BdotN_surf[:, :, None]}
        surf_freeb.to_vtk(os.path.join(OUT_DIR, "surf_freeb"), extra_data=pointData)
    bs.set_points(surf_freeb_big.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
    BdotN_surf = np.sum(Bbs * surf_freeb_big.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
    if comm_world.rank == 0:
        pointData = {"Bcoils.n/B": BdotN_surf[:, :, None]}
        surf_freeb_big.to_vtk(os.path.join(OUT_DIR, "surf_freeb_big"), extra_data=pointData)
    bs.set_points(surf.gamma().reshape((-1, 3)))

# import os
# import numpy as np
# from pathlib import Path
# from subprocess import run
# import matplotlib.pyplot as plt
# import booz_xform as bx
# from simsopt import load
# from simsopt.geo import curves_to_vtk
# from simsopt.util import MpiPartition
# from simsopt.field.coil import coils_to_makegrid
# from simsopt.mhd import Vmec, QuasisymmetryRatioResidual, Boozer, VirtualCasing
# ###################################################################################
# mpi = MpiPartition()
# parent_path = str(Path(__file__).parent.resolve())
# os.chdir(parent_path)
# directory = f'optimization_QH_finitebeta'
# this_path = os.path.join(parent_path, directory)
# vmec_results_path = os.path.join(this_path, "vmec")
# coils_results_path = os.path.join(this_path, "coils")
# vmec_input_filename = os.path.join(this_path, 'input.final')
# coils_filename = os.path.join(coils_results_path, 'biot_savart_opt.json')
# OUT_DIR = os.path.join(this_path, "figures")
# os.makedirs(OUT_DIR, exist_ok=True)

# bs = load(coils_filename)

# nphi_VMEC = 34
# ntheta_VMEC = 34
# vmec_final = Vmec(vmec_input_filename, mpi=mpi, verbose=False, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
# surf = vmec_final.boundary

# ncoils = int(len(bs.coils)/surf.nfp/2)
# base_curves   = [coil.curve for coil in bs.coils[0:ncoils]]
# base_currents = [coil.current for coil in bs.coils[0:ncoils]]

# coils_to_makegrid(os.path.join(coils_results_path,'coils.opt_coils'), base_curves, base_currents, nfp=surf.nfp, stellsym=True)

# r0 = np.sqrt(surf.gamma()[:, :, 0] ** 2 + surf.gamma()[:, :, 1] ** 2)
# z0 = surf.gamma()[:, :, 2]
# nzeta = 45
# nr = 47
# nz = 49
# rmin=0.9*np.min(r0)
# rmax=1.1*np.max(r0)
# zmin=1.1*np.min(z0)
# zmax=1.1*np.max(z0)

# with open(os.path.join(coils_results_path,'input_xgrid.dat'), 'w') as f:
#     f.write('opt_coils\n')
#     f.write('S\n')
#     f.write('y\n')
#     f.write(f'{rmin}\n')
#     f.write(f'{rmax}\n')
#     f.write(f'{zmin}\n')
#     f.write(f'{zmax}\n')
#     f.write(f'{nzeta}\n')
#     f.write(f'{nr}\n')
#     f.write(f'{nz}\n')

# print("Running makegrid")
# os.chdir(coils_results_path)
# mgrid_executable = '/Users/rogeriojorge/bin/xgrid'
# run_string = f"{mgrid_executable} < {os.path.join(coils_results_path,'input_xgrid.dat')} > {os.path.join(coils_results_path,'log_xgrid.opt_coils')}"
# run(run_string, shell=True, check=True)
# os.chdir(this_path)
# print(" done")

# os.chdir(coils_results_path)
# mgrid_file = os.path.join(coils_results_path,'mgrid_opt_coils.nc')

# vmec_final.indata.lfreeb = True
# vmec_final.indata.mgrid_file = os.path.join(coils_results_path,'mgrid_opt_coils.nc')
# vmec_final.indata.extcur[0:len(bs.coils)] = [-c.current.get_value()*1.515*1e-7 for c in bs.coils]
# vmec_final.indata.nvacskip = 6
# vmec_final.indata.nzeta = nzeta
# vmec_final.indata.phiedge = vmec_final.indata.phiedge

# vmec_final.indata.ns_array[:4]    = [   9,    29,    49,   101]
# vmec_final.indata.niter_array[:4] = [4000,  6000,  6000,  8000]
# vmec_final.indata.ftol_array[:4]  = [1e-5,  1e-6, 1e-12, 1e-15]

# vmec_final.write_input(os.path.join(this_path,'input.final_freeb'))

# # vmec_final.run()

# print("Running VMEC")
# os.chdir(vmec_results_path)
# vmec_executable = '/Users/rogeriojorge/bin/xvmec2000'
# run_string = f"{vmec_executable} {os.path.join(this_path,'input.final_freeb')}"
# run(run_string, shell=True, check=True)
# os.chdir(this_path)
# print(" done")

# print("Plotting VMEC result")
# if os.path.isfile(os.path.join(vmec_results_path, f"wout_final_freeb.nc")):
#     print('Found final vmec file')
#     print("Plot VMEC result")
#     import vmecPlot2
#     vmecPlot2.main(file=os.path.join(vmec_results_path, f"wout_final_freeb.nc"), name='free_b', figures_folder=OUT_DIR)
#     vmec_freeb = Vmec(os.path.join(vmec_results_path, f"wout_final_freeb.nc"), nphi=nphi_VMEC, ntheta=ntheta_VMEC)
#     quasisymmetry_target_surfaces = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#     if 'QA' in directory:
#         qs = QuasisymmetryRatioResidual(vmec_freeb, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=0)
#     else:
#         qs = QuasisymmetryRatioResidual(vmec_freeb, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1)

#     print('####################################################')
#     print('####################################################')
#     print('Quasisymmetry objective free boundary =',qs.total())
#     print('Mean iota free boundary =',vmec_freeb.mean_iota())
#     print('####################################################')
#     print('####################################################')

#     print('Creating Boozer class for vmec_freeb')
#     b1 = Boozer(vmec_freeb, mpol=64, ntor=64)
#     print('Defining surfaces where to compute Boozer coordinates')
#     boozxform_nsurfaces = 10
#     booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
#     print(f' booz_surfaces={booz_surfaces}')
#     b1.register(booz_surfaces)
#     print('Running BOOZ_XFORM')
#     try:
#         b1.run()
#         # b1.bx.write_boozmn(os.path.join(vmec_results_path,'vmec',"boozmn_free_b.nc"))
#         print("Plot BOOZ_XFORM")
#         fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
#         plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
#         fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
#         plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
#         fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
#         plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
#         fig = plt.figure(); bx.symplot(b1.bx, helical_detail = True if 'QH' in directory else False, sqrts=True)
#         plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
#         fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
#         plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_free_b.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
#     except Exception as e: print(e)

#     s_final = vmec_freeb.boundary
#     B_on_surface_final = bs.set_points(s_final.gamma().reshape((-1, 3))).AbsB()
#     Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
#     vc_src_nphi = nphi_VMEC
#     vc_final = VirtualCasing.from_vmec(vmec_freeb, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
#     BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc_final.B_external_normal
#     curves = [c.curve for c in bs.coils]
#     curves_to_vtk(curves, os.path.join(coils_results_path, "curves_freeb"))
#     pointData = {"B_N": BdotN_surf[:, :, None]}
#     s_final.to_vtk(os.path.join(coils_results_path, "surf_freeb"), extra_data=pointData)

# os.remove(os.path.join(parent_path,'threed1.final'))