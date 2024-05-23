#!/usr/bin/env python3
import re
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from simsopt import load
from simsopt.mhd import Vmec, VirtualCasing
from simsopt.util import MpiPartition
from simsopt.geo import SurfaceRZFourier
from simsopt.util import MpiPartition, proc0_print, comm_world
this_path = os.path.dirname(os.path.abspath(__file__))
mpi = MpiPartition()

# cd optimization_finitebeta_nfp3_QA_ncoils3_stage23
# ../src/vmecPlot2.py wout_final.nc;../src/vmecPlot2.py wout_final_freeb.nc;../src/booz_plot.py wout_final.nc;../src/booz_plot.py wout_final_freeb.nc

results_folder = f'optimization_finitebeta_nfp3_QA_ncoils3_stage23'
run_freeb_and_original_input = True

nphi_plot = 32
ntheta_plot = 80
s_array = [0.05, 0.25, 0.55, 1.0]
phi_plot=np.pi/2#np.pi/3

filename_input  = f'input.final'
filename_output = f"wout_final.nc"
coils_file = f'biot_savart_opt.json'
ncoils = int(re.search(r'ncoils(\d+)', results_folder).group(1))


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
input_freeb_file_from_wout = os.path.join(out_dir, "input.final_freeb_from_wout")

coils_filename = os.path.join(OUT_DIR,coils_file)
bs = load(coils_filename)

coils = bs.coils
base_curves = [coils[i]._curve for i in range(ncoils)]
base_currents = [coils[i]._current for i in range(ncoils)]

if run_freeb_and_original_input:
    nphi_mgrid = 32
    if comm_world.rank == 0:
        mgrid_file = os.path.join(OUT_DIR, "mgrid.nc")
        bs.to_mgrid(
            mgrid_file, nr=64, nz=65, nphi=nphi_mgrid,
            rmin=0.7*np.min(r0), rmax=1.3*np.max(r0),
            zmin=1.3*np.min(z0), zmax=1.3*np.max(z0), nfp=surf.nfp,
        )
    mpi.comm_world.Barrier()
    vmec = Vmec(vmec_file_input, mpi=mpi)
    vmec.indata.lfreeb = True
    vmec.indata.mgrid_file = mgrid_file
    vmec.indata.nzeta = nphi_mgrid
    vmec.indata.extcur[0] = 1.0
    
    vmec.indata.ns_array   [:5] = [ 5,      16,   21,    51,   101]
    vmec.indata.niter_array[:5] = [ 100,   300,  400,   400, 22000]
    vmec.indata.ftol_array [:5] = [ 1e-9, 1e-9, 1e-9, 1e-10, 1e-14]
    
    vmec.write_input(input_freeb_file)
    # print('Running once to get the free boundary')
    vmec.run()
    if comm_world.rank == 0:
        shutil.move(os.path.join(out_dir, f"{filename_output[:-3]}_000_000000.nc"), wout_freeb_file)
        os.remove(os.path.join(out_dir, f'{filename_input}_000_000000'))
    # vmec.boundary = SurfaceRZFourier.from_wout(vmec.output_file)
    vmec.boundary = SurfaceRZFourier.from_wout(wout_freeb_file)
    # print('Running again to get the input file')
    vmec.write_input(input_freeb_file_from_wout)
    # vmec.run()

if run_freeb_and_original_input:
    vmec = Vmec(vmec_file_input, mpi=mpi)
    vmec.run()
    if comm_world.rank == 0:
        shutil.move(os.path.join(out_dir, f"{filename_output[:-3]}_000_000000.nc"), os.path.join(out_dir, filename_output))
        os.remove(os.path.join(out_dir, f'{filename_input}_000_000000'))

if os.path.isfile(wout_freeb_file):
    nphi_vmec_freeb = 28
    ntheta_vmec_freeb = 28
    vmec_freeb = Vmec(wout_freeb_file, mpi=mpi, verbose=False, nphi=nphi_vmec_freeb, ntheta=ntheta_vmec_freeb, range_surface='half period')
    # vmec_freeb = Vmec(input_freeb_file_from_wout, mpi=mpi, verbose=False, nphi=nphi_vmec_freeb, ntheta=ntheta_vmec_freeb, range_surface='half period')
    surf_freeb = vmec_freeb.boundary
    nphi_big   = nphi_vmec_freeb * 2 * surf_freeb.nfp + 1
    ntheta_big = ntheta_vmec_freeb + 1
    surf_freeb_big = SurfaceRZFourier(dofs=surf_freeb.dofs, nfp=surf_freeb.nfp, mpol=surf_freeb.mpol, ntor=surf_freeb.ntor, quadpoints_phi=np.linspace(0, 1, nphi_big), quadpoints_theta=np.linspace(0, 1, ntheta_big), stellsym=surf_freeb.stellsym)
    
    vc = VirtualCasing.from_vmec(wout_freeb_file, src_nphi=nphi_vmec_freeb, trgt_nphi=nphi_vmec_freeb, trgt_ntheta=ntheta_vmec_freeb, filename=None)
    
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

if os.path.exists(wout_freeb_file) and os.path.exists(os.path.join(out_dir, filename_output)):
    print('Do not use wout for plotting, use input')
    fig = plt.figure()
    fig.set_size_inches(6,6)
    ax=fig.add_subplot(111, label="1")
    for i, s in enumerate(s_array):
        # vmec_freeb = Vmec(wout_freeb_file, mpi=mpi, verbose=False, nphi=nphi_vmec_freeb, ntheta=ntheta_plot, range_surface='half period')
        # surf_freeb = vmec_freeb.boundary
        surf_freeb = SurfaceRZFourier.from_wout(wout_freeb_file, quadpoints_phi=np.linspace(0, 1, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1), s=s)
        surf_freeb_input = SurfaceRZFourier.from_vmec_input(input_freeb_file_from_wout, quadpoints_phi=np.linspace(0, 1, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1))
        cross_section_freeb = surf_freeb.cross_section(phi=phi_plot)
        cross_section_freeb_input = surf_freeb_input.cross_section(phi=phi_plot)
        r_interp_freeb = np.sqrt(cross_section_freeb[:, 0] ** 2 + cross_section_freeb[:, 1] ** 2)
        r_interp_freeb_input = np.sqrt(cross_section_freeb_input[:, 0] ** 2 + cross_section_freeb_input[:, 1] ** 2)
        z_interp_freeb = cross_section_freeb[:, 2]
        z_interp_freeb_input = cross_section_freeb_input[:, 2]
        
        # vmec_final = Vmec(os.path.join(out_dir, filename_output), mpi=mpi, verbose=False, nphi=nphi_plot, ntheta=ntheta_plot, range_surface='half period')
        # surf_final = vmec_final.boundary
        surf_final = SurfaceRZFourier.from_wout(os.path.join(out_dir, filename_output), quadpoints_phi=np.linspace(0, 1, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1), s=s)
        surf_final_input = SurfaceRZFourier.from_vmec_input(os.path.join(out_dir, filename_input), quadpoints_phi=np.linspace(0, 1, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1))
        cross_section_final = surf_final.cross_section(phi=phi_plot)
        cross_section_final_input = surf_final_input.cross_section(phi=phi_plot)
        r_interp_final = np.sqrt(cross_section_final[:, 0] ** 2 + cross_section_final[:, 1] ** 2)
        r_interp_final_input = np.sqrt(cross_section_final_input[:, 0] ** 2 + cross_section_final_input[:, 1] ** 2)
        z_interp_final = cross_section_final[:, 2]
        z_interp_final_input = cross_section_final_input[:, 2]
        
        plt.plot(r_interp_final, z_interp_final, 'r', linewidth=3, label=f'Fixed Boundary' if i==0 else None)
        plt.plot(r_interp_freeb, z_interp_freeb, 'k-.', linewidth=2, label=f'Free Boundary' if i==0 else None)
        # plt.plot(r_interp_final_input, z_interp_final_input, 'g--', linewidth=3, label=f'Fixed Boundary input' if i==0 else None)
        # plt.plot(r_interp_freeb_input, z_interp_freeb_input, 'b.-', linewidth=1, label=f'Free Boundary input' if i==0 else None)
    plt.gca().set_aspect('equal',adjustable='box')
    plt.legend(fontsize=12)
    plt.xlabel('R', fontsize=22)
    plt.ylabel('Z', fontsize=22)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'Fixed_vs_Free_boundary.png'), dpi=300)
    # plt.show()
    plt.close()
    
if 'stage23' in results_folder:
    path_with_stage12 = os.path.join(this_path,results_folder.replace('stage23','stage12'))
    try:
        fig = plt.figure()
        fig.set_size_inches(6,6)
        ax=fig.add_subplot(111, label="1")
        print('Do not use wout for plotting, use input')
        for i, s in enumerate(s_array):
            surf_freeb_stage3 = SurfaceRZFourier.from_wout(wout_freeb_file, quadpoints_phi=np.linspace(0, 1/2/surf.nfp, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1), s=s)
            # surf_freeb_stage3 = SurfaceRZFourier.from_vmec_input(input_freeb_file_from_wout, quadpoints_phi=np.linspace(0, 1/2/surf.nfp, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1))
            # surf_freeb_stage3.to_vtk(os.path.join(OUT_DIR,"surf_freeb_stage3"))
            cross_section_freeb_stage3 = surf_freeb_stage3.cross_section(phi=phi_plot)
            r_interp_freeb_stage3 = np.sqrt(cross_section_freeb_stage3[:, 0] ** 2 + cross_section_freeb_stage3[:, 1] ** 2)
            z_interp_freeb_stage3 = cross_section_freeb_stage3[:, 2]
            
            surf_final_stage3 = SurfaceRZFourier.from_wout(os.path.join(out_dir, filename_output), quadpoints_phi=np.linspace(0, 1/2/surf.nfp, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1), s=s)
            # surf_final_stage3 = SurfaceRZFourier.from_vmec_input(os.path.join(out_dir, filename_input), quadpoints_phi=np.linspace(0, 1/2/surf.nfp, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1))
            # surf_final_stage3.to_vtk(os.path.join(OUT_DIR,"surf_final_stage3"))
            cross_section_final_stage3 = surf_final_stage3.cross_section(phi=phi_plot)
            r_interp_final_stage3 = np.sqrt(cross_section_final_stage3[:, 0] ** 2 + cross_section_final_stage3[:, 1] ** 2)
            z_interp_final_stage3 = cross_section_final_stage3[:, 2]
            
            
            surf_freeb_stage1 = SurfaceRZFourier.from_wout(os.path.join(path_with_stage12, "wout_final_freeb.nc"), quadpoints_phi=np.linspace(0, 1/2/surf.nfp, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1), s=s)
            # surf_freeb_stage1 = SurfaceRZFourier.from_vmec_input(os.path.join(path_with_stage12, "input.final_freeb_from_wout"), quadpoints_phi=np.linspace(0, 1/2/surf.nfp, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1)) 
            # surf_freeb_stage1.to_vtk(os.path.join(OUT_DIR,"surf_freeb_stage1"))
            cross_section_freeb_stage1 = surf_freeb_stage1.cross_section(phi=phi_plot)
            r_interp_freeb_stage1 = np.sqrt(cross_section_freeb_stage1[:, 0] ** 2 + cross_section_freeb_stage1[:, 1] ** 2)
            z_interp_freeb_stage1 = cross_section_freeb_stage1[:, 2]
            
            surf_final_stage1 = SurfaceRZFourier.from_wout(os.path.join(path_with_stage12, filename_output), quadpoints_phi=np.linspace(0, 1/2/surf.nfp, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1), s=s)
            # surf_final_stage1 = SurfaceRZFourier.from_vmec_input(os.path.join(path_with_stage12, filename_input), quadpoints_phi=np.linspace(0, 1/2/surf.nfp, nphi_plot * 2 * surf_freeb.nfp + 1), quadpoints_theta=np.linspace(0, 1, ntheta_plot + 1))
            # surf_final_stage1.to_vtk(os.path.join(OUT_DIR,"surf_final_stage1"))
            cross_section_final_stage1 = surf_final_stage1.cross_section(phi=phi_plot)
            r_interp_final_stage1 = np.sqrt(cross_section_final_stage1[:, 0] ** 2 + cross_section_final_stage1[:, 1] ** 2)
            z_interp_final_stage1 = cross_section_final_stage1[:, 2]
            
            plt.plot(r_interp_final_stage1, z_interp_final_stage1, 'b-', linewidth=2, label=f'Stage 1 (Fixed Boundary)' if i==0 else None)
            plt.plot(r_interp_freeb_stage1, z_interp_freeb_stage1, 'g--', linewidth=2, label=f'Stage 2 (Free Boundary)' if i==0 else None)
            plt.plot(r_interp_final_stage3, z_interp_final_stage3, 'r.-', linewidth=2, label=f'Single-Stage (Fixed Boundary)' if i==0 else None)
            plt.plot(r_interp_freeb_stage3, z_interp_freeb_stage3, 'k-.', linewidth=2, label=f'Single-Stage (Free Boundary)' if i==0 else None)
            
        plt.gca().set_aspect('equal',adjustable='box')
        plt.legend(fontsize=12)
        plt.xlabel('R', fontsize=22)
        plt.ylabel('Z', fontsize=22)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,'stage123_comparison.png'), dpi=300)
        plt.show()
        plt.close()
    except Exception as e:
        print(e)