#!/usr/bin/env python3
import os
import imageio
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec
from simsopt.objectives import SquaredFlux, LeastSquaresProblem
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import create_equally_spaced_curves, curves_to_vtk, SurfaceRZFourier
# from single_stage_movie import ncoils, nmodes_coils
####################################################################################
####################################################################################
parent_path='/Users/rogeriojorge/local/single_stage_optimization_finite_beta/'
this_path = os.path.join(parent_path, 'optimization_nfp2')
os.chdir(this_path)
####################################################################################
csv_path = os.path.join(this_path,'opt_dofs.csv')
vmec_input_filename = os.path.join(parent_path, 'vmec_inputs', 'input.nfp2_torus')
output_path = os.path.join(this_path, 'output_movie')
Path(output_path).mkdir(parents=True, exist_ok=True)
max_current = 1e5
ncoils = 4
nmodes_coils = 8
ntheta=32
nphi=32
####################################################################################
df = pd.read_csv(csv_path)
vmec_columns = [col for col in df.columns if 'vmec' in col]
coils_columns = [col for col in df.columns if 'coils' in col]
vmec_data = df[vmec_columns].values
coils_data = df[coils_columns].values
vmec = Vmec(vmec_input_filename, mpi=None, verbose=False, ntheta=ntheta, nphi=nphi, range_surface='full torus')
surf = vmec.boundary

nphi_big = nphi * 2 * surf.nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi = np.linspace(0, 1, nphi_big)

vmec.run()
R0_coils = np.sum(vmec.wout.raxis_cc)
R1_coils = np.min((vmec.wout.Aminor_p*2.6,R0_coils/1.4))
base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0_coils, R1=R1_coils, order=nmodes_coils, numquadpoints=128)
base_currents = [Current(1) * max_current for _ in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))

JF = SquaredFlux(surf, bs, definition="local")

surf.fix_all()
surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
surf.fix("rc(0,0)")

image_files = []
plotter = pv.Plotter(off_screen=True)
plotter.camera_position = (40,15,33)
plotter.camera_set = True
plotter.camera.Zoom(0.7)
for index, row in df.iterrows():
    surf.x = vmec_data[index]
    JF.x = coils_data[index]
    surf_big = SurfaceRZFourier(dofs=surf.dofs,nfp=surf.nfp,mpol=surf.mpol,ntor=surf.ntor,quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta,)
    bs.set_points(surf_big.gamma().reshape((-1, 3)))
    BdotN = (np.sum(bs.B().reshape((nphi_big, ntheta_big, 3)) * surf_big.unitnormal(), axis=2)) / np.linalg.norm(bs.B().reshape((nphi_big, ntheta_big, 3)), axis=2)
    surf_big.to_vtk(os.path.join(output_path, f"surf_between_halfnfp_{index}"), extra_data={"B.n/B": BdotN[:, :, None]})
    vtk_file = os.path.join(output_path, f"surf_between_halfnfp_{index}.vts")
    png_file = os.path.join(output_path, f"surf_between_halfnfp_{index}.png")
    curves_file = os.path.join(output_path,f'curves_between_halfnfp_{index}.vtu')
    curves_to_vtk(curves, curves_file[:-4], close=True)
    coils_vtu = pv.read(curves_file)
    surf_between_vtk = pv.read(vtk_file)
    args_cbar = dict(height=0.1, title_font_size=24, label_font_size=16, position_x=0.22, color="k")
    surf_mesh = plotter.add_mesh(surf_between_vtk, scalars="B.n/B", cmap="coolwarm", name='surf', scalar_bar_args=args_cbar)
    for coil_index, coil in enumerate(bs.coils):
        coil_points = coils_vtu.extract_cells(coil_index)
        if coil_points.n_points > 0:
            plotter.add_mesh(coil_points, line_width=11, color="white", render_lines_as_tubes=True, lighting=True, show_scalar_bar=False, name=f'cube{coil_index}')
    plotter.set_background("white")    
    plotter.screenshot(png_file, return_img=False)
    image_files.append(png_file)
plotter.close()
pv.close_all()
    
gif_file = os.path.join(output_path, "surf_between_animation.gif")
with imageio.get_writer(gif_file, mode='I') as writer:
    for image_file in image_files:
        image = imageio.v2.imread(image_file)
        writer.append_data(image)

print(f"GIF created: {gif_file}")

save_indices = [0, len(df)-1]
for vtk_file in os.listdir(output_path):
    if vtk_file.endswith(".vts"): os.remove(os.path.join(output_path, vtk_file))
    if vtk_file.endswith(".vtu"): os.remove(os.path.join(output_path, vtk_file))
    if vtk_file.endswith(".png"):
        if vtk_file not in [f"surf_between_halfnfp_{index}.png" for index in save_indices]:
            os.remove(os.path.join(output_path, vtk_file))