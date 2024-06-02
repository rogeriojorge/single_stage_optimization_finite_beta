#!/usr/bin/env python3
import os
import shutil
import numpy as np
from simsopt import load
from simsopt.mhd import Vmec, VirtualCasing
from simsopt.util import MpiPartition
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.util import MpiPartition
mpi = MpiPartition()
################## Input parameters
filename_input  = f'input.final'
input_freeb_file = f'input.final_freeb'
filename_output = f"wout_final.nc"
wout_freeb_file = f"wout_final_freeb.nc"
coils_file = f'biot_savart_opt.json'
mgrid_file = "mgrid.nc"
ncoils = 5
nphi_vmec = 32
ntheta_vmec = 32
nphi_mgrid = 32
################## Run Script
## Create the surface
surf = SurfaceRZFourier.from_vmec_input(filename_input, nphi=nphi_vmec, ntheta=ntheta_vmec, range="half period")
r0 = np.sqrt(surf.gamma()[:, :, 0] ** 2 + surf.gamma()[:, :, 1] ** 2)
z0 = surf.gamma()[:, :, 2]
## Load the coils
bs = load(coils_file)
coils = bs.coils
base_curves = [coils[i]._curve for i in range(ncoils)]
base_currents = [coils[i]._current for i in range(ncoils)]
bs.set_points(surf.gamma().reshape((-1, 3)))
## Create the virtual casing
vc = VirtualCasing.from_vmec(filename_output, src_nphi=nphi_vmec, trgt_nphi=nphi_vmec, trgt_ntheta=ntheta_vmec, filename=None)
## Create surf and curves figures
Bbs = bs.B().reshape((nphi_vmec, ntheta_vmec, 3))
BdotN_surf = (np.sum(Bbs * surf.unitnormal(), axis=2) + vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
Bmod = bs.AbsB().reshape((nphi_vmec,ntheta_vmec,1))
pointData = {"B.n/B": BdotN_surf[:, :, None], "B": Bmod}
surf.to_vtk("surf", extra_data=pointData)
curves_to_vtk(base_curves, "curves", close=True)
## Create the mgrid file
bs.to_mgrid(mgrid_file, nfp=surf.nfp,
            nr=32, nz=32, nphi=nphi_mgrid,
            rmin=0.7*np.min(r0), rmax=1.3*np.max(r0),
            zmin=1.3*np.min(z0), zmax=1.3*np.max(z0))
## Run VMEC free boundary
vmec = Vmec(filename_input, mpi=mpi)
vmec.indata.lfreeb = True
vmec.indata.mpol = 6
vmec.indata.ntor = 6
vmec.indata.mgrid_file = mgrid_file
vmec.indata.nzeta = nphi_mgrid
vmec.indata.extcur[0] = 1.0
vmec.indata.ns_array   [:3] = [ 5,      16,   31]#,    51,   101]
vmec.indata.niter_array[:3] = [ 100,   200,  3000]#,   400, 20000]
vmec.indata.ftol_array [:3] = [ 1e-9, 1e-9, 1e-11]#, 1e-10, 1e-14]
vmec.run()
shutil.move(os.path.join(f"{filename_output[:-3]}_000_000000.nc"), wout_freeb_file)
os.remove(os.path.join(f'{filename_input}_000_000000'))
vmec.boundary = SurfaceRZFourier.from_wout(wout_freeb_file)
vmec.write_input(input_freeb_file)