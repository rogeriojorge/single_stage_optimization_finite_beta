#!/usr/bin/env python3
import os
import sys
import numpy as np
import booz_xform as bx
from pathlib import Path
from simsopt.mhd import Vmec, Boozer
import matplotlib.pyplot as plt

def main(file, OUT_DIR=""):
    vmec_final_class = Vmec(file, verbose=False)
    b1 = Boozer(vmec_final_class, mpol=42, ntor=42)
    boozxform_nsurfaces=10
    print('Defining surfaces where to compute Boozer coordinates')
    booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
    print(f' booz_surfaces={booz_surfaces}')
    b1.register(booz_surfaces)
    print('Running BOOZ_XFORM')
    b1.run()
    # b1.bx.write_boozmn(os.path.join(OUT_DIR,"boozmn.nc"))
    print("Plot BOOZ_XFORM")
    fig = plt.figure(figsize=(4, 3)); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, f"Boozxform_surfplot_1_{file[5:-3]}.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(figsize=(4, 3)); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, f"Boozxform_surfplot_2_{file[5:-3]}.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(figsize=(4, 3)); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, f"Boozxform_surfplot_3_{file[5:-3]}.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(figsize=(4, 3)); bx.symplot(b1.bx, helical_detail = True, sqrts=True)
    plt.savefig(os.path.join(OUT_DIR, f"Boozxform_symplot_{file[5:-3]}.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(figsize=(4, 3)); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
    plt.savefig(os.path.join(OUT_DIR, f"Boozxform_modeplot_{file[5:-3]}.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()

if __name__ == "__main__":
    # Create results folders if not present
    try:
        Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
        figures_results_path = str(Path(sys.argv[2]).resolve())
        main(sys.argv[1], sys.argv[2])
    except:
        main(sys.argv[1])