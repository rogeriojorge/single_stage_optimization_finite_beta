#!/usr/bin/env python3
import os
from simsopt.mhd import Vmec, Boozer
import matplotlib.pyplot as plt
from qsc import Qsc
import numpy as np
import logging
logger = logging.getLogger(__name__)
from scipy.optimize import curve_fit
import sys

def from_boozxform(vmec_file, max_s_for_fit = 0.25, N_phi = 200, max_n_to_plot = 2, show=False, nNormal=None, savefig=True):

    name=os.path.basename(vmec_file)[5:-3]
    
    vmec = Vmec(vmec_file, verbose=False)
    
    b1 = Boozer(vmec, mpol=20, ntor=20)
    boozxform_nsurfaces=vmec.wout.ns
    booz_surfaces = np.linspace(0.0,1.0,boozxform_nsurfaces,endpoint=False)
    b1.register(booz_surfaces)
    print('Running BOOZ_XFORM')
    b1.run()
    
    print('Done running BOOZ_XFORM')
    bmnc = np.transpose(b1.bx.bmnc_b)
    ixm = b1.bx.xm_b
    ixn = b1.bx.xn_b
    jlist = b1.bx.compute_surfs+2
    ns = vmec.wout.ns
    nfp = b1.bx.nfp
    Psi_a = np.abs(vmec.wout.phi[-1])
    iotaVMECt = b1.bx.iota[0]
    rc = vmec.wout.raxis_cc
    zs = -vmec.wout.zaxis_cs
    if vmec.wout.lasym__logical__:
        rs = -vmec.wout.raxis_cs
        zc = vmec.wout.zaxis_cc
    else:
        rs = []
        zc = []
        
    print('Preparing coordinates for fit')
    s_full = np.linspace(0,1,ns)
    ds = s_full[1] - s_full[0]
    #s_half = s_full[1:] - 0.5*ds
    s_half = s_full[jlist-1] - 0.5*ds
    mask = s_half < max_s_for_fit
    s_fine = np.linspace(0,1,400)
    sqrts_fine = s_fine
    Boozer_I = b1.bx.Boozer_I
    Boozer_G = b1.bx.Boozer_G
    phi = np.linspace(0,2*np.pi / nfp, N_phi)
    B0  = np.zeros(N_phi)
    B1s = np.zeros(N_phi)
    B1c = np.zeros(N_phi)
    B20 = np.zeros(N_phi)
    B2s = np.zeros(N_phi)
    B2c = np.zeros(N_phi)
    ####################################################################################################
    print('Performing fit to Boozer I')
    def model_1(x, a): return a*x**(5/4)
    def model_2(x, a): return a*x**(4/4)
    params_1, params_covariance_1 = curve_fit(model_1, s_half[mask], Boozer_I[mask])
    params_2, params_covariance_2 = curve_fit(model_2, s_half[mask], Boozer_I[mask])
    print(f'  I_5/2 = {params_1[0]}')
    print(f'  I_2 = {params_2[0]}')
    plt.plot(np.sqrt(s_half), Boozer_I, 'o-', label='Boozer I(r)', linewidth=4.0)
    plt.plot(np.sqrt(s_fine), model_2(s_fine, *params_2), 'g--', label=r'Near-Axis Model $I=r^2 I_2$', linewidth=3.5)
    plt.plot(np.sqrt(s_fine), model_1(s_fine, *params_1), 'r--', label=r'Modified Model $I=r^{5/2} I_{5/2}$', linewidth=3.5)
    plt.legend(fontsize=14)
    plt.xlabel(r'$r=\sqrt{s}$', fontsize=14)
    plt.ylabel(r'$I$', fontsize=14)
    plt.xlim([0,np.sqrt(max_s_for_fit)])
    plt.ylim([0.0,Boozer_I[mask][-1]])
    if show: plt.show()
    if savefig: plt.savefig(f'NAE_Boozer_I_fit_{name}.png',dpi=300)
    plt.close()
    ####################################################################################################
    print('Performing fit to Boozer iota')
    Boozer_iota = b1.bx.iota
    print(f' iotaVMEC = {iotaVMECt}')
    def model_1(x, a, b): return a + b*x**(1/2)
    def model_2(x, a, b): return a + b*x
    params_1, params_covariance_1 = curve_fit(model_1, s_half[mask], Boozer_iota[mask])
    params_2, params_covariance_2 = curve_fit(model_2, s_half[mask], Boozer_iota[mask])
    print(f' Modified  iota: iota_0 = {params_1[0]}, iota_1 = {params_1[1]}')
    print(f' Near-Axis iota: iota_0 = {params_2[0]}, iota_2 = {params_2[1]}')
    plt.plot(np.sqrt(s_half), Boozer_iota, 'o-', label=r'Boozer $\iota(r)$', linewidth=4.0)
    plt.plot(np.sqrt(s_fine), model_2(s_fine, *params_2), 'g--', label=r'Near-Axis Model $\iota=\iota_0 + r^2 \iota_2$', linewidth=3.5)
    plt.plot(np.sqrt(s_fine), model_1(s_fine, *params_1), 'r--', label=r'Modified Model $\iota=\iota_0 + r \iota_1$', linewidth=3.5)
    plt.xlim([0,np.sqrt(max_s_for_fit)])
    if Boozer_iota[mask][-1] > 0: plt.ylim([0.95*np.min(Boozer_iota[mask]),1.05*np.max(Boozer_iota[mask])])
    else: plt.ylim([1.05*np.min(Boozer_iota[mask]),0.95*np.max(Boozer_iota[mask])])
    plt.legend(fontsize=14)
    plt.xlabel(r'$r=\sqrt{s}$', fontsize=14)
    plt.ylabel(r'$\iota$', fontsize=14)
    plt.tight_layout()
    if show: plt.show()
    if savefig: plt.savefig(f'NAE_Boozer_iota_fit_{name}.png',dpi=300)
    plt.close()
    ####################################################################################################
    print('Performing fit to Boozer G')
    def model_1(x, a, b, c): return a + b*x + c*x**(5/4)
    def model_2(x, a, b):    return a + b*x
    params_1, params_covariance_1 = curve_fit(model_1, s_half[mask], Boozer_G[mask])
    params_2, params_covariance_2 = curve_fit(model_2, s_half[mask], Boozer_G[mask])
    print(f' Modified  G: G_0 = {params_1[0]}, G_2 = {params_1[1]}, G_5/2 = {params_1[2]}')
    print(f' Near-Axis G: G_0 = {params_2[0]}, G_2 = {params_2[1]}')
    plt.plot(np.sqrt(s_half), Boozer_G, 'o-', label=r'Boozer $G(r)$', linewidth=4.0)
    plt.plot(np.sqrt(s_fine), model_2(s_fine, *params_2), 'g--', label=r'Near-Axis Model $G=G_0 + r^2 G_2$', linewidth=3.5)
    plt.plot(np.sqrt(s_fine), model_1(s_fine, *params_1), 'r--', label=r'Modified Model  $G=G_0 + r^2 G_2 + r^{5/2} G_{5/2}$', linewidth=3.5)
    plt.xlim([0,np.sqrt(max_s_for_fit)])
    if Boozer_G[mask][-1] > 0: plt.ylim([0.999*np.min(Boozer_G[mask]),1.001*np.max(Boozer_G[mask])])
    else: plt.ylim([1.001*np.min(Boozer_G[mask]),0.999*np.max(Boozer_G[mask])])
    plt.legend(fontsize=14)
    plt.xlabel(r'$r=\sqrt{s}$', fontsize=14)
    plt.ylabel(r'$G$', fontsize=14)
    plt.tight_layout()
    if show: plt.show()
    if savefig: plt.savefig(f'NAE_Boozer_G_fit_{name}.png',dpi=300)
    plt.close()
    ####################################################################################################
    print('Computing normal vector')
    stel = Qsc(rc=rc, rs=rs, zc=zc, zs=zs, nfp=nfp, nphi=21)
    nNormal = stel.iotaN - stel.iota
    ####################################################################################################
    print('Performing fit')
    numRows=3
    numCols=max_n_to_plot*2+1
    fig=plt.figure(figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
    for jmn in range(len(ixm)):
        m = ixm[jmn]
        n = ixn[jmn] / nfp
        if m>2:
            continue
        doplot = (np.abs(n) <= max_n_to_plot)
        row = m
        col = n+max_n_to_plot
        if doplot:
            plt.subplot(int(numRows),int(numCols),int(row*numCols + col + 1))
            plt.plot(np.sqrt(s_half), bmnc[:,jmn],'.-')
            # plt.xlabel(r'$\sqrt{s}$')
            plt.title('bmnc(m='+str(m)+' n='+str(n)+')')
        if m==0:
            # For m=0, fit a polynomial in s (not sqrt(s)) that does not need to go through the origin.
            degree = 4
            p = np.polyfit(s_half[mask], bmnc[mask,jmn], degree)
            B0 += p[-1] * np.cos(n*nfp*phi)
            B20 += p[-2] * np.cos(n*nfp*phi)
            if doplot:
                plt.plot(np.sqrt(s_fine), np.polyval(p, s_fine),'r')
        if m==1:
            # For m=1, fit a polynomial in sqrt(s) to an odd function
            x1 = np.sqrt(s_half[mask])
            y1 = bmnc[mask,jmn]
            x2 = np.concatenate((-x1,x1))
            y2 = np.concatenate((-y1,y1))
            degree = 5
            p = np.polyfit(x2,y2, degree)
            B1c += p[-2] * (np.sin(n*nfp*phi) * np.sin(nNormal*phi) + np.cos(n*nfp*phi) * np.cos(nNormal*phi))
            B1s += p[-2] * (np.sin(n*nfp*phi) * np.cos(nNormal*phi) - np.cos(n*nfp*phi) * np.sin(nNormal*phi))
            if doplot:
                plt.plot(sqrts_fine, np.polyval(p, sqrts_fine),'r')
        if m==2:
            # For m=2, fit a polynomial in s (not sqrt(s)) that does need to go through the origin.
            x1 = s_half[mask]
            y1 = bmnc[mask,jmn]
            degree = 4
            p = np.polyfit(x1,y1, degree)
            B2c += p[-2] * (np.sin(n*nfp*phi) * np.sin(nNormal*phi) + np.cos(n*nfp*phi) * np.cos(nNormal*phi))
            B2s += p[-2] * (np.sin(n*nfp*phi) * np.cos(nNormal*phi) - np.cos(n*nfp*phi) * np.sin(nNormal*phi))
            if doplot:
                plt.plot(np.sqrt(s_fine), np.polyval(p, s_fine),'r')
    if savefig:
        plt.savefig(f'NAE_fit_{name}.png',dpi=300)
    # Convert expansion in sqrt(s) to an expansion in r
    BBar = np.mean(B0)
    sqrt_s_over_r = np.sqrt(np.pi * BBar / Psi_a)
    B1s *= sqrt_s_over_r
    B1c *= -sqrt_s_over_r
    B20 *= sqrt_s_over_r*sqrt_s_over_r
    B2c *= sqrt_s_over_r*sqrt_s_over_r
    B2s *= sqrt_s_over_r*sqrt_s_over_r
    eta_bar = np.mean(B1c) / BBar
    ####################################################################################################
    print('Performing final plot')
    numRows=3
    numCols=2
    fig=plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(int(numRows),int(numCols),1);plt.plot(phi, B0, label='B0');plt.xlabel(r'$\phi$');plt.ylabel(r'$B_{0}$')
    plt.subplot(int(numRows),int(numCols),2);plt.plot(phi, B1c, label='B1c');plt.xlabel(r'$\phi$');plt.ylabel(r'$B_{1c}$')
    plt.subplot(int(numRows),int(numCols),3);plt.plot(phi, B1s, label='B1s');plt.xlabel(r'$\phi$');plt.ylabel(r'$B_{1s}$')
    plt.subplot(int(numRows),int(numCols),4);plt.plot(phi, B20, label='B20');plt.xlabel(r'$\phi$');plt.ylabel(r'$B_{20}$')
    plt.subplot(int(numRows),int(numCols),5);plt.plot(phi, B2c, label='B2c');plt.xlabel(r'$\phi$');plt.ylabel(r'$B_{2c}$')
    plt.subplot(int(numRows),int(numCols),6);plt.plot(phi, B2s, label='B2s');plt.xlabel(r'$\phi$');plt.ylabel(r'$B_{2s}$')
    if savefig: plt.savefig(f'NAE_params_{name}.png',dpi=300)
    if show: plt.show()
    
    ## THIS I2 IS INCORRECT    
    mu0 = 4 * np.pi * 1e-7
    I2=0#params_2[0]*sqrt_s_over_r*sqrt_s_over_r/2/np.pi
    # MAYBE ?? vmec.wout.ctor*vmec.wout.ac[1]/sqrt_s_over_r/sqrt_s_over_r/2/np.pi*mu0
    
    print(f'B0={BBar}, eta_bar={eta_bar}, B2c={np.mean(B2c)}, iotaVMEC={iotaVMECt}, iotaNAE={Qsc(rc=rc, rs=rs, zc=zc, zs=zs, nfp=nfp, etabar=eta_bar, nphi=51, I2=I2).iota}')    

    return [B0,B1c,B1s,B20,B2c,B2s,iotaVMECt]

if __name__ == "__main__":
    # from_boozxform(vmec_file=None)
    if len(sys.argv) == 2:
        from_boozxform(vmec_file=sys.argv[1], show=False, savefig=True)
    else:
        print("Usage: fit_boozxform_to_nearaxis.py vmec_wout_file")