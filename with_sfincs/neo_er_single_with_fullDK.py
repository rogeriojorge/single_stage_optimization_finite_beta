#!/usr/bin/env python3
import os
import sys
import argparse
from simsopt.mhd import Vmec
import numpy as np
#import pandas as pd
import shutil
import stat
# needed for the job at hand
import subprocess
#from .input_file import sfincs_input_file_monoenergetic
#from .utils import write_input_file, get_transport_matrix, get_FSABjHat
import mpi4py# as MPI2
#import plotly.graph_objects as go
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
#from simsopt.util import MpiPartition, proc0_print
#from neo_er import neoclassical_objective_function
from simsopt._core.optimizable import make_optimizable
import time
from simsopt.util import MpiPartition, proc0_print
#import h5py 



#This function does sfincs scan of the fluxes eihter with Er or with radial position
def electron_root_scan(
    indir: str, 
    workdir: str,
    Z_I: int = 2, 
    m_I: float = 5.445e-4,
    T_I: float = 1.0,
    T_e: float = 1.0,
    n_I: float = 1.0,
    n_e: float = 1.0,
    Er: float= 0.0,
    Ntheta: int = 25,
    Nzeta: int = 31,
    Nxi: int = 31,
    Nx: int = 8,
    Ln: float = -2.0,
    Lt: float = -2.0,
    s_coordinate: float = 0.375,
    flux11_I: float = 1.0,
    flux13_I: float = 1.0,
    flux31_I: float = 1.0, 
    flux33_I: float = 1.0,
    flux11_e: float = 1.0,
    flux13_e: float = 1.0,
    flux31_e: float = 1.0,
    flux33_e: float = 1.0,
    R_bar: float = 1.0,
    B_bar: float = 1.0,
    nuprime: int = 0,
    mode: int = 0,
    coll_operator: int=1,
    Er_Energy_drift: str='false',
    Er_Pitch_drift: str='false',
    DKES_drift: str='true',
    B_drift: int=0,
    mpi: MpiPartition=None  
    ):
    
    sfincs_exe='/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs'

    #Run Ions and electrons to get flux scan
    # if(mpi.rank_leaders!=-1):
    workdir_I=os.path.join(workdir,indir.split('/')[-1].split('.')[0])+'IonScan'
    workdir_e=os.path.join(workdir,indir.split('/')[-1].split('.')[0])+'ElectronScan'

    #try:
    #	os.mkdir(workdir_I)
#    try:
#	os.makedirs(workdir_I,exist_ok=True)
#	print("Directory Ion created successfully")
#    except OSError as error:
#	print("Directory Ion can not be created")

#    try:
#	os.makedirs(workdir_e,exist_ok=True)
#	print("Directory electron created successfully")
#    except OSError as error:
#	print("Directory electron can not be created")
    #except Exception as e:
    #    print(e)
    #    print('WTF')
    
    os.makedirs(workdir_I,exist_ok=True)
    os.makedirs(workdir_e,exist_ok=True)
    #print('Passed')
    copyComplete(sfincs_exe,os.path.join(workdir_I,'sfincs'))
    copyComplete(sfincs_exe,os.path.join(workdir_e,'sfincs'))
    
    print('Running Ions')
    run_SFINCS_NonMonoenergetic(indir,workdir_I,1,m_I,T_I,n_I,Er,Ntheta,Nzeta,Nxi,Nx,s_coordinate,                            R_bar,B_bar,nuprime,mode,coll_operator,Er_Energy_drift,Er_Pitch_drift,DKES_drift,B_drift,mpi=mpi)
    Lij_I=np.zeros((3,3))
    Lij_I = get_transport_matrix(workdir_I)
    Gamma_I=np.zeros((3,1))    
    Gamma_I=sfincs_fluxes(Lij_I,Er,Ln,Lt,0,flux11_I,flux13_I,flux31_I,flux33_I)
#    Gamma_I_pos=np.zeros((3,1))    
#    Gamma_I_pos=sfincs_fluxes(Lij_I,-Er,Ln=-2.0,Lt=-2.0)
    print('running_electrons')
    run_SFINCS_NonMonoenergetic(indir, workdir_e,-1,5.445e-4,T_e,n_e,Er,Ntheta,Nzeta,Nxi,Nx,s_coordinate,                            R_bar,B_bar,nuprime,mode,coll_operator,Er_Energy_drift,Er_Pitch_drift,DKES_drift,B_drift,mpi=mpi)
    Lij_e=np.zeros((3,3))
    Lij_e=get_transport_matrix(workdir_e)
    Gamma_e=np.zeros((3,1)) 
#    Gamma_e_pos=np.zeros((3,1)) 
    #-Er for accounting for Za on drive term   
    Gamma_e=sfincs_fluxes(Lij_e,-Er,Ln,Lt,0,flux11_e,flux13_e,flux31_e,flux33_e)
#    Gamma_e_pos=sfincs_fluxes(Lij_e,Er,Ln=-2.0,Lt=-2.0)

    Gammas=np.zeros((1,6)) 
#    Gammas_pos=np.zeros((1,6))
    Gammas = [Gamma_I[0], Gamma_e[0],Gamma_I[1], Gamma_e[1], Gamma_I[2], Gamma_e[2]]
#    Gammas_pos = [Gamma_I_pos[0], Gamma_e_pos[0],Gamma_I_pos[1], Gamma_e_pos[1], Gamma_I_pos[2], Gamma_e_pos[2]]
    
    
    return Gammas,Lij_I,Lij_e


#Cost function of epsiloon effective with sfincs (development ongoing, do not use)
def ripple_objective_function(
    indir: str, 
    workdir: str,
    Z_I: int = 1, 
    m_I: float = 5.445e-4,
    T_I: float = 1.0,
    T_e: float = 1.0,
    n_I: float = 1.0,
    n_e: float = 1.0,
    Er: float= 0.0,
    Ntheta: int = 25,
    Nzeta: int = 31,
    Nxi: int = 31,
    Nx: int = 8,
    Ln: float = -2.0,
    Lt: float = -2.0,
    s_coordinate: float = 0.375,
    R_bar: float = 1.0,
    B_bar: float = 1.0,
    nuprime: int = 0,
    mode: int = 0,
    coll_operator: int=1,
    Er_Energy_drift: str='false',
    Er_Pitch_drift: str='false',
    DKES_drift: str='true',
    B_drift: int=0,
    mpi: MpiPartition=None  
    ):
    
    sfincs_exe='/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs'
    #indir = v.output_file
    #return_id = run_SFINCS_NonMonoenergetic(indir, workdir)
    #Run ions+electrons
   # if(mpi.rank_leaders!=-1):
    workdir_e=os.path.join(workdir,indir.split('/')[-1].split('.')[0])+'Ripple'
    #try:
    #	os.mkdir(workdir_I)
#    try:
#	os.makedirs(workdir_I,exist_ok=True)
#	print("Directory Ion created successfully")
#    except OSError as error:
#	print("Directory Ion can not be created")

#    try:
#	os.makedirs(workdir_e,exist_ok=True)
#	print("Directory electron created successfully")
#    except OSError as error:
#	print("Directory electron can not be created")
    #except Exception as e:
    #    print(e)

    os.makedirs(workdir_e,exist_ok=True)
    copyComplete(sfincs_exe,os.path.join(workdir_e,'sfincs'))
    print('running_electrons')
    Er=0.00001
    run_SFINCS_NonMonoenergetic(indir, workdir_e,-1,5.445e-4,T_e,n_e,Er,Ntheta,Nzeta,Nxi,Nx,s_coordinate,                            R_bar,B_bar,nuprime,mode,coll_operator,Er_Energy_drift,Er_Pitch_drift,DKES_drift,B_drift,mpi=mpi)
    Lij_e=np.zeros((3,3))
    Lij_e=get_transport_matrix(workdir_e)

    J = Lij_e[0,0]*np.sqrt(5.446e-4)     

    return J


#Calculates electron root cost function mode=0 default cost function == L11_I/L11_e, 
#other modes target the fluxes 
def neoclassical_objective_function(
    indir: str, 
    workdir: str,
    Z_I: int = 1, 
    m_I: float = 5.445e-4,
    T_I: float = 1.0,
    T_e: float = 1.0,
    n_I: float = 1.0,
    n_e: float = 1.0,
    Er: float= 0.0,
    Ntheta: int = 25,
    Nzeta: int = 31,
    Nxi: int = 31,
    Nx: int = 8,
    Ln: float = -2.0,
    Lt: float = -2.0,
    s_coordinate: float = 0.375,
    R_bar: float = 1.0,
    B_bar: float = 1.0,
    nuprime: int = 0,
    mode: int = 0,
    coll_operator: int=1,
    Er_Energy_drift: str='false',
    Er_Pitch_drift: str='false',
    DKES_drift: str='true',
    B_drift: int=0,
    mpi: MpiPartition=None  
    ):
    
    sfincs_exe='/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs'
    #indir = v.output_file
    #return_id = run_SFINCS_NonMonoenergetic(indir, workdir)
    #Run ions+electrons
   # if(mpi.rank_leaders!=-1):
    workdir_I=os.path.join(workdir,indir.split('/')[-1].split('.')[0])+'Ion'
    workdir_e=os.path.join(workdir,indir.split('/')[-1].split('.')[0])+'Electron'
    if(mode==1):
        workdir_IL=os.path.join(workdir,indir.split('/')[-1].split('.')[0])+'IonLarge'
        workdir_eL=os.path.join(workdir,indir.split('/')[-1].split('.')[0])+'ElectronLarge'
    #try:
    #	os.mkdir(workdir_I)
#    try:
#	os.makedirs(workdir_I,exist_ok=True)
#	print("Directory Ion created successfully")
#    except OSError as error:
#	print("Directory Ion can not be created")

#    try:
#	os.makedirs(workdir_e,exist_ok=True)
#	print("Directory electron created successfully")
#    except OSError as error:
#	print("Directory electron can not be created")
    #except Exception as e:
    #    print(e)

    
    os.makedirs(workdir_I,exist_ok=True)
    os.makedirs(workdir_e,exist_ok=True)
    copyComplete(sfincs_exe,os.path.join(workdir_I,'sfincs'))
    copyComplete(sfincs_exe,os.path.join(workdir_e,'sfincs'))
    if(mode==1):
        os.makedirs(workdir_IL,exist_ok=True)
        os.makedirs(workdir_eL,exist_ok=True)    
        copyComplete(sfincs_exe,os.path.join(workdir_IL,'sfincs'))
        copyComplete(sfincs_exe,os.path.join(workdir_eL,'sfincs'))

    print('Running Ions')
    run_SFINCS_NonMonoenergetic(indir,workdir_I,1,m_I,T_I,n_I,Er,Ntheta,Nzeta,Nxi,Nx,s_coordinate,                            R_bar,B_bar,nuprime,mode,coll_operator,Er_Energy_drift,Er_Pitch_drift,DKES_drift,B_drift,mpi=mpi)
    Lij_I=np.zeros((3,3))
    Lij_I = get_transport_matrix(workdir_I)
    if(mode==1):
        Gamma_I=np.zeros((3,1))    
        Gamma_I=sfincs_fluxes(Lij_I,Er,Ln,Lt)
    print('running_electrons')
    run_SFINCS_NonMonoenergetic(indir, workdir_e,-1,5.445e-4,T_e,n_e,Er,Ntheta,Nzeta,Nxi,Nx,s_coordinate,                            R_bar,B_bar,nuprime,mode,coll_operator,Er_Energy_drift,Er_Pitch_drift,DKES_drift,B_drift,mpi=mpi)
    Lij_e=np.zeros((3,3))
    Lij_e=get_transport_matrix(workdir_e)
    if(mode==1):
        Gamma_e=np.zeros((3,1))    
        Gamma_e=sfincs_fluxes(Lij_e,-Er,Ln,Lt)

    if(mode==2):
        Er=3.0
        run_SFINCS_NonMonoenergetic(indir,workdir_IL,1,m_I,T_I,n_I,Er,Ntheta,Nzeta,Nxi,Nx,s_coordinate,                            R_bar,B_bar,nuprime,mode,coll_operator,Er_Energy_drift,Er_Pitch_drift,DKES_drift,B_drift,mpi=mpi)
        Lij_IL=np.zeros((3,3))
        Lij_IL = get_transport_matrix(workdir_IL)
        Gamma_IL=np.zeros((3,1))    
        Gamma_IL=sfincs_fluxes(Lij_IL,Er,Ln,Lt)
        run_SFINCS_NonMonoenergetic(indir,workdir_eL,1,m_I,T_I,n_I,Er,Ntheta,Nzeta,Nxi,Nx,s_coordinate,                            R_bar,B_bar,nuprime,mode,coll_operator,Er_Energy_drift,Er_Pitch_drift,DKES_drift,B_drift,mpi=mpi)
        Lij_eL=np.zeros((3,3))
        Lij_eL = get_transport_matrix(workdir_eL)
        Gamma_eL=np.zeros((3,1))    
        Gamma_eL=sfincs_fluxes(Lij_eL,-Er,Ln,Lt)        

    if(mode==0):
        J = Lij_I[0,0]/Lij_e[0,0]*np.sqrt(m_I/5.446e-4)
        #J = np.sqrt(np.square(Lij_I[0,0]/Lij_e[0,0])+np.square(Lij_I[0,2])+np.square(Lij_e[0,2]))
    else:
        J = np.sqrt(m_I/5.446e-4)*Gamma_I[0]/Gamma_e[0]
        #J = np.sqrt(m_I/5.446e-4*np.square(Gamma_I[0]/Gamma_e[0])+5.446e-4/m_I*np.square(Gamma_eL[0]/Gamma_IL[0]))
        #J=Gamma_I[0]*np.sqrt(m_I)-Gamma_e[0]*np.sqrt(5.446e-4)       

    return J

#Calculates bootstrap via fullDK e+i  and compares  against J\cdotB VMEC  at one radial point
def bootstrap_objective_function(
    indir: str, 
    workdir: str,
    Z_i: int = 1,
    Z_e: int = -1,
    m_i: float = 2.0,          
    m_e: float = 5.445e-4,
    ni: float = 1.0,
    ne: float = 1.0,    
    Ti: float = 1.0,
    Te: float = 1.0,
    dnidr: float = -0.1,
    dnedr: float = -0.1,    
    dTidr: float = -0.1,
    dTedr: float = -0.1,   
    Er: float= 0.0,
    Ntheta: int = 25,
    Nzeta: int = 31,
    Nxi: int = 34,
    Nx: int = 8,
    s_coordinate: float = 0.375,
    R_bar: float = 1.0,
    B_bar: float = 1.0,
    coll_operator: int=0,
    Er_Energy_drift: str='true',
    Er_Pitch_drift: str='true',
    DKES_drift: str='false',
    B_drift: int=0,
    mpi: MpiPartition=None  
    ):
    
    sfincs_exe='/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs'
    #indir = v.output_file
    #return_id = run_SFINCS_NonMonoenergetic(indir, workdir)
    #Run ions+electrons
   # if(mpi.rank_leaders!=-1):
    workdir_bootstrap=os.path.join(workdir,indir.split('/')[-1].split('.')[0])+'Bootstrap'
    #try:
    #	os.mkdir(workdir_I)
#    try:
#	os.makedirs(workdir_I,exist_ok=True)
#	print("Directory Ion created successfully")
#    except OSError as error:
#	print("Directory Ion can not be created")

#    try:
#	os.makedirs(workdir_e,exist_ok=True)
#	print("Directory electron created successfully")
#    except OSError as error:
#	print("Directory electron can not be created")
    #except Exception as e:
    #    print(e)

    
    os.makedirs(workdir_bootstrap,exist_ok=True)
    copyComplete(sfincs_exe,os.path.join(workdir_bootstrap,'sfincs'))

    # print('Running SFINCS DK full bootstrap ')
    run_SFINCS_fullDK(indir,workdir_bootstrap,Z_i,Z_e,m_i,m_e,ni,ne,Ti,Te,dnidr,dnedr,dTidr,dTedr,
                Er,Ntheta,Nzeta,Nxi,Nx,s_coordinate,R_bar,B_bar,coll_operator,Er_Energy_drift,
                Er_Pitch_drift,DKES_drift,B_drift,mpi=mpi)
    Jbootstrap = get_Bootstrap(workdir_bootstrap)

    return Jbootstrap

#Function that runs sfincs in non-monoenergetic mode (will be called one time for ions and one time for electrons 
#in the cost function)
def run_SFINCS_NonMonoenergetic(
    indir: str, 
    workdir: str,
    Z_a: int = -1,
    m_a: float = 5.445e-4,
    T_a: float = 1.0,
    n_a: float = 1.0,
    Er: float= 0.0,
    Ntheta: int = 25,
    Nzeta: int = 51,
    Nxi: int = 64,
    Nx: int = 8,
    s_coordinate: float = 0.375,
    R_bar: float = 1.0,
    B_bar: float = 1.0,
    nuprime: int = 0,
    mode: int = 0,
    coll_operator: int =1,
    Er_Energy_drift: str='false',
    Er_Pitch_drift: str='false',
    DKES_drift: str='true',
    B_drift: int=0,
    mpi: MpiPartition=None 
):

    # print("running SFINCSparallelMonoenergeticCoefficient ...")
    # print(f"    indir = '{indir}'")
    # print(f"  workdir = '{workdir}'")

    SFINCS_INPUT_NAMELIST = "input.namelist"

    # See if vmec free boundary exists, otherwise take the fixed one

    # #TODO: replace with your filename
    # if os.path.exists(indir):
    #     print("wout found!!")
    # else:
    #     print("wout not found!!")


    in_par=sfincs_NonMonoenergetic_parameters(Z_a,m_a,T_a,n_a,R_bar,B_bar,nuprime)
    sfincs_content_body = sfincs_input_file_NonMonoenergetic(indir, Z_a, in_par[0], in_par[1],in_par[2],Er,
            Ntheta,Nzeta,Nxi,Nx,s_coordinate,coll_operator,Er_Energy_drift,Er_Pitch_drift,DKES_drift,B_drift)

    write_input_file(os.path.join(workdir, SFINCS_INPUT_NAMELIST), sfincs_content_body)
 #   bashCommand=f"srun --ntasks=1 --exclusive -c 1 --mem-per-cpu=3750 {workdir+'/sfincs'} {SFINCS_INPUT_NAMELIST}"
    bashCommand=f"{workdir+'/sfincs'} {workdir+'/input.namelist'}"
    # print(bashCommand.split())
    #os.chdir(workdir)
    filed=open(os.path.join(workdir, f'output.log'), 'w')
    #p=subprocess.Popen(bashCommand.split(),stdout=filed,stderr=subprocess.STDOUT)
    #p.wait()

    #directory=os.path.join(cwd,str(task))
    #os.chdir(workdir)
    #info = mpi4py.MPI.Info.Create()
    #info.update({"wdir": workdir})
    #info.update({"stdout": 'output.log'})
    #print("before spawn in slave:", rank)
    #new_comm=mpi.comm_groups.Spawn('./sfincs',
    #              args=['./input.namelist', '>', './output.log'],maxprocs=1,info=info)
    #print("after spawn in slave:", rank)

#    mpi.new_comm.Disconnect()
    #print("Before barrier")
    #new_comm.Barrier()
    #print("After Barrier")
    #new_comm.Disconnect()
    #print("After Disconnect")
    #new_comm.Wait()
    #info.Free()

   # os.chdir(workdir)
   # bashCommand = f"./sfincs input.namelist"
   # with open(os.path.join(workdir, f'output.log'), 'w') as file:
   #     p = subprocess.Popen(bashCommand.split(), stdout=file, stderr=subprocess.STDOUT)
   #     p.wait()
   # p = subprocess.Popen(["./sfincs", SFINCS_INPUT_NAMELIST], cwd=workdir,stdout=filed,stderr=subprocess.STDOUT)
    p = subprocess.Popen(["./sfincs", SFINCS_INPUT_NAMELIST], cwd=workdir,stdout=filed,stderr=subprocess.STDOUT)
  # p = subprocess.Popen(["srun --ntasks=1 --overlap ./sfincs", SFINCS_INPUT_NAMELIST], cwd=workdir,stdout=filed,stderr=subprocess.STDOUT)
    p.wait()
    #print(workdir)
    #bashCommand = f"{workdir+'/sfincs'} {SFINCS_INPUT_NAMELIST}"
    #with open(os.path.join(workdir, f'output-{f_wout}.log'), 'w') as file:
    #p = subprocess.Popen(bashCommand.split())#, stdout=file, stderr=subprocess.STDOUT)
    #p.wait()
   # f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
 #   hdf5_file = h5py.File(os.path.join(workdir, "sfincsOutput.h5"), "r", driver='mpio',comm=new_comm)
    #Lij=np.zeros((3,3))+1.
   # Lij=hdf5_file["transportMatrix"][()]
   #### Lij = get_transport_matrix(workdir)
    #if(mode==1):
#	Gamma=sfincs_fluxes(Lij,Ln=-2.0,Lt=-2.0,E_ind=0.0)

    #return Lij #, Gamma
 #   return new_comm


#Function that runs sfincs in non-monoenergetic mode (will be called one time for ions and one time for electrons 
#in the cost function)
def run_SFINCS_fullDK(
    indir: str, 
    workdir: str,
    Z_i: int = 1,
    Z_e: int = -1,   
    m_i: float = 2.0,
    m_e: float = 5.445e-4,    
    ni: float = 1.0,
    ne: float = 1.0,
    Ti: float = 1.0,
    Te: float = 1.0,  
    dnidr: float = -0.1,
    dnedr: float = -0.1,
    dTidr: float = -0.1,
    dTedr: float = -0.1,       
    Er: float= 0.0,
    Ntheta: int = 25,
    Nzeta: int = 31,
    Nxi: int = 34,
    Nx: int = 8,
    s_coordinate: float = 0.375,
    R_bar: float = 1.0,
    B_bar: float = 1.0,
    coll_operator: int =0,
    Er_Energy_drift: str='true',
    Er_Pitch_drift: str='true',
    DKES_drift: str='false',
    B_drift: int=0,
    mpi: MpiPartition=None 
):

    # print("running SFINCSparallelMonoenergeticCoefficient ...")
    # print(f"    indir = '{indir}'")
    # print(f"  workdir = '{workdir}'")

    SFINCS_INPUT_NAMELIST = "input.namelist"

    # See if vmec free boundary exists, otherwise take the fixed one

    # #TODO: replace with your filename
    # if os.path.exists(indir):
    #     print("wout found!!")
    # else:
    #     print("wout not found!!")

    sfincs_content_body = sfincs_input_file_fullDK(indir, Z_i, Z_e,m_i,m_e, ni, ne, Ti, Te,  dnidr, dnedr, dTidr, 
                dTedr, Er, Ntheta,Nzeta,Nxi,Nx,s_coordinate,coll_operator,Er_Energy_drift,Er_Pitch_drift,DKES_drift,B_drift)
    # print('I have been here')
    write_input_file(os.path.join(workdir, SFINCS_INPUT_NAMELIST), sfincs_content_body)
 #   bashCommand=f"srun --ntasks=1 --exclusive -c 1 --mem-per-cpu=3750 {workdir+'/sfincs'} {SFINCS_INPUT_NAMELIST}"
    bashCommand=f"{workdir+'/sfincs'} {workdir+'/input.namelist'}"
    # print(bashCommand.split())
    #os.chdir(workdir)
    filed=open(os.path.join(workdir, f'output.log'), 'w')
    
    p = subprocess.Popen(["./sfincs", SFINCS_INPUT_NAMELIST], cwd=workdir,stdout=filed,stderr=subprocess.STDOUT)
  
    p.wait()
    



#To copy a file with all permissions
def copyComplete(source, target):
    # copy content, stat-info (mode too), timestamps...
    shutil.copy2(source, target)
    # copy owner and group
    st = os.stat(source)
    os.chown(target, st.st_uid, st.st_gid)


#Write input sfincs
def sfincs_input_file_fullDK(
    wout_file_path: str,
    Z_i: int = 1,
    Z_e: int =-1,
    m_i: float = 2.0,
    m_e: float =5.445e-4,    
    ni: float = 1.0,
    ne: float = 1.0,
    Ti: float = 1.0,
    Te: float = 1.0,
    dnidr: float = -0.1,
    dnedr: float = -0.1,
    dTidr: float = -0.1,
    dTedr: float = -0.1,                
    Er: float = 0.0,
    Ntheta: int = 25,
    Nzeta: int = 31,
    Nxi: int = 34,
    Nx: int = 8,
    s_coordinate: float = 0.375,
    coll_operator: int = 0,
    Er_Energy_drift: str = 'true',
    Er_Pitch_drift: str = 'true',
    DKES_drift: str='false',
    B_drift: int =0
):
    return f"""
  ! Input file for SFINCS version 3.
  ! See the user manual for documentation of the parameters in this file.
  !----------------------------------------------------------------------

  &general
    RHSMode = 1  ! Full DK
  /

  &geometryParameters
    geometryScheme = 5 ! input from wout file
    equilibriumFile = "{wout_file_path}"

    inputRadialCoordinate = 3  ! VMEC s coordinate
    rN_wish = {s_coordinate}

    VMECRadialOption = 1
    min_Bmn_to_load = 0
  /

  &speciesParameters
    ! T_a=1, n_a, Z_a and m_a for species a.
    Zs= {Z_i} {Z_e}  !So that Zs_bar =Z_a
    mHats = {m_i} {m_e} !so that m_bar=m_a 
    THats =  {Ti} {Te} !T_bar=T_a necessary for SFINCS to work properly
    nHats =  {ni} {ne} !n_bar=n_a 
    dTHatdrHats = {dTidr} {dTedr} !T_bar=T_a necessary for SFINCS to work properly
    dnHatdrHats = {dnidr} {dnedr}!n_bar=n_a           

  /

  &physicsParameters
    ! We change Delta, alpha, nu_n, or Er because changing T_a is not worign at the moment.
    Delta = 4.443e-4 !Defines the temperature and mass, B_bar=1, R_bar=1 for VMEC inputs
    alpha = 1.0    !Defines units of Electric field, related with temperature
    nu_n= 8.833e-4    !Defnes units of species parameters
    Er = {Er}          !in fraction of given temperature in ev

    collisionOperator = {coll_operator}           !Only Lorentz operator
    includeXDotTerm = .{Er_Energy_drift}.                 !Extra E drift term
    includeElectricFieldTermInXiDot = .{Er_Pitch_drift}. !Extra E drift term
    useDKESExBDrift = .{DKES_drift}.                !Exact compressible ExB term, 
    includePhi1 = .false.                    !Phi1 only available in RHSMode=1   
    magneticDriftScheme = {B_drift}
  /

  &resolutionParameters
    Ntheta = {Ntheta}
    Nzeta = {Nzeta}
    Nxi = {Nxi}
    Nx = {Nx}
    solverTolerance = 1d-6
  /

  &otherNumericalParameters
  /

  &preconditionerOptions
  /

  &export_f
    !export_full_f = .true.
    !export_delta_f = .true.

    !export_f_theta_option = 1
    !export_f_theta = 0

    !export_f_zeta_option = 0

    !export_f_xi_option = 1
    !export_f_xi = -1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1

    !export_f_x_option = 0
  /
  """


#Write input sfincs
def sfincs_input_file_NonMonoenergetic(
    wout_file_path: str,
    Z_a: int = -1,
    Delta: float = 4.443e-4,
    nu_n: float = 8.833e-4,
    alpha: float= 2.72,
    Er: float = 0.0,
    Ntheta: int = 25,
    Nzeta: int = 31,
    Nxi: int = 31,
    Nx: int = 4,
    s_coordinate: float = 0.375,
    coll_operator: int = 1,
    Er_Energy_drift: str = 'false',
    Er_Pitch_drift: str = 'false',
    DKES_drift: str='true',
    B_drift: int =0
):
    return f"""
  ! Input file for SFINCS version 3.
  ! See the user manual for documentation of the parameters in this file.
  !----------------------------------------------------------------------

  &general
    RHSMode = 2  ! Non-Monoenergetic coefficients
  /

  &geometryParameters
    geometryScheme = 5 ! input from wout file
    equilibriumFile = "{wout_file_path}"

    inputRadialCoordinate = 3  ! VMEC s coordinate
    rN_wish = {s_coordinate}

    VMECRadialOption = 1
    min_Bmn_to_load = 0
  /

  &speciesParameters
    ! T_a=1, n_a, Z_a and m_a for species a.
    Zs= {Z_a}!So that Zs_bar =Z_a
    mHats = 1.0 !so that m_bar=m_a 
    THats = 1.0 !T_bar=T_a necessary for SFINCS to work properly
    nHats = 1.0 !n_bar=n_a    
  /

  &physicsParameters
    ! We change Delta, alpha, nu_n, or Er because changing T_a is not worign at the moment.
    Delta = {Delta}  !Defines the temperature and mass, B_bar=1, R_bar=1 for VMEC inputs
    alpha = {alpha}     !Defines units of Electric field, related with temperature
    nu_n={nu_n}     !Defnes units of species parameters
    Er = {Er}          !in fraction of given temperature in ev

    collisionOperator = {coll_operator}           !Only Lorentz operator
    includeXDotTerm = .{Er_Energy_drift}.                 !Extra E drift term
    includeElectricFieldTermInXiDot = .{Er_Pitch_drift}. !Extra E drift term
    useDKESExBDrift = .{DKES_drift}.                !Exact compressible ExB term, 
    includePhi1 = .false.                    !Phi1 only available in RHSMode=1   
    magneticDriftScheme = {B_drift}
  /

  &resolutionParameters
    Ntheta = {Ntheta}
    Nzeta = {Nzeta}
    Nxi = {Nxi}
    Nx = {Nx}
    solverTolerance = 1d-6
  /

  &otherNumericalParameters
  /

  &preconditionerOptions
  /

  &export_f
    !export_full_f = .true.
    !export_delta_f = .true.

    !export_f_theta_option = 1
    !export_f_theta = 0

    !export_f_zeta_option = 0

    !export_f_xi_option = 1
    !export_f_xi = -1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1

    !export_f_x_option = 0
  /
  """


#Function to calculate delta, alpha and nu_n
def sfincs_NonMonoenergetic_parameters(
    Z_a: int = -1,
    m_a: float = 5.445e-4,
    T_a: float = 1.0,
    n_a: float= 1.0,
    R_bar:float = 1.0,
    B_bar: float =1.0,
    nuprime: int=0
):


    N=n_a*1.e+20
    qp=1.6e-19
    ln=17.
    Z4=np.power(Z_a,4)
    qp2=np.power(qp,2)
    T=T_a*1.e+3
    epsilon02=np.square(8.854187817e-12)
    m=m_a*1.67262192e-27
    pisqrtpi=np.pi*np.sqrt(np.pi)

    Delta=np.sqrt(m*T*2.*qp)/(qp*R_bar*B_bar)
    nu_n=R_bar*N*Z4*qp2*ln/(3.*pisqrtpi*4.*epsilon02*T*T)
    alpha=1.0
    NMpar=np.zeros((1,3))
    NMpar=[Delta, nu_n, alpha] 

    return NMpar





#Function to calculate flux, heat flux and flow/bootstrap from L_ij
def sfincs_fluxes(Lij,Er, Ln=-2.0,Lt=-2.0,E_ind=0.0,fac11=10.,fac13=1.0,fac31=1.0,fac33=1.0):

    Iflux=np.zeros([3])
    A1=Ln-1.5*Lt-Er 
    A2=Lt
    A3=E_ind


    Iflux[0]=fac11*Lij[0,0]*A1+fac11*Lij[1,0]*A2+fac13*Lij[2,0]*A3
    Iflux[1]=fac11*Lij[0,1]*A1+fac11*Lij[1,1]*A2+fac13*Lij[2,1]*A3
    Iflux[2]=fac31*Lij[0,2]*A1+fac31*Lij[1,2]*A2+fac33*Lij[2,2]*A3

    return Iflux


#Function to get the transport matrix from the sfincs output file
def get_transport_matrix(workdir: str):
    import h5py

    hdf5_file = h5py.File(os.path.join(workdir, "sfincsOutput.h5"), "r")
    return hdf5_file["transportMatrix"][()]


#Get bootstrap
def get_Bootstrap(workdir: str):
    import h5py

    hdf5_file = h5py.File(os.path.join(workdir, "sfincsOutput.h5"), "r")
    JB=hdf5_file["FSABjHat"][()]
    #B_FSA=hdf5_file["FSABHat"][()]
    Jbootstrap=JB
    return Jbootstrap  # for some reason this exports two values

#Write input file in a folder
def write_input_file(filepath: str, content: str):
    with open(filepath, "w") as f:
        f.write(content)


