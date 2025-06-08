#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Run free-bdry DESC on a finite beta eq given from a VMEC .nc file
First cmd line arg: path to the VMEC wout file
Second cmd line arg: path to the MAKEGRID coils file for this equilibrium
Third cmd line arg : path to the VMEC input file (so DESC can get the pressure profile)
"""


import os
import sys
import shutil
import subprocess
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))

def make_slurm_for_freeb(wout_filepath, path_to_coilset, launch_script=False):
    slurm_str = """#!/bin/bash
#SBATCH --job-name=DESC_FB_single_stage_beta            # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH -n 16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G         # memory per cpu-core (4G is default)
#SBATCH --time=01:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when process begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=dpanici@princeton.edu
#SBATCH --constraint=a100 
#SBATCH --gres=gpu:1




"""    

    coilfolder_path = "/" +  os.path.join(*path_to_coilset.split("/")[0:-1]) # path to coilset's folder (which we will save stuff at, since the coilset names are too long to include with the other information in saving filenames)
    slurmpath = os.path.join(coilfolder_path, "job.slurm_desc_freebdry")
    slurmoutpath = os.path.join(coilfolder_path, "slurm-%j.out")
    slurm_str += f"#SBATCH --output={slurmoutpath}\n"
    slurm_str += "source setup_DESC_env\n"
    slurm_str += f"python run_free_boundary_given_wout_and_coilset.py {wout_filepath} {path_to_coilset}"

    free_bdry_script_path = os.path.join(coilfolder_path, "run_free_boundary_given_wout_and_coilset.py")
    # place freebdry script into the path
    shutil.copyfile("./run_free_boundary_given_wout_and_coilset.py", free_bdry_script_path)
    # now copy the slurm script

    with open(slurmpath,"w+") as f:
        f.write(slurm_str)
    if launch_script:
        subprocess.run(["sbatch", slurmpath])




# we always know the wout file is "wout_final.nc" and is inside the folder starting with "optimization"
# so let's here just look for the path to the DESC coilset, then from there, find the woutfile path to pass to the above free bdry script
for dirpath, dirnames, filenames in os.walk(os.getcwd()):
    if dirpath.find("optimal_coil") != -1 and filenames and dirpath.find("ipynb") == -1: # we reached one of the folders below optimal_coils that contains coil files (and we are not in a jupyter checkpoint folder)
        if dirpath.find("ipynb"): 
            pass # skip jupyter stuff
        found_a_DESC_coilset = False
        coilset_filename = ""
        for file in filenames:
            if file.find("coilset_desc") !=1 and file.find(".h5") != -1 and file.find("nfp") != -1: # is a coilset_DESC...h5 file
                found_a_DESC_coilset = True
                coilset_filename = file
                
        if found_a_DESC_coilset:
            path_to_coilset = os.path.join(dirpath, coilset_filename)
            # now get path to the folder starting with optimization... where the wout_final.nc lives
            path_split = dirpath.split("/")
            for i in range(len(path_split)):
                if path_split[i].find("optimization_") != -1 and path_split[i].find("single_stage_optimization_finite_beta") == -1: # last part is to jump past the topmost folder
                    break
            wout_filepath = "/" + os.path.join(*path_split[0:i+1], "wout_final.nc")
            print(f"{wout_filepath=}")
            print(f"{path_to_coilset=}")
            make_slurm_for_freeb(wout_filepath, path_to_coilset,True)



