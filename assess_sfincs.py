#!/usr/bin/env python3
import os
import shutil
import subprocess
home_directory = os.path.expanduser("~")

QA_or_QH = "nfp2_QA_ncoils4_stage123"
mpi_command = "~/miniforge3/bin/mpiexec.hydra"
vmec_wout_file = "wout_maxmode4.nc"
number_of_cores = 1
Nradius = 5
beta = 2.5
ne0 = 3  * (beta/100/0.05)**(1/3)
Te0 = 15 * (beta/100/0.05)**(2/3)
finite_beta_folder = f'{home_directory}/local/single_stage_optimization_finite_beta/src/sfincs_scripts'
files_to_copy = ['input.namelist', 'job.sfincsScan', 'profiles']

sfincs_result_folder = 'sfincs'
OUT_DIR_APPENDIX=f"optimization_finitebeta_{QA_or_QH}"
this_path = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(this_path,OUT_DIR_APPENDIX)
OUT_DIR = os.path.join(OUT_DIR,sfincs_result_folder)
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)

def copy_files(source_folder, destination_folder, filenames):
    for filename in filenames:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        try:
            shutil.copy2(source_path, destination_path)
            # print(f"Successfully copied {filename} from {source_folder} to {destination_folder}")
        except FileNotFoundError: print(f"File {filename} not found in {source_folder}")
        except Exception as e: print(f"Error copying {filename} from {source_folder} to {destination_folder}: {e}")

def change_file_content(file_path, replace_dict):
    backup_path = file_path + ".bak"
    with open(file_path, 'r') as original, open(backup_path, 'w') as backup:
        backup.writelines(original.readlines())
    with open(backup_path, 'r') as backup, open(file_path, 'w') as original:
        for line in backup:
            for old_value, new_value in replace_dict.items():
                if isinstance(new_value, (int, float)):
                    line = line.replace(str(old_value), f"{new_value:.2f}")
                else:
                    line = line.replace(str(old_value), str(new_value))
            original.write(line)
    os.remove(backup_path)

# Copy necessary files to the output directory
copy_files(finite_beta_folder, OUT_DIR, files_to_copy)

# Change equilibrium file and profiles file content
equilibrium_nr_replace_dict = {'wout_final.nc': os.path.join(OUT_DIR, "..", vmec_wout_file), "!ss Nradius = 5": f"!ss Nradius = {Nradius}"}
profiles_replace_dict = {2.38: ne0, -2.38: -ne0, 9.45: Te0, -9.45: -Te0}
job_replace_dict = {'mpiexec -n 6': f'{mpi_command} -n {number_of_cores}'}

change_file_content(os.path.join(OUT_DIR, "input.namelist"), equilibrium_nr_replace_dict)
change_file_content(os.path.join(OUT_DIR, "profiles"), profiles_replace_dict)
change_file_content(os.path.join(OUT_DIR, "job.sfincsScan"), job_replace_dict)

# Run sfincsScan, sfincsScanPlot_4, and convertSfincsToVmecCurrentProfile
for script_name in ["sfincsScan", "sfincsScanPlot_4", "convertSfincsToVmecCurrentProfile"]:
    script_path = os.path.join(finite_beta_folder, script_name)
    command = ["python", script_path]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e: print(f"Error running {script_name}: {e}")

# run sfuncsScan for convergence and plot - replace in the input.namelist file !ss scanType = 4 to !ss scanType = 1
change_file_content(os.path.join(OUT_DIR, "input.namelist"), {'!ss scanType = 4': '!ss scanType = 1'})
try: subprocess.run(["python", os.path.join(finite_beta_folder, "sfincsScan")], check=True)
except subprocess.CalledProcessError as e: print(f"Error running sfincsScan: {e}")
subprocess.run(["python", os.path.join(finite_beta_folder, "sfincsScanPlot_1")], check=True)