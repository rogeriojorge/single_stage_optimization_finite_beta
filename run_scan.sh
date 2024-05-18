#!/bin/bash

# User-defined variables
type=${1:-1}
echo "The value of type is $type"
output_folder="scan_output_$type"
RUN_PY_PATH="/Users/rogeriojorge/local/single_stage_optimization_finite_beta/coil_optimization_scan.py"
run_configurations=("2:2" "3:5" "4:4" "5:2" "6:0" "7:0")
create_output_txt_files=false #true

# Counter for the total number of Python runs
total_runs=0

# Function to run the script with a specified number of times and argument
run_script() {
  local ncoils=$1
  local times=$2

  for ((i = 1; i <= times; i++)); do
    if [ "$create_output_txt_files" = true ]; then
      # Redirect both stdout and stderr to a text file in the output folder
      python3 "$RUN_PY_PATH" --ncoils "$ncoils" --type "$type" > "$output_folder/output_${ncoils}_${i}.txt" 2>&1 &
    else
      # Run the script without redirecting output to files
      python3 "$RUN_PY_PATH" --ncoils "$ncoils" --type "$type" > /dev/null 2>&1 &
    fi
    # Store the process ID for each background process
    pids+=($!)
    # Increment the total runs counter
    ((total_runs++))
  done

  echo "Launched Python script with $ncoils coils $times times."
}

# Array to store process IDs
pids=()

# Create the output folder if it doesn't exist (only if creating files)
if [ "$create_output_txt_files" = true ]; then
  mkdir -p "$output_folder"
fi

# Run the script for each specified configuration
for config in "${run_configurations[@]}"; do
  IFS=":" read -r ncoils times <<< "$config"
  run_script "$ncoils" "$times"
done

echo "Total Python runs launched: $total_runs"

# Function to terminate background processes
terminate_background_processes() {
  echo "Terminating background processes..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2> /dev/null
  done
  wait "${pids[@]}"  # Wait for all background processes to terminate
  echo "All background processes terminated."
}

# Terminate all background processes if the script is interrupted
trap terminate_background_processes EXIT

# Explicitly print the message and wait for Enter key press
echo "Press Enter at any time to stop."
read -n 1 -s -r

echo "Exiting..."
echo "Script finished."
