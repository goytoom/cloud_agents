#!/bin/bash
# Revised script that reclaims free GPUs when jobs finish (even if a job crashes)
exec 2>>"logs/bash_errors.log"
trap 'echo "Received SIGTERM at $(date)" >> logs/bash_errors.log' SIGTERM
trap 'echo "Received SIGINT at $(date)" >> logs/bash_errors.log' SIGINT
trap 'echo "Received SIGHUP at $(date)" >> logs/bash_errors.log' SIGHUP

shopt -s extglob  # Ensure extended globbing if needed

N_SIMS=2                   # Number of simulation sets (sim_nr 0,1,2,...)
conditions=(6)       # Array of conditions to run for each simulation
start_nr=0                 # Simulation numbering starts here
gpu="0,1,2,3,4,5,6,7"       # List of available GPUs (total = 8)

# Activate virtual environment and navigate to working directory
source ../agents/bin/activate
cd reverie/backend_server/
mkdir -p ../../logs

# Split the GPU list into an array
IFS=',' read -ra available_gpus <<< "$gpu"
num_gpus=${#available_gpus[@]}

# Initialize free GPUs pool with all available GPUs.
free_gpus=("${available_gpus[@]}")

# Declare an associative array to map job PIDs to their assigned GPU.
declare -A job_gpu_map

# A helper function to check running jobs and free GPUs whose jobs have ended.
reclaim_gpus() {
    for pid in "${!job_gpu_map[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            gpu_id=${job_gpu_map[$pid]}
            # only add it back if it's not already there
            if [[ ! " ${free_gpus[*]} " =~ " ${gpu_id} " ]]; then
                free_gpus+=("$gpu_id")
            fi
            echo "Job (PID=$pid) finished; GPU ${job_gpu_map[$pid]} is now free."
            unset job_gpu_map[$pid]
        fi
    done
}

# Function to wait for at least one GPU to be free.
wait_for_free_gpu() {
    while [ ${#free_gpus[@]} -eq 0 ]; do
        # Wait for any one job to finish.
        wait -n
        reclaim_gpus
    done
}

# Loop over simulation numbers and conditions.
for ((sim=start_nr; sim<start_nr+N_SIMS; sim++)); do
    for gc in "${conditions[@]}"; do
        # Wait until there is at least one free GPU.
        wait_for_free_gpu
        
        # Pop one GPU from the free pool.
        assigned_gpu=${free_gpus[0]}
        free_gpus=("${free_gpus[@]:1}")  # remove the first GPU from free_gpus
        
        echo "Starting reverie.py for sim_nr=${sim}, condition=${gc} on GPU ${assigned_gpu}"
        log_file="../../logs/output_gc_${gc}_sim_${sim}"
        
        # # Launch the job in the background.
        # CUDA_VISIBLE_DEVICES="$assigned_gpu" python3 reverie.py \
        #     --sim_nr "$sim" --gc "$gc" --model_name "Mistral-Small-24B-Instruct-2501-6.5bpw-h8-exl2" \
        #     > "${log_file}.out" 2> "${log_file}.err" &

        CUDA_VISIBLE_DEVICES="$assigned_gpu" \
stdbuf -oL -eL python3 reverie.py \
        --sim_nr "$sim" \
        --gc "$gc" \
        --model_name "Mistral-Small-24B-Instruct-2501-6.5bpw-h8-exl2" \
        --model_dir "$HOME/models/" \
        > >(gzip  -1  -c > "${log_file}.out.gz") \
        2> >(gzip  -1  -c > "${log_file}.err.gz") &


        job_pid=$!
        # Record the job's assigned GPU.
        job_gpu_map[$job_pid]="$assigned_gpu"
        
        # Optional short sleep to help stagger launches.
        sleep 0.5
    done
done

# Wait for all remaining jobs to finish and reclaim their GPUs.
while [ ${#job_gpu_map[@]} -gt 0 ]; do
    wait -n
    reclaim_gpus
done

echo "All reverie.py simulations have completed."
