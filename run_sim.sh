#!/bin/bash
# Revised script that reclaims free GPUs when jobs finish (even if a job crashes)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"
exec 1>>"$LOG_DIR/bash_output.log" \
     2>>"$LOG_DIR/bash_errors.log"

cleanup() {
  echo "⏹️ Cleanup at $(date)" >>"$LOG_DIR/bash_errors.log"
  for pid in "${!job_gpu_map[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
  reclaim_gpus     # upload logs for any remaining jobs
  exit 1
}
trap 'cleanup' ERR SIGTERM SIGINT SIGHUP
trap 'reclaim_gpus; echo "Exited normally."' EXIT

N_SIMS=2                   # Number of simulation sets (sim_nr 0,1,2,...)
conditions=(6)       # Array of conditions to run for each simulation
start_nr=0                 # Simulation numbering starts here
gpu="0,1,2,3,4,5,6,7"       # List of available GPUs (total = 8)

# Activate virtual environment and navigate to working directory
source ../agents/bin/activate
cd reverie/backend_server/

# Split the GPU list into an array
IFS=',' read -ra available_gpus <<< "$gpu"
num_gpus=${#available_gpus[@]}

# Initialize free GPUs pool with all available GPUs.
free_gpus=("${available_gpus[@]}")

# Declare an associative array to map job PIDs to their assigned GPU.
declare -A job_gpu_map
declare -A job_info_map

# A helper function to check running jobs and free GPUs whose jobs have ended.
reclaim_gpus() {
    for pid in "${!job_gpu_map[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            gpu_id=${job_gpu_map[$pid]}
            sim_gc=${job_info_map[$pid]}
            sim=${sim_gc%%_*}
            gc=${sim_gc##*_}

            # put GPU back (no dupes)
            if [[ ! " ${free_gpus[*]} " =~ " ${gpu_id} " ]]; then
                free_gpus+=("$gpu_id")
            fi

            echo "Job (PID=$pid) for sim=${sim}, condition=${gc} finished; GPU $gpu_id freed."

            # ─── upload logs for this sim ───────────────
            out="$LOG_DIR/output_gc_${gc}_sim_${sim}.out.gz"
            err="$LOG_DIR/output_gc_${gc}_sim_${sim}.err.gz"

            echo "📤 Uploading logs for sim=${sim},gc=${gc}…"
            aws s3 cp "$out" \
                s3://${R2_BUCKET}/logs/ \
                --endpoint-url "${AWS_ENDPOINT_URL}" \
                --only-show-errors \
             || echo "⚠️ failed to upload $out"

            aws s3 cp "$err" \
                s3://${R2_BUCKET}/logs/ \
                --endpoint-url "${AWS_ENDPOINT_URL}" \
                --only-show-errors \
             || echo "⚠️ failed to upload $err"

            # ─── upload any new results ───────────────
            # (sync is cheap—only sends new files)
            echo "📤 Syncing sim_results_conflicts…"
            aws s3 sync "../../sim_results_conflicts" \
                s3://${R2_BUCKET}/sim_results_conflicts \
                --endpoint-url "${AWS_ENDPOINT_URL}" \
                --only-show-errors \
             || echo "⚠️ failed to sync sim_results_conflicts"

            # cleanup
            unset job_gpu_map[$pid]
            unset job_info_map[$pid]
        fi
    done
}


# Function to wait for at least one GPU to be free.
wait_for_free_gpu() {
    while [ ${#free_gpus[@]} -eq 0 ]; do
        # Wait for any one job to finish.
        wait -n
        sleep 0.5
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
        
        CUDA_VISIBLE_DEVICES="$assigned_gpu" \
stdbuf -oL -eL python3 reverie.py \
        --sim_nr "$sim" \
        --gc "$gc" \
        --model_name "Mistral-Small-24B-Instruct-2501-6.5bpw-h8-exl2" \
        --model_dir "$HOME/models/" \
        > >(gzip -c > "$LOG_DIR/output_gc_${gc}_sim_${sim}.out.gz") \
      2> >(gzip -c > "$LOG_DIR/output_gc_${gc}_sim_${sim}.err.gz") &


        job_pid=$!
        # Record the job's assigned GPU.
        job_gpu_map[$job_pid]="$assigned_gpu"
        job_info_map[$job_pid]="${sim}_${gc}"

        
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
