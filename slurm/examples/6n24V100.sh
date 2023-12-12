#!/bin/bash

#SBATCH --job-name=6n24v100_test
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=200000M
#SBATCH --cpus-per-task=1
#SBATCH --partition=bii-gpu
#SBATCH -A bii_dsc_community
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --error="slurm/6n24v100_test.err"
#SBATCH --output="slurm/6n24v100_test.output"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

NCCL_DEBUG=INFO

module purge
source /home/fad3ew/.bashrc
source /scratch/fad3ew/rtp/.venv/bin/activate
cd /scratch/fad3ew/rtp

SCRIPTS=(
multi_dp_benchmark.py
multi_fsdp_benchmark.py
multi_rtp_benchmark.py
multi_rtp_benchmark_inplace.py
)


for script in "${SCRIPTS[@]}"; do
    srun --export=ALL /scratch/fad3ew/rtp/.venv/bin/python \
    benchmarks/$script \
    --use_synthetic_data 
done