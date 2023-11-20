#!/bin/bash

#SBATCH --job-name=8n32v100_throughput
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=384000M
#SBATCH --cpus-per-task=1
#SBATCH --partition=bii-gpu
#SBATCH -A bii_dsc_community
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error="slurm/throughput/8n32v100_throughput.err"
#SBATCH --output="slurm/throughput/8n32v100_throughput.output"

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
multi_tp_benchmark.py
multi_rtp_benchmark.py
multi_rtp_benchmark_inplace.py
)

CONFIGS=(
gpt2-large
)


for config in "${CONFIGS[@]}"; do
    for script in "${SCRIPTS[@]}"; do
        for i in {1..16}; do
            srun --export=ALL /scratch/fad3ew/rtp/.venv/bin/python \
            benchmarks/$script \
            --use_synthetic_data \
            --model_config=$config \
            --batch_size $i
        done
    done
done