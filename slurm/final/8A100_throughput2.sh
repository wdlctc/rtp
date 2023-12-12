#!/bin/bash

#SBATCH --job-name=8A100_lc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:8
#SBATCH -C gpupod
#SBATCH --mem=2000000M
#SBATCH --partition=gpu
#SBATCH -A bii_dsc_community
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error="slurm/final/8A100_lc.err"
#SBATCH --output="slurm/final/8A100_lc.output"


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

CONFIGS=(
gpt2-large
)

BATCHS=(
2
4
6
8
10
12
14
16
18
20
22
24
26
28
30
)

for config in "${CONFIGS[@]}"; do
    for script in "${SCRIPTS[@]}"; do
        for i in "${BATCHS[@]}"; do
            srun --export=ALL /scratch/fad3ew/rtp/.venv/bin/python \
            benchmarks/$script \
            --use_synthetic_data \
            --model_config=$config \
            --max_batch 10\
            --full_fp16 \
            --batch_size $i
        done
    done
done