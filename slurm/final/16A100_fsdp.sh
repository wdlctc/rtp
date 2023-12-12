#!/bin/bash

#SBATCH --job-name=8A100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:8
#SBATCH -C gpupod
#SBATCH --mem=2000000M
#SBATCH --partition=gpu
#SBATCH -A bii_dsc_community
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --error="slurm/final/8A100.err"
#SBATCH --output="slurm/final/8A100.output"

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
# multi_benchmark.py
# multi_dp_benchmark.py
multi_fsdp_benchmark.py
# multi_rtp_benchmark.py
)

CONFIGS=(
Llama-2-6b-for-llm
Llama-2-7b-for-llm
Llama-2-10b-for-llm
Llama-2-11b-for-llm
Llama-2-12b-for-llm
Llama-2-13b-for-llm
Llama-2-14b-for-llm
)


for config in "${CONFIGS[@]}"; do
    for script in "${SCRIPTS[@]}"; do
        for i in {1..1}; do
            srun --export=ALL /scratch/fad3ew/rtp/.venv/bin/python \
            benchmarks/$script \
            --use_synthetic_data \
            --batch_size $i \
            --max_batch 10 \
            --model_config=$config
        done
    done
done