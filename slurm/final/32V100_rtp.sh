#!/bin/bash

#SBATCH --job-name=32V100_rtp
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=384000M
#SBATCH --cpus-per-task=8
#SBATCH --partition=bii-gpu
#SBATCH -A bii_dsc_community
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error="slurm/final/32V100_rtp.err"
#SBATCH --output="slurm/final/32V100_rtp.output"

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
# multi_fsdp_benchmark.py
multi_rtp_benchmark.py
)

CONFIGS=(
Llama-2-10b-for-llm
Llama-2-11b-for-llm
Llama-2-12b-for-llm
Llama-2-13b-for-llm
Llama-2-14b-for-llm
Llama-2-15b-for-llm
Llama-2-16b-for-llm
Llama-2-17b-for-llm
Llama-2-18b-for-llm
Llama-2-19b-for-llm
Llama-2-20b-for-llm
)


for config in "${CONFIGS[@]}"; do
    for script in "${SCRIPTS[@]}"; do
        for i in {1..1}; do
            srun --export=ALL /scratch/fad3ew/rtp/.venv/bin/python \
            benchmarks/$script \
            --use_synthetic_data \
            --batch_size $i \
            --max_batch 2 \
            --full_fp16 \
            --model_config=$config
        done
    done
done
