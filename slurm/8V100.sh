#!/bin/bash

#SBATCH --job-name=multinode-example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100:8
#SBATCH --mem=200000M
#SBATCH -C gpupod
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu # gpupod only available to -p gpu
#SBATCH -A bii_dsc_community
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --error="8V100.err"
#SBATCH --output="8V100.output"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
source /home/fad3ew/.bashrc
source /scratch/fad3ew/venv/rfsdp/bin/activate
cd /scratch/fad3ew/rfsdp

srun /scratch/fad3ew/venv/rfsdp/bin/torchrun \
--nnodes 2 \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:29500 \
benchmarks/multi_dp_benchmark.py \
    --use_synthetic_data \

srun /scratch/fad3ew/venv/rfsdp/bin/torchrun \
--nnodes 2 \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:29500 \
benchmarks/multi_tp_benchmark.py \
    --use_synthetic_data \

srun /scratch/fad3ew/venv/rfsdp/bin/torchrun \
--nnodes 2 \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:29500 \
benchmarks/multi_fsdp_benchmark.py \
    --use_synthetic_data \

srun /scratch/fad3ew/venv/rfsdp/bin/torchrun \
--nnodes 2 \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:29500 \
benchmarks/multi_rtp_benchmark.py \
    --use_synthetic_data \

srun /scratch/fad3ew/venv/rfsdp/bin/torchrun \
--nnodes 2 \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:29500 \
benchmarks/multi_rtp_benchmark_inplace.py \
    --use_synthetic_data \