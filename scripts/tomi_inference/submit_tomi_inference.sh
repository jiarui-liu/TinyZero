#!/bin/bash
#SBATCH --job-name=code
# SBATCH --partition=general
# SBATCH --partition=debug
#SBATCH --partition=preempt
# SBATCH --partition=r3lit
#SBATCH --output=log.out
#SBATCH --error=log.err
#SBATCH --nodes=1
#SBATCH --ntasks=1  
# SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8 
# SBATCH --gres=gpu:L40:1
# SBATCH --gres=gpu:H100:1
#SBATCH --gres=gpu:A100_80GB:2
# SBATCH --gres=gpu:A100_80GB:1
# SBATCH --gres=gpu:6000Ada:10
#SBATCH --time=2-00:00:00
# SBATCH --nodelist=babel-8-7
#SBATCH --mem=64G
#SBATCH --mail-user=jiaruiliu999@gmail.com   # Your email address
#SBATCH --mail-type=BEGIN                    # Send email when the job starts
#SBATCH --mail-type=END                      # Send email when the job ends
#SBATCH --mail-type=FAIL                     # Send email if the job fails

source ~/.bashrc
conda activate tinyzero

export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
export ACTOR_MODEL=/compute/babel-2-29/jiaruil5/social_reasoning/tinyzero/tomi_distilled_1.5b/actor/global_step_270
export DATA_DIR=/home/jiaruil5/social_reasoning_rl/data/tomi
export ROLLOUT_TP_SIZE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export DATA_OUTPUT_PATH=outputs_gen/test_gen.parquet

export NCCL_TIMEOUT=1800  # 30 minutes instead of default 10
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0  # Replace with your network interface

bash inference_tiny_zero.sh