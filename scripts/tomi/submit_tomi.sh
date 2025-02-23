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

export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=2
export BASE_MODEL=/compute/babel-4-33/jiaruil5/.cache/DeepSeek-R1-Distill-Qwen-1.5B
export DATA_DIR=/home/jiaruil5/social_reasoning_rl/data/tomi
export EXPERIMENT_NAME=tomi-tinyzero
export ROLLOUT_TP_SIZE=2
export VLLM_ATTENTION_BACKEND=XFORMERS
export SAVE_DIR=/compute/babel-2-29/jiaruil5/social_reasoning/tinyzero/tomi_distilled_1.5b

export NCCL_TIMEOUT=1800  # 30 minutes instead of default 10
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0  # Replace with your network interface

bash train_tiny_zero.sh