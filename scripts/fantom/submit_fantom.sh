#!/bin/bash
#SBATCH --job-name=code
#SBATCH --partition=preempt
#SBATCH --output=log.out
#SBATCH --error=log.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --time=2-00:00:00
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
export DATA_DIR=/home/jiaruil5/social_reasoning_rl/data/fantom
export EXPERIMENT_NAME=fantom-tinyzero
export ROLLOUT_TP_SIZE=2
export VLLM_ATTENTION_BACKEND=XFORMERS
export SAVE_DIR=/compute/babel-2-29/jiaruil5/social_reasoning/tinyzero/fantom_distilled_1.5b

export NCCL_TIMEOUT=1800  # 30 minutes instead of default 10
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export HYDRA_FULL_ERROR=1

bash train_tiny_zero.sh