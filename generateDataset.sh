#!/bin/bash
#SBATCH --job-name=generate-data-opinion-1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4

#SBATCH --partition=granite-gpu-guest
#SBATCH --account=chpc
#SBATCH --qos=granite-gpu-guest

cd $SLURM_SUBMIT_DIR
module load fall2026indpstdy
module load vllm


vllm_serve Qwen/Qwen3.8-27B --tensor-parallel-size 4 --gpu-memory-utilization 0.80 --download-dir /scratch/general/vast/u1645697/vllm_cache --port 3200


python -m humanual_datasets.process_raw --dataset_name humanual_opinion --splits train --pull_from_hub snap-stanford/humanual-opinion --save_dir /scratch/general/vast/u1645697/data/humanual-opinion-processed --llm_model hosted_vllm/Qwen/Qwen3.8-27B --partition_by post --config $SLURM_SUBMIT_DIR/humanual_datasets/configs/humanual_opinion.yaml --llm_api_base http://localhost:3200/v1