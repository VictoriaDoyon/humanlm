#!/bin/bash
#SBATCH --job-name=generate-data-opinion-1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1

#SBATCH --partition=granite-gpu
#SBATCH --account=chpc
#SBATCH --qos=granite-gpu

cd $SLURM_SUBMIT_DIR
module load fall2026indpstdy
module load vllm


vllm_serve Qwen/Qwen3.8-27B --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --download-dir /scratch/general/vast/u1645697/vllm_cache --port 3200


python -m humanual_datasets.process_raw --dataset_name humanual_opinion --splits train --pull_from_hub snap-stanford/humanual-opinion --save_dir /scratch/general/vast/u1645697/data/humanual-opinion-processed --llm_model hosted_vllm/Qwen/Qwen3.8-27B --partition_by post --config $SLURM_SUBMIT_DIR/humanual_datasets/configs/humanual_opinion.yaml --llm_api_base http://localhost:3200/v1