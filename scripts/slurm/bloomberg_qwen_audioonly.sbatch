#!/bin/bash
#SBATCH -c 2
#SBATCH --mem 64GB
#SBATCH --time=6:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=bloomberg_exps
#SBATCH --output logs/bloomberg_audioonly.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gv2167@nyu.edu
#SBATCH --account=pr_230_tandon_priority

base_path="/scratch/gv2167/2025_investigating_mmshap"
environment="hpc"
system_instruction="default"
model="qwenaudio"
input_path="${base_path}/data/input_data/bloomberg.json"

$base_path/sing << EOF
python \
	$base_path/experiments/qwenaudio_bloomberg_audioonly.py \
	--input_path=$input_path	\
	--model=$model 			\
	--environment=$environment
EOF
