#!/bin/bash
#SBATCH -c 2
#SBATCH --mem 64GB
#SBATCH --time=4:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=qwenaudio_toy
#SBATCH --output logs/qwenaudio_toy.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gv2167@nyu.edu
#SBATCH --account=pr_230_tandon_priority

base_path="/scratch/gv2167/2025_investigating_mmshap"
environment="hpc"
system_instruction="default"
range=1
index=0
model="qwenaudio"
input_path="${base_path}/data/input_data/toy_example.json"

$base_path/sing << EOF
python \
	$base_path/experiments/qwenaudio_toy_example.py \
	--input_path=$input_path	\
	--model=$model 			\
	--environment=$environment 	\
	--range=$range 			\
	--index=$index
EOF
