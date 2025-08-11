#!/bin/bash
#SBATCH -c 2
#SBATCH --mem 64GB
#SBATCH --time=4:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=mullama_zs_%j
#SBATCH --output logs/mullama_zs_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gv2167@nyu.edu
#SBATCH --account=pr_230_tandon_priority
#SBATCH --array=0-858

base_path="/scratch/gv2167/2025_investigating_mmshap"
environment="hpc"
system_instruction="default"
range=1
index=$SLURM_ARRAY_TASK_ID
model="mullama"
input_path="${base_path}/data/input_data/muchomusic_musiccaps_zs.json"

$base_path/sing_mu << EOF
python \
	$base_path/src/experiments/mullama_experiment.py \
	--input_path=$input_path	\
	--model=$model 			\
	--environment=$environment 	\
	--range=$range 			\
	--index=$index
EOF
