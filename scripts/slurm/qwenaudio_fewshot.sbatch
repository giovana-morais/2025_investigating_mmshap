#!/bin/bash
#SBATCH -c 2
#SBATCH --mem 64GB
#SBATCH --time=4:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=qwenaudio_fs_%j
#SBATCH --output logs/qwenaudio_fs_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gv2167@nyu.edu
#SBATCH --account=pr_230_tandon_priority
#SBATCH --array=0-858

base_path="/scratch/gv2167/2025_investigating_mmshap"
environment="hpc"
system_instruction="default"
range=1
index=$SLURM_ARRAY_TASK_ID
model="qwenaudio"
input_path="${base_path}/data/input_data/muchomusic_musiccaps_fs.json"

$base_path/sing << EOF
python \
	$base_path/src/experiments/qwenaudio_experiment.py \
	--input_path=$input_path	\
	--model=$model 			\
	--environment=$environment 	\
	--range=$range 			\
	--index=$index
EOF
