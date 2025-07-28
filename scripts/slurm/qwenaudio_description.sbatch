#!/bin/bash
#SBATCH -c 2
#SBATCH --mem 64GB
#SBATCH --time=8:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=qwenaudio_desc_%j
#SBATCH --output logs/qwenaudio_desc_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gv2167@nyu.edu
#SBATCH --account=pr_230_tandon_priority
#SBATCH --array=0-9

base_path="/scratch/gv2167/2025_investigating_mmshap"
environment="hpc"
system_instruction="default"
range=120
index=$SLURM_ARRAY_TASK_ID
input_path="${base_path}/data/input_data/muchomusic_musiccaps_desc.json"
output_path="${base_path}/data/output_data/qwenaudio_${prompt_type}_${range}_${index}.json"

/scratch/gv2167/ismir2025/ismir2025_exploration/sing << EOF
python \
	$base_path/src/experiments/qwenaudio_experiment.py \
	--input_path=$input_path	\
	--output_path=$output_path 	\
	--environment=$environment 	\
	--range=$range 			\
	--index=$index
EOF
