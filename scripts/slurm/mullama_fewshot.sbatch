#!/bin/bash
#SBATCH -c 2
#SBATCH --mem 64GB
#SBATCH --time=4:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=mullama_fs.out
#SBATCH --output logs/mullama_fs_20251017.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gv2167@nyu.edu
#SBATCH --account=pr_230_tandon_priority
#SBATCH --array=4,5,6,13,18,19,21,29,33,37,48,49,51,52,57,58,65,66,68,74,76,90,95,115,117,121,133,137,146,147,154,157,175,176,182,192,195,199,201,202,208,214,220,228,238,243,252,276,280,293,294,298,300,305,315,317,341,345,347,353,362,368,369,370,373,374,385,393,406,411,416,423,436,437,441,448,457,458,463,474,476,481,482,504,506,512,516,521,525,529,534,545,546,561,562,572,581,584,604,609,617,618,621,623,637,642,653,662,663,666,679,680,681,684,685,691,692,695,707,711,736,770,777,779,782,788,789,807,808,809,815,821,822,831,836,837,849,850,856

base_path="/scratch/gv2167/2025_investigating_mmshap"
environment="hpc"
system_instruction="default"
range=1
index=$SLURM_ARRAY_TASK_ID
model="mullama"
input_path="${base_path}/data/input_data/muchomusic_musiccaps_fs.json"

$base_path/sing_mu << EOF
python \
	$base_path/experiments/mullama_experiment.py \
	--input_path=$input_path	\
	--model=$model 			\
	--environment=$environment 	\
	--range=$range 			\
	--index=$index
EOF
