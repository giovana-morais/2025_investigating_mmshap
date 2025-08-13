#!/bin/bash
base_path="/scratch/gv2167/2025_investigating_mmshap"
environment="hpc"
system_instruction="default"
range=1
index=0 #$SLURM_ARRAY_TASK_ID
model="mullama"
input_path="${base_path}/data/input_data/muchomusic_musiccaps_fs.json"

python \
	$base_path/src/experiments/mullama_experiment.py \
	--input_path=$input_path	\
	--model=$model 			\
	--environment=$environment 	\
	--range=$range 			\
	--index=$index
