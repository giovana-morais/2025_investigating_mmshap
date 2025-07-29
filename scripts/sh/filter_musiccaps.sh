#!/bin/bash -e
# create filtered version of muchomusic with musiccaps tracks only

base_folder="/home/gigibs/Documents/2025_investigating_mmshap"
input_path="${base_folder}/data/input_data/muchomusic_original.json"
output_path="${base_folder}/data/input_data/muchomusic_musiccaps.json"

echo "fixing muchomusic path"
python $base_folder/src/data/filter_musiccaps.py \
	--input_path=$input_path \
	--output_path=$output_path

echo "done! final json is available at $output_path"
