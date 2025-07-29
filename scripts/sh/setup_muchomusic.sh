#!/bin/bash -e
# preprocess and add unique id. this is only necessary if the muchomusic.json was
# generated from their `prepare_prompt` script, available on
# https://github.com/mulab-mir/muchomusic/blob/main/prepare_prompts.py

base_folder="/home/gigibs/Documents/2025_investigating_mmshap"
input_path="${base_folder}/data/input_data/muchomusic_original.json"
output_path="${base_folder}/data/input_data/muchomusic_original.json"

echo "fixing muchomusic path"
python $base_folder/src/data/fix_muchomusic_musiccaps.py \
	--input_path=$input_path \
	--output_path=$output_path

echo "adding unique id to q&a pairs"
python $base_folder/src/data/add_unique_id.py \
	--input_path=$input_path \
	--output_path=$output_path


echo "done! final json is available at $output_path"
