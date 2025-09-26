#!/bin/bash
# this script creates the json files for the experiments

base_folder="/home/gigibs/Documents/research/2025_investigating_mmshap"
base_file="$base_folder/data/input_data/muchomusic_musiccaps.json"
output_folder="$base_folder/data/input_data"
experiments=("few_shot" "description" "question_only" "zero_shot")

for experiment in "${experiments[@]}"; do
    echo "Creating $experiment file"

	python $base_folder/src/data/create_experiments_json.py \
        --experiment="$experiment" \
        --base_file="$base_file" \
        --output_folder="$output_folder"

    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "Successfully completed: $experiment"
    else
        echo "Error occurred while running: $experiment"
        exit 1
    fi
done

echo "All set!"
