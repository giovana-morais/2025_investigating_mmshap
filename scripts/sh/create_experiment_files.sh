#!/bin/bash
# this script creates the json files for the experiments

experiments=("few_shot" "description" "question_only" "zero_shot")
base_file="/home/gigibs/Documents/research/2025_investigating_mmshap/data/input_data/muchomusic_original.json"
output_folder="/home/gigibs/Documents/research/2025_investigating_mmshap/data/input_data"

for experiment in "${experiments[@]}"; do
    echo "Creating $experiment file"

	python /home/gigibs/Documents/research/2025_investigating_mmshap/src/data/create_experiments_json.py \
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
