#!/bin/sh


model="mullama"
exp="fs"
dir="../../data/output_data/${model}_muchomusic_musiccaps_${exp}"
# Loop through all subdirectories
for dir in */; do
    dir="${dir%/}"  # Remove trailing '/'
    npz_file="$dir/${dir}_info.npz"  # Expected file path (e.g., id/id.npz)

    # Check if the .npz file exists
    if [ -f "$npz_file" ]; then
        echo "Moving $npz_file"
        mv "$npz_file" .  # Move to current dir (exp/)
        rmdir "$dir"      # Remove the now-empty directory
    else
        echo "Warning: $npz_file not found (skipping $dir)"
    fi
done

echo "Done! All .npz files moved to exp/, and subdirectories removed."
