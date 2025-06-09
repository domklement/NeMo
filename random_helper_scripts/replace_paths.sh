#!/bin/bash

# Define old and new paths
old_path="/storage/brno12-cerit/home/dklement/speech/dicow_followup/CHIME2024/data"
new_path="/tmp/dicow_data/data"

# Find all regular text files and replace the path
find . -type f -name '*.jsonl' | while read -r file; do
    echo $file
    sed -i "s|$old_path|$new_path|g" "$file"
done

echo "Replacement complete."
