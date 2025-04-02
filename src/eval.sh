#!/bin/bash

# Check for file argument
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file="$1"

# Initialize counters
correct=0
incorrect=0

# Read the file line by line
while IFS= read -r line; do
    classification=$(echo "$line" | grep -oP 'Classification = \K\d+')
    label=$(echo "$line" | grep -oP 'Label = \K\d+')

    if [[ "$classification" == "$label" ]]; then
        ((correct++))
    else
        ((incorrect++))
    fi
done < "$input_file"

# Total predictions
total=$((correct + incorrect))

if [[ $total -gt 0 ]]; then
    accuracy=$(echo "scale=4; $correct / $total * 100" | bc)
    echo "Correct: $correct"
    echo "Incorrect: $incorrect"
    echo "Accuracy: $accuracy%"
else
    echo "No predictions found in file."
fi