#!/bin/bash

# Initialize counters
correct=0
incorrect=0

# Read from stdin
while IFS= read -r line; do
    classification=$(echo "$line" | grep -oP 'Classification = \K\d+')
    label=$(echo "$line" | grep -oP 'Label = \K\d+')

    if [[ "$classification" == "$label" ]]; then
        ((correct++))
    else
        ((incorrect++))
    fi
done

# Total predictions
total=$((correct + incorrect))

if [[ $total -gt 0 ]]; then
    accuracy=$(echo "scale=4; $correct / $total * 100" | bc)
    echo "Correct: $correct"
    echo "Incorrect: $incorrect"
    echo "Accuracy: $accuracy%"
else
    echo "No predictions found in input."
fi
