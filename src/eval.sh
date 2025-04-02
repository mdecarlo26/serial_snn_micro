#!/bin/bash

# File paths
MODEL_OUTPUT="./C/model_output.txt"
LABELS="labels.txt"

# Check if files exist
if [[ ! -f "$MODEL_OUTPUT" || ! -f "$LABELS" ]]; then
    echo "Error: One or both files do not exist."
    exit 1
fi

# Read labels into an array
mapfile -t labels < <(awk '{print int($1)}' "$LABELS")

# Initialize counters
total_samples=0
correct_classifications=0
incorrect_classifications=0

# Read model output and compare with labels
while IFS= read -r line; do
    # Extract classification from model output
    classification=$(echo "$line" | awk -F'=' '{print $2}' | awk '{print $1}' | tr -d '[:space:],')

    # Compare with label
    if [[ "$classification" -eq "${labels[total_samples]}" ]]; then
        ((correct_classifications++))
    else
        ((incorrect_classifications++))
    fi
    
    ((total_samples++))
done < <(grep "Classification" "$MODEL_OUTPUT")

# Calculate accuracy
accuracy=$(echo "scale=2; $correct_classifications / $total_samples * 100" | bc)

# Print summary statistics
echo "Total samples: $total_samples"
echo "Correct classifications: $correct_classifications"
echo "Incorrect classifications: $incorrect_classifications"
echo "Accuracy: $accuracy%"

exit 0