#!/bin/bash

# Usage: ./generate_c_variable_with_commas.sh input_file output_file variable_name rows cols
# Example: ./generate_c_variable_with_commas.sh weights_fc1.txt weights_fc1.c weights_fc1_data 256 784

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 input_file output_file variable_name rows cols"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_FILE=$2
VARIABLE_NAME=$3
ROWS=$4
COLS=$5

# Start the C variable declaration
echo "// Auto-generated C variable from $INPUT_FILE" > "$OUTPUT_FILE"
echo "static float $VARIABLE_NAME[$ROWS][$COLS] = {" >> "$OUTPUT_FILE"

# Read the input file and format it into C array syntax
ROW_COUNT=0
while IFS= read -r line; do
    # Replace spaces with commas for inner array elements
    FORMATTED_LINE=$(echo "$line" | sed 's/ \+/, /g')

    # Add a comma at the end of each row except the last one
    if [ "$ROW_COUNT" -lt "$(($ROWS - 1))" ]; then
        echo "    { $FORMATTED_LINE }," >> "$OUTPUT_FILE"
    else
        echo "    { $FORMATTED_LINE }" >> "$OUTPUT_FILE"
    fi
    ROW_COUNT=$((ROW_COUNT + 1))
done < "$INPUT_FILE"

# Close the C variable declaration
echo "};" >> "$OUTPUT_FILE"

echo "C variable declaration written to $OUTPUT_FILE"