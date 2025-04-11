#!/bin/bash

# Source and destination directories
SRC_DIR="./C"
DEST_DIR="../arduino_stuff/ard_code"

# Array of filenames to copy
files=("snn_network.c" "snn_network.h" "rate_encoding.h" "rate_encoding.c" "dummy.c" "dummy.h" "define.h")

# Copy each file from source to destination
for file in "${files[@]}"; do
    if [[ -f "$SRC_DIR/$file" ]]; then
        cp "$SRC_DIR/$file" "$DEST_DIR/"
        echo "Copied $file to $DEST_DIR"
    else
        echo "File $file not found in $SRC_DIR"
    fi
done