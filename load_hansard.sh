#!/bin/bash

# Usage: ./copy_files.sh /path/to/source /path/to/destination

SOURCE_DIR="/home/tytan216/q4_volume/hansard/clean"
DEST_DIR="/home/tytan216/volume/tzeyoung/Policy_RLHF/hansard_clean_compiled"

if [[ -z "$SOURCE_DIR" || -z "$DEST_DIR" ]]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

mkdir -p "$DEST_DIR"

# Copy only regular files (not directories)
find "$SOURCE_DIR" -maxdepth 5 -type f -exec cp {} "$DEST_DIR" \;

echo "Files copied from $SOURCE_DIR to $DEST_DIR"
