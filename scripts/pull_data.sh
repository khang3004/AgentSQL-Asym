#!/bin/bash
# Script to automate downloading BIRD Mini-Dev dataset from Google Drive
# and extracting it into the correct logical structure.

set -e

DATA_DIR="data_minidev"
DRIVE_FILE_ID="13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG"
ZIP_NAME="MINIDEV.zip"

echo "======================================================"
echo "🚀 BIRD MINI-DEV AUTOMATED DOWNLOADER"
echo "======================================================"

if [ -d "$DATA_DIR/MINIDEV" ]; then
    echo "Dataset already exists at $DATA_DIR/MINIDEV. Skipping download."
    exit 0
fi

# Ensure data_minidev directory exists
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading Complete Package from Google Drive..."
gdown $DRIVE_FILE_ID -O $ZIP_NAME

echo "Extracting $ZIP_NAME..."
# Handle case when unzip command isn't present
if ! command -v unzip &> /dev/null; then
    echo "Installing unzip utility..."
    apt-get update -y && apt-get install -y unzip
fi

unzip -q $ZIP_NAME
rm $ZIP_NAME

echo "Download and Extraction Complete!"
echo "Data is safely stored in: $DATA_DIR/MINIDEV"
