#!/bin/bash

# Kaggle competition slug
KAGGLE_COMPETITION="soil-classification"
TARGET_DIR="./data"

echo "Downloading competition data: $KAGGLE_COMPETITION"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download and unzip competition data
kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$TARGET_DIR" --unzip

echo "Download complete. Files saved to $TARGET_DIR"
