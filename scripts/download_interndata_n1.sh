#!/bin/bash

# Script to download InternData-N1 dataset from Hugging Face
# Usage: ./download_interndata_n1.sh <target_folder> [branch]
# Example: ./download_interndata_n1.sh /path/to/data v0.1-mini

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <target_folder> [branch]"
    echo "  target_folder: Directory where dataset will be downloaded"
    echo "  branch: Dataset branch (main, v0.1-full, or v0.1-mini). Default: v0.1-mini"
    echo ""
    echo "Examples:"
    echo "  $0 /data/interndata_n1 v0.1-mini    # Download mini dataset (~220GB)"
    echo "  $0 /data/interndata_n1 v0.1-full    # Download full dataset (~5TB+)"
    echo "  $0 /data/interndata_n1 main         # Download latest version"
    exit 1
fi

TARGET_FOLDER="$1"
BRANCH="${2:-v0.1-mini}"

echo "=========================================="
echo "InternData-N1 Dataset Download Script"
echo "=========================================="
echo "Target folder: $TARGET_FOLDER"
echo "Branch: $BRANCH"
echo ""

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "ERROR: git-lfs is not installed."
    echo "Please install it first:"
    echo "  Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  Or visit: https://git-lfs.com"
    exit 1
fi

# Initialize git-lfs (if not already done)
echo "Initializing git-lfs..."
git lfs install

# Create target directory if it doesn't exist
mkdir -p "$TARGET_FOLDER"
cd "$TARGET_FOLDER"

# Check if directory is empty
if [ "$(ls -A $TARGET_FOLDER 2>/dev/null)" ]; then
    echo "WARNING: Target folder is not empty!"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "IMPORTANT: Authentication Required"
echo "=========================================="
echo "This dataset requires:"
echo "1. Hugging Face account with access granted"
echo "2. Hugging Face access token"
echo ""
echo "If you haven't done so:"
echo "1. Visit: https://huggingface.co/datasets/InternRobotics/InternData-N1"
echo "2. Accept the license agreement"
echo "3. Generate an access token at: https://huggingface.co/settings/tokens"
echo ""
read -p "Press Enter to continue..."

# Check for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "Hugging Face token not found in environment."
    echo "You can either:"
    echo "  1. Set HF_TOKEN environment variable: export HF_TOKEN=your_token"
    echo "  2. Or login using: huggingface-cli login"
    echo ""
    read -p "Do you want to login now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    else
        echo "Please set HF_TOKEN or login before continuing."
        exit 1
    fi
fi

echo ""
echo "Starting download..."
echo "This may take a while depending on your connection and the branch selected."
echo ""

# Clone the dataset
REPO_URL="https://huggingface.co/datasets/InternRobotics/InternData-N1"

if [ -n "$HF_TOKEN" ]; then
    # Use token for authentication
    AUTH_URL="https://${HF_TOKEN}@huggingface.co/datasets/InternRobotics/InternData-N1"
    echo "Cloning with authentication token..."
    git clone -b "$BRANCH" "$AUTH_URL" .
else
    # Use huggingface-cli authentication
    echo "Cloning (using huggingface-cli authentication)..."
    git clone -b "$BRANCH" "$REPO_URL" .
fi

echo ""
echo "=========================================="
echo "Download completed!"
echo "=========================================="
echo "Dataset location: $TARGET_FOLDER"
echo "Branch: $BRANCH"
echo ""
echo "Note: If you see LFS pointer files, run 'git lfs pull' to download actual files."
echo "Or use selective download commands (see script comments)."

