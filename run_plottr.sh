#!/bin/zsh

# Load conda configuration
source ~/.zshrc

# Activate conda environment
conda activate py3.11

# Start plottr-inspectr with the provided database path
plottr-inspectr --dbpath="$1"
