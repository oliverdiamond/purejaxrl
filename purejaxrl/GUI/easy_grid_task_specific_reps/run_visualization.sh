#!/bin/bash

# Script to run the Streamlit visualization
# Usage: ./run_visualization.sh <filename>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    echo "Example: $0 activations_N=5_n_tasks=3_shared_bottleneck=8_task_features=32_conv1=32_conv2=16_seed=0.pkl"
    exit 1
fi

filename="$1"

echo "Starting Streamlit visualization for file: $filename"
echo "The GUI will open in your default web browser..."
echo "Press Ctrl+C to stop the server"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

streamlit run "$SCRIPT_DIR/visualize_reps_easy_grid_with_task_specific_reps.py" -- "$filename"
