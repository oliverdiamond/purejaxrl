#!/bin/bash

# Script to run the Streamlit visualization
# Usage: ./run_visualization.sh <results_directory>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <results_directory>"
    echo "Example: $0 purejaxrl/results/dqn_multitask/TwoRoomsMultiTask5/MTQNet/25-01-29-10-30-45"
    echo "Example: $0 ../../results/dqn_multitask/TwoRoomsMultiTask5/MTQNet/25-01-29-10-30-45"
    exit 1
fi

results_dir="$1"

# Check if the results directory exists and contains activations.pkl
if [ ! -d "$results_dir" ]; then
    echo "Error: Results directory '$results_dir' does not exist."
    exit 1
fi

if [ ! -f "$results_dir/activations.pkl" ]; then
    echo "Error: activations.pkl not found in '$results_dir'"
    echo "Make sure you're pointing to a directory that contains activations.pkl"
    exit 1
fi

echo "Starting Streamlit visualization for results directory: $results_dir"
echo "The GUI will open in your default web browser..."
echo "Press Ctrl+C to stop the server"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

streamlit run "$SCRIPT_DIR/visualize_reps.py" -- "$results_dir"
