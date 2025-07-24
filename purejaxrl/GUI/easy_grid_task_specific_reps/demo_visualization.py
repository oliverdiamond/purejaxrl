#!/usr/bin/env python3
"""
Demo script to show available data files and launch the visualization.
"""

import os
import subprocess
import sys

def list_available_files():
    """List all available activation files."""
    # Navigate up to the project root to find the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    data_dir = os.path.join(project_root, "purejaxrl", "data")
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found!")
        print(f"Current directory: {current_dir}")
        print(f"Looking for data in: {data_dir}")
        return []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    return files

def main():
    print("🧠 Neural Activation Visualizer Demo")
    print("=" * 50)
    
    # List available files
    files = list_available_files()
    
    if not files:
        print("No activation files found in purejaxrl/data/")
        print("Please run the training script first to generate activation data.")
        return
    
    print("Available activation files:")
    for i, filename in enumerate(files, 1):
        print(f"  {i}. {filename}")
    
    print()
    
    # Get user selection
    if len(files) == 1:
        selected_file = files[0]
        print(f"Using the only available file: {selected_file}")
    else:
        try:
            choice = input(f"Select a file (1-{len(files)}) or press Enter for file 1: ").strip()
            if not choice:
                choice = "1"
            
            file_idx = int(choice) - 1
            if 0 <= file_idx < len(files):
                selected_file = files[file_idx]
            else:
                print("Invalid selection, using first file.")
                selected_file = files[0]
        except ValueError:
            print("Invalid input, using first file.")
            selected_file = files[0]
    
    print(f"\nStarting visualization for: {selected_file}")
    print("The GUI will open in your web browser...")
    print("Press Ctrl+C in this terminal to stop the server.")
    print()
    
    # Launch streamlit
    try:
        # Get the path to the visualization script in the same directory
        script_path = os.path.join(os.path.dirname(__file__), "visualize_reps_easy_grid_with_task_specific_reps.py")
        
        cmd = [
            "streamlit", "run", 
            script_path,
            "--", selected_file
        ]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    except FileNotFoundError:
        print("Error: streamlit command not found. Please install streamlit:")
        print("pip install streamlit")
    except Exception as e:
        print(f"Error starting visualization: {e}")

if __name__ == "__main__":
    main()
