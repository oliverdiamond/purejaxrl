#!/usr/bin/env python3
"""
Demo script to show available results directories and launch the visualization.

This script will automatically find all available results directories containing
activations.pkl files and allow you to select which one to visualize.

Usage:
    python demo_visualization.py

The script will look for results in:
    purejaxrl/results/dqn_multitask/<ENV_NAME>/<NETWORK_NAME>/<TIMESTAMP>/
"""

import os
import subprocess
import sys

def list_available_results():
    """List all available results directories containing activations.pkl."""
    # Get the directory where this script is located (purejaxrl/GUI/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # The script is in purejaxrl/GUI/, so go up two levels to get to project root
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Verify we found the correct project root by checking for purejaxrl directory
    if not os.path.exists(os.path.join(project_root, "purejaxrl")):
        # Fallback: search upward from current working directory
        test_path = os.getcwd()
        while test_path != os.path.dirname(test_path):  # Not at filesystem root
            if os.path.exists(os.path.join(test_path, "purejaxrl")):
                project_root = test_path
                break
            test_path = os.path.dirname(test_path)
        else:
            # Final fallback: use current working directory
            project_root = os.getcwd()
    
    results_base_dir = os.path.join(project_root, "purejaxrl", "results", "dqn_multitask")
    
    if not os.path.exists(results_base_dir):
        print(f"Results directory {results_base_dir} not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script location: {script_dir}")
        print(f"Project root detected as: {project_root}")
        print(f"Looking for results in: {results_base_dir}")
        return []
    
    # Find all directories containing activations.pkl
    results_dirs = []
    
    # Walk through the results directory structure
    for env_name in os.listdir(results_base_dir):
        env_path = os.path.join(results_base_dir, env_name)
        if not os.path.isdir(env_path):
            continue
            
        for network_name in os.listdir(env_path):
            network_path = os.path.join(env_path, network_name)
            if not os.path.isdir(network_path):
                continue
                
            for run_timestamp in os.listdir(network_path):
                run_path = os.path.join(network_path, run_timestamp)
                if not os.path.isdir(run_path):
                    continue
                    
                # Check if this directory contains activations.pkl
                activations_file = os.path.join(run_path, "activations.pkl")
                if os.path.exists(activations_file):
                    # Store relative path from project root for easier display
                    relative_path = os.path.relpath(run_path, project_root)
                    results_dirs.append({
                        'path': run_path,
                        'relative_path': relative_path,
                        'env_name': env_name,
                        'network_name': network_name,
                        'timestamp': run_timestamp
                    })
    
    # Sort by timestamp (most recent first)
    results_dirs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return results_dirs

def main():
    print("🧠 Neural Activation Visualizer Demo")
    print("=" * 50)
    
    # List available results directories
    results_dirs = list_available_results()
    
    if not results_dirs:
        print("No results directories with activations.pkl found in purejaxrl/results/dqn_multitask/")
        print("Please run the training script first to generate activation data.")
        print("\nExpected directory structure:")
        print("purejaxrl/results/dqn_multitask/")
        print("  └── <ENV_NAME>/")
        print("      └── <NETWORK_NAME>/")
        print("          └── <TIMESTAMP>/")
        print("              └── activations.pkl")
        return
    
    print("Available results directories:")
    for i, result_info in enumerate(results_dirs, 1):
        print(f"  {i}. {result_info['env_name']}/{result_info['network_name']}/{result_info['timestamp']}")
        print(f"     Path: {result_info['relative_path']}")
        if i < len(results_dirs):
            print()
    
    print()
    
    # Get user selection
    if len(results_dirs) == 1:
        selected_result = results_dirs[0]
        print(f"Using the only available results directory: {selected_result['relative_path']}")
    else:
        try:
            choice = input(f"Select a results directory (1-{len(results_dirs)}) or press Enter for the most recent: ").strip()
            if not choice:
                choice = "1"
            
            result_idx = int(choice) - 1
            if 0 <= result_idx < len(results_dirs):
                selected_result = results_dirs[result_idx]
            else:
                print("Invalid selection, using most recent results.")
                selected_result = results_dirs[0]
        except ValueError:
            print("Invalid input, using most recent results.")
            selected_result = results_dirs[0]
    
    print(f"\nStarting visualization for: {selected_result['relative_path']}")
    print("The GUI will open in your web browser...")
    print("Press Ctrl+C in this terminal to stop the server.")
    print()
    
    # Launch streamlit
    try:
        # Get the path to the visualization script in the same directory
        script_path = os.path.join(os.path.dirname(__file__), "visualize_reps.py")
        
        cmd = [
            "streamlit", "run", 
            script_path,
            "--", selected_result['path']
        ]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    except FileNotFoundError:
        print("Error: streamlit command not found. Please install streamlit:")
        print("pip install streamlit")
    except Exception as e:
        print(f"Error starting visualization: {e}")
        print(f"Command attempted: {' '.join(cmd)}")

if __name__ == "__main__":
    main()
