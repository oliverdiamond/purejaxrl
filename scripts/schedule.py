#!/usr/bin/env python3
"""
Schedule experiments using JSON configuration files or directories containing JSON files.
This script collects JSON files from specified paths and submits them to the SLURM scheduler.
"""

from html import parser
import os
import sys
import argparse
import subprocess
from pathlib import Path


def collect_json_files(paths, exclude_files=None, verbose=False):
    """
    Collect all JSON files from the given paths (files or directories) recursively.
    
    Args:
        paths (list): List of file or directory paths to process
        exclude_files (list): List of relative file paths to exclude
        verbose (bool): Print debug information
        
    Returns:
        list: List of JSON file paths
    """
    json_files = []
    exclude_files = exclude_files or []
    
    if verbose and exclude_files:
        print(f"Debug: Exclude patterns: {exclude_files}", file=sys.stderr)
    
    for path in paths:
        path_obj = Path(path)
        
        if not path_obj.exists():
            print(f"Warning: Path '{path}' does not exist", file=sys.stderr)
            continue
        
        if path_obj.is_file():
            # Handle individual files
            if path_obj.suffix == '.json':
                # Check if the file should be excluded
                relative_path = str(path_obj)
                should_exclude = False
                for exclude_path in exclude_files:
                    if relative_path.endswith(exclude_path):
                        should_exclude = True
                        if verbose:
                            print(f"Debug: Excluding {relative_path} (matched pattern: {exclude_path})", file=sys.stderr)
                        break
                
                if not should_exclude:
                    json_files.append(str(path_obj))
                elif verbose:
                    print(f"Debug: Would have included {relative_path} but it was excluded", file=sys.stderr)
            else:
                print(f"Warning: File '{path}' is not a JSON file", file=sys.stderr)
        elif path_obj.is_dir():
            # Handle directories (existing logic)
            # Find all JSON files recursively
            for json_file in path_obj.rglob("*.json"):
                if json_file.is_file():
                    # Check if the relative path matches any excluded paths
                    relative_path = str(json_file)
                    should_exclude = False
                    for exclude_path in exclude_files:
                        # Try multiple matching strategies
                        if relative_path.endswith(exclude_path):
                            should_exclude = True
                            if verbose:
                                print(f"Debug: Excluding {relative_path} (matched pattern: {exclude_path})", file=sys.stderr)
                            break
                    
                    if not should_exclude:
                        json_files.append(str(json_file))
                    elif verbose:
                        print(f"Debug: Would have included {relative_path} but it was excluded", file=sys.stderr)
        else:
            print(f"Warning: '{path}' is neither a file nor a directory", file=sys.stderr)
    
    return sorted(json_files)


def run_slurm_command(json_files, cluster_config, gpu_count, gpu_mem, entry_script, mode="par", verbose=False):
    """
    Run the SLURM command with the collected JSON files.
    
    Args:
        json_files (list): List of JSON configuration files
        cluster_config (str): Path to cluster configuration file
        gpu_count (str): Number of GPUs to request
        gpu_mem (int): Total GPU memory to request in GB
        entry_script (str): Entry point script path
        mode (str): Execution mode ('par' or 'seq')
        verbose (bool): Whether to print verbose output
    """
    cmd = [
        "python", "./scripts/slurm.py",
        "-e", *json_files,
        "--cluster", cluster_config,
        "--gpu", gpu_count,
        "--gpu-mem", str(gpu_mem),
        "--entry", entry_script,
        "--preallocate",
        "--mode", mode
    ]
    
    if verbose:
        print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        print()
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error: SLURM command failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print("Error: Could not find './scripts/slurm.py'. Make sure you're running from the correct directory.", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Schedule experiments using JSON configuration files or directories containing JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python schedule_experiments.py experiments/exp5/env_batch_0
  python schedule_experiments.py experiments/exp5/env_batch_0/config.json
  python schedule_experiments.py experiments/exp5/env_batch_0 experiments/exp5/env_batch_1
  python schedule_experiments.py experiments/exp5/config.json experiments/exp5/settings.json
  python schedule_experiments.py experiments/exp5 --exclude experiments/exp5/config.json experiments/exp5/settings.json
  python schedule_experiments.py experiments/exp5 --exclude env_batch_0/temp.json
  python schedule_experiments.py experiments/exp5 --cluster clusters/fir.json --gpu 2
  python schedule_experiments.py experiments/exp5 --entry src/sac_continuous_action.py --mode seq
        """
    )
    
    parser.add_argument(
        "directories",
        nargs="*",
        help="Directories to search for JSON files or individual JSON files"
    )
    
    parser.add_argument(
        "--exclude", "-e",
        nargs="*",
        default=[],
        help="Relative file paths to exclude (e.g., 'env_batch_0/config.json' or 'experiments/exp5/settings.json')"
    )
    
    parser.add_argument(
        "--cluster",
        default="clusters/vulcan.json",
        help="Cluster configuration file (default: clusters/vulcan.json)"
    )
    
    parser.add_argument(
        "--gpu",
        default="1",
        help="Number of GPUs to request (default: 1)"
    )
    parser.add_argument(
        "--gpu-mem",
        type=int,
        choices=[10, 20, 40, 80],
        required=False,
        default=80,
        help="(fir cluster only) Specify total gpu memory to use in GB, must be one of [10, 20, 40, 80]."+
            "Smaller values result in faster scheduling and more efficient usage." + 
            "If not specified, will use the full gpu memory.",
    )
    parser.add_argument(
        "--entry",
        default="src/ppo_continuous_action.py",
        help="Entry point script (default: src/ppo_continuous_action.py)"
    )
    
    parser.add_argument(
        "--mode",
        default="par",
        choices=["par", "seq", "vec"],
        help="Execution mode: 'par' for parallel, 'seq' for sequential (default: par)"
    )
    parser.add_argument(
    "--fix_cores",
    action="store_true",
    default=False,
    help="When mode=par, fix the number of cores requested to be the same for all jobs,"
    + "as specified in the cluster config."
    + "Otherwise number of cores will be reduced depending on group size.",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without actually running the command"
    )
    
    args = parser.parse_args()
    
    # Handle case where no paths are specified
    if not args.directories:
        print("Error: No files or directories specified", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    # Build exclude list
    exclude_files = list(args.exclude)
    
    if args.verbose and exclude_files:
        print(f"Excluding files: {', '.join(exclude_files)}")
    
    # Collect JSON files
    print(f"Collecting JSON files from paths: {', '.join(args.directories)}")
    json_files = collect_json_files(args.directories, exclude_files, args.verbose)
    
    if not json_files:
        print(f"Error: No JSON files found in specified paths: {', '.join(args.directories)}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(json_files)} JSON files:")
    for json_file in json_files:
        print(f"  {json_file}")
    print()
    
    # Check if cluster config exists
    if not Path(args.cluster).exists():
        print(f"Warning: Cluster configuration file '{args.cluster}' does not exist", file=sys.stderr)
    
    # Check if entry script exists
    if not Path(args.entry).exists():
        print(f"Warning: Entry script '{args.entry}' does not exist", file=sys.stderr)
    
    if args.dry_run:
        print("DRY RUN - Would execute:")
        cmd_preview = [
            "python", "./scripts/slurm.py",
            "-e", *json_files,
            "--cluster", args.cluster,
            "--gpu", args.gpu,
            "--gpu-mem", str(args.gpu_mem),
            "--entry", args.entry,
            "--preallocate",
            "--mode", args.mode,
            "--fix_cores" if args.mode == "par" and args.fix_cores else ""
        ]
        print(f"  {' '.join(cmd_preview)}")
        sys.exit(0)
    
    # Run the SLURM command
    print("Submitting to SLURM...")
    exit_code = run_slurm_command(
        json_files, 
        args.cluster, 
        args.gpu, 
        args.gpu_mem,
        args.entry, 
        args.mode,
        args.verbose
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
