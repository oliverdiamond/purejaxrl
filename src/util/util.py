import os
import json

def load_env_dir(parent_dir, env_name):
    # Find the correct exp directory for given env by recursively searching subdirectories
    env_parent_dir = parent_dir
    env_dir = None

    # Use os.walk to recursively search through all subdirectories
    for root, dirs, files in os.walk(env_parent_dir):
        params_file = os.path.join(root, "params.json")
        if os.path.exists(params_file):
            try:
                with open(params_file, "r") as f:
                    stored_params = json.load(f)
                if stored_params["metaParameters"]["env_name"] == env_name:
                    env_dir = root
                    break
            except (KeyError, json.JSONDecodeError):
                continue

    if env_dir is None:
        raise ValueError(f"Could not find env directory for env_name: {env_name}")

    return env_dir

def get_time_str(run_time):
    hours = int(run_time // 3600)
    minutes = int((run_time % 3600) // 60)
    seconds = int(run_time % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"