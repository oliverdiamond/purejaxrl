# If local_settings.json exists in the project root, it will override the default
# WANDB_ENTITY and WANDB_PROJECT values.
#
# This allows us to keep personal wandb settings out of version control.

import json
import os

WANDB_ENTITY = "odiamond-personal"
WANDB_PROJECT = "feature-attainment"

# Check if local_settings.json exists in the project root
local_settings_path = os.path.join(os.path.dirname(__file__), "../../local_settings.json")

if os.path.exists(local_settings_path):
    with open(local_settings_path, "r") as f:
        local_settings = json.load(f)
        WANDB_ENTITY = local_settings.get("WANDB_ENTITY", WANDB_ENTITY)
        WANDB_PROJECT = local_settings.get("WANDB_PROJECT", WANDB_PROJECT)
