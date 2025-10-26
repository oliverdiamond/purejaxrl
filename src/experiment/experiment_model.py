# adapted from https://github.com/andnp/rl-control-template/tree/main
import json
import sys

from PyExpUtils.models.ExperimentDescription import ExperimentDescription
from typing_extensions import override



class ExperimentModel(ExperimentDescription):
    def __init__(self, d, path):
        super().__init__(d, path)
        self.total_steps = d.get("total_timesteps", None)
        self.agent = d['agent']
        self.env = d["metaParameters"]["env_name"]

    @override
    def getRun(self, idx: int) -> int:
        # For our experiment model, we always return a run of 0, since we
        # specify seeds in configuration files.
        return 0


def load(path=None) -> ExperimentModel:
    path = path if path is not None else sys.argv[1]
    with open(path, "r") as f:
        d = json.load(f)

    exp = ExperimentModel(d, path)
    return exp
