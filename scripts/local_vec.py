#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

import argparse
import random
import subprocess
from functools import partial
from util import gather_missing

import src.experiment.experiment_model as experiment

parser = argparse.ArgumentParser()
parser.add_argument("-e", type=str, nargs="+", required=True)
parser.add_argument("--entry", type=str, default="src/ppo_vec.py")
parser.add_argument("--results", type=str, default="./")


if __name__ == "__main__":
    cmdline = parser.parse_args()

    assert "vec" in cmdline.entry, "Warning: This script will pass all missing permutation idxs for each provided experiment config to a single process, running each experiment sequnetially."

    cmds = []
    e_to_missing = gather_missing(cmdline.e)
    for i, path in enumerate(cmdline.e):
        missing = list(e_to_missing[path])
        if not missing:
            continue
        idx_arg = " ".join(map(str, missing))
        exe = f"python {cmdline.entry} -e {path} -i {idx_arg} --no-gpu"
        cmds.append(exe)
    
    # Run each command sequentially and print progress
    total = len(cmds)
    for idx, exe in enumerate(cmds, start=1):
        print(f"Running experiment ({idx}/{total}). Command: {exe}")
        subprocess.run(exe, shell=True, check=True)