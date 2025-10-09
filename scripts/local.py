#!/usr/bin/env python3

import sys

sys.path.insert(0, ".")

import argparse
import random
import subprocess
from functools import partial
from multiprocessing.pool import Pool

from util import gather_missing

import src.experiment.experiment_model as experiment

parser = argparse.ArgumentParser()
parser.add_argument("-e", type=str, nargs="+", required=True)
parser.add_argument("--cpus", type=int, default=8)
parser.add_argument("--entry", type=str, default="src/ppo.py")
parser.add_argument("--results", type=str, default="./")


def count(pre, it):
    print(pre, 0, end="\r")
    for i, x in enumerate(it):
        print(pre, i + 1, end="\r")
        yield x

    print()


if __name__ == "__main__":
    cmdline = parser.parse_args()

    pool = Pool(cmdline.cpus)

    cmds = []
    e_to_missing = gather_missing(cmdline.e)
    for path in cmdline.e:
        exp = experiment.load(path)

        indices = count(path, e_to_missing[path])
        for idx in indices:
            exe = f"python {cmdline.entry} --silent -e {path} -i {idx}"
            cmds.append(exe)

    print(len(cmds))
    random.shuffle(cmds)
    res = pool.imap_unordered(
        partial(subprocess.run, shell=True, stdout=subprocess.PIPE),
        cmds,
        chunksize=1,
    )
    for i, _ in enumerate(res):
        sys.stderr.write(f"\r{i + 1}/{len(cmds)}")
    sys.stderr.write("\n")
