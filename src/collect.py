import os
import jax
import jax.numpy as jnp
import math
import datetime
from functools import partial
import argparse
import json
import time
import pickle
import filelock as fl
from typing import Mapping, cast

import chex
import flax
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from util.wrappers import LogWrapper, FlattenObservationWrapper
import gymnax
import flashbax as fbx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

from environments import make
from environments.gridworld import EnvState as GridworldEnvState
from util import get_time_str, WANDB_ENTITY, WANDB_PROJECT
from util.fta import fta
from experiment import experiment_model

def make_collect(config):

    env, env_params = make(config["ENV_NAME"])

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def collect(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        def _env_step(runner_state, unused):

            env_state, last_obs, rng = runner_state

            # STEP THE ENV
            rng, _rng = jax.random.split(rng)
            action_rng = jax.random.split(_rng, config["NUM_ENVS"])
            action = jax.vmap(env.action_space().sample)(action_rng)
            
            rng, rng_s = jax.random.split(rng)
            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, env_state, action
            )

            runner_state = (env_state, obs, rng)
            transition = {
                "obs": last_obs,
                "action": action,
                "next_obs": obs,
                "reward": reward,
                "done": done,
                "truncated": info["truncated"],
            }

            return runner_state, transition

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (env_state, init_obs, _rng)

        runner_state, dataset = jax.lax.scan(
            _env_step, runner_state, None, config["TOTAL_TIMESTEPS"]
        )

        return {"runner_state": runner_state, "dataset": dataset}

    return collect

if __name__ == "__main__":
    # ------------------
    # -- Command Args --
    # ------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp",
        type=str,
        required=True,
        help="the experiment configuration file to use",
    )
    parser.add_argument(
        "-i",
        "--idxs",
        nargs="+",
        type=int,
        required=True,
        help=" ".join(
            [
                "the hyperparameter index to use, must be less than the maximum",
                "number of hyperparameter settings in the configuration file",
                "specified with -e",
            ]
        ),
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="use GPU",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="print episode info while running",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        help="when using multiple GPUs, sets the device index to use",
    )
    print("jax devices", jax.devices())
    args = parser.parse_args()

    if not args.gpu:
        jax.config.update("jax_platform_name", "cpu")
    else:
        n_gpus = jax.device_count("gpu")
        jax.default_device(jax.devices("gpu")[args.device % n_gpus])

    exp = experiment_model.load(args.exp)
    print(f"Experiment: {args.exp}")
    indices = args.idxs
    assert (np.array(indices) < exp.numPermutations()).all()

    start_time_all = time.time()

    for idx in indices:
        params = exp.getPermutation(idx)

        hypers = params["metaParameters"]

        print(f"Specified hypers for idx {idx}:")
        print(json.dumps(hypers, indent=2))
        config = {
            "TOTAL_TIMESTEPS": hypers["total_timesteps"],
            "SEED": hypers["seed"],
            "NUM_ENVS": hypers.get("num_envs", 1),
            "ENV_NAME": hypers["env_name"],
            "IMG_OBS": hypers.get('img_obs', True),
            "VERBOSE": args.verbose,
        }

        file_context = exp.buildSaveContext(idx)
        file_context.ensureExists()
        base_dir = file_context.resolve()
        save_dir = os.path.join(base_dir, f"{idx}")
        os.makedirs(save_dir, exist_ok=True)

        log_file_path = os.path.join(save_dir, "run_log.txt")
        with open(log_file_path, "w") as f:
            job_id = os.getenv("SLURM_JOB_ID")
            if job_id is not None:
                f.write(f"job_id = {job_id}\n")
            print(f"Full config for idx {idx}:\n")
            print(json.dumps(config, indent=2) + "\n")
            f.write(f"Full config for idx {idx}:\n")
            f.write(json.dumps(config, indent=2) + "\n")

            rng = jax.random.PRNGKey(config['SEED'])

            collect = make_collect(config)
            collect_jit = jax.jit(collect)

            # Ahead-of-time compilation
            start_compilation_time = time.time()

            compiled_collect = collect_jit.lower(rng).compile()
            jax.block_until_ready(compiled_collect)

            compilation_time_str = get_time_str(time.time() - start_compilation_time)
            print(f"idx {idx} compilation time: {compilation_time_str}")
            f.write(f"idx {idx} compilation time: {compilation_time_str}\n")

            start_time = time.time()

            results = compiled_collect(rng)
            jax.block_until_ready(results)

            run_time_str = get_time_str(time.time() - start_time)
            print(f"idx {idx} runtime: {run_time_str}")
            f.write(f"idx {idx} runtime: {run_time_str}\n")
        

        dataset = results["dataset"]  # each leaf has shape (TOTAL_TIMESTEPS, N_ENVS, ...)
        dataset = jax.tree_util.tree_map(
            lambda x: x.reshape(
                (config["TOTAL_TIMESTEPS"] * config["NUM_ENVS"],) + x.shape[2:]
            ),
            dataset
        )  # Shape (TOTAL_TIMESTEPS * N_ENVS, ...)

        # Save to save_dir
        np.savez(
            os.path.join(save_dir, "dataset.npz"),
            obs=np.array(dataset["obs"]),
            action=np.array(dataset["action"]),
            next_obs=np.array(dataset["next_obs"]),
            reward=np.array(dataset["reward"]),
            done=np.array(dataset["done"]),
            truncated=np.array(dataset["truncated"]),
        )
        print(f"Saved dataset to {os.path.join(save_dir, 'dataset.npz')}")

        try:
            with open(os.path.join(save_dir, "params.json"), "w") as pf:
                json.dump(params, pf, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not serialize params to JSON for idx {idx}: {e}")
            # Fallback to pickle if JSON fails
            with open(os.path.join(save_dir, "params.pkl"), "wb") as pf:
                pickle.dump(params, pf)
        
        # ---------------------------
        # -- Caching finished jobs --
        # ---------------------------
        # Save a list of finished indices and metaParameters
        # keep finished_indices.pkl in the experiment base folder (not per-run)
        fpath = os.path.join(base_dir, "finished_indices.pkl")

        lock = fl.FileLock(fpath + ".lock")
        with lock:
            if os.path.exists(fpath):
                with open(fpath, "rb") as finished_jobs:
                    finished_idxs, params_dict = pickle.load(finished_jobs)
            else:
                finished_idxs = set()
                params_dict = dict()

            finished_idxs.add(idx)
            params_dict[idx] = params
            with open(fpath, "wb") as finished_jobs:
                out_pickle = (finished_idxs, params_dict)
                pickle.dump(out_pickle, finished_jobs)

        print(f"Finished saving results for idx {idx}")

        del results