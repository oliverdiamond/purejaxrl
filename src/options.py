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
from flax.core import freeze, unfreeze
from util.util import load_env_dir
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
from dqn import QNet, QNetFTA

def make_stopping_condition(config, feature_network, feature_network_params, get_features, options_network, dataset, n_options):
    #TODO Determine if feature should be maximized or minimized based on gradient on the entire dataset (so we can do attainment for hidden layers)
    if not "no_val" in config["STOPPING_CONDITION"]:
        assert config["USE_LAST_HIDDEN"] == True, "value-based stopping val needs features to be from last hidden layer"
        main_params_batched = jax.tree_util.tree_map(
            lambda arr: jnp.stack([arr] * n_options),
            feature_network_params
        )
        feature_idxs = jnp.arange(n_options)
        main_params_batched = unfreeze(main_params_batched)
        new_output_weights = main_params_batched['params']['output']['kernel'].at[feature_idxs, feature_idxs, :].set(config["BONUS_WEIGHT"])
        main_params_batched['params']['output']['kernel'] = new_output_weights
        main_params_batched = freeze(main_params_batched)
        vmap_main_net_apply = jax.vmap(
                lambda params, inputs: feature_network.apply(params, inputs),
                in_axes=(0, None)
            )
    if "percentile" in config["STOPPING_CONDITION"]:
        dataset_features = feature_network.apply(
            feature_network_params, 
            dataset,
            method=get_features
        ) # (dataset_size, n_features)
        percentile = jnp.percentile(
            dataset_features, 
            config["STOPPING_PERCENTILE"], 
            axis=0
        ) # (n_features,)
    
    if config["STOPPING_CONDITION"] == "stomp":
        def stomp_stop_cond(obs, params):
            greedy_action_idxs = feature_network.apply(feature_network_params, obs).argmax(axis=-1)  # (batch_size,)
            action_vals_with_bonus = vmap_main_net_apply(feature_net_params, obs)  # (n_features, batch_size, n_actions)
            state_vals_with_bonus = action_vals_with_bonus[:, jnp.arange(obs.shape[0]), greedy_action_idxs]  # (n_features, batch_size)
            option_state_vals = options_network.apply(params, obs).max(axis=-1)  # (n_features, batch_size)
            stop = (state_vals_with_bonus > option_state_vals).astype(jnp.int32)  # (n_features, batch_size)

            return stop, state_vals_with_bonus
        
        return stomp_stop_cond

    elif config["STOPPING_CONDITION"] == "stomp_no_val":
        def stomp_no_val_stop_cond(obs, params):
            option_state_vals = options_network.apply(params, obs).max(axis=-1)  # (n_features, batch_size)
            obs_features = feature_network.apply(
                        feature_network_params, 
                        obs,
                        method=get_features
                    ) # (batch_size, n_features)
            stop_val = jnp.transpose(config["BONUS_WEIGHT"] * obs_features) # (n_features, batch_size)
            stop = (stop_val > option_state_vals).astype(jnp.int32)  # (n_features, batch_size)
            return stop, stop_val

        return stomp_no_val_stop_cond

    elif config["STOPPING_CONDITION"] == "percentile":
        assert config["USE_LAST_HIDDEN"] == True, "value-based stopping val needs features to be from last hidden layer"
        def percentile_stop_cond(obs):
            obs_features = feature_network.apply(
                        feature_network_params, 
                        obs,
                        method=get_features
                    ) # (batch_size, n_features)
            stop = jnp.transpose(obs_features > percentile).astype(jnp.int32)  # (n_features, batch_size)
            greedy_action_idxs = feature_network.apply(feature_network_params, obs).argmax(axis=-1)  # (batch_size,)
            action_vals_with_bonus = vmap_main_net_apply(feature_net_params, obs)  # (n_features, batch_size, n_actions)
            state_vals_with_bonus = action_vals_with_bonus[:, jnp.arange(obs.shape[0]), greedy_action_idxs]  # (n_features, batch_size)
            
            return stop, state_vals_with_bonus
        
        return percentile_stop_cond


    elif config["STOPPING_CONDITION"] == "percentile_no_val":
        #TODO Figure out what the stopping value should be. Normalized (or unnormalized) feature value with weight coefficient?
        def percentile_no_val_stop_cond(obs):
            obs_features = feature_network.apply(
                        feature_network_params, 
                        obs,
                        method=get_features
                    ) # (batch_size, n_features)
            stop_val = jnp.transpose(config["BONUS_WEIGHT"] * obs_features) # (n_features, batch_size)
            stop = jnp.transpose(obs_features > percentile).astype(jnp.int32)  # (n_features, batch_size)
            
            return stop, stop_val

        return percentile_no_val_stop_cond
    else:
        raise ValueError(f"Unknown stopping condition: {config['STOPPING_CONDITION']}")
    
def make_options_network(config, action_dim, n_options):
    """Returns the appropriate network based on the configuration."""
    if config["ACTIVATION"] == "relu":
        # Option Networks
        return nn.vmap(
            QNet,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=(None,),
            out_axes=0,
            axis_size=n_options,
            methods=['__call__']
        )(action_dim=action_dim, 
          conv1_dim=config["CONV1_DIM"], 
          conv2_dim=config["CONV2_DIM"], 
          rep_dim=config["REP_DIM"])
    else:
        raise ValueError("Option learning currently only supports ReLU activations")
    
def make_feature_network(config, action_dim):
    """Returns the appropriate network based on the configuration."""
    feature_net_agent = config["FEATURE_PARAMS"]["agent"].lower()
    feature_net_hypers = config["FEATURE_PARAMS"]["metaParameters"]
    if feature_net_agent == "dqn":
        if feature_net_hypers["activation"] == "relu":
            return QNet(
                action_dim=action_dim,
                conv1_dim=feature_net_hypers.get("conv1_dim", 32),
                conv2_dim=feature_net_hypers.get("conv2_dim", 16),
                rep_dim=feature_net_hypers.get("rep_dim", 32),
                head_hidden_dim=feature_net_hypers.get("head_hidden_dim", 64)
            )
        elif feature_net_hypers["activation"] == "fta":
            return QNetFTA(
                action_dim=action_dim,
                conv1_dim=feature_net_hypers.get('conv1_dim', 32),
                conv2_dim=feature_net_hypers.get('conv2_dim', 16),
                rep_dim=feature_net_hypers.get('rep_dim', 32),
                head_hidden_dim=feature_net_hypers.get("head_hidden_dim", 64),
                fta_eta=feature_net_hypers.get('fta_eta', 2),
                fta_tiles=feature_net_hypers.get("fta_tiles", 20),
                fta_lower_bound=feature_net_hypers.get("fta_lower_bound", -20.0),
                fta_upper_bound=feature_net_hypers.get("fta_upper_bound", 20.0),
            )            
    else:
        raise ValueError(f"Unknown feature network: {feature_net_agent}")



class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict # type: ignore
    n_updates: int

def make_train(config):
    def train(feature_net_params, dataset, rng):
        # FEATURE NETWORK
        if config["IMG_OBS"]:
            init_x = jnp.zeros((1,) + config["OBS_SHAPE"])  # Conv layers need extra dimension for batch size
        else:
            init_x = jnp.zeros(config["OBS_SHAPE"])

        feature_network = make_feature_network(config, action_dim=config["ACTION_DIM"])

        if config['USE_LAST_HIDDEN']:
            get_features = feature_network.get_last_hidden # type: ignore
        else:
            get_features = feature_network.get_features # type: ignore

        _features = feature_network.apply( # type: ignore
            feature_net_params,
            init_x,
            method=get_features, # type: ignore
        )
        n_options = _features.shape[-1] # type: ignore

        # OPTIONS NETWORK (seperate ff for each option for now)
        options_network = make_options_network(
            config, 
            action_dim=config["ACTION_DIM"], 
            n_options=n_options
        )
        rng, _rng = jax.random.split(rng)
        options_network_params = options_network.init(_rng, init_x)

        # STOPPING CONDITION
        stop_cond = make_stopping_condition(
            config, 
            feature_network, 
            feature_net_params, 
            get_features, 
            options_network,
            dataset, 
            n_options
        )

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LEARNING_RATE"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LEARNING_RATE"]
        tx = optax.adam(learning_rate=lr)

        train_state = CustomTrainState.create(
            apply_fn=options_network.apply,
            params=options_network_params,
            target_network_params=jax.tree_util.tree_map(
                lambda x: jnp.copy(x), 
                options_network_params),
            tx=tx,
            n_updates=0,
        )

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            train_state, rng = runner_state
            # NETWORK UPDATE
            rng, _rng = jax.random.split(rng)
            idxs = jax.random.choice(_rng, config['DATASET_SIZE'], (config["BATCH_SIZE"],), replace=True)
            learn_batch = jax.tree_util.tree_map(
                lambda x: x[idxs],
                dataset
            )
            q_next_target = options_network.apply(
                train_state.target_network_params, 
                learn_batch['next_obs']
            )  # (num_options, batch_size, num_actions)
            q_next_target = jnp.max(q_next_target, axis=-1)  # (num_options, batch_size) # type: ignore

            stop, stop_val = stop_cond(
                learn_batch['next_obs'], 
                params=train_state.target_network_params) # (num_options, batch_size)  # type: ignore
            
            # Assumes all trunc transitions are discarded during collection
            # TODO: Double check this is correct!
            target = jax.lax.stop_gradient(
                learn_batch['reward']
                + (1 - learn_batch['done']) 
                * (
                    (1 - stop) * config["GAMMA"] * q_next_target 
                  + (stop) * stop_val
                  )
            )

            def _loss_fn(params):
                q_vals = options_network.apply(
                    params, 
                    learn_batch['obs']
                )  # (num_options, batch_size, num_actions)
                chosen_action_qvals = jnp.take_along_axis(
                    q_vals, # type: ignore
                    jnp.expand_dims(learn_batch['action'], axis=(0, -1)), # (1, batch_size, 1)
                    axis=-1,
                ).squeeze(axis=-1)
                losses = jnp.mean((chosen_action_qvals - target) ** 2, axis=-1)
                return jnp.sum(losses), losses

            (total_loss, losses), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            metrics = {
                "losses": losses,
                "updates": train_state.n_updates,
            }

            if config['VERBOSE'] == True:
                def print_callback(metrics):
                    if metrics["updates"] % 500 == 0:
                        jax.debug.print(
                            "updates: {updates}, losses: {losses}",
                            updates=metrics["updates"],
                            losses=metrics["losses"],
                        )
                jax.debug.callback(print_callback, metrics)

            runner_state = (train_state, rng)

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        
        return {"runner_state": runner_state, "metrics": metrics}

    return train

def plot_qvals(network_params, config, save_dir):
    """Performs a forward pass with final params and visualizes Q-values for all locations in the maze for each option."""
    # Create environment
    basic_env, env_params = make(config["ENV_NAME"])
    
    # Get grid dimensions
    N = basic_env.N
    
    # Determine n_options from network_params
    n_options = jax.tree_util.tree_leaves(network_params)[0].shape[0]

    # Create options network
    options_network = make_options_network(
        config, 
        action_dim=basic_env.action_space(env_params).n, 
        n_options=n_options
    )

    # Store Q-values for all agent_location pairs for each option
    q_values_grid = jnp.zeros((n_options, N, N, 4))  # (n_options, N, N, 4 actions)
    
    # Iterate over each location in the grid
    for row in range(N):
        for col in range(N):
            agent_loc = jnp.array([row, col])
            
            # Check if this is a valid location for the agent (not an obstacle)
            is_obstacle = basic_env._obstacles_map[row, col] == 1.0
            is_goal = jnp.array_equal(agent_loc, basic_env.goal_loc)
            
            if not is_obstacle and not is_goal:
                # Create new state with current agent location
                current_state = GridworldEnvState(
                    time=0,
                    agent_loc=agent_loc
                )
                
                # Generate observation for this state
                obs = basic_env.get_obs(current_state, params=env_params)
                obs_batch = jnp.expand_dims(obs, 0) # Add batch dimension
                
                # Forward pass through network for all options
                q_vals_all_options = options_network.apply(network_params, obs_batch) # (n_options, 1, n_actions)
                q_values_grid = q_values_grid.at[:, row, col].set(q_vals_all_options[:, 0, :])

    # Create visualization
    cols = int(math.ceil(math.sqrt(n_options)))
    rows = int(math.ceil(n_options / cols))
    
    base_fig_size = max(8, N * 0.8)
    fig_size_w = min(base_fig_size * cols, 25 * cols)
    fig_size_h = min(base_fig_size * rows, 25 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_size_w, fig_size_h), squeeze=False)
    axes_flat = axes.flat

    q_value_fontsize = max(4, min(10, 80 / N))
    label_fontsize = max(8, min(24, 120 / N))
    edge_linewidth = max(0.3, min(1.0, 8 / N))

    for option_idx in range(n_options):
        ax = axes_flat[option_idx]
        
        # Normalize Q-values for color mapping for this option
        valid_q_values = []
        for row in range(N):
            for col in range(N):
                is_obstacle = basic_env._obstacles_map[row, col] == 1.0
                is_goal = jnp.array_equal(jnp.array([row, col]), basic_env.goal_loc)
                if not is_obstacle and not is_goal:
                    valid_q_values.extend(q_values_grid[option_idx, row, col])
        
        if valid_q_values:
            q_min, q_max = min(valid_q_values), max(valid_q_values)
            q_range = q_max - q_min if q_max > q_min else 1.0
        else:
            q_min, q_max, q_range = 0, 1, 1

        # Draw the grid for this option
        for row in range(N):
            for col in range(N):
                agent_loc = jnp.array([row, col])
                is_obstacle = basic_env._obstacles_map[row, col] == 1.0
                is_goal = jnp.array_equal(agent_loc, basic_env.goal_loc)
                
                plot_row = N - 1 - row
                
                if is_obstacle:
                    rect = patches.Rectangle((col, plot_row), 1, 1, linewidth=edge_linewidth, edgecolor='black', facecolor='black')
                    ax.add_patch(rect)
                elif is_goal:
                    rect = patches.Rectangle((col, plot_row), 1, 1, linewidth=edge_linewidth, edgecolor='black', facecolor='green')
                    ax.add_patch(rect)
                    ax.text(col + 0.5, plot_row + 0.5, 'G', ha='center', va='center', fontsize=label_fontsize, color='white', weight='bold')
                else:
                    q_vals = q_values_grid[option_idx, row, col]
                    
                    triangles = [
                        [(col, plot_row + 1), (col + 1, plot_row + 1), (col + 0.5, plot_row + 0.5)],  # up
                        [(col + 1, plot_row + 1), (col + 1, plot_row), (col + 0.5, plot_row + 0.5)],    # right
                        [(col + 1, plot_row), (col, plot_row), (col + 0.5, plot_row + 0.5)],      # down
                        [(col, plot_row), (col, plot_row + 1), (col + 0.5, plot_row + 0.5)]       # left
                    ]
                    
                    for q_val, triangle_verts in zip(q_vals, triangles):
                        intensity = (q_val - q_min) / q_range if q_range > 0 else 0.5
                        color_intensity = float(max(0.1, min(1.0, intensity)))
                        
                        triangle = patches.Polygon(triangle_verts, closed=True, facecolor=(0, 0, color_intensity, 0.8), edgecolor='black', linewidth=edge_linewidth * 0.5)
                        ax.add_patch(triangle)
                        
                        triangle_center_x = sum(v[0] for v in triangle_verts) / 3
                        triangle_center_y = sum(v[1] for v in triangle_verts) / 3
                        
                        q_text = f'{float(q_val):.2f}'
                        ax.text(triangle_center_x, triangle_center_y, q_text, ha='center', va='center', fontsize=q_value_fontsize, color='white', weight='bold')
                
                rect = patches.Rectangle((col, plot_row), 1, 1, linewidth=edge_linewidth, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
        
        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        title_fontsize = max(10, min(16, 100 / N))
        ax.set_title(f'Option {option_idx} Q-Values', fontsize=title_fontsize)
        ax.grid(True, alpha=0.3, linewidth=edge_linewidth * 0.5)

    for i in range(n_options, len(axes_flat)):
        axes_flat[i].axis('off')

    fig.suptitle(f'Option Action Values for {config["ENV_NAME"]}. \n Stopping Condition: {config["STOPPING_CONDITION"]} \n Bonus weight: {config["BONUS_WEIGHT"]} \n Use last hidden layer: {config["USE_LAST_HIDDEN"]}', fontsize=max(12, min(20, 120 / N)))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'q_vals_{config["ENV_NAME"]}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)

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
        assert params["agent"].lower() == "dqn"

        hypers = params["metaParameters"]

        feature_dir = load_env_dir(hypers['feature_dir'], hypers["env_name"])
        with open(os.path.join(feature_dir, "params.json"), "r") as f:
            feature_params = json.load(f)

        dataset_dir = load_env_dir(hypers['dataset_dir'], hypers["env_name"])
        with open(os.path.join(dataset_dir, "params.json"), "r") as f:
            dataset_params = json.load(f)

        print(f"Specified hypers for idx {idx}:")
        print(json.dumps(hypers, indent=2))
        config = {
            "SEED": hypers["seed"],
            "N_SEEDS": hypers["n_seeds"],
            "ENV_NAME": hypers["env_name"],
            "IMG_OBS": dataset_params["metaParameters"].get("img_obs", True),
            "NUM_UPDATES": hypers["num_updates"],
            "GAMMA": 0.99,
            "LEARNING_RATE": hypers.get("learning_rate", 1e-4),
            "LR_LINEAR_DECAY": hypers.get("lr_linear_decay", False),
            "BATCH_SIZE": hypers.get("batch_size", 32),
            "TAU": hypers.get("tau", 1.0),
            "EPSILON_START": hypers.get("epsilon_start", 0.1),
            "EPSILON_FINISH": hypers.get("epsilon_finish", 0.1),
            "EPSILON_ANNEAL_TIME": hypers.get("epsilon_anneal_time", 1),
            "TARGET_UPDATE_INTERVAL": hypers.get("target_update_interval", 64),
            "CONV1_DIM": hypers.get("conv1_dim", 32),
            "CONV2_DIM": hypers.get("conv2_dim", 16),
            "REP_DIM": hypers.get("rep_dim", 32),
            "ACTIVATION": hypers.get("activation", "relu"),
            "USE_LAST_HIDDEN": hypers.get("use_last_hidden", False),
            "STOPPING_CONDITION": hypers.get("stopping_condition", "stomp"),
            "BONUS_WEIGHT": hypers.get("bonus_weight", 10),
            "FEATURE_PARAMS": feature_params,
            "FEATURE_DIR": feature_dir,
            "DATASET_DIR": dataset_dir,
            "VERBOSE": args.verbose,
        }
        if "percentile" in config["STOPPING_CONDITION"]:
            config["STOPPING_PERCENTILE"] = hypers.get("stopping_percentile", 90)
        
        if config["USE_LAST_HIDDEN"]:
            print(feature_params)
            assert feature_params['agent'].lower() == 'dqn', "Agent must be 'dqn' when using last hidden layer as features"

        dummy_env, env_params = make(config["ENV_NAME"])
        config["ACTION_DIM"] = dummy_env.action_space(env_params).n
        config["OBS_SHAPE"] = dummy_env.observation_space(env_params).shape

        # run = wandb.init(
        #     entity=WANDB_ENTITY,
        #     project=WANDB_PROJECT,
        #     tags=["DQN", "OPTIONS", config["ENV_NAME"].upper()],
        #     name=f'options_dqn_{config["ENV_NAME"]}_{config["ACTIVATION"]}_idx{idx}_{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}',
        #     config=config,
        #     mode=params.get("wandb_mode", "disabled")
        # )

        # Load feature network weights
        with open(os.path.join(config["FEATURE_DIR"], "network_weights.pkl"), "rb") as f:
            feature_net_params = pickle.load(f)
        print(f"Loaded feature network params from {os.path.join(config['FEATURE_DIR'], 'network_weights.pkl')}")

        # Load dataset
        dataset_path = os.path.join(config["DATASET_DIR"], "dataset.npz")
        data = np.load(dataset_path)
        # Assumes all trunc transitions are discarded during collection
        dataset = {
            "obs": jnp.array(data["obs"]),
            "action": jnp.array(data["action"]),
            "next_obs": jnp.array(data["next_obs"]),
            "reward": jnp.array(data["reward"]),
            "done": jnp.array(data["done"]),
        }
        config["DATASET_SIZE"] = dataset["obs"].shape[0]
        print(f"Loaded dataset from {dataset_path}")
        print(f"Dataset shapes - obs: {dataset['obs'].shape}, action: {dataset['action'].shape}, next_obs: {dataset['next_obs'].shape}, reward: {dataset['reward'].shape}, done: {dataset['done'].shape}")

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
            rngs = jax.random.split(rng, config['N_SEEDS'])

            train = make_train(config)
            # Outer vmap over num unique feature_net_weights, inner vmap over seeds
            train_jit = jax.jit(
                jax.vmap(
                    jax.vmap(train, in_axes=(None, None, 0)),
                    in_axes=(0, None, None)
                    )
                )

            # Ahead-of-time compilation
            start_compilation_time = time.time()

            compiled_train = train_jit.lower(
                feature_net_params, 
                dataset, 
                rngs
            ).compile()
            jax.block_until_ready(compiled_train)

            compilation_time_str = get_time_str(time.time() - start_compilation_time)
            print(f"idx {idx} compilation time: {compilation_time_str}")
            f.write(f"idx {idx} compilation time: {compilation_time_str}\n")

            start_time = time.time()

            results = compiled_train(
                feature_net_params,
                dataset,
                rngs
            )
            jax.block_until_ready(results)

            run_time_str = get_time_str(time.time() - start_time)
            print(f"idx {idx} runtime: {run_time_str}")
            f.write(f"idx {idx} runtime: {run_time_str}\n")

        metrics = jax.device_get(results["metrics"])

        losses = metrics["losses"]
        jnp.save(os.path.join(save_dir, "loss.npy"), losses)
        del losses  # Free memory after saving
        
        options_weights = results["runner_state"][0].params
        
        # Plot Q-values if we run for 1 seed
        if config["N_SEEDS"] == 1 and feature_params['metaParameters']['n_seeds'] == 1:
            plotting_weights = jax.tree_util.tree_map(lambda x: x[0][0], options_weights) # remove leading num unique feature_net_weights and seeds dimensions
            if config["ENV_NAME"] in ["Maze", "TwoRooms"]:
                plot_qvals(plotting_weights, config, save_dir)
                # plot_rep_heatmaps(plotting_weights, config, save_dir) # This would need to be adapted for options

        # Save network weights
        if params.get("save_weights", False):
            # Save network weights as pickle
            with open(os.path.join(save_dir, "options_weights.pkl"), "wb") as f:
                pickle.dump(options_weights, f)
            print(f"Saved options network weights for idx {idx}")

        # save params for this idx as JSON
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
        del metrics