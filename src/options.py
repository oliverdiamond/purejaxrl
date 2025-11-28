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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

from environments import make
from environments.gridworld import EnvState as GridworldEnvState, Gridworld
from util import get_time_str, WANDB_ENTITY, WANDB_PROJECT
from util.fta import fta
from experiment import experiment_model
from dqn import QNet, QNet2, QNetFTA, QNetLinear

def make_stopping_condition(
    config, 
    feature_network, 
    main_network, 
    feature_network_params, 
    main_network_params, 
    get_features, 
    options_network, 
    dataset, 
    n_options,
    feature_idxs
    ):
    #TODO Determine if feature should be maximized or minimized based on the sign of the derivative (so we can do "avoidance" for features with negative outgoing weights)
    if "replacement" in config["STOPPING_CONDITION"]:
        assert config["USE_LAST_HIDDEN"] == True, "replacement stopping val needs features to be from last hidden layer"
        main_net_params_batched = jax.tree_util.tree_map(
            lambda arr: jnp.stack([arr] * n_options),
            main_network_params
        )
        main_net_params_batched = unfreeze(main_net_params_batched)
        new_output_weights = main_net_params_batched['params']['output']['kernel'].at[jnp.arange(n_options), feature_idxs, :].set(config["BONUS_WEIGHT"])
        main_net_params_batched['params']['output']['kernel'] = new_output_weights
        main_net_params_batched = freeze(main_net_params_batched)
        vmap_main_net_apply = jax.vmap(
                lambda params, inputs: main_network.apply(params, inputs),
                in_axes=(0, None)
            )
        get_action_vals_with_bonus = lambda obs: vmap_main_net_apply(main_net_params_batched, obs)  # (n_features, batch_size, n_actions)
        
    if "percentile" in config["STOPPING_CONDITION"]:
        dataset_features = get_features(
            feature_network_params, 
            dataset["obs"]
        ) # (dataset_size, n_features)
        percentile = jnp.percentile(
            dataset_features, 
            config["STOPPING_PERCENTILE"], 
            axis=0
        ) # (n_features,)

    if "zscore" in config["STOPPING_CONDITION"]:
        dataset_features = get_features(
            feature_network_params, 
            dataset["obs"]
        ) # (dataset_size, n_features)
        feature_mean = jnp.mean(dataset_features, axis=0)  # (n_features,)
        feature_std = jnp.std(dataset_features, axis=0)  # (n_features,)
        zscore_threshold = feature_mean + 2.0 * feature_std  # (n_features,)

    if config["STOPPING_CONDITION"] == "stomp_replacement":
        def stomp_replacement_stop_cond(obs, params):
            greedy_action_idxs = main_network.apply(main_network_params, obs).argmax(axis=-1)  # (batch_size,)
            action_vals_with_bonus = get_action_vals_with_bonus(obs)  # (n_features, batch_size, n_actions)
            state_vals_with_bonus = action_vals_with_bonus[:, jnp.arange(obs.shape[0]), greedy_action_idxs]  # (n_features, batch_size)
            option_state_vals = options_network.apply(params, obs).max(axis=-1)  # (n_features, batch_size)
            stop = (state_vals_with_bonus >= option_state_vals).astype(jnp.int32)  # (n_features, batch_size)

            return stop, state_vals_with_bonus
        
        return stomp_replacement_stop_cond

    if config["STOPPING_CONDITION"] == "stomp_addition":
        def stomp_addition_stop_cond(obs, params):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            state_val = main_network.apply(main_network_params, obs).max(axis=-1)  # (batch_size,)
            state_val = jnp.tile(state_val, (n_options, 1))  # (n_features, batch_size)
            stop_bonus = jnp.transpose(config["BONUS_WEIGHT"] * obs_features) # (n_features, batch_size)
            stop_val = state_val + stop_bonus
            option_state_val = options_network.apply(params, obs).max(axis=-1)  # (n_features, batch_size)
            stop = (stop_val >= option_state_val).astype(jnp.int32)  # (n_features, batch_size)

            return stop, stop_val
        
        return stomp_addition_stop_cond

    elif config["STOPPING_CONDITION"] == "stomp_no_val":
        def stomp_no_val_stop_cond(obs, params):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop_val = jnp.transpose(config["BONUS_WEIGHT"] * obs_features) # (n_features, batch_size)
            option_state_vals = options_network.apply(params, obs).max(axis=-1)  # (n_features, batch_size)
            stop = (stop_val >= option_state_vals).astype(jnp.int32)  # (n_features, batch_size)
            return stop, stop_val

        return stomp_no_val_stop_cond

    if config["STOPPING_CONDITION"] == "percentile_replacement":
        def percentile_replacement_stop_cond(obs, params=None):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop = jnp.transpose(obs_features >= percentile).astype(jnp.int32)  # (n_features, batch_size)
            greedy_action_idxs = main_network.apply(main_network_params, obs).argmax(axis=-1)  # (batch_size,)
            action_vals_with_bonus = get_action_vals_with_bonus(obs)  # (n_features, batch_size, n_actions)
            state_vals_with_bonus = action_vals_with_bonus[:, jnp.arange(obs.shape[0]), greedy_action_idxs]  # (n_features, batch_size)
            
            return stop, state_vals_with_bonus
        
        return percentile_replacement_stop_cond

    elif config["STOPPING_CONDITION"] == "percentile_addition":
        def percentile_addition_stop_cond(obs, params=None):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop = jnp.transpose(obs_features >= percentile).astype(jnp.int32)  # (n_features, batch_size)
            state_val = main_network.apply(main_network_params, obs).max(axis=-1)  # (batch_size,)
            state_val = jnp.tile(state_val, (n_options, 1))  # (n_features, batch_size)
            stop_bonus = jnp.transpose(config["BONUS_WEIGHT"] * obs_features) # (n_features, batch_size)
            stop_val = state_val + stop_bonus

            return stop, stop_val
        
        return percentile_addition_stop_cond


    elif config["STOPPING_CONDITION"] == "percentile_no_val":
        #TODO Figure out what the stopping value should be. Normalized (or unnormalized) feature value with weight coefficient?
        def percentile_no_val_stop_cond(obs, params=None):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop_bonus = jnp.transpose(config["BONUS_WEIGHT"] * obs_features) # (n_features, batch_size)
            stop = jnp.transpose(obs_features >= percentile).astype(jnp.int32)  # (n_features, batch_size)
            
            return stop, stop_bonus

        return percentile_no_val_stop_cond
    
    elif config["STOPPING_CONDITION"] == "percentile_val_only":
        def percentile_val_only_stop_cond(obs, params=None):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop = jnp.transpose(obs_features >= percentile).astype(jnp.int32)  # (n_features, batch_size)
            state_val = main_network.apply(main_network_params, obs).max(axis=-1)  # (batch_size,)
            state_val = jnp.tile(state_val, (n_options, 1))  # (n_features, batch_size)
            return stop, state_val
        
        return percentile_val_only_stop_cond

    elif config["STOPPING_CONDITION"] == "percentile_no_bonus":
        #TODO Figure out what the stopping value should be. Normalized (or unnormalized) feature value with weight coefficient?
        def percentile_no_bonus_stop_cond(obs, params=None):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop = jnp.transpose(obs_features >= percentile).astype(jnp.int32)  # (n_features, batch_size)
            return stop, jnp.zeros_like(stop, dtype=jnp.float32)

        return percentile_no_bonus_stop_cond

    elif config["STOPPING_CONDITION"] == "zscore_replacement":
        def zscore_replacement_stop_cond(obs, params=None):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop = jnp.transpose(obs_features >= zscore_threshold).astype(jnp.int32)  # (n_features, batch_size)
            greedy_action_idxs = main_network.apply(main_network_params, obs).argmax(axis=-1)  # (batch_size,)
            action_vals_with_bonus = get_action_vals_with_bonus(obs)  # (n_features, batch_size, n_actions)
            state_vals_with_bonus = action_vals_with_bonus[:, jnp.arange(obs.shape[0]), greedy_action_idxs]  # (n_features, batch_size)

            return stop, state_vals_with_bonus
        
        return zscore_replacement_stop_cond

    elif config["STOPPING_CONDITION"] == "zscore_addition":
        def zscore_addition_stop_cond(obs, params=None):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop = jnp.transpose(obs_features >= zscore_threshold).astype(jnp.int32)  # (n_features, batch_size)
            state_val = main_network.apply(main_network_params, obs).max(axis=-1)  # (batch_size,)
            state_val = jnp.tile(state_val, (n_options, 1))  # (n_features, batch_size)
            stop_bonus = jnp.transpose(config["BONUS_WEIGHT"] * obs_features) # (n_features, batch_size)
            stop_val = state_val + stop_bonus

            return stop, stop_val
        
        return zscore_addition_stop_cond

    elif config["STOPPING_CONDITION"] == "zscore_no_val":
        def zscore_no_val_stop_cond(obs, params=None):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop_bonus = jnp.transpose(config["BONUS_WEIGHT"] * obs_features) # (n_features, batch_size)
            stop = jnp.transpose(obs_features >= zscore_threshold).astype(jnp.int32)  # (n_features, batch_size)
            
            return stop, stop_bonus

        return zscore_no_val_stop_cond
    
    elif config["STOPPING_CONDITION"] == "zscore_val_only":
        def zscore_val_only_stop_cond(obs, params=None):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop = jnp.transpose(obs_features >= zscore_threshold).astype(jnp.int32)  # (n_features, batch_size)
            state_val = main_network.apply(main_network_params, obs).max(axis=-1)  # (batch_size,)
            state_val = jnp.tile(state_val, (n_options, 1))  # (n_features, batch_size)
            return stop, state_val
        
        return zscore_val_only_stop_cond

    elif config["STOPPING_CONDITION"] == "zscore_no_bonus":
        def zscore_no_bonus_stop_cond(obs, params=None):
            obs_features = get_features(feature_network_params, obs) # (batch_size, n_features)
            stop = jnp.transpose(obs_features >= zscore_threshold).astype(jnp.int32)  # (n_features, batch_size)
            return stop, jnp.zeros_like(stop, dtype=jnp.float32)
        
        return zscore_no_bonus_stop_cond
    
    else:
        raise ValueError(f"Unknown stopping condition: {config['STOPPING_CONDITION']}")
    
def make_options_network(config, action_dim, n_options):
    """Returns the appropriate network based on the configuration."""
    if config["NETWORK_NAME"] == "QNet":
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
    elif config["NETWORK_NAME"] == "QNetLinear":
        return nn.vmap(
            QNetLinear,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=(None,),
            out_axes=0,
            axis_size=n_options,
            methods=['__call__']
        )(action_dim=action_dim)
    elif config["NETWORK_NAME"] == "QNet2":
        return nn.vmap(
            QNet2,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=(None,),
            out_axes=0,
            axis_size=n_options,
            methods=['__call__']
        )(action_dim=action_dim)
    else:
        raise ValueError("Option learning currently only supports ReLU activations")
    
def make_feature_network(config, action_dim):
    """Returns the appropriate network based on the configuration."""
    feature_net_type = config["FEATURE_PARAMS"]["agent"].lower()
    feature_net_hypers = config["FEATURE_PARAMS"]["metaParameters"]
    if feature_net_type == "dqn":
        if feature_net_hypers["network_name"] == "QNet":
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
        elif feature_net_hypers["network_name"] == "QNetLinear":
            return QNetLinear(
                action_dim=action_dim
            )
        elif feature_net_hypers["network_name"] == "QNet2":
            return QNet2(
            action_dim=action_dim
        )
    else:
        raise ValueError(f"Unknown feature network: {feature_net_type}")

def make_main_network(config, action_dim):
    """Returns the appropriate network based on the configuration."""
    main_net_type = config["MAIN_PARAMS"]["agent"].lower()
    main_net_hypers = config["MAIN_PARAMS"]["metaParameters"]
    if main_net_type == "dqn":
        if main_net_hypers["network_name"] == "QNet":
            if main_net_hypers["activation"] == "relu":
                return QNet(
                    action_dim=action_dim,
                    conv1_dim=main_net_hypers.get("conv1_dim", 32),
                    conv2_dim=main_net_hypers.get("conv2_dim", 16),
                    rep_dim=main_net_hypers.get("rep_dim", 32),
                    head_hidden_dim=main_net_hypers.get("head_hidden_dim", 64)
                )
            elif main_net_hypers["activation"] == "fta":
                return QNetFTA(
                    action_dim=action_dim,
                    conv1_dim=main_net_hypers.get('conv1_dim', 32),
                    conv2_dim=main_net_hypers.get('conv2_dim', 16),
                    rep_dim=main_net_hypers.get('rep_dim', 32),
                    head_hidden_dim=main_net_hypers.get("head_hidden_dim", 64),
                    fta_eta=main_net_hypers.get('fta_eta', 2),
                    fta_tiles=main_net_hypers.get("fta_tiles", 20),
                    fta_lower_bound=main_net_hypers.get("fta_lower_bound", -20.0),
                    fta_upper_bound=main_net_hypers.get("fta_upper_bound", 20.0),
                )        
        elif main_net_hypers["network_name"] == "QNetLinear":
            return QNetLinear(
                action_dim=action_dim
            )
        elif main_net_hypers["network_name"] == "QNet2":
            return QNet2(
            action_dim=action_dim
        )
    else:
        raise ValueError(f"Unknown feature network: {main_net_type}")



class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict # type: ignore
    n_updates: int

def make_train(config):
    def train(feature_net_params, main_net_params, dataset, rng):
        # FEATURE NETWORK
        if config["IMG_OBS"]:
            init_x = jnp.zeros((1,) + config["OBS_SHAPE"])  # Conv layers need extra dimension for batch size
        else:
            init_x = jnp.zeros(config["OBS_SHAPE"])

        feature_network = make_feature_network(config, action_dim=config["ACTION_DIM"])
        main_network = make_main_network(config, action_dim=config["ACTION_DIM"])

        if config['USE_LAST_HIDDEN']:
            get_all_features = feature_network.get_last_hidden # type: ignore
        else:
            get_all_features = feature_network.get_features # type: ignore

        _all_features = feature_network.apply( # type: ignore
            feature_net_params,
            init_x,
            method=get_all_features, # type: ignore
        )

        if config['FEATURE_IDXS'] is not None:
            feature_idxs = jnp.array(config['FEATURE_IDXS'])
        else:
            feature_idxs = jnp.arange(_all_features.shape[-1]) # type: ignore
            
        get_features = lambda params, x: feature_network.apply( # type: ignore
            params,
            x,
            method=get_all_features, # type: ignore
        )[:, feature_idxs]  # type: ignore
            
        _features = get_features(feature_net_params, init_x)

        n_options = _features.shape[-1] # type: ignore

        # OPTIONS NETWORK (seperate ff net for each option for now)
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
            main_network,
            feature_net_params, 
            main_net_params,
            get_features, 
            options_network,
            dataset, 
            n_options,
            feature_idxs
        )

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LEARNING_RATE"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LEARNING_RATE"]
        if config["OPT"] == 'adam':
            tx = optax.adam(learning_rate=lr)
        elif config["OPT"] == 'sgd':
            tx = optax.sgd(learning_rate=lr)

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
                            "updates: {updates}",#, losses: {losses}",
                            updates=metrics["updates"],
                            #losses=metrics["losses"],
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

def _get_all_observations_vectorized(basic_env, env_params, has_key=None):
    """Generate all valid observations in a vectorized manner.
    
    Returns:
        obs_batch: Array of observations for all valid locations
        locations: List of (row, col) tuples corresponding to each observation
        valid_mask: Boolean array indicating valid (non-obstacle, non-goal) locations
    """
    H = basic_env.H
    W = basic_env.W
    
    # Create list of all locations and check validity
    locations = []
    valid_mask = np.zeros((H, W), dtype=bool)
    
    for row in range(H):
        for col in range(W):
            agent_loc = jnp.array([row, col])
            is_obstacle = basic_env._obstacles_map[row, col] == 1.0
            is_goal = jnp.array_equal(agent_loc, basic_env.goal_loc)
            
            if not is_obstacle and not is_goal:
                locations.append((row, col))
                valid_mask[row, col] = True
    
    # Generate all states and observations in one go
    if len(locations) == 0:
        return jnp.array([]), [], valid_mask
    
    # Create all states
    agent_locs = jnp.array([[row, col] for row, col in locations])
    key_loc = basic_env.fixed_key_loc if (has_key is not None and hasattr(basic_env, 'fixed_key_loc')) else jnp.array([0, 0])
    
    if has_key is not None:
        states = jax.vmap(lambda loc: GridworldEnvState(
            time=0,
            agent_loc=loc,
            has_key=jnp.array(has_key),
            key_loc=key_loc
        ))(agent_locs)
    else:
        states = jax.vmap(lambda loc: GridworldEnvState(
            time=0,
            agent_loc=loc
        ))(agent_locs)
    
    # Vectorized observation generation
    obs_batch = jax.vmap(lambda state: basic_env.get_obs(state, params=env_params))(states)
    
    return obs_batch, locations, valid_mask

def _get_static_locs(basic_env):
    """Pre-calculates lists of static environment features to avoid looping later."""
    H, W = basic_env.H, basic_env.W
    obstacle_locs = []
    penalty_locs = []
    start_locs = []
    
    # Identify key locations once
    for r in range(H):
        for c in range(W):
            if basic_env._obstacles_map[r, c] == 1.0:
                obstacle_locs.append((r, c))
            elif basic_env._penalty_map[r, c] != 0.0 and not np.array_equal([r,c], basic_env.goal_loc):
                penalty_locs.append((r, c))
            
            # Start locations
            agent_loc = jnp.array([r, c])
            is_start = any(jnp.array_equal(agent_loc, sl) for sl in basic_env._start_locs)
            if is_start and basic_env._obstacles_map[r, c] != 1.0:
                start_locs.append((r, c))
                
    return obstacle_locs, penalty_locs, start_locs

def plot_stopping_values(network_params, config, save_dir):
    """Visualizes stopping values for all locations in the maze for each option."""
    # Create environment
    basic_env, env_params = make(config["ENV_NAME"])
    # Check if environment has a key
    has_key_mechanism = hasattr(basic_env, 'fixed_key_loc') and hasattr(basic_env, 'use_fixed_key_loc')
    if has_key_mechanism and basic_env.use_fixed_key_loc:
        for has_key in [False, True]:
            _plot_stopping_values_single(network_params, config, save_dir, basic_env, env_params, has_key)
    else:
        _plot_stopping_values_single(network_params, config, save_dir, basic_env, env_params, None)

def _plot_stopping_values_single(network_params, config, save_dir, basic_env, env_params, has_key):
    """Optimized plotter using imshow and vectorized inference."""
    H = basic_env.H
    W = basic_env.W
    
    # 1. Pre-calculate static lists
    obstacle_locs, penalty_locs, start_locs_list = _get_static_locs(basic_env)

    # Determine n_options from network_params
    n_options = jax.tree_util.tree_leaves(network_params)[0].shape[0]

    # Create options network
    options_network = make_options_network(
        config, 
        action_dim=basic_env.action_space(env_params).n, 
        n_options=n_options
    )

    # Load feature network and dataset for stopping condition
    with open(os.path.join(config["FEATURE_DIR"], "network_weights.pkl"), "rb") as f:
        feature_net_params = pickle.load(f)
    
    # Handle case where feature_net_params might have seed dimension
    feature_net_params = jax.tree_util.tree_map(lambda x: x[0], feature_net_params)
    
    # Load main network parameters
    if config["MAIN_DIR"] != config["FEATURE_DIR"]:
        with open(os.path.join(config["MAIN_DIR"], "network_weights.pkl"), "rb") as f:
            main_net_params = pickle.load(f)
        main_net_params = jax.tree_util.tree_map(lambda x: x[0], main_net_params)
    else:
        main_net_params = jax.tree_util.tree_map(lambda x: x.copy(), feature_net_params)
    
    dataset_path = os.path.join(config["DATASET_DIR"], "dataset.npz")
    data = np.load(dataset_path)
    dataset = {
        "obs": jnp.array(data["obs"]),
        "action": jnp.array(data["action"]),
        "next_obs": jnp.array(data["next_obs"]),
        "reward": jnp.array(data["reward"]),
        "done": jnp.array(data["done"]),
    }
    
    # Create feature network and main network
    feature_network = make_feature_network(config, action_dim=basic_env.action_space(env_params).n)
    main_network = make_main_network(config, action_dim=basic_env.action_space(env_params).n)
    
    if config['USE_LAST_HIDDEN']:
        get_all_features = feature_network.get_last_hidden
    else:
        get_all_features = feature_network.get_features
    
    # Initialize to get feature dimensions
    if config["IMG_OBS"]:
        init_x = jnp.zeros((1,) + config["OBS_SHAPE"])
    else:
        init_x = jnp.zeros(config["OBS_SHAPE"])
    
    _all_features = feature_network.apply(
        feature_net_params,
        init_x,
        method=get_all_features,
    )
    
    if config['FEATURE_IDXS'] is not None:
        feature_idxs = jnp.array(config['FEATURE_IDXS'])
    else:
        feature_idxs = jnp.arange(_all_features.shape[-1])
    
    get_features = lambda params, x: feature_network.apply(
        params,
        x,
        method=get_all_features,
    )[:, feature_idxs]
    
    stop_cond = make_stopping_condition(
        config, 
        feature_network, 
        main_network,
        feature_net_params, 
        main_net_params,
        get_features, 
        options_network,
        dataset, 
        n_options,
        feature_idxs
    )
    stop_cond = jax.jit(stop_cond)
    
    # --- 2. VECTORIZED INFERENCE (Same as before) ---
    obs_batch, locations, valid_mask = _get_all_observations_vectorized(basic_env, env_params, has_key)
    
    # Initialize grids
    stop_val_grid = np.zeros((n_options, H, W))
    stop_grid = np.zeros((n_options, H, W))
    
    if len(locations) > 0:
        if "percentile" in config["STOPPING_CONDITION"]:
            stop_all, stop_val_all = stop_cond(obs_batch)
        else:
            stop_all, stop_val_all = stop_cond(obs_batch, network_params)
            
        # Move to CPU for plotting
        stop_val_all = np.array(stop_val_all)
        stop_all = np.array(stop_all)
        
        # Fill grids
        for idx, (row, col) in enumerate(locations):
            stop_val_grid[:, row, col] = stop_val_all[:, idx]
            stop_grid[:, row, col] = stop_all[:, idx]

    # --- 3. FAST PLOTTING ---
    cols = int(math.ceil(math.sqrt(n_options)))
    rows = int(math.ceil(n_options / cols))
    
    max_dim = max(H, W)
    base_fig_size = max(8, max_dim * 0.8)
    fig_size_w = min(base_fig_size * cols, 25 * cols)
    fig_size_h = min(base_fig_size * rows, 25 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_size_w, fig_size_h), squeeze=False)
    axes_flat = axes.flat

    value_fontsize = max(6, min(14, 90 / max(1, max_dim)))
    label_fontsize = max(8, min(24, 120 / max_dim))
    edge_linewidth = max(0.3, min(1.0, 8 / max_dim))

    cmap = mcolors.LinearSegmentedColormap.from_list("stopping_red", ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"])

    for option_idx in range(n_options):
        ax = axes_flat[option_idx]
        
        current_stop_vals = stop_val_grid[option_idx]
        
        # Normalize
        valid_vals = current_stop_vals[valid_mask]
        if valid_vals.size:
            v_min, v_max = float(valid_vals.min()), float(valid_vals.max())
            v_range = 1.0 if math.isclose(v_max, v_min) else v_max - v_min
        else:
            v_min, v_max, v_range = 0.0, 1.0, 1.0

        # Prepare Mask for imshow (Flip UD for correct orientation)
        obs_mask = basic_env._obstacles_map.astype(bool)
        goal_mask = np.zeros((H, W), dtype=bool)
        goal_mask[int(basic_env.goal_loc[0]), int(basic_env.goal_loc[1])] = True
        total_mask = obs_mask | goal_mask
        
        data_flipped = np.flipud(current_stop_vals)
        mask_flipped = np.flipud(total_mask)
        masked_data = np.ma.array(data_flipped, mask=mask_flipped)

        # Plot Heatmap
        im = ax.imshow(masked_data, cmap=cmap, vmin=v_min, vmax=v_max, 
                       origin='lower', interpolation='nearest', extent=[0, W, 0, H])

        # --- Draw Overlays (Vectorized-ish) ---
        
        # Obstacles
        for r, c in obstacle_locs:
            plot_row = H - 1 - r
            rect = patches.Rectangle((c, plot_row), 1, 1, linewidth=edge_linewidth, 
                                    edgecolor='black', facecolor='grey')
            ax.add_patch(rect)
            
        # Goal
        goal_row, goal_col = int(basic_env.goal_loc[0]), int(basic_env.goal_loc[1])
        plot_goal_row = H - 1 - goal_row
        rect = patches.Rectangle((goal_col, plot_goal_row), 1, 1, linewidth=edge_linewidth, 
                                edgecolor='black', facecolor='green')
        ax.add_patch(rect)
        ax.text(goal_col + 0.5, plot_goal_row + 0.5, 'G', ha='center', va='center', 
               fontsize=label_fontsize, color='white', weight='bold')

        # Text Values
        for idx, (row, col) in enumerate(locations):
            val = current_stop_vals[row, col]
            plot_row = H - 1 - row
            intensity = (val - v_min) / v_range if v_range > 0 else 0.5
            text_color = "white" if intensity > 0.6 else "black"
            ax.text(col + 0.5, plot_row + 0.5, f'{val:.2f}', 
                   ha='center', va='center', fontsize=value_fontsize, color=text_color)

        # Green Outlines for Stopping
        # Get coordinates where stopping is true
        stop_rows, stop_cols = np.where(stop_grid[option_idx] == 1)
        for r, c in zip(stop_rows, stop_cols):
             plot_row = H - 1 - r
             rect_outline = patches.Rectangle((c, plot_row), 1, 1, linewidth=edge_linewidth * 3, 
                                             edgecolor='lime', facecolor='none', zorder=10)
             ax.add_patch(rect_outline)

        # Penalties
        for r, c in penalty_locs:
            plot_row = H - 1 - r
            dot_size = max(10, min(50, 200 / max(1, max_dim)))
            ax.scatter(c + 0.85, plot_row + 0.85, s=dot_size, c='yellow', 
                      edgecolors='black', linewidths=edge_linewidth * 0.5, zorder=10)

        # Starts
        for r, c in start_locs_list:
            plot_row = H - 1 - r
            ax.text(c + 0.85, plot_row + 0.85, 'S', ha='center', va='center', 
                   fontsize=label_fontsize, color='green', weight='bold', zorder=11)
                   
        # Key
        if has_key is not None and not has_key and hasattr(basic_env, 'fixed_key_loc'):
            key_row, key_col = basic_env.fixed_key_loc
            plot_key_row = H - 1 - key_row
            ax.text(key_col + 0.5, plot_key_row + 0.5, 'K', ha='center', va='center', 
                    fontsize=label_fontsize * 1.2, color='gold', weight='bold', zorder=15,
                    bbox=dict(boxstyle='circle', facecolor='black', edgecolor='gold', linewidth=1.5))

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        title_fontsize = max(10, min(16, 100 / max_dim))
        ax.set_title(f'Option {option_idx} Stopping Values', fontsize=title_fontsize)

    for i in range(n_options, len(axes_flat)):
        axes_flat[i].axis('off')
        
    # [Rest of saving logic remains the same]
    if has_key is not None:
        key_state_str = "with key" if has_key else "without key"
        title_suffix = f" ({key_state_str})"
    else:
        title_suffix = ""
    
    fig.suptitle(f'Stopping Values for {config["ENV_NAME"]}{title_suffix}.', fontsize=max(12, min(20, 120 / max_dim)))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    
    if has_key is not None:
        key_suffix = "_with_key" if has_key else "_without_key"
        save_path = os.path.join(save_dir, f'stopping_vals_{config["ENV_NAME"]}{key_suffix}.png')
    else:
        save_path = os.path.join(save_dir, f'stopping_vals_{config["ENV_NAME"]}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_qvals(network_params, config, save_dir):
    """Performs a forward pass with final params and visualizes Q-values for all locations in the maze for each option."""
    # Create environment
    basic_env, env_params = make(config["ENV_NAME"])
    
    # Check if environment has a key
    has_key_mechanism = hasattr(basic_env, 'fixed_key_loc') and hasattr(basic_env, 'use_fixed_key_loc')
    
    if has_key_mechanism and basic_env.use_fixed_key_loc:
        # Plot for both has_key=False and has_key=True
        for has_key in [False, True]:
            _plot_qvals_single(network_params, config, save_dir, basic_env, env_params, has_key)
    else:
        # Plot without key consideration
        _plot_qvals_single(network_params, config, save_dir, basic_env, env_params, None)

def _plot_qvals_single(network_params, config, save_dir, basic_env, env_params, has_key):
    """Optimized Q-value plotter using Quiver for arrows."""
    H = basic_env.H
    W = basic_env.W
    
    # 1. Pre-calculate static lists
    obstacle_locs, penalty_locs, start_locs_list = _get_static_locs(basic_env)

    # Determine n_options from network_params
    n_options = jax.tree_util.tree_leaves(network_params)[0].shape[0]

    # Create options network
    options_network = make_options_network(
        config, 
        action_dim=basic_env.action_space(env_params).n, 
        n_options=n_options
    )

    # Load feature network and dataset for stopping condition
    with open(os.path.join(config["FEATURE_DIR"], "network_weights.pkl"), "rb") as f:
        feature_net_params = pickle.load(f)
    
    # Handle seed dimension
    feature_net_params = jax.tree_util.tree_map(lambda x: x[0], feature_net_params)
    
    # Load main network parameters
    if config["MAIN_DIR"] != config["FEATURE_DIR"]:
        with open(os.path.join(config["MAIN_DIR"], "network_weights.pkl"), "rb") as f:
            main_net_params = pickle.load(f)
        main_net_params = jax.tree_util.tree_map(lambda x: x[0], main_net_params)
    else:
        main_net_params = jax.tree_util.tree_map(lambda x: x.copy(), feature_net_params)
    
    dataset_path = os.path.join(config["DATASET_DIR"], "dataset.npz")
    data = np.load(dataset_path)
    dataset = {
        "obs": jnp.array(data["obs"]),
        "action": jnp.array(data["action"]),
        "next_obs": jnp.array(data["next_obs"]),
        "reward": jnp.array(data["reward"]),
        "done": jnp.array(data["done"]),
    }
    
    # Create feature network and main network
    feature_network = make_feature_network(config, action_dim=basic_env.action_space(env_params).n)
    main_network = make_main_network(config, action_dim=basic_env.action_space(env_params).n)
    
    if config['USE_LAST_HIDDEN']:
        get_all_features = feature_network.get_last_hidden
    else:
        get_all_features = feature_network.get_features
    
    # Initialize to get feature dimensions
    if config["IMG_OBS"]:
        init_x = jnp.zeros((1,) + config["OBS_SHAPE"])
    else:
        init_x = jnp.zeros(config["OBS_SHAPE"])
    
    _all_features = feature_network.apply(
        feature_net_params,
        init_x,
        method=get_all_features,
    )
    
    if config['FEATURE_IDXS'] is not None:
        feature_idxs = jnp.array(config['FEATURE_IDXS'])
    else:
        feature_idxs = jnp.arange(_all_features.shape[-1])
    
    get_features = lambda params, x: feature_network.apply(
        params,
        x,
        method=get_all_features,
    )[:, feature_idxs]
    
    stop_cond = make_stopping_condition(
        config, 
        feature_network, 
        main_network,
        feature_net_params, 
        main_net_params,
        get_features, 
        options_network,
        dataset, 
        n_options,
        feature_idxs
    )
    stop_cond = jax.jit(stop_cond)
    get_options_qvals = jax.jit(options_network.apply)

    # 2. VECTORIZED INFERENCE
    obs_batch, locations, valid_mask = _get_all_observations_vectorized(basic_env, env_params, has_key)
    
    q_values_grid = np.zeros((n_options, H, W, 4))
    stop_grid = np.zeros((n_options, H, W))
    
    if len(locations) > 0:
        q_vals_all_options = get_options_qvals(network_params, obs_batch)
        
        # Helper for stop cond (mocking the call logic slightly for brevity)
        if "percentile" in config["STOPPING_CONDITION"]:
            stop_all, _ = stop_cond(obs_batch)
        else:
            stop_all, _ = stop_cond(obs_batch, network_params)
            
        q_vals_all_options = np.array(q_vals_all_options)
        stop_all = np.array(stop_all)

        for idx, (row, col) in enumerate(locations):
            q_values_grid[:, row, col] = q_vals_all_options[:, idx, :]
            stop_grid[:, row, col] = stop_all[:, idx]

    # 3. PLOTTING
    cols = int(math.ceil(math.sqrt(n_options)))
    rows = int(math.ceil(n_options / cols))
    
    max_dim = max(H, W)
    base_fig_size = max(8, max_dim * 0.8)
    fig_size_w = min(base_fig_size * cols, 25 * cols)
    fig_size_h = min(base_fig_size * rows, 25 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_size_w, fig_size_h), squeeze=False)
    axes_flat = axes.flat

    q_value_fontsize = max(4, min(10, 80 / max_dim))
    label_fontsize = max(8, min(24, 120 / max_dim))
    edge_linewidth = max(0.3, min(1.0, 8 / max_dim))

    # Directions for arrows (Up, Right, Down, Left)
    # (U, V) components
    arrow_dirs = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)])

    for option_idx in range(n_options):
        print("Plotting Q-values for option", option_idx)
        ax = axes_flat[option_idx]
        current_q_grid = q_values_grid[option_idx]
        
        # Calculate range
        valid_q = current_q_grid[valid_mask]
        if valid_q.size:
            q_min, q_max = float(valid_q.min()), float(valid_q.max())
            q_range = 1.0 if math.isclose(q_max, q_min) else q_max - q_min
        else:
            q_min, q_max, q_range = 0, 1, 1

        # Draw Obstacles (Vectorized)
        for r, c in obstacle_locs:
            plot_row = H - 1 - r
            rect = patches.Rectangle((c, plot_row), 1, 1, linewidth=edge_linewidth, edgecolor='black', facecolor='black')
            ax.add_patch(rect)

        # Draw Goal
        goal_row, goal_col = int(basic_env.goal_loc[0]), int(basic_env.goal_loc[1])
        plot_goal_row = H - 1 - goal_row
        rect = patches.Rectangle((goal_col, plot_goal_row), 1, 1, linewidth=edge_linewidth, edgecolor='black', facecolor='green')
        ax.add_patch(rect)
        ax.text(goal_col + 0.5, plot_goal_row + 0.5, 'G', ha='center', va='center', fontsize=label_fontsize, color='white', weight='bold')

        # PREPARE ARROW DATA FOR QUIVER (Batching Arrows)
        quiver_X, quiver_Y, quiver_U, quiver_V = [], [], [], []

        # Draw Triangles and Prepare Arrows (Only loop valid locs)
        for row, col in locations:
            plot_row = H - 1 - row
            q_vals = current_q_grid[row, col]
            
            # Draw 4 triangles
            triangles = [
                [(col, plot_row + 1), (col + 1, plot_row + 1), (col + 0.5, plot_row + 0.5)], # up
                [(col + 1, plot_row + 1), (col + 1, plot_row), (col + 0.5, plot_row + 0.5)],   # right
                [(col + 1, plot_row), (col, plot_row), (col + 0.5, plot_row + 0.5)],     # down
                [(col, plot_row), (col, plot_row + 1), (col + 0.5, plot_row + 0.5)]      # left
            ]
            
            for i, (q_val, verts) in enumerate(zip(q_vals, triangles)):
                intensity = (q_val - q_min) / q_range if q_range > 0 else 0.5
                color_intensity = float(max(0.1, min(1.0, intensity)))
                
                # We still add patches individually here as PolyCollection is complex with text
                # But we've stripped all other logic out of this loop
                poly = patches.Polygon(verts, closed=True, facecolor=(0, 0, color_intensity, 0.8), 
                                     edgecolor='black', linewidth=edge_linewidth * 0.5)
                ax.add_patch(poly)
                
                # Text
                cx = sum(v[0] for v in verts) / 3
                cy = sum(v[1] for v in verts) / 3
                ax.text(cx, cy, f'{q_val:.2f}', ha='center', va='center', fontsize=q_value_fontsize, color='white', weight='bold')

            # Collect Arrow Data
            q_rounded = np.round(q_vals, 4)
            best_actions = np.where(q_rounded == q_rounded.max())[0]
            
            for action_idx in best_actions:
                u, v = arrow_dirs[action_idx]
                quiver_X.append(col + 0.5)
                quiver_Y.append(plot_row + 0.5)
                quiver_U.append(u)
                quiver_V.append(v)

        # PLOT ALL ARROWS AT ONCE (Quiver)
        if quiver_X:
            ax.quiver(quiver_X, quiver_Y, quiver_U, quiver_V, 
                     color='orange', scale=None, scale_units='xy', angles='xy',
                     width=0.015, headwidth=4, headlength=4, pivot='mid', zorder=12, alpha=0.8)

        # Overlays
        stop_rows, stop_cols = np.where(stop_grid[option_idx] == 1)
        for r, c in zip(stop_rows, stop_cols):
             plot_row = H - 1 - r
             rect = patches.Rectangle((c, plot_row), 1, 1, linewidth=edge_linewidth * 3, 
                                     edgecolor='lime', facecolor='none', zorder=10)
             ax.add_patch(rect)

        for r, c in penalty_locs:
            plot_row = H - 1 - r
            ax.scatter(c + 0.85, plot_row + 0.85, s=max(20, min(100, 400/max_dim)), c='yellow', edgecolors='black', zorder=10)
            
        for r, c in start_locs_list:
            plot_row = H - 1 - r
            ax.text(c + 0.85, plot_row + 0.85, 'S', ha='center', va='center', fontsize=label_fontsize, color='green', weight='bold', zorder=11)

        if has_key is not None and not has_key and hasattr(basic_env, 'fixed_key_loc'):
            key_row, key_col = basic_env.fixed_key_loc
            plot_key_row = H - 1 - key_row
            ax.text(key_col + 0.5, plot_key_row + 0.5, 'K', ha='center', va='center', 
                    fontsize=label_fontsize * 1.2, color='gold', weight='bold', zorder=15,
                    bbox=dict(boxstyle='circle', facecolor='black', edgecolor='gold', linewidth=1.5))

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Option {option_idx} Q-Values', fontsize=max(10, min(16, 100 / max_dim)))

    for i in range(n_options, len(axes_flat)):
        axes_flat[i].axis('off')
        
    # [Rest of saving logic remains the same]
    if has_key is not None:
        key_state_str = "with key" if has_key else "without key"
        title_suffix = f" ({key_state_str})"
    else:
        title_suffix = ""
    
    fig.suptitle(f'Option Action Values for {config["ENV_NAME"]}{title_suffix}', fontsize=max(12, min(20, 120 / max_dim)))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    
    if has_key is not None:
        key_suffix = "_with_key" if has_key else "_without_key"
        save_path = os.path.join(save_dir, f'q_vals_{config["ENV_NAME"]}{key_suffix}.png')
    else:
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

        if 'main_dir' in hypers:
            main_dir = load_env_dir(hypers['main_dir'], hypers["env_name"])
            with open(os.path.join(main_dir, "params.json"), "r") as f:
                main_params = json.load(f)
        else:
            main_dir = feature_dir
            main_params = feature_params

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
            "NETWORK_NAME": hypers["network_name"],
            "OPT": hypers.get("opt", "adam"),
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
            "HEAD_HIDDEN_DIM": hypers.get("head_hidden_dim", 64),
            "ACTIVATION": hypers.get("activation", "relu"),
            "USE_LAST_HIDDEN": hypers.get("use_last_hidden", False),
            "STOPPING_CONDITION": hypers.get("stopping_condition", "stomp"),
            "BONUS_WEIGHT": hypers.get("bonus_weight", 10),
            "FEATURE_PARAMS": feature_params,
            "FEATURE_DIR": feature_dir,
            "MAIN_PARAMS": main_params,
            "MAIN_DIR": main_dir,
            "FEATURE_IDXS": hypers.get("feature_idxs", None),
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

        # Load main network weights
        if config["MAIN_DIR"] != config["FEATURE_DIR"]:
            with open(os.path.join(config["MAIN_DIR"], "network_weights.pkl"), "rb") as f:
                main_net_params = pickle.load(f)
            print(f"Loaded main network params from {os.path.join(config['MAIN_DIR'], 'network_weights.pkl')}")
        else:
            main_net_params = jax.tree_util.tree_map(lambda x: x.copy(), feature_net_params)
            print("Main network params are the same as feature network params.")

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
                    jax.vmap(train, in_axes=(None, None, None, 0)),
                    in_axes=(0, 0, None, None)
                    )
                )

            # Ahead-of-time compilation
            start_compilation_time = time.time()

            compiled_train = train_jit.lower(
                feature_net_params, 
                main_net_params,
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
                main_net_params,
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
            base_env, _ = make(config["ENV_NAME"])
            if isinstance(base_env, Gridworld):
                plot_qvals(plotting_weights, config, save_dir)
                # plot_stopping_values(plotting_weights, config, save_dir)
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