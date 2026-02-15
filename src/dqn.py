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
from environments.gridworld import EnvState as GridworldEnvState, Gridworld
from util import get_time_str, WANDB_ENTITY, WANDB_PROJECT
from util.fta import fta
from experiment import experiment_model

class QNet(nn.Module):
    action_dim: int
    conv1_dim: int = 32
    conv2_dim: int = 16
    rep_dim: int = 32
    head_hidden_dim: int = 64

    def setup(self):
        w_conv_init = variance_scaling(scale=math.sqrt(5), mode='fan_avg', distribution='uniform')
        w_init = variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')

        # Conv Backbone
        self.conv1 = nn.Conv(
            features=self.conv1_dim, kernel_size=(4, 4), strides=(1, 1), padding=[(1, 1), (1, 1)],
            kernel_init=w_conv_init, name="conv1"
        )
        self.conv2 = nn.Conv(
            features=self.conv2_dim, kernel_size=(4, 4), strides=(2, 2), padding=[(2, 2), (2, 2)],
            kernel_init=w_conv_init, name="conv2"
        )
        self.conv_backbone = nn.Sequential([
            self.conv1,
            nn.relu,
            self.conv2,
            nn.relu
        ], name="conv_backbone")

        self.rep = nn.Sequential([
            nn.Dense(self.rep_dim, kernel_init=w_init, name="linear"),
            nn.relu
        ], name="rep")

        self.head = nn.Sequential(
            [
            nn.Dense(self.head_hidden_dim, kernel_init=w_init, name="post_rep_hidden1"),
            nn.relu,
            nn.Dense(self.head_hidden_dim, kernel_init=w_init, name="post_rep_hidden2"),
            nn.relu,
            nn.Dense(self.action_dim, kernel_init=w_init, name="output")
            ],
            name="head",
        )

    def __call__(self, x: jnp.ndarray):
        # conv backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Representation layer
        rep = self.rep(x)
        # q values
        q = self.head(rep)
        return q

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        # conv backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Representation layer
        rep = self.rep(x)
        # q values
        q = self.head(rep)
        
        return {
            "rep": rep,
            "q_values": q
        }

    def get_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns rep."""
        # conv backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Representation layer
        rep = self.rep(x)

        return rep
    
    def get_last_hidden(self, x: jnp.ndarray) -> jnp.ndarray:
        "Returns last hidden layer"
        # conv backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Representation layer
        x = self.rep(x)
        # Manually apply layers of the head to get the last hidden layer
        x = self.head.layers[0](x)
        x = self.head.layers[1](x)
        x = self.head.layers[2](x)
        last_hidden = self.head.layers[3](x)

        return last_hidden
    
class QNet2(nn.Module):
    action_dim: int
    
    # Adjusted hyperparameters
    conv1_dim: int = 16   # Start smaller
    conv2_dim: int = 32   # Increase depth as image shrinks
    dense_dim: int = 256  # Wide enough to hold spatial info

    def setup(self):
        w_conv_init = nn.initializers.variance_scaling(
            scale=math.sqrt(2), mode='fan_avg', distribution='uniform') # He init is usually better for ReLU
        w_init = nn.initializers.variance_scaling(
            scale=1.0, mode='fan_avg', distribution='uniform')

        # Conv Backbone
        self.conv_backbone = nn.Sequential([
            # Conv 1: Capture local walls/paths
            nn.Conv(features=self.conv1_dim, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_init=w_conv_init),
            nn.relu,
            # Conv 2: Downsample slightly but increase feature richness
            nn.Conv(features=self.conv2_dim, kernel_size=(3, 3), strides=(2, 2), padding='SAME', kernel_init=w_conv_init),
            nn.relu
        ])

        # Combined Head (Simpler and wider)
        self.head = nn.Sequential([
            nn.Dense(self.dense_dim, kernel_init=w_init),
            nn.relu,
            nn.Dense(self.action_dim, kernel_init=w_init)
        ])

    def __call__(self, x: jnp.ndarray):
        # 1. Conv Backbone
        x = self.conv_backbone(x)
        
        # 2. Flatten 
        # Output will be roughly 8x8x32 = 2048 features
        x = x.reshape((x.shape[0], -1)) 
        
        # 3. Q-values
        # We feed 2048 -> 256. This is a healthy 8:1 compression, much safer than 32:1
        q = self.head(x)
        return q

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
            """Returns intermediate activations."""
            # 1. Conv Backbone
            x = self.conv_backbone(x)
            x = x.reshape((x.shape[0], -1)) 
            
            # 2. Manual traversal of the head to capture the 'rep'
            # Layer 0: Dense(256)
            x = self.head.layers[0](x)
            # Layer 1: ReLU
            rep = self.head.layers[1](x)
            
            # Layer 2: Output Dense
            q = self.head.layers[2](rep)
            
            return {
                "rep": rep, 
                "q_values": q
            }

    def get_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns the learned representation (output of the 256-unit layer)."""
        # 1. Conv Backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))
        
        # 2. Apply first part of head
        x = self.head.layers[0](x) # Dense
        rep = self.head.layers[1](x) # ReLU

        return rep
    
    def get_last_hidden(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns last hidden layer (Same as features in this shallower net)."""
        # This is functionally identical to get_features in this specific 
        # architecture, but kept separate to maintain your API.
        return self.get_features(x)


class QNetLinear(nn.Module):
    action_dim: int

    def setup(self):
        #NOTE Does not use bias
        self.head = nn.Dense(self.action_dim, use_bias=False, name="output")

    def __call__(self, x: jnp.ndarray):
        q = self.head(x)
        return q

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        q = self.head(x)
        return {
            "rep": x,
            "q_values": q
        }

    def get_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns rep."""
        return x

    def get_last_hidden(self, x: jnp.ndarray) -> jnp.ndarray:
        "Returns last hidden layer"
        return x  # No hidden layers, so just return input


class QNetFTA(nn.Module):
    action_dim: int
    conv1_dim: int = 32
    conv2_dim: int = 16
    rep_dim: int = 32
    head_hidden_dim: int = 64
    fta_eta: float = 2.0
    fta_tiles: int = 20
    fta_lower_bound: float = -20.0
    fta_upper_bound: float = 20.0

    def setup(self):
        w_conv_init = variance_scaling(scale=math.sqrt(5), mode='fan_avg', distribution='uniform')
        w_init = variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')
        fta_activation = partial(fta, eta=self.fta_eta, tiles=self.fta_tiles, lower_bound=self.fta_lower_bound, upper_bound=self.fta_upper_bound)

        # Conv Backbone
        self.conv1 = nn.Conv(
            features=self.conv1_dim, kernel_size=(4, 4), strides=(1, 1), padding=[(1, 1), (1, 1)],
            kernel_init=w_conv_init, name="conv1"
        )
        self.conv2 = nn.Conv(
            features=self.conv2_dim, kernel_size=(4, 4), strides=(2, 2), padding=[(2, 2), (2, 2)],
            kernel_init=w_conv_init, name="conv2"
        )
        self.conv_backbone = nn.Sequential([
            self.conv1,
            nn.relu,
            self.conv2,
            nn.relu
        ], name="conv_backbone")

        self.rep = nn.Sequential([
            nn.Dense(self.rep_dim, kernel_init=w_init, name="linear"),
            fta_activation
        ], name="rep")

        self.head = nn.Sequential(
            [
            nn.Dense(self.head_hidden_dim, kernel_init=w_init, name="post_rep_hidden1"),
            nn.relu,
            nn.Dense(self.head_hidden_dim, kernel_init=w_init, name="post_rep_hidden2"),
            nn.relu,
            nn.Dense(self.action_dim, kernel_init=w_init, name="output")
            ], 
            name="head",
        )

    def __call__(self, x: jnp.ndarray):
        # conv backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Representation layer
        rep = self.rep(x)
        # q values
        q = self.head(rep)
        return q

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        # conv backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Representation layer
        rep = self.rep(x)
        # q values
        q = self.head(rep)
        
        return {
            "rep": rep,
            "q_values": q
        }

    def get_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns rep."""
        # conv backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Representation layer
        rep = self.rep(x)

        return rep
    
    def get_last_hidden(self, x: jnp.ndarray) -> jnp.ndarray:
        "Returns last hidden layer"
        # conv backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Representation layer
        x = self.rep(x)
        # Manually apply layers of the head to get the last hidden layer
        x = self.head.layers[0](x)
        x = self.head.layers[1](x)
        x = self.head.layers[2](x)
        last_hidden = self.head.layers[3](x)

        return last_hidden

class QNet2Bernoulli(nn.Module):
    """
    Implements the Bernoulli Bottleneck using QNet2 architecture style.
    It replaces the standard Dense+ReLU in the middle with Dense+Sigmoid+STE.
    """
    action_dim: int
    conv1_dim: int = 16   # Matches QNet2
    conv2_dim: int = 32   # Matches QNet2
    dense_dim: int = 256  # Matches QNet2

    def setup(self):
        w_conv_init = nn.initializers.variance_scaling(
            scale=math.sqrt(2), mode='fan_avg', distribution='uniform')
        w_init = nn.initializers.variance_scaling(
            scale=1.0, mode='fan_avg', distribution='uniform')

        # 1. Conv Backbone (Same as QNet2)
        self.conv_backbone = nn.Sequential([
            nn.Conv(features=self.conv1_dim, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_init=w_conv_init),
            nn.relu,
            nn.Conv(features=self.conv2_dim, kernel_size=(3, 3), strides=(2, 2), padding='SAME', kernel_init=w_conv_init),
            nn.relu
        ])

        # 2. Encoder to Logits
        # This replaces the first Dense layer of QNet2's head, but outputs logits for sigmoid
        self.encoder_linear = nn.Dense(self.dense_dim, kernel_init=w_init, name="encoder_linear")

        # 3. Readout Head
        # This maps the binary features to Q-values. 
        # Note: In QNet2, there was a ReLU after the dense_dim. Here, the activation IS the bottleneck.
        # We project straight to action_dim.
        self.head = nn.Dense(self.action_dim, kernel_init=w_init, name="output")

    def get_bernoulli_features(self, x: jnp.ndarray):
        # 1. Conv Backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))
        
        # 2. Logits
        logits = self.encoder_linear(x)
        probs = nn.sigmoid(logits)
        
        # 3. Straight-Through Estimator (STE)
        # Forward: Binary (0 or 1)
        # Backward: Gradient of Sigmoid (probs)
        binary_mask = (probs > 0.5).astype(jnp.float32)
        features = probs + jax.lax.stop_gradient(binary_mask - probs)
        
        return features, probs

    def __call__(self, x: jnp.ndarray):
        features, _ = self.get_bernoulli_features(x)
        q = self.head(features)
        return q

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        features, probs = self.get_bernoulli_features(x)
        q = self.head(features)
        
        return {
            "rep": features, # Strictly 0 or 1
            "q_values": q,
            "probs": probs  # Probabilities before STE
        }

    def get_features(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.get_bernoulli_features(x)[0]
    
    def get_last_hidden(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.get_features(x)

def make_network(config, action_dim):
    """Returns the appropriate network based on the configuration."""
    if config["NETWORK_NAME"] == "QNet":
        if config["ACTIVATION"] == "relu":
            return QNet(
                action_dim=action_dim,
                conv1_dim=config["CONV1_DIM"],
                conv2_dim=config["CONV2_DIM"],
                rep_dim=config["REP_DIM"],
                head_hidden_dim=config["HEAD_HIDDEN_DIM"]
            )
        elif config["ACTIVATION"] == "fta":
            return QNetFTA(
                action_dim=action_dim,
                conv1_dim=config["CONV1_DIM"],
                conv2_dim=config["CONV2_DIM"],
                rep_dim=config["REP_DIM"],
                head_hidden_dim=config["HEAD_HIDDEN_DIM"],
                fta_eta=config["FTA_ETA"],
                fta_tiles=config["FTA_TILES"],
                fta_lower_bound=config["FTA_LOWER_BOUND"],
                fta_upper_bound=config["FTA_UPPER_BOUND"]
            )
    elif config["NETWORK_NAME"] == "QNet2":
        if config["ACTIVATION"] == "relu":
            return QNet2(
                action_dim=action_dim
            )
        elif config["ACTIVATION"] == "bernoulli":
            return QNet2Bernoulli(
                action_dim=action_dim,
                dense_dim=config["HEAD_HIDDEN_DIM"]
            )
    elif config["ACTIVATION"] == "linear":
        return QNetLinear(
            action_dim=action_dim
        )
    else:
        raise ValueError(f"Unknown network name: {config['NETWORK_NAME']}")

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    next_obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict # type: ignore
    timesteps: int
    n_updates: int

def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]

    base_env, env_params = make(config["ENV_NAME"])
    env = LogWrapper(base_env, gamma=config["GAMMA"])

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # INIT BUFFER
        buffer = fbx.make_item_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batches=True,
        )
        buffer = buffer.replace( # type: ignore
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        _action = base_env.action_space().sample(rng)
        _last_obs, _last_env_state = base_env.reset(rng, env_params)
        _obs, _, _reward, _done, _ = base_env.step(rng, _last_env_state, _action, env_params)
        _timestep = TimeStep(obs=_last_obs, next_obs=_obs, action=_action, reward=_reward, done=_done) # type: ignore
        buffer_state = buffer.init(_timestep)

        # INIT NETWORK AND OPTIMIZER
        network = make_network(config, action_dim=env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1,) + env.observation_space(env_params).shape) # Conv layers need extra dimension for batch size
        network_params = network.init(_rng, init_x)

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LEARNING_RATE"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LEARNING_RATE"]
        if config["OPT"] == 'adam':
            tx = optax.adam(learning_rate=lr)
        elif config["OPT"] == 'sgd':
            tx = optax.sgd(learning_rate=lr)

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            target_network_params=jax.tree_util.tree_map(lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        # epsilon-greedy exploration
        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(
                rng, 2
            )  # a key for sampling random actions and one for picking
            eps = jnp.clip(  # get epsilon
                (
                    (config["EPSILON_FINISH"] - config["EPSILON_START"])
                    / config["EPSILON_ANNEAL_TIME"]
                )
                * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
            chosed_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape)
                < eps,  # pick the actions that should be random
                jax.random.randint(
                    rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
                ),  # sample random actions,
                greedy_actions,
            )
            return chosed_actions

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, env_state, last_obs, rng = runner_state

            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = network.apply(train_state.params, last_obs)
            action = eps_greedy_exploration(
                rng_a, q_vals, train_state.timesteps
            )  # explore with epsilon greedy_exploration
            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, env_state, action
            )
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, next_obs=obs, action=action, reward=reward, done=done) # type: ignore
            buffer_state = jax.lax.cond(
                info['truncated'].any(),  # if any envs are truncated, do not add to buffer. Note this breaks parallel envs! In the future can change to traj buffer and get half without truncated obs if single env is slow
                lambda: buffer_state,
                lambda: buffer.add(buffer_state, timestep),
            )

            # NETWORKS UPDATE
            def _learn_phase(train_state, rng):

                learn_batch = buffer.sample(buffer_state, rng).experience
                q_next_target = network.apply(
                    train_state.target_network_params, 
                    learn_batch.next_obs
                )  # (batch_size, num_actions)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,) # type: ignore
                target = (
                    learn_batch.reward
                    + (1 - learn_batch.done) * config["GAMMA"] * q_next_target
                )

                def _loss_fn(params):
                    activations = network.apply(
                        params, 
                        learn_batch.obs,
                        method=network.get_activations
                    )
                    q_vals = activations["q_values"]
                    features = activations["rep"]
                    probs = activations["probs"]
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals, # type: ignore
                        jnp.expand_dims(learn_batch.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)
                    loss_q = jnp.mean((chosen_action_qvals - target) ** 2)
                    
                    # MODIFIED: Add sparsity penalty if defined in config
                    loss_sparsity = config["SPARSITY_COEF"] * jnp.mean(probs)
                    
                    return loss_q + loss_sparsity, features
                (loss, features), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                percent_active = jnp.mean(features)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss, percent_active

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    train_state.timesteps > config["LEARNING_STARTS"]
                )
                & (  # pure exploration phase ended
                    train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                )  # training interval
            )
            train_state, loss, percent_active = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (train_state, jnp.array(0.0), jnp.array(0.0)),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
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

            metric = {
                "reward_sum": info["reward_sum"],
                "discounted_reward_sum": info["discounted_reward_sum"],
                "timestep": info["timestep"],
                "episode_step": info["episode_step"],
                "done": info["done"],
                "loss": loss,
                "percent_active": percent_active,
            }

            # report on wandb if required
            if ((params.get("wandb_mode", False) == "online"
                or params.get("wandb_mode", False) == "offline" ) and config["N_SEEDS"] == 1):
                def wandb_callback(metric):
                    wandb.log(
                        {
                        "reward_sum": metric["reward_sum"][0],
                        "discounted_reward_sum": metric["discounted_reward_sum"][0],
                        "timestep": metric["timestep"][0],
                        "episode_step": metric["episode_step"][0],
                        "loss": metric["loss"],
                        "done": metric["done"][0],
                        "percent_active": metric["percent_active"]
                        }
                    )
                jax.lax.cond(
                    metric["done"][0],
                    lambda metric: jax.debug.callback(wandb_callback, metric),
                    lambda metric: None,
                    metric,
                )

            if config.get("VERBOSE"):
                def callback(info):
                    returns = info["reward_sum"][
                        info["done"]
                    ]
                    steps = info["episode_step"][
                        info["done"]
                    ]
                    timesteps = info["timestep"][info["done"]]
                    loss = info["loss"]
                    percent_active = float(info['percent_active'])

                    for t in range(len(timesteps)):
                        print(
                            " ".join(
                                [
                                    f"global step={timesteps[t]},",
                                    f"return={returns[t]}",
                                    f"episode_step={steps[t]}",
                                    f"loss={loss:.4f}",
                                    f"percent_active={percent_active:.4f}"
                                ]
                            )
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, buffer_state, env_state, obs, rng)

            return runner_state, metric

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def _get_all_observations_vectorized(basic_env, env_params, has_key=None, has_key2=None, key_loc=None, key_loc2=None, has_key3=None, key_loc3=None):
    """Generate all valid observations in a vectorized manner.
    
    Args:
        has_key: Boolean for first key state (None if no key mechanism)
        has_key2: Boolean for second key state (None if no second key)
        key_loc: Specific location for first key (None to use fixed_key_loc)
        key_loc2: Specific location for second key (None to use fixed_key_loc2)
        has_key3: Boolean for third key state (None if no third key)
        key_loc3: Specific location for third key (None to use fixed_key_loc3)
    
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
    
    # Determine key locations to use
    if key_loc is not None:
        use_key_loc = key_loc
    elif has_key is not None and hasattr(basic_env, 'fixed_key_loc'):
        use_key_loc = basic_env.fixed_key_loc
    else:
        use_key_loc = jnp.array([0, 0])
    
    if key_loc2 is not None:
        use_key_loc2 = key_loc2
    elif has_key2 is not None and hasattr(basic_env, 'fixed_key_loc2'):
        use_key_loc2 = basic_env.fixed_key_loc2
    else:
        use_key_loc2 = jnp.array([0, 0])
    
    if key_loc3 is not None:
        use_key_loc3 = key_loc3
    elif has_key3 is not None and hasattr(basic_env, 'fixed_key_loc3'):
        use_key_loc3 = basic_env.fixed_key_loc3
    else:
        use_key_loc3 = jnp.array([0, 0])
    
    if has_key is not None and has_key2 is not None and has_key3 is not None:
        # Three-key environment
        states = jax.vmap(lambda loc: GridworldEnvState(
            time=0,
            agent_loc=loc,
            has_key=jnp.array(has_key),
            key_loc=use_key_loc,
            has_key2=jnp.array(has_key2),
            key_loc2=use_key_loc2,
            has_key3=jnp.array(has_key3),
            key_loc3=use_key_loc3
        ))(agent_locs)
    elif has_key is not None and has_key2 is not None:
        # Two-key environment
        states = jax.vmap(lambda loc: GridworldEnvState(
            time=0,
            agent_loc=loc,
            has_key=jnp.array(has_key),
            key_loc=use_key_loc,
            has_key2=jnp.array(has_key2),
            key_loc2=use_key_loc2
        ))(agent_locs)
    elif has_key is not None:
        # One-key environment
        states = jax.vmap(lambda loc: GridworldEnvState(
            time=0,
            agent_loc=loc,
            has_key=jnp.array(has_key),
            key_loc=use_key_loc
        ))(agent_locs)
    else:
        # No-key environment
        states = jax.vmap(lambda loc: GridworldEnvState(
            time=0,
            agent_loc=loc
        ))(agent_locs)
    
    # Vectorized observation generation
    obs_batch = jax.vmap(lambda state: basic_env.get_obs(state, params=env_params))(states)
    
    return obs_batch, locations, valid_mask


def plot_qvals(network_params, config, save_dir):
    """Performs a forward pass with final params and visualizes Q-values for all locations in the maze."""
    # Create environment
    basic_env, env_params = make(config["ENV_NAME"])
    network = make_network(config, action_dim=basic_env.action_space(env_params).n)
    
    # Check if environment has keys
    has_one_key = hasattr(basic_env, 'fixed_key_loc') and hasattr(basic_env, 'use_fixed_key_loc')
    has_two_keys = hasattr(basic_env, 'fixed_key_loc2') and hasattr(basic_env, 'use_fixed_key_loc')
    has_key_locs_set = hasattr(basic_env, 'key_locs_set')
    
    if has_key_locs_set:
        # Environment with random key locations from a set (3 locations, 3 keys sampled without replacement)
        # Plot for all combinations of key possession and key locations
        # Order: none, k3, k1, k1+k3, k1+k2, k2, k2+k3, k1+k2+k3
        key_possession_order = [
            (False, False, False),  # none
            (False, False, True),   # k3
            (True, False, False),   # k1
            (True, False, True),    # k1+k3
            (True, True, False),    # k1+k2
            (False, True, False),   # k2
            (False, True, True),    # k2+k3
            (True, True, True),     # k1+k2+k3
        ]
        for has_k1, has_k2, has_k3 in key_possession_order:
            for k1_idx in range(basic_env.key_locs_set.shape[0]):
                for k2_idx in range(basic_env.key_locs_set.shape[0]):
                    # Skip if both keys at same location (we sample without replacement)
                    if k1_idx == k2_idx:
                        continue
                    # The third key goes to the remaining location
                    for k3_idx in range(basic_env.key_locs_set.shape[0]):
                        if k3_idx == k1_idx or k3_idx == k2_idx:
                            continue
                        k_loc = basic_env.key_locs_set[k1_idx]
                        k_loc2 = basic_env.key_locs_set[k2_idx]
                        k_loc3 = basic_env.key_locs_set[k3_idx]
                        _plot_qvals_single(
                            network_params, config, save_dir, network, basic_env, env_params, 
                            has_k1, has_k2, k_loc, k_loc2, k1_idx, k2_idx, has_key3=has_k3, key_loc3=k_loc3, key_loc3_idx=k3_idx
                        )
    elif has_two_keys and basic_env.use_fixed_key_loc:
        # Two-key environment: plot for all four combinations
        for has_key in [False, True]:
            for has_key2 in [False, True]:
                _plot_qvals_single(network_params, config, save_dir, network, basic_env, env_params, has_key, has_key2)
    elif has_one_key and basic_env.use_fixed_key_loc:
        # One-key environment: plot for both has_key=False and has_key=True
        for has_key in [False, True]:
            _plot_qvals_single(network_params, config, save_dir, network, basic_env, env_params, has_key, None)
    else:
        # Plot without key consideration
        _plot_qvals_single(network_params, config, save_dir, network, basic_env, env_params, None, None)


def _plot_qvals_single(network_params, config, save_dir, network, basic_env, env_params, has_key, has_key2=None, key_loc=None, key_loc2=None, key_loc_idx=None, key_loc2_idx=None, has_key3=None, key_loc3=None, key_loc3_idx=None):
    """Helper function to plot Q-values for a single key state (or no key)."""
    if key_loc_idx is not None and key_loc2_idx is not None and key_loc3_idx is not None:
        print(f"Generating Q-value plot for has_key={has_key}, has_key2={has_key2}, has_key3={has_key3}, key_loc_idx={key_loc_idx}, key_loc2_idx={key_loc2_idx}, key_loc3_idx={key_loc3_idx}")
    elif key_loc_idx is not None and key_loc2_idx is not None:
        print(f"Generating Q-value plot for has_key={has_key}, has_key2={has_key2}, key_loc_idx={key_loc_idx}, key_loc2_idx={key_loc2_idx}")
    elif has_key2 is not None:
        print(f"Generating Q-value plot for has_key={has_key}, has_key2={has_key2}")
    else:
        print("Generating Q-value plot for has_key =", has_key)
    
    # Get grid dimensions
    H = basic_env.H
    W = basic_env.W
    
    # Store Q-values for all agent_location pairs
    q_values_grid = jnp.zeros((H, W, 4))  # 4 actions: up, right, down, left
    
    # Vectorized observation generation and forward pass
    obs_batch, locations, valid_mask = _get_all_observations_vectorized(basic_env, env_params, has_key, has_key2, key_loc, key_loc2, has_key3, key_loc3)
    
    if len(locations) > 0:
        # Single jitted forward pass for all locations
        q_vals_all = jax.jit(network.apply)(network_params, obs_batch)
        
        # Fill in the grid with computed Q-values
        for idx, (row, col) in enumerate(locations):
            q_values_grid = q_values_grid.at[row, col].set(q_vals_all[idx])
    
    # Create visualization
    max_dim = max(H, W)
    base_fig_size = max(8, max_dim * 0.8)
    fig_size = min(base_fig_size, 25)
    q_value_fontsize = max(4, min(10, 80 / max_dim))
    label_fontsize = max(8, min(24, 120 / max_dim))
    edge_linewidth = max(0.3, min(1.0, 8 / max_dim))
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_size * W / max_dim, fig_size * H / max_dim))
    
    # Normalize Q-values for color mapping
    valid_q_values = []
    for row in range(H):
        for col in range(W):
            is_obstacle = basic_env._obstacles_map[row, col] == 1.0
            is_goal = jnp.array_equal(jnp.array([row, col]), basic_env.goal_loc)
            if not is_obstacle and not is_goal:
                valid_q_values.extend(q_values_grid[row, col])
    
    if valid_q_values:
        q_min, q_max = min(valid_q_values), max(valid_q_values)
        q_range = q_max - q_min if q_max > q_min else 1.0
    else:
        q_min, q_max, q_range = 0, 1, 1

    # Draw the grid
    for row in range(H):
        for col in range(W):
            agent_loc = jnp.array([row, col])
            is_obstacle = basic_env._obstacles_map[row, col] == 1.0
            is_goal = jnp.array_equal(agent_loc, basic_env.goal_loc)
            
            plot_row = H - 1 - row
            
            if is_obstacle:
                rect = patches.Rectangle((col, plot_row), 1, 1, linewidth=edge_linewidth, edgecolor='black', facecolor='black')
                ax.add_patch(rect)
            elif is_goal:
                rect = patches.Rectangle((col, plot_row), 1, 1, linewidth=edge_linewidth, edgecolor='black', facecolor='green')
                ax.add_patch(rect)
                ax.text(col + 0.5, plot_row + 0.5, 'G', ha='center', va='center', fontsize=label_fontsize, color='white', weight='bold')
            else:
                q_vals = q_values_grid[row, col]
                
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
                
                # Add white arrows for best action(s)
                # Round Q-values to thousandths place for comparison
                rounded_q_vals = [round(float(q), 3) for q in q_vals]
                max_q = max(rounded_q_vals)
                best_actions = [i for i, q in enumerate(rounded_q_vals) if q == max_q]
                
                # Direction vectors for each action: up, right, down, left
                directions = [
                    (0, 0.2),   # up
                    (0.2, 0),   # right
                    (0, -0.2),  # down
                    (-0.2, 0)   # left
                ]
                
                arrow_width = max(0.06, min(0.1, 0.6 / max_dim))
                arrow_head_width = arrow_width * 3
                arrow_head_length = arrow_width * 1.5
                
                for action_idx in best_actions:
                    dx, dy = directions[action_idx]
                    ax.arrow(col + 0.5, plot_row + 0.5, dx, dy, 
                            head_width=arrow_head_width, head_length=arrow_head_length, 
                            fc='white', ec='black', linewidth=edge_linewidth * 0.5, 
                            zorder=12, length_includes_head=True)
            
            # Add yellow dot in upper right if this state is in a penalty region
            has_penalty = basic_env._penalty_map[row, col] != 0.0
            if has_penalty and not is_obstacle and not is_goal:
                dot_size = max(20, min(100, 400 / max_dim))
                ax.scatter(col + 0.85, plot_row + 0.85, s=dot_size, c='yellow', 
                          edgecolors='black', linewidths=edge_linewidth, zorder=10)
            
            # Add green 'S' in upper right if this is a start state
            is_start = any(jnp.array_equal(agent_loc, start_loc) for start_loc in basic_env._start_locs)
            if is_start and not is_obstacle and not is_goal:
                ax.text(col + 0.85, plot_row + 0.85, 'S', ha='center', va='center', 
                       fontsize=label_fontsize, color='green', weight='bold', zorder=11)
            
            rect = patches.Rectangle((col, plot_row), 1, 1, linewidth=edge_linewidth, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
    
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    title_fontsize = max(10, min(16, 100 / max_dim))
    
    # Add key state to title if applicable
    if key_loc_idx is not None and key_loc2_idx is not None and key_loc3_idx is not None:
        key_state_str = f"key1={'Y' if has_key else 'N'}, key2={'Y' if has_key2 else 'N'}, K1@{key_loc_idx}, K2@{key_loc2_idx}, K3@{key_loc3_idx}"
        ax.set_title(f'Action Values for {config["ENV_NAME"]} ({key_state_str})', fontsize=title_fontsize)
    elif key_loc_idx is not None and key_loc2_idx is not None:
        key_state_str = f"key1={'Y' if has_key else 'N'}, key2={'Y' if has_key2 else 'N'}, K1@{key_loc_idx}, K2@{key_loc2_idx}"
        ax.set_title(f'Action Values for {config["ENV_NAME"]} ({key_state_str})', fontsize=title_fontsize)
    elif has_key2 is not None:
        key_state_str = f"key1={'Y' if has_key else 'N'}, key2={'Y' if has_key2 else 'N'}"
        ax.set_title(f'Action Values for {config["ENV_NAME"]} ({key_state_str})', fontsize=title_fontsize)
    elif has_key is not None:
        key_state_str = "with key" if has_key else "without key"
        ax.set_title(f'Action Values for {config["ENV_NAME"]} ({key_state_str})', fontsize=title_fontsize)
    else:
        ax.set_title(f'Action Values for {config["ENV_NAME"]}', fontsize=title_fontsize)
    
    ax.grid(True, alpha=0.3, linewidth=edge_linewidth * 0.5)
    
    # Determine which key locations to display
    display_key_loc = key_loc if key_loc is not None else (basic_env.fixed_key_loc if hasattr(basic_env, 'fixed_key_loc') else None)
    display_key_loc2 = key_loc2 if key_loc2 is not None else (basic_env.fixed_key_loc2 if hasattr(basic_env, 'fixed_key_loc2') else None)
    display_key_loc3 = key_loc3 if key_loc3 is not None else (basic_env.fixed_key_loc3 if hasattr(basic_env, 'fixed_key_loc3') else None)
    
    # Add key location marker if applicable
    if has_key is not None and not has_key and display_key_loc is not None:
        key_row, key_col = int(display_key_loc[0]), int(display_key_loc[1])
        plot_key_row = H - 1 - key_row
        # Draw a small red 'k1' in upper right corner
        ax.text(key_col + 0.85, plot_key_row + 0.85, 'k1', ha='center', va='center', 
                fontsize=label_fontsize * 0.6, color='darkred', weight='bold', zorder=15)
    
    # Add second key location marker if applicable
    if has_key2 is not None and not has_key2 and display_key_loc2 is not None:
        key2_row, key2_col = int(display_key_loc2[0]), int(display_key_loc2[1])
        plot_key2_row = H - 1 - key2_row
        # Draw a small blue 'k2' in upper right corner
        ax.text(key2_col + 0.85, plot_key2_row + 0.85, 'k2', ha='center', va='center', 
                fontsize=label_fontsize * 0.6, color='darkblue', weight='bold', zorder=15)
    
    # Add third key location marker if applicable (distractor key)
    if has_key3 is not None and not has_key3 and display_key_loc3 is not None:
        key3_row, key3_col = int(display_key_loc3[0]), int(display_key_loc3[1])
        plot_key3_row = H - 1 - key3_row
        # Draw a small purple 'k3' in upper right corner
        ax.text(key3_col + 0.85, plot_key3_row + 0.85, 'k3', ha='center', va='center', 
                fontsize=label_fontsize * 0.6, color='purple', weight='bold', zorder=15)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    
    # Adjust filename based on key state
    if has_key is None and has_key2 is None:
        save_path = os.path.join(save_dir, f'q_vals_{config["ENV_NAME"]}.png')
    elif key_loc_idx is not None and key_loc2_idx is not None and key_loc3_idx is not None:
        # Environment with random key locations from set (3 keys)
        key1_suffix = "_key1" if has_key else "_nokey1"
        key2_suffix = "_key2" if has_key2 else "_nokey2"
        loc_suffix = f"_k1loc{key_loc_idx}_k2loc{key_loc2_idx}_k3loc{key_loc3_idx}"
        save_path = os.path.join(save_dir, f'q_vals_{config["ENV_NAME"]}{key1_suffix}{key2_suffix}{loc_suffix}.png')
    elif key_loc_idx is not None and key_loc2_idx is not None:
        # Environment with random key locations from set (2 keys)
        key1_suffix = "_key1" if has_key else "_nokey1"
        key2_suffix = "_key2" if has_key2 else "_nokey2"
        loc_suffix = f"_k1loc{key_loc_idx}_k2loc{key_loc2_idx}"
        save_path = os.path.join(save_dir, f'q_vals_{config["ENV_NAME"]}{key1_suffix}{key2_suffix}{loc_suffix}.png')
    elif has_key2 is not None:
        # Two-key environment
        key1_suffix = "_key1" if has_key else "_nokey1"
        key2_suffix = "_key2" if has_key2 else "_nokey2"
        save_path = os.path.join(save_dir, f'q_vals_{config["ENV_NAME"]}{key1_suffix}{key2_suffix}.png')
    else:
        # One-key environment
        key_suffix = "_with_key" if has_key else "_without_key"
        save_path = os.path.join(save_dir, f'q_vals_{config["ENV_NAME"]}{key_suffix}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Log figure to wandb if enabled
    if wandb.run is not None:
        if key_loc_idx is not None and key_loc2_idx is not None and key_loc3_idx is not None:
            log_key = f"q_values_key1{'Y' if has_key else 'N'}_key2{'Y' if has_key2 else 'N'}_k1loc{key_loc_idx}_k2loc{key_loc2_idx}_k3loc{key_loc3_idx}"
        elif key_loc_idx is not None and key_loc2_idx is not None:
            log_key = f"q_values_key1{'Y' if has_key else 'N'}_key2{'Y' if has_key2 else 'N'}_k1loc{key_loc_idx}_k2loc{key_loc2_idx}"
        elif has_key2 is not None:
            log_key = f"q_values_key1{'Y' if has_key else 'N'}_key2{'Y' if has_key2 else 'N'}"
        elif has_key is not None:
            key_suffix = "with_key" if has_key else "without_key"
            log_key = f"q_values_{key_suffix}"
        else:
            log_key = "q_values"
        wandb.log({log_key: wandb.Image(fig)})

    plt.close(fig)


def _plot_feature_heatmaps(network_params, config, save_dir, method_name, title, filename):
    """Helper function to create heatmaps for any feature extraction method."""
    basic_env, env_params = make(config["ENV_NAME"])
    network = make_network(config, action_dim=basic_env.action_space(env_params).n)
    
    # Check if environment has keys
    has_one_key = hasattr(basic_env, 'fixed_key_loc') and hasattr(basic_env, 'use_fixed_key_loc')
    has_two_keys = hasattr(basic_env, 'fixed_key_loc2') and hasattr(basic_env, 'use_fixed_key_loc')
    has_key_locs_set = hasattr(basic_env, 'key_locs_set')
    
    if has_key_locs_set:
        # Environment with random key locations from a set (3 keys)
        # Generate all combinations: (has_key, has_key2, has_key3, key_loc_idx, key_loc2_idx, key_loc3_idx)
        # Keys are sampled without replacement, so skip cases where any two indices match
        # Order: none, k3, k1, k1+k3, k1+k2, k2, k2+k3, k1+k2+k3
        key_combinations = []
        key_possession_order = [
            (False, False, False),  # none
            (False, False, True),   # k3
            (True, False, False),   # k1
            (True, False, True),    # k1+k3
            (True, True, False),    # k1+k2
            (False, True, False),   # k2
            (False, True, True),    # k2+k3
            (True, True, True),     # k1+k2+k3
        ]
        for has_k1, has_k2, has_k3 in key_possession_order:
            for k1_idx in range(basic_env.key_locs_set.shape[0]):
                for k2_idx in range(basic_env.key_locs_set.shape[0]):
                    # Skip if both keys at same location (we sample without replacement)
                    if k1_idx == k2_idx:
                        continue
                    # The third key goes to the remaining location
                    for k3_idx in range(basic_env.key_locs_set.shape[0]):
                        if k3_idx == k1_idx or k3_idx == k2_idx:
                            continue
                        key_combinations.append((has_k1, has_k2, has_k3, k1_idx, k2_idx, k3_idx))
        _plot_feature_heatmaps_grid(
            network_params, config, save_dir, method_name, 
            title, filename, network, basic_env, env_params, key_combinations, 
            two_keys=True, key_locs_set=True
        )
    elif has_two_keys and basic_env.use_fixed_key_loc:
        # Two-key environment: create single plot with 4 columns
        key_combinations = [(False, False), (True, False), (False, True), (True, True)]
        _plot_feature_heatmaps_grid(
            network_params, config, save_dir, method_name, 
            title, filename, network, basic_env, env_params, key_combinations, two_keys=True
        )
    elif has_one_key and basic_env.use_fixed_key_loc:
        # One-key environment: create single plot with 2 columns
        key_combinations = [(False, None), (True, None)]
        _plot_feature_heatmaps_grid(
            network_params, config, save_dir, method_name, 
            title, filename, network, basic_env, env_params, key_combinations, two_keys=False
        )
    else:
        # Plot without key consideration - single column
        key_combinations = [(None, None)]
        _plot_feature_heatmaps_grid(
            network_params, config, save_dir, method_name, 
            title, filename, network, basic_env, env_params, key_combinations, two_keys=False
        )
def _plot_feature_heatmaps_grid(network_params, config, save_dir, method_name, title, filename, network, basic_env, env_params, key_combinations, two_keys=False, key_locs_set=False):
    """
    Optimized heatmap plotter with colored key squares and matching colored titles.
    """
    H = basic_env.H
    W = basic_env.W
    
    # --- 1. OPTIMIZATION: Pre-calculate static map features ---
    obstacle_locs = []
    penalty_locs = []
    
    # Get key locations if they exist (for fixed key environments)
    key_loc = None
    key_loc2 = None
    if hasattr(basic_env, 'fixed_key_loc'):
        key_loc = (int(basic_env.fixed_key_loc[0]), int(basic_env.fixed_key_loc[1]))
    if hasattr(basic_env, 'fixed_key_loc2'):
        key_loc2 = (int(basic_env.fixed_key_loc2[0]), int(basic_env.fixed_key_loc2[1]))
    
    for r in range(H):
        for c in range(W):
            if basic_env._obstacles_map[r, c] == 1.0:
                obstacle_locs.append((r, c))
            elif basic_env._penalty_map[r, c] != 0.0 and not np.array_equal([r,c], basic_env.goal_loc):
                penalty_locs.append((r, c))

    # --- 2. VECTORIZED INFERENCE FOR ALL KEY COMBINATIONS ---
    print(f"Generating features for {len(key_combinations)} key combinations...")
    all_feature_grids = []
    valid_mask = None
    
    for combo in key_combinations:
        if key_locs_set:
            # Unpack: (has_key, has_key2, has_key3, key_loc_idx, key_loc2_idx, key_loc3_idx)
            has_key, has_key2, has_key3, k1_idx, k2_idx, k3_idx = combo
            k_loc = basic_env.key_locs_set[k1_idx]
            k_loc2 = basic_env.key_locs_set[k2_idx]
            k_loc3 = basic_env.key_locs_set[k3_idx]
            obs_batch, locations, valid_mask = _get_all_observations_vectorized(
                basic_env, env_params, has_key, has_key2, k_loc, k_loc2, has_key3=has_key3, key_loc3=k_loc3
            )
        else:
            # Original format: (has_key, has_key2) or (has_key, None) or (None, None)
            has_key, has_key2 = combo if len(combo) == 2 else (combo[0], None)
            obs_batch, locations, valid_mask = _get_all_observations_vectorized(
                basic_env, env_params, has_key, has_key2
            )

        
        if len(locations) == 0:
            return
        
        # JIT the network application for maximum speed
        apply_fn = jax.jit(lambda params, obs: network.apply(
            params, obs, method=getattr(network, method_name)
        ))
        
        # One single call to the GPU for all grid cells
        features_all = apply_fn(network_params, obs_batch)
        features_all = np.asarray(features_all)
        
        # Fill the 3D grid (H, W, Features)
        feature_dim = features_all.shape[-1]
        feature_grid = np.full((H, W, feature_dim), np.nan, dtype=np.float32)
        
        # Map linear batch results back to (H, W) grid
        for idx, (row, col) in enumerate(locations):
            feature_grid[row, col, :] = features_all[idx]
        
        all_feature_grids.append(feature_grid)

    # --- 3. PLOTTING SETUP ---
    num_cols = len(key_combinations)
    num_rows = feature_dim
    
    max_dim = max(H, W)
    cell_size = max(0.5, min(1.0, 5.0 / max(1, max_dim / 5)))
    
    fig_width = num_cols * cell_size
    fig_height = num_rows * cell_size
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False)

    title_fontsize = max(4, min(7, 60 / max(1, max_dim)))
    edge_linewidth = max(0.2, min(0.6, 6 / max_dim))

    cmap = mcolors.LinearSegmentedColormap.from_list("feature_red", ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"])

    # Prepare Mask (Obstacles + Goal)
    obstacle_mask = basic_env._obstacles_map.astype(bool)
    goal_mask = np.zeros((H, W), dtype=bool)
    goal_mask[int(basic_env.goal_loc[0]), int(basic_env.goal_loc[1])] = True
    mask = obstacle_mask | goal_mask

    # --- 4. FAST PLOTTING LOOP ---
    print(f"Plotting {feature_dim} features rows...")
    for feature_idx in range(feature_dim):
        if feature_idx % 20 == 0:
             print(f"  Feature {feature_idx}/{feature_dim}...")
        
        # Calculate feature min/max across ALL key combinations for this feature
        all_valid_values = []
        for feature_grid in all_feature_grids:
            feature_values_temp = feature_grid[:, :, feature_idx]
            valid_values_temp = feature_values_temp[valid_mask]
            if valid_values_temp.size:
                all_valid_values.extend(valid_values_temp.flatten())
        
        if len(all_valid_values) > 0:
            feature_min, feature_max = float(np.min(all_valid_values)), float(np.max(all_valid_values))
        else:
            feature_min, feature_max = 0.0, 1.0
        
        for col_idx, combo in enumerate(key_combinations):
            ax = axes[feature_idx, col_idx]
            feature_grid = all_feature_grids[col_idx]
            feature_values = feature_grid[:, :, feature_idx]

            feature_values_flipped = np.flipud(feature_values)
            mask_flipped = np.flipud(mask)
            masked_features = np.ma.array(feature_values_flipped, mask=mask_flipped)

            im = ax.imshow(masked_features, cmap=cmap, vmin=feature_min, vmax=feature_max, 
                           origin='lower', interpolation='nearest', extent=[0, W, 0, H])

            # --- DRAW OVERLAYS ---
            # Obstacles
            for r, c in obstacle_locs:
                plot_row = H - 1 - r
                rect = patches.Rectangle((c, plot_row), 1, 1, linewidth=edge_linewidth, 
                                        edgecolor='black', facecolor='grey', zorder=5)
                ax.add_patch(rect)

            # Goal
            goal_row, goal_col = int(basic_env.goal_loc[0]), int(basic_env.goal_loc[1])
            plot_goal_row = H - 1 - goal_row
            rect = patches.Rectangle((goal_col, plot_goal_row), 1, 1, linewidth=edge_linewidth, 
                                    edgecolor='black', facecolor='green', zorder=5)
            ax.add_patch(rect)

            # Penalty Dots
            for r, c in penalty_locs:
                plot_row = H - 1 - r
                dot_size = max(5, min(20, 100 / max(1, max_dim)))
                ax.scatter(c + 0.85, plot_row + 0.85, s=dot_size, c='yellow', 
                          edgecolors='black', linewidths=edge_linewidth * 0.5, zorder=10)
            
            # Determine key locations for this combo
            if key_locs_set:
                has_key, has_key2, has_key3, k1_idx, k2_idx, k3_idx = combo
                combo_key_loc = (int(basic_env.key_locs_set[k1_idx][0]), int(basic_env.key_locs_set[k1_idx][1]))
                combo_key_loc2 = (int(basic_env.key_locs_set[k2_idx][0]), int(basic_env.key_locs_set[k2_idx][1]))
                combo_key_loc3 = (int(basic_env.key_locs_set[k3_idx][0]), int(basic_env.key_locs_set[k3_idx][1]))
            else:
                has_key, has_key2 = combo if len(combo) == 2 else (combo[0], None)
                combo_key_loc = key_loc
                combo_key_loc2 = key_loc2
                combo_key_loc3 = None
            
            # --- Key 1 Square (gold) ---
            if has_key is not None and not has_key and combo_key_loc is not None:
                key_row, key_col = combo_key_loc
                plot_key_row = H - 1 - key_row
                # Solid gold square
                rect = patches.Rectangle((key_col, plot_key_row), 1, 1, linewidth=edge_linewidth,
                                         edgecolor='black', facecolor='gold', zorder=5)
                ax.add_patch(rect)
            
            # --- Key 2 Square (Blue) ---
            if has_key2 is not None and not has_key2 and combo_key_loc2 is not None:
                key2_row, key2_col = combo_key_loc2
                plot_key2_row = H - 1 - key2_row
                # Solid blue square
                rect = patches.Rectangle((key2_col, plot_key2_row), 1, 1, linewidth=edge_linewidth,
                                         edgecolor='black', facecolor='blue', zorder=5)
                ax.add_patch(rect)
            
            # --- Key 3 Square (Purple) - Distractor Key ---
            if key_locs_set and has_key3 is not None and not has_key3 and combo_key_loc3 is not None:
                key3_row, key3_col = combo_key_loc3
                plot_key3_row = H - 1 - key3_row
                # Solid purple square for distractor key (only show if not collected)
                rect = patches.Rectangle((key3_col, plot_key3_row), 1, 1, linewidth=edge_linewidth,
                                         edgecolor='black', facecolor='purple', zorder=5)
                ax.add_patch(rect)

            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # --- TITLE GENERATION ---
            title_color = 'black'
            if key_locs_set:
                # For key_locs_set environments, show key possession
                has_key, has_key2, has_key3, k1_idx, k2_idx, k3_idx = combo
                # Check all 8 combinations
                if has_key and has_key2 and has_key3:
                    col_title = f"Has (K1, K2, K3)"
                elif has_key and has_key2:
                    col_title = f"Has (K1, K2)"
                elif has_key and has_key3:
                    col_title = f"Has (K1, K3)"
                elif has_key2 and has_key3:
                    col_title = f"Has (K2, K3)"
                elif has_key:
                    col_title = f"Has K1"
                    title_color = 'gold'  # Match K1 color
                elif has_key2:
                    col_title = f"Has K2"
                    title_color = 'blue'   # Match K2 color
                elif has_key3:
                    col_title = f"Has K3"
                    title_color = 'purple'   # Match K3 color
                else:
                    col_title = f"Has None"
            elif has_key and has_key2:
                col_title = "Has (K1, K2)"
            elif has_key:
                col_title = "Has K1"
                title_color = 'gold'  # Match K1 color
            elif has_key2:
                col_title = "Has K2"
                title_color = 'blue'   # Match K2 color
            elif has_key is False and has_key2 is False:
                col_title = "Has None"
            else:
                col_title = "Has Key" if has_key else "None"
                if has_key: title_color = 'gold'
            
            ax.set_title(col_title, fontsize=title_fontsize, color=title_color)
            
            # Feature Label
            if col_idx == 0:
                ax.set_ylabel(f"F{feature_idx}", fontsize=title_fontsize, rotation=90, labelpad=2)

    print("adding tight layout...")
    fig.suptitle(title, fontsize=title_fontsize + 2)
    plt.tight_layout(rect=(0, 0, 1, 0.98))

    print("saving figure...")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rep_heatmaps(network_params, config, save_dir):
    """Visualizes representation features and last hidden layer as heatmaps across all valid agent locations."""
    # Plot representation layer heatmaps
    # _plot_feature_heatmaps(
    #     network_params, 
    #     config, 
    #     save_dir, 
    #     "get_features", 
    #     "Representation Layer Activations", 
    #     f"rep_heatmaps_{config['ENV_NAME']}.png"
    # )
    
    # Plot last hidden layer heatmaps
    if config["ACTIVATION"] != "linear":
        _plot_feature_heatmaps(
            network_params, 
            config, 
            save_dir, 
            "get_last_hidden", 
            "Last Hidden Layer Activations", 
            f"last_hidden_heatmaps_{config['ENV_NAME']}.png"
        )

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

        print(f"Specified hypers for idx {idx}:")
        print(json.dumps(hypers, indent=2))
        config = {
            "SEED": hypers["seed"],
            "N_SEEDS": hypers["n_seeds"],
            "ENV_NAME": hypers["env_name"],
            "NUM_ENVS": hypers.get("num_envs", 1),
            "TOTAL_TIMESTEPS": hypers["total_timesteps"],
            "LEARNING_STARTS": 1000,
            "TRAINING_INTERVAL": 1,
            "GAMMA": 0.99,
            "NETWORK_NAME": hypers["network_name"],
            "OPT": hypers.get("opt", "adam"),
            "LEARNING_RATE": hypers.get("learning_rate", 1e-4),
            "LR_LINEAR_DECAY": hypers.get("lr_linear_decay", False),
            "BUFFER_SIZE": hypers.get("buffer_size", 100000),
            "BUFFER_BATCH_SIZE": hypers.get("buffer_batch_size", 32),
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
            "SPARSITY_COEF": hypers.get("sparsity_coef", 0.0),
            "VERBOSE": args.verbose,
        }
        if config["ACTIVATION"] == "fta":
            config["FTA_ETA"] = hypers.get("fta_eta", 2)
            config["FTA_TILES"] = hypers.get("fta_tiles", 20)
            config["FTA_LOWER_BOUND"] = hypers.get("fta_lower_bound", -20)
            config["FTA_UPPER_BOUND"] = hypers.get("fta_upper_bound", 20)

        assert config['NUM_ENVS'] == 1, "Parallel envs is broken with our current truncation handling"

        print("Wandb entity:", WANDB_ENTITY)
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            tags=["DQN", config["ENV_NAME"].upper()],
            name=f'dqn_{config["ENV_NAME"]}_{config["ACTIVATION"]}_idx{idx}_{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}',
            config=config,
            mode=params.get("wandb_mode", "disabled")
        )

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
            train_jit = jax.jit(jax.vmap(train))

            # Ahead-of-time compilation
            start_compilation_time = time.time()

            compiled_train = train_jit.lower(rngs).compile()
            jax.block_until_ready(compiled_train)

            compilation_time_str = get_time_str(time.time() - start_compilation_time)
            print(f"idx {idx} compilation time: {compilation_time_str}")
            f.write(f"idx {idx} compilation time: {compilation_time_str}\n")

            start_time = time.time()

            results = compiled_train(rngs)
            jax.block_until_ready(results)

            run_time_str = get_time_str(time.time() - start_time)
            print(f"idx {idx} runtime: {run_time_str}")
            f.write(f"idx {idx} runtime: {run_time_str}\n")

        metrics = jax.device_get(results["metrics"])

        # save standard metrics into the per-run directory
        metrics = results["metrics"]
        for metric in metrics:
            metric_data = metrics[metric]
            jnp.save(os.path.join(save_dir, f"{metric}.npy"), metric_data)
            del metric_data  # Free memory after saving

        train_state = results["runner_state"][0]
        network_weights = train_state.params
        
        # Plot Q-values if we run for 1 seed
        if config["N_SEEDS"] == 1:
            plotting_weights = jax.tree_util.tree_map(lambda x: x[0], network_weights) # remove leading seed dimension
            base_env, _ = make(config["ENV_NAME"])
            if isinstance(base_env, Gridworld):
                plot_qvals(plotting_weights, config, save_dir)
                plot_rep_heatmaps(plotting_weights, config, save_dir)

        # Save network weights
        if params.get("save_weights", False):
            # Save network weights as pickle
            with open(os.path.join(save_dir, "network_weights.pkl"), "wb") as f:
                pickle.dump(network_weights, f)
            print(f"Saved network weights for idx {idx}")

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

        print(f"Finished saving results for idx {idx} into '{base_dir}'.")

        del results
        del metrics

        run.finish()

    total_time_str = get_time_str(time.time() - start_time_all)
    print(f"All done! Total time for all {config['N_SEEDS']} idxs: {total_time_str}.")
