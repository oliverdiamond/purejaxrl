import os
import datetime
import copy
import math 
import json
import argparse
import itertools
from functools import partial

import wandb
import jax
import jax.numpy as jnp
import chex
import optax
import flax
import flax.linen as nn
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
import flashbax as fbx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from wrappers import MultiTaskLogWrapper
from environments import make_multitask
from environments.rooms_multitask import EnvState as TwoRoomsEnvStateMultiTask
from purejaxrl.util.fta import fta


class TaskNet(nn.Module):
    action_dim: int
    n_features: int
    linear_rep: bool = False

    def setup(self):
        if self.linear_rep:
            self.task_rep = nn.Dense(self.n_features, name="task_rep")
        else:
            self.task_rep = nn.Sequential([
                nn.Dense(self.n_features, name="task_rep_dense"),
                nn.relu
            ], name="task_rep")
        self.task_head = nn.Dense(self.action_dim, name="task_head")
    
    def __call__(self, common_input: jnp.ndarray, shared_rep: jnp.ndarray) -> jnp.ndarray:
        # Task-specific representation layer
        task_rep = self.task_rep(common_input)

        # Concatenate with the shared representation
        combined_rep = jnp.concatenate([shared_rep, task_rep], axis=-1)

        # Task-specific output head
        q_values = self.task_head(combined_rep)
        return q_values
    
    def get_activations(self, common_input: jnp.ndarray, shared_rep: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        task_rep = self.task_rep(common_input)

        combined_rep = jnp.concatenate([shared_rep, task_rep], axis=-1)

        q_values = self.task_head(combined_rep)

        return {
            "task_rep": task_rep,
            "q_values": q_values
        }

class MTQNetWithTaskReps(nn.Module):
    action_dim: int
    n_tasks: int
    linear_task_rep: bool
    n_features_per_task: int
    n_shared_expand: int
    n_shared_bottleneck: int
    n_features_conv1: int
    n_features_conv2: int

    def setup(self):
        w_conv_init = variance_scaling(scale=math.sqrt(5), mode='fan_avg', distribution='uniform')
        w_init = variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')

        # Conv Backbone
        self.conv1 = nn.Conv(
            features=self.n_features_conv1, kernel_size=(4, 4), strides=(1, 1), padding=[(1, 1), (1, 1)],
            kernel_init=w_conv_init, name="conv1"
        )
        self.conv2 = nn.Conv(
            features=self.n_features_conv2, kernel_size=(4, 4), strides=(2, 2), padding=[(2, 2), (2, 2)],
            kernel_init=w_conv_init, name="conv2"
        )
        self.conv_backbone = nn.Sequential([
            self.conv1,
            nn.relu,
            self.conv2,
            nn.relu
        ], name="conv_backbone")

        self.shared_rep = nn.Sequential([
            nn.Dense(self.n_shared_expand, kernel_init=w_init, name="shared_rep_expand"),
            nn.Dense(self.n_shared_bottleneck, kernel_init=w_init, name="shared_rep_bottleneck"),
            nn.relu
        ], name="shared_rep")

        # Task-specific networks
        self.TaskNets = nn.vmap(
            TaskNet,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=(None, None),
            out_axes=0,
            axis_size=self.n_tasks,
            methods=['__call__', 'get_activations']
        )(self.action_dim, self.n_features_per_task, self.linear_task_rep, name="TaskNets")

    def __call__(self, x: jnp.ndarray, task: jnp.ndarray):
        # Apply the convolutional backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output

        # Shared representation layer
        shared_rep = self.shared_rep(x)

        # Task-specific outputs
        q_vals = self.TaskNets(x, shared_rep)
        batch_indices = jnp.arange(x.shape[0])
        selected_outputs = q_vals[task, batch_indices]

        return selected_outputs

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        # Conv Backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Shared representation layer
        shared_rep = self.shared_rep(x)
        # Task-specific representations and heads
        task_data = self.TaskNets.get_activations(x, shared_rep)

        task_reps = task_data["task_rep"]
        q_vals = task_data["q_values"]
        
        return {
            "shared_rep": shared_rep,
            "task_rep": task_reps,
            "q_values": q_vals
        }

class MTQNet(nn.Module):
    action_dim: int
    n_tasks: int
    n_shared_expand: int
    n_features_conv1: int
    n_features_conv2: int

    def setup(self):
        w_conv_init = variance_scaling(scale=math.sqrt(5), mode='fan_avg', distribution='uniform')
        w_init = variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')

        # Conv Backbone
        self.conv1 = nn.Conv(
            features=self.n_features_conv1, kernel_size=(4, 4), strides=(1, 1), padding=[(1, 1), (1, 1)],
            kernel_init=w_conv_init, name="conv1"
        )
        self.conv2 = nn.Conv(
            features=self.n_features_conv2, kernel_size=(4, 4), strides=(2, 2), padding=[(2, 2), (2, 2)],
            kernel_init=w_conv_init, name="conv2"
        )
        self.conv_backbone = nn.Sequential([
            self.conv1,
            nn.relu,
            self.conv2,
            nn.relu
        ], name="conv_backbone")

        self.shared_rep = nn.Sequential([
            nn.Dense(self.n_shared_expand, kernel_init=w_init, name="shared_rep_expand"),
            nn.relu
        ], name="shared_rep")

        # Task-specific networks
        self.TaskHeads = nn.vmap(
            nn.Dense,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_tasks,
        )(self.action_dim, name="TaskHeads")

    def __call__(self, x: jnp.ndarray, task: jnp.ndarray):
        # Apply the convolutional backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output

        # Shared representation layer
        shared_rep = self.shared_rep(x)

        # Task-specific outputs
        q_vals = self.TaskHeads(shared_rep)
        batch_indices = jnp.arange(x.shape[0])
        selected_outputs = q_vals[task, batch_indices]

        return selected_outputs

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        # Conv Backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Shared representation layer
        shared_rep = self.shared_rep(x)
        # Task-specific representations and heads
        q_vals = self.TaskHeads(shared_rep)
        
        return {
            "shared_rep": shared_rep,
            "q_values": q_vals
        }
        

class MTQNetWithBottleneck(nn.Module):
    action_dim: int
    n_tasks: int
    n_shared_expand: int
    n_shared_bottleneck: int
    n_features_conv1: int
    n_features_conv2: int

    def setup(self):
        w_conv_init = variance_scaling(scale=math.sqrt(5), mode='fan_avg', distribution='uniform')
        w_init = variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')

        # Conv Backbone
        self.conv1 = nn.Conv(
            features=self.n_features_conv1, kernel_size=(4, 4), strides=(1, 1), padding=[(1, 1), (1, 1)],
            kernel_init=w_conv_init, name="conv1"
        )
        self.conv2 = nn.Conv(
            features=self.n_features_conv2, kernel_size=(4, 4), strides=(2, 2), padding=[(2, 2), (2, 2)],
            kernel_init=w_conv_init, name="conv2"
        )
        self.conv_backbone = nn.Sequential([
            self.conv1,
            nn.relu,
            self.conv2,
            nn.relu
        ], name="conv_backbone")

        self.shared_rep = nn.Sequential([
            nn.Dense(self.n_shared_expand, kernel_init=w_init, name="shared_rep_expand"),
            nn.Dense(self.n_shared_bottleneck, kernel_init=w_init, name="shared_rep_bottleneck"),
            nn.relu
        ], name="shared_rep")

        # Task-specific networks
        self.TaskHeads = nn.vmap(
            nn.Dense,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_tasks,
        )(self.action_dim, name="TaskHeads")

    def __call__(self, x: jnp.ndarray, task: jnp.ndarray):
        # Apply the convolutional backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output

        # Shared representation layer
        shared_rep = self.shared_rep(x)

        # Task-specific outputs
        q_vals = self.TaskHeads(shared_rep)
        batch_indices = jnp.arange(x.shape[0])
        selected_outputs = q_vals[task, batch_indices]

        return selected_outputs

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        # Conv Backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Shared representation layer
        shared_rep = self.shared_rep(x)
        # Task-specific representations and heads
        q_vals = self.TaskHeads(shared_rep)
        
        return {
            "shared_rep": shared_rep,
            "q_values": q_vals
        }


class MTQNetFTA(nn.Module):
    action_dim: int
    n_tasks: int
    n_shared_expand: int
    n_features_conv1: int
    n_features_conv2: int
    fta_eta: float = 2
    fta_tiles: int = 20
    fta_lower_bound: float = -20
    fta_upper_bound: float = 20

    def setup(self):
        w_conv_init = variance_scaling(scale=math.sqrt(5), mode='fan_avg', distribution='uniform')
        w_init = variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')
        fta_activation = partial(fta, eta=self.fta_eta, tiles=self.fta_tiles, lower_bound=self.fta_lower_bound, upper_bound=self.fta_upper_bound)

        # Conv Backbone
        self.conv1 = nn.Conv(
            features=self.n_features_conv1, kernel_size=(4, 4), strides=(1, 1), padding=[(1, 1), (1, 1)],
            kernel_init=w_conv_init, name="conv1"
        )
        self.conv2 = nn.Conv(
            features=self.n_features_conv2, kernel_size=(4, 4), strides=(2, 2), padding=[(2, 2), (2, 2)],
            kernel_init=w_conv_init, name="conv2"
        )
        self.conv_backbone = nn.Sequential([
            self.conv1,
            nn.relu,
            self.conv2,
            nn.relu
        ], name="conv_backbone")

        self.shared_rep = nn.Sequential([
            nn.Dense(self.n_shared_expand, kernel_init=w_init, name="shared_rep_expand"),
            fta_activation
        ], name="shared_rep")

        # Task-specific networks
        self.TaskHeads = nn.vmap(
            nn.Dense,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_tasks,
        )(self.action_dim, name="TaskHeads")

    def __call__(self, x: jnp.ndarray, task: jnp.ndarray):
        # Apply the convolutional backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output

        # Shared representation layer
        shared_rep = self.shared_rep(x)

        # Task-specific outputs
        q_vals = self.TaskHeads(shared_rep)
        batch_indices = jnp.arange(x.shape[0])
        selected_outputs = q_vals[task, batch_indices]

        return selected_outputs

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        # Conv Backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Shared representation layer
        shared_rep = self.shared_rep(x)
        # Task-specific representations and heads
        q_vals = self.TaskHeads(shared_rep)
        
        return {
            "shared_rep": shared_rep,
            "q_values": q_vals
        }

class MTQNetFTAWithBottleneck(nn.Module):
    action_dim: int
    n_tasks: int
    n_shared_expand: int
    n_shared_bottleneck: int
    n_features_conv1: int
    n_features_conv2: int
    fta_eta: float = 2
    fta_tiles: int = 20
    fta_lower_bound: float = -20
    fta_upper_bound: float = 20

    def setup(self):
        w_conv_init = variance_scaling(scale=math.sqrt(5), mode='fan_avg', distribution='uniform')
        w_init = variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')
        fta_activation = partial(fta, eta=self.fta_eta, tiles=self.fta_tiles, lower_bound=self.fta_lower_bound, upper_bound=self.fta_upper_bound)

        # Conv Backbone
        self.conv1 = nn.Conv(
            features=self.n_features_conv1, kernel_size=(4, 4), strides=(1, 1), padding=[(1, 1), (1, 1)],
            kernel_init=w_conv_init, name="conv1"
        )
        self.conv2 = nn.Conv(
            features=self.n_features_conv2, kernel_size=(4, 4), strides=(2, 2), padding=[(2, 2), (2, 2)],
            kernel_init=w_conv_init, name="conv2"
        )
        self.conv_backbone = nn.Sequential([
            self.conv1,
            nn.relu,
            self.conv2,
            nn.relu
        ], name="conv_backbone")

        self.shared_rep = nn.Sequential([
            nn.Dense(self.n_shared_expand, kernel_init=w_init, name="shared_rep_expand"),
            nn.Dense(self.n_shared_bottleneck, kernel_init=w_init, name="shared_rep_bottleneck"),
            fta_activation
        ], name="shared_rep")

        # Task-specific networks
        self.TaskHeads = nn.vmap(
            nn.Dense,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_tasks,
        )(self.action_dim, name="TaskHeads")

    def __call__(self, x: jnp.ndarray, task: jnp.ndarray):
        # Apply the convolutional backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output

        # Shared representation layer
        shared_rep = self.shared_rep(x)

        # Task-specific outputs
        q_vals = self.TaskHeads(shared_rep)
        batch_indices = jnp.arange(x.shape[0])
        selected_outputs = q_vals[task, batch_indices]

        return selected_outputs

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        # Conv Backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Shared representation layer
        shared_rep = self.shared_rep(x)
        # Task-specific representations and heads
        q_vals = self.TaskHeads(shared_rep)
        
        return {
            "shared_rep": shared_rep,
            "q_values": q_vals
        }
        
        
def make_network(config, action_dim, n_tasks):
    """Returns the appropriate network based on the configuration."""
    if config["NETWORK_NAME"] == "MTQNet":
        return MTQNet(
            action_dim=action_dim,
            n_tasks=n_tasks,
            n_shared_expand=config["N_SHARED_EXPAND"],
            n_features_conv1=config["N_FEATURES_CONV1"],
            n_features_conv2=config["N_FEATURES_CONV2"]
        )
    elif config["NETWORK_NAME"] == "MTQNetWithBottleneck":
        return MTQNetWithBottleneck(
            action_dim=action_dim,
            n_tasks=n_tasks,
            n_shared_expand=config["N_SHARED_EXPAND"],
            n_shared_bottleneck=config["N_SHARED_BOTTLENECK"],
            n_features_conv1=config["N_FEATURES_CONV1"],
            n_features_conv2=config["N_FEATURES_CONV2"]
        )
    elif config["NETWORK_NAME"] == "MTQNetWithTaskReps":
        return MTQNetWithTaskReps(
            action_dim=action_dim,
            n_tasks=n_tasks,
            linear_task_rep=config["LINEAR_TASK_REP"],
            n_features_per_task=config["N_FEATURES_PER_TASK"],
            n_shared_expand=config["N_SHARED_EXPAND"],
            n_shared_bottleneck=config["N_SHARED_BOTTLENECK"],
            n_features_conv1=config["N_FEATURES_CONV1"],
            n_features_conv2=config["N_FEATURES_CONV2"]
        )
    elif config["NETWORK_NAME"] == "MTQNetFTA":
        return MTQNetFTA(
            action_dim=action_dim,
            n_tasks=n_tasks,
            n_shared_expand=config["N_SHARED_EXPAND"],
            n_features_conv1=config["N_FEATURES_CONV1"],
            n_features_conv2=config["N_FEATURES_CONV2"],
            fta_eta=config["FTA_ETA"],
            fta_tiles=config["FTA_TILES"],
            fta_lower_bound=config["FTA_LOWER_BOUND"],
            fta_upper_bound=config["FTA_UPPER_BOUND"]
        )
    elif config["NETWORK_NAME"] == "MTQNetFTAWithBottleneck":
        return MTQNetFTAWithBottleneck(
            action_dim=action_dim,
            n_tasks=n_tasks,
            n_shared_expand=config["N_SHARED_EXPAND"],
            n_shared_bottleneck=config["N_SHARED_BOTTLENECK"],
            n_features_conv1=config["N_FEATURES_CONV1"],
            n_features_conv2=config["N_FEATURES_CONV2"],
            fta_eta=config["FTA_ETA"],
            fta_tiles=config["FTA_TILES"],
            fta_lower_bound=config["FTA_LOWER_BOUND"],
            fta_upper_bound=config["FTA_UPPER_BOUND"]
        )
    else:
        raise ValueError(f"Unknown network name: {config['NETWORK_NAME']}")

@chex.dataclass(frozen=True)
class MultiTaskTimeStep:
    obs: chex.Array
    next_obs: chex.Array
    task: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class MultiTaskTrainState(TrainState):
    target_network_params: flax.core.FrozenDict # type: ignore
    timesteps: int
    n_updates: int


def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]

    basic_env, env_params = make_multitask(config["ENV_NAME"])
    env = MultiTaskLogWrapper(basic_env) # type: ignore

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None, None)
    )(jax.random.split(rng, n_envs), env_state, action, config["GAMMA"], env_params)

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
        rng = jax.random.PRNGKey(0)  # use a dummy rng here
        _action = basic_env.action_space().sample(rng)
        _last_obs, _last_env_state = env.reset(rng, env_params)
        _obs, _env_state, _reward, _done, _ = env.step(rng, _last_env_state, _action, config["GAMMA"], env_params)
        _timestep = MultiTaskTimeStep(obs=_last_obs, next_obs=_obs, task=_last_env_state.env_state.task, action=_action, reward=_reward, done=_done) # type: ignore
        buffer_state = buffer.init(_timestep)

        network = make_network(config, action_dim=env.action_space(env_params).n, n_tasks=env_params.n_tasks)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1,) + env.observation_space(env_params).shape) # Conv layers need extra dimension for batch size
        init_task = jnp.zeros((1,), dtype=jnp.int32)
        network_params = network.init(_rng, init_x, init_task)

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        tx = optax.adam(learning_rate=lr)

        train_state = MultiTaskTrainState.create(
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

            train_state, buffer_state, last_env_state, last_obs, rng = runner_state

            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = network.apply(train_state.params, last_obs, last_env_state.env_state.task)
            action = eps_greedy_exploration(
                rng_a, q_vals, train_state.timesteps
            )  # explore with epsilon greedy_exploration
            obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, last_env_state, action
            )
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            timestep = MultiTaskTimeStep(obs=last_obs, next_obs=obs, task=last_env_state.env_state.task , action=action, reward=reward, done=done) # type: ignore
            buffer_state = jax.lax.cond(
                info['truncated'].any(),  # if any envs are truncated, do not add to buffer. Note this makes parallel envs not work!
                lambda: buffer_state,
                lambda: buffer.add(buffer_state, timestep),
            )
            # NETWORKS UPDATE
            def _learn_phase(train_state, rng):

                learn_batch = buffer.sample(buffer_state, rng).experience

                q_next_target = network.apply(
                    train_state.target_network_params, 
                    learn_batch.next_obs, 
                    learn_batch.task # Actual task for next obs will only differ from stored task if transition was terminal, in which case q_next_target is not used anyways
                )  # (batch_size, num_actions)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,) # type: ignore
                target = (
                    learn_batch.reward
                    + (1 - learn_batch.done) * config["GAMMA"] * q_next_target
                )

                def _loss_fn(params):
                    q_vals = network.apply(
                        params, 
                        learn_batch.obs, 
                        learn_batch.task
                    )  # (batch_size, num_actions)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals, # type: ignore
                        jnp.expand_dims(learn_batch.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)
                    return jnp.mean((chosen_action_qvals - target) ** 2)

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state)) # enough experience in buffer
                & (
                    train_state.timesteps > config["LEARNING_STARTS"] # pure exploration phase ended
                )
                & (
                    train_state.timesteps % config["TRAINING_INTERVAL"] == 0 # training interval
                )
            )
            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (train_state, jnp.array(0.0)),  # do nothing
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

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                "undiscounted_returns": info["returned_episode_returns"].mean(),
                "discounted_returns": info["returned_episode_discounted_returns"].mean(),
                "task": info["returned_episode_tasks"][0] # only works for single env (no parallelization)
            }
            # report on wandb if required
            wandb_mode = config.get("WANDB_MODE", "disabled")
            if wandb_mode == "online":
                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics)
            
            if config['PRINT_METRICS'] == True:
                def print_callback(metrics):
                    if metrics["timesteps"] % 2000 == 0:
                        jax.debug.print(
                            "timesteps: {timesteps}, updates: {updates}, loss: {loss:.4f}, undiscounted_returns: {undiscounted_returns:.4f}, discounted_returns: {discounted_returns:.4f} task: {task}",
                            timesteps=metrics["timesteps"],
                            updates=metrics["updates"],
                            loss=metrics["loss"],
                            undiscounted_returns=metrics["undiscounted_returns"],
                            discounted_returns=metrics["discounted_returns"],
                            task=metrics["task"]
                        )
                jax.debug.callback(print_callback, metrics)

            runner_state = (train_state, buffer_state, env_state, obs, rng)

            
            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def plot_and_store_data(network_params, rng, config):
    """Performs a forward pass with final params and visualizes Q-values for all (goal, hallway) combinations."""
    basic_env, env_params = make_multitask(config["ENV_NAME"])
    
    # Initialize the appropriate network based on config
    network = make_network(config, action_dim=basic_env.action_space(env_params).n, n_tasks=env_params.n_tasks)
    
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1,) + basic_env.observation_space(env_params).shape) # Conv layers need extra dimension for batch size # type: ignore
    init_task = jnp.zeros((1,), dtype=jnp.int32)
    _ = network.init(_rng, init_x, init_task)
    
    # Get grid dimensions
    N = basic_env.N
    num_hallways = env_params.hallway_locs.shape[0]
    num_goals = env_params.goal_locs.shape[0]
    
    # Store Q-values and activations for all (goal, hallway, agent_location) combinations
    all_q_values = {}
    all_activations = {}
    
    # Determine if this is (task defined by hallway) or regular (task defined by goal)
    hallway_is_task = "HallwayAsTask" in config["ENV_NAME"]
    
    # Unified loop: iterate over all (goal, hallway) combinations
    for goal_idx in range(num_goals):
        for hallway_idx in range(num_hallways):
            # Determine task based on environment type
            if hallway_is_task:
                task = jnp.array([hallway_idx])  # Task defined by hallway
            else:
                task = jnp.array([goal_idx])  # Task defined by goal
            
            goal_loc = env_params.goal_locs[goal_idx]
            hallway_loc = env_params.hallway_locs[hallway_idx]
            combo_key = (goal_idx, hallway_idx)
            q_values_grid = jnp.zeros((N, N, 4))  # 4 actions: up, right, down, left
            
            # Store activations for this combination
            activations_data = {
                'shared_rep': {},
                'q_values': {}
            }
            if config["NETWORK_NAME"] == "MTQNetWithTaskReps":
                activations_data['task_rep'] = {}
            
            # Inner loop: iterate over each location in the grid
            for row in range(N):
                for col in range(N):
                    agent_loc = jnp.array([row, col])
                    
                    # Check if this is a valid location for the agent
                    is_wall = (col == hallway_loc[1] and row != hallway_loc[0])
                    is_goal_loc = jnp.array_equal(agent_loc, goal_loc)
                    
                    if not is_wall and not is_goal_loc:
                        # Create new state with current agent location                    
                        current_state = TwoRoomsEnvStateMultiTask(
                            time=0,
                            task=task,
                            start_loc=env_params.start_locs[0],
                            hallway_loc=hallway_loc,
                            agent_loc=agent_loc,
                            goal_loc=goal_loc,
                        )
                        
                        # Generate observation for this state
                        obs = basic_env.get_obs(current_state, params=env_params)
                        obs_batch = jnp.expand_dims(obs, 0) # Add batch dimension
                        
                        # Forward pass through network
                        q_vals = network.apply(network_params, obs_batch, task)
                        q_values_grid = q_values_grid.at[row, col].set(q_vals[0])
                        
                        # Get activations
                        activations = network.apply(network_params, obs_batch, method=network.get_activations)
                        
                        # Store activations for this location (squeeze out dimensions of size 1)
                        if config["NETWORK_NAME"] == "MTQNetWithTaskReps":
                            activations_data['task_rep'][(row, col)] = jnp.squeeze(activations["task_rep"], axis=1)  # shape: (n_tasks, N_FEATURES_PER_TASK) # type: ignore
                        activations_data['shared_rep'][(row, col)] = jnp.squeeze(activations["shared_rep"], axis=0)  # shape: (N_SHARED_BOTTLENECK,) or (N_SHARED_EXPAND,) # type: ignore
                        activations_data['q_values'][(row, col)] = jnp.squeeze(activations["q_values"], axis=1)  # shape: (n_tasks, n_actions) # type: ignore
            
            all_q_values[combo_key] = q_values_grid
            all_activations[combo_key] = activations_data
    
    # Create visualization with dynamic sizing
    # Scale figure size and font sizes based on grid size
    base_fig_size = max(8, N * 0.5)  # Minimum 8, grows with grid size
    fig_size = min(base_fig_size, 20)  # Cap at 20 to avoid huge figures
    
    # Dynamic font sizes based on grid size
    q_value_fontsize = max(4, min(12, 60 / N))  # Scale inversely with grid size
    label_fontsize = max(8, min(24, 120 / N))   # For S and G labels
    edge_linewidth = max(0.3, min(1.0, 8 / N)) # Thinner lines for larger grids
    
    # Calculate subplot arrangement - use a simple grid layout for all cases
    num_combinations = len(all_q_values)
    n_cols = min(num_combinations, 3)  # Max 3 columns
    n_rows = (num_combinations + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_size * n_cols, fig_size * n_rows))
    
    # Always flatten axes to handle consistently
    if num_combinations == 1:
        axes = [axes]
    elif n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    
    # Simple iteration through combinations
    for plot_idx, combo_key in enumerate(all_q_values.keys()):
        goal_idx, hallway_idx = combo_key
        ax = axes[plot_idx]
        
        hallway_loc = env_params.hallway_locs[hallway_idx]
        goal_loc = env_params.goal_locs[goal_idx]
        q_values_grid = all_q_values[combo_key]
        
        # Normalize Q-values for color mapping
        valid_q_values = []
        for row in range(N):
            for col in range(N):
                agent_loc = jnp.array([row, col])
                is_wall = (col == hallway_loc[1] and row != hallway_loc[0])
                is_goal = jnp.array_equal(agent_loc, goal_loc)
                if not is_wall and not is_goal:
                    valid_q_values.extend(q_values_grid[row, col])
        
        if valid_q_values:
            q_min, q_max = min(valid_q_values), max(valid_q_values)
            q_range = q_max - q_min if q_max > q_min else 1.0
        else:
            q_min, q_max, q_range = 0, 1, 1
        
        # Draw the grid
        for row in range(N):
            for col in range(N):
                agent_loc = jnp.array([row, col])
                is_wall = (col == hallway_loc[1] and row != hallway_loc[0])
                is_goal = jnp.array_equal(agent_loc, goal_loc)
                
                # Flip row for plotting (matplotlib uses bottom-left origin)
                plot_row = N - 1 - row
                
                if is_wall:
                    # Draw wall as black square
                    rect = patches.Rectangle((col, plot_row), 1, 1, 
                                            linewidth=edge_linewidth, edgecolor='black', facecolor='black')
                    ax.add_patch(rect)
                elif is_goal:
                    # Draw goal as green square
                    rect = patches.Rectangle((col, plot_row), 1, 1, 
                                            linewidth=edge_linewidth, edgecolor='black', facecolor='green')
                    ax.add_patch(rect)
                    ax.text(col+0.5, plot_row+0.5, f'G', 
                                                    ha='center', va='center', fontsize=label_fontsize, 
                                                    color='white', weight='bold')
                else:
                    # Valid agent location - draw Q-value triangles
                    q_vals = q_values_grid[row, col]
                    
                    # Define triangle vertices for each action
                    # up, right, down, left
                    triangles = [
                        [(col, plot_row + 1), (col + 1, plot_row + 1), (col + 0.5, plot_row + 0.5)],  # up
                        [(col + 1, plot_row + 1), (col + 1, plot_row), (col + 0.5, plot_row + 0.5)],    # right
                        [(col + 1, plot_row), (col, plot_row), (col + 0.5, plot_row + 0.5)],      # down
                        [(col, plot_row), (col, plot_row + 1), (col + 0.5, plot_row + 0.5)]       # left
                    ]
                    
                    for action_idx, (q_val, triangle_verts) in enumerate(zip(q_vals, triangles)):
                        # Normalize Q-value for color intensity
                        intensity = (q_val - q_min) / q_range if q_range > 0 else 0.5
                        color_intensity = float(max(0.1, min(1.0, intensity)))  # Clamp between 0.1 and 1.0
                        
                        triangle = patches.Polygon(triangle_verts, closed=True,
                                                    facecolor=(0, 0, color_intensity, 0.8),
                                                    edgecolor='black', linewidth=edge_linewidth * 0.5)
                        ax.add_patch(triangle)
                        
                        # Add Q-value text in triangle center, but shift slightly towards square center
                        triangle_center_x = sum(v[0] for v in triangle_verts) / 3
                        triangle_center_y = sum(v[1] for v in triangle_verts) / 3
                        square_center_x = col + 0.5
                        square_center_y = plot_row + 0.5
                        
                        # Move the text 30% of the way from triangle center to square center
                        shift_factor = 0.15
                        center_x = triangle_center_x + shift_factor * (square_center_x - triangle_center_x)
                        center_y = triangle_center_y + shift_factor * (square_center_y - triangle_center_y)
                        
                        # Format Q-value text based on magnitude for better readability
                        if abs(q_val) >= 100:
                            q_text = f'{float(q_val):.0f}'
                        elif abs(q_val) >= 10:
                            q_text = f'{float(q_val):.1f}'
                        else:
                            q_text = f'{float(q_val):.2f}'
                            
                        ax.text(center_x, center_y, q_text, 
                                ha='center', va='center', fontsize=q_value_fontsize, 
                                color='white', weight='bold')
                
                if jnp.array_equal(agent_loc, env_params.start_locs[0]):
                    # Draw S at start location
                    ax.text(col + 0.5, plot_row + 0.5, 'S', 
                            ha='center', va='center', fontsize=label_fontsize, 
                            color='white', weight='bold')
                # Draw grid lines
                rect = patches.Rectangle((col, plot_row), 1, 1, 
                                        linewidth=edge_linewidth, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
        
        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        title_fontsize = max(10, min(16, 100 / N))  # Dynamic title font size
        
        if hallway_is_task:
            ax.set_title(f'Task {hallway_idx}: Hallway at ({hallway_loc[0]}, {hallway_loc[1]})', 
                         fontsize=title_fontsize)
        else:
            ax.set_title(f'Task {goal_idx}: Goal at ({goal_loc[0]}, {goal_loc[1]}), Hallway at ({hallway_loc[0]}, {hallway_loc[1]})', 
                         fontsize=title_fontsize)
        
        ax.grid(True, alpha=0.3, linewidth=edge_linewidth * 0.5)
    
    plt.tight_layout()
    
    # Add extra padding around the figure borders
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    

    file_prefix = f'purejaxrl/results/dqn_multitask/{config["ENV_NAME"]}/{config["NETWORK_NAME"]}/{config["CURRENT_TIME"]}'
    # Save figure locally
    fig_path = f'{file_prefix}/qvals.png'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    # Log figure to wandb if enabled
    if wandb.run is not None:
        wandb.log({"q_values": wandb.Image(fig)})
    
    plt.close(fig)  # Close the figure to free memory
    
    # Save activations data to file
    import pickle
    import numpy as np
    import json

    data_path = f"{file_prefix}/activations.pkl"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Convert JAX arrays to numpy for saving using tree_map
    activations_to_save = jax.tree_util.tree_map(np.asarray, all_activations)
    
    # Also save metadata
    metadata = {
        'config': config,
        'grid_size': N,
        'num_hallways': num_hallways,
        'num_goals': num_goals,
        'hallway_locs': np.array(env_params.hallway_locs).tolist(),
        'goal_locs': np.array(env_params.goal_locs).tolist(),
        'start_loc': np.array(env_params.start_locs[0]).tolist()
    }
    
    save_data = {
        'activations': activations_to_save,
        'metadata': metadata
    }
    
    with open(data_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    # Save metadata as JSON in both directories
    metadata_json_data = f"{file_prefix}/metadata.json"

    with open(metadata_json_data, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Log activations data to wandb if enabled
    if wandb.run is not None:
        # Create a wandb artifact for the activations data
        artifact = wandb.Artifact(
            name=f"activations_data_{wandb.run.name}",
            type="activations",
            )
        artifact.add_file(data_path)
        artifact.add_file(metadata_json_data)
        wandb.log_artifact(artifact)
        
        # Also log some summary statistics
        wandb.log({
            "activations_file": data_path,
        })

def load_config(filename):
    """Load configuration from a JSON file."""
    with open(f"purejaxrl/configs/{filename}", 'r') as f:
        config = json.load(f)
    return config


def generate_config_combinations(base_config):
    """
    Generate all unique combinations of configurations based on list parameters.
    
    Args:
        base_config: Dictionary with potentially list-valued parameters
        
    Returns:
        List of config dictionaries, one for each unique combination
    """
    # Find all parameters that are lists
    list_params = {}
    scalar_params = {}
    
    for key, value in base_config.items():
        if isinstance(value, list):
            list_params[key] = value
        else:
            scalar_params[key] = value
    
    # If no list parameters, return the original config
    if not list_params:
        return [base_config]
    
    # Generate all combinations of list parameters
    param_names = list(list_params.keys())
    param_values = list(list_params.values())
    
    configs = []
    for combination in itertools.product(*param_values):
        # Create a new config for this combination
        config = scalar_params.copy()
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value
        configs.append(config)
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='DQN Multi-Task Training')
    parser.add_argument('-c', '--config', type=str, 
                       default='test.json',
                       help='Path to configuration JSON file')
    args = parser.parse_args()
    
    # Load base configuration from JSON file
    base_config = load_config(args.config)
    
    # Generate all combinations of configurations
    config_combinations = generate_config_combinations(base_config)
    
    print(f"Running {len(config_combinations)} configuration combination(s)")
    
    # Run training for each configuration combination
    for config_idx, config in enumerate(config_combinations):
        print(f"\nRunning configuration {config_idx + 1}/{len(config_combinations)}")
        
        # Print the current configuration parameters that are different from base
        if len(config_combinations) > 1:
            print("Configuration parameters:")
            for key, value in config.items():
                if isinstance(base_config.get(key), list):
                    print(f"  {key}: {value}")

        # Handle multiple seeds within each configuration
        seeds = config["SEED"]  # SEED is always a list in config files
        
        for seed in seeds:
            current_config = config.copy()
            current_config["SEED"] = seed
            current_config["CURRENT_TIME"] = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            
            run = wandb.init(
                entity=current_config["ENTITY"],
                project=current_config["PROJECT"],
                tags=["DQN_MULTITASK", current_config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
                name=f'dqn_multitask_{current_config["ENV_NAME"]}_{current_config["NETWORK_NAME"]}_{current_config["CURRENT_TIME"]}',
                config=current_config,
                mode=current_config["WANDB_MODE"],
            )

            rng = jax.random.PRNGKey(current_config["SEED"])
            rngs = jax.random.split(rng, current_config["NUM_SEEDS"])
            train_vjit = jax.jit(jax.vmap(make_train(current_config)))
            outs = jax.block_until_ready(train_vjit(rngs))
            runner_state = outs["runner_state"]
            # This is a quick fix for now but eventually we might want to vmap the plotting over all of the seeds. 
            # NOTE This will currently fail if we try to squeeze after runnning with more than 1 random seed
            params = jax.tree_util.tree_map(lambda x: x.squeeze(axis=0), runner_state[0].params)
            if config["SAVE_ACTIVATIONS"]:
                plot_and_store_data(params, rng, current_config)
            run.finish()

if __name__ == "__main__":
    main()
