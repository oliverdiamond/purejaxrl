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

import chex
import flax
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from src.util.wrappers import LogWrapper, FlattenObservationWrapper
import gymnax
import flashbax as fbx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from environments.maze import Maze
from environments.maze import EnvState as MazeEnvState
from environments import make
from util import get_time_str, WANDB_ENTITY, WANDB_PROJECT
from util.fta import fta
from experiment import experiment_model



class QNet(nn.Module):
    action_dim: int
    conv1_dim: int = 32
    conv2_dim: int = 16
    rep_dim: int = 64
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
            nn.Dense(self.head_hidden_dim, kernel_init=w_init, name="head_hidden1"),
            nn.relu,
            nn.Dense(self.head_hidden_dim, kernel_init=w_init, name="head_hidden2"),
            nn.relu,
            nn.Dense(self.action_dim, kernel_init=w_init, name="head_output")
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


class QNetFTA(nn.Module):
    action_dim: int
    conv1_dim: int = 32
    conv2_dim: int = 16
    rep_dim: int = 64
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
            nn.Dense(self.head_hidden_dim, kernel_init=w_init, name="head_hidden1"),
            nn.relu,
            nn.Dense(self.head_hidden_dim, kernel_init=w_init, name="head_hidden2"),
            nn.relu,
            nn.Dense(self.action_dim, kernel_init=w_init, name="head_output")
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

def make_network(config, action_dim):
    """Returns the appropriate network based on the configuration."""
    if config["ACTIVATION"] == "relu":
        return QNet(
            action_dim=action_dim,
            conv1_dim=config["CONV1_DIM"],
            conv2_dim=config["CONV2_DIM"],
            rep_dim=config["REP_DIM"]
        )
    elif config["ACTIVATION"] == "fta":
        return QNetFTA(
            action_dim=action_dim,
            conv1_dim=config["CONV1_DIM"],
            conv2_dim=config["CONV2_DIM"],
            rep_dim=config["REP_DIM"],
            fta_eta=config["FTA_ETA"],
            fta_tiles=config["FTA_TILES"],
            fta_lower_bound=config["FTA_LOWER_BOUND"],
            fta_upper_bound=config["FTA_UPPER_BOUND"]
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

    # basic_env, env_params = gymnax.make(config["ENV_NAME"])
    # env = FlattenObservationWrapper(basic_env)
    # env = LogWrapper(env) # type: ignore
    base_env, env_params = make(config["ENV_NAME"]) # type: ignore
    env = LogWrapper(base_env) # type: ignore

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
        rng = jax.random.PRNGKey(0)  # use a dummy rng here
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
        tx = optax.adam(learning_rate=lr)

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
                info['truncated'].any(),  # if any envs are truncated, do not add to buffer. Note this breaks parallel envs!
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
                    q_vals = network.apply(
                        params, 
                        learn_batch.obs
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
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    train_state.timesteps > config["LEARNING_STARTS"]
                )
                & (  # pure exploration phase ended
                    train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                )  # training interval
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
                "returns": info["returned_episode_returns"].mean(),  # avg most recent returns for each par env
            }

            # report on wandb if required
            if config.get("WANDB_MODE", False) == "online":

                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            if config['VERBOSE'] == True:
                def print_callback(metrics):
                    if metrics["timesteps"] % 500 == 0:
                        jax.debug.print(
                            "timesteps: {timesteps}, updates: {updates}, loss: {loss:.4f}, undiscounted_returns: {undiscounted_returns:.4f}",
                            timesteps=metrics["timesteps"],
                            updates=metrics["updates"],
                            loss=metrics["loss"],
                            undiscounted_returns=metrics["returns"],
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


def plot_qvals(network_params, config, save_dir):
    """Performs a forward pass with final params and visualizes Q-values for all locations in the maze."""
    basic_env = Maze()
    env_params = basic_env.default_params
    network = make_network(config, action_dim=basic_env.action_space(env_params).n)
    
    # Get grid dimensions
    N = basic_env.N
    
    # Store Q-values for all agent_location pairs
    q_values_grid = jnp.zeros((N, N, 4))  # 4 actions: up, right, down, left
    
    # Iterate over each location in the grid
    for row in range(N):
        for col in range(N):
            agent_loc = jnp.array([row, col])
            
            # Check if this is a valid location for the agent (not an obstacle)
            is_obstacle = basic_env._obstacles_map[row, col] == 1.0
            is_goal = jnp.array_equal(agent_loc, basic_env.goal_loc)
            
            if not is_obstacle and not is_goal:
                # Create new state with current agent location
                current_state = MazeEnvState(
                    time=0,
                    agent_loc=agent_loc
                )
                
                # Generate observation for this state
                obs = basic_env.get_obs(current_state, params=env_params)
                obs_batch = jnp.expand_dims(obs, 0) # Add batch dimension
                
                # Forward pass through network
                q_vals = network.apply(network_params, obs_batch)
                q_values_grid = q_values_grid.at[row, col].set(q_vals[0])
    
    # Create visualization
    base_fig_size = max(8, N * 0.8)
    fig_size = min(base_fig_size, 25)
    q_value_fontsize = max(4, min(10, 80 / N))
    label_fontsize = max(8, min(24, 120 / N))
    edge_linewidth = max(0.3, min(1.0, 8 / N))
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    
    # Normalize Q-values for color mapping
    valid_q_values = []
    for row in range(N):
        for col in range(N):
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
            
            is_start_loc = any(jnp.array_equal(agent_loc, sl) for sl in basic_env._start_locs)
            if is_start_loc:
                ax.text(col + 0.5, plot_row + 0.5, 'S', ha='center', va='center', fontsize=label_fontsize, color='yellow', weight='bold')

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
    ax.set_title(f'Action Values for {config["ENV_NAME"]}', fontsize=title_fontsize)
    ax.grid(True, alpha=0.3, linewidth=edge_linewidth * 0.5)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'q_vals_{config["ENV_NAME"]}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Log figure to wandb if enabled
    if wandb.run is not None:
        wandb.log({"q_values": wandb.Image(fig)})

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
            "LEARNING_RATE": hypers.get("learning_rate", 1e-4),
            "LR_LINEAR_DECAY": hypers.get("lr_linear_decay", False),
            "BUFFER_SIZE": hypers.get("buffer_size", 100000),
            "BUFFER_BATCH_SIZE": hypers.get("buffer_batch_size", 32),
            "TAU": hypers.get("tau", 1.0),
            "EPSILON_START": hypers.get("epsilon_start", 1.0),
            "EPSILON_FINISH": hypers.get("epsilon_finish", 0.1),
            "EPSILON_ANNEAL_TIME": hypers.get("epsilon_anneal_time", 1),
            "TARGET_UPDATE_INTERVAL": hypers.get("target_update_interval", 64),
            "CONV1_DIM": hypers.get("conv1_dim", 32),
            "CONV2_DIM": hypers.get("conv2_dim", 16),
            "REP_DIM": hypers.get("rep_dim", 64),
            "ACTIVATION": hypers.get("activation", "relu"),
            "VERBOSE": args.verbose,
        }
        if config["ACTIVATION"] == "fta":
            config["FTA_ETA"] = hypers.get("fta_eta", 2)
            config["FTA_TILES"] = hypers.get("fta_tiles", 20)
            config["FTA_LOWER_BOUND"] = hypers.get("fta_lower_bound", -20)
            config["FTA_UPPER_BOUND"] = hypers.get("fta_upper_bound", 20)

        assert config['NUM_ENVS'] == 1, "Parallel envs is broken with our current truncation handling"

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
        returns = metrics["returns"]
        jnp.save(os.path.join(save_dir, "returns.npy"), returns)
        del returns  # Free memory after saving

        loss = metrics["loss"]
        jnp.save(os.path.join(save_dir, "loss.npy"), loss)
        del loss  # Free memory after saving


        train_state = results["runner_state"][0]
        network_weights = train_state.params
        
        # Plot Q-values if we run for 1 seed
        if config["N_SEEDS"] == 1:
            plotting_weights = jax.tree_util.tree_map(lambda x: x[0], network_weights) # remove leading seed dimension
            if config["ENV_NAME"] == "Maze":
                plot_qvals(plotting_weights, config, save_dir)

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

        print(f"Finished saving results for idx {idx}")

        del results
        del metrics

        run.finish()