import os
import datetime
import copy
import math 

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
from environments.rooms_multitask import TwoRoomsMultiTask5 as TwoRoomsMultiTask
from environments.rooms_multitask import EnvState as TwoRoomsEnvStateMultiTask #TODO Make function to select based on conifg, will have easy and hard varaints and varaints with smaller/larger N


class TaskNet(nn.Module):
    action_dim: int
    n_features: int

    def setup(self):
        self.task_rep = nn.Dense(self.n_features, name="task_rep")
        self.task_head = nn.Dense(self.action_dim, name="task_head")
    
    def __call__(self, common_input: jnp.ndarray, shared_rep: jnp.ndarray) -> jnp.ndarray:
        # Task-specific representation layer
        task_rep = self.task_rep(common_input)
        task_rep = nn.relu(task_rep)

        # Concatenate with the shared representation
        combined_rep = jnp.concatenate([shared_rep, task_rep], axis=-1)

        # Task-specific output head
        q_values = self.task_head(combined_rep)
        return q_values
    
    def get_activations(self, common_input: jnp.ndarray, shared_rep: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        task_rep = self.task_rep(common_input)
        task_rep = nn.relu(task_rep)

        combined_rep = jnp.concatenate([shared_rep, task_rep], axis=-1)

        q_values = self.task_head(combined_rep)

        return {
            "task_rep": task_rep,
            "q_values": q_values
        }

class MultiTaskMazeQNetwork(nn.Module):
    action_dim: int
    n_tasks: int
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
        )(self.action_dim, self.n_features_per_task, name="TaskNets")

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

    basic_env = TwoRoomsMultiTask()
    env_params = basic_env.default_params
    env = MultiTaskLogWrapper(basic_env) # type: ignore

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
        _action = basic_env.action_space().sample(rng)
        _last_obs, _last_env_state = env.reset(rng, env_params)
        _obs, _env_state, _reward, _done, _ = env.step(rng, _last_env_state, _action, env_params)
        _timestep = MultiTaskTimeStep(obs=_last_obs, next_obs=_obs, task=_last_env_state.env_state.task, action=_action, reward=_reward, done=_done) # type: ignore
        buffer_state = buffer.init(_timestep)

        # INIT NETWORK AND OPTIMIZER
        network = MultiTaskMazeQNetwork(
            action_dim=env.action_space(env_params).n, 
            n_tasks=env_params.n_tasks, # type: ignore
            n_features_per_task=config["N_FEATURES_PER_TASK"],
            n_shared_expand=config["N_SHARED_EXPAND"],
            n_shared_bottleneck=config["N_SHARED_BOTTLENECK"],
            n_features_conv1=config["N_FEATURES_CONV1"],
            n_features_conv2=config["N_FEATURES_CONV2"],
            )
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
                    if metrics["timesteps"] % 500 == 0:
                        jax.debug.print(
                            "timesteps: {timesteps}, updates: {updates}, loss: {loss:.4f}, undiscounted_returns: {undiscounted_returns:.4f}, task: {task}",
                            timesteps=metrics["timesteps"],
                            updates=metrics["updates"],
                            loss=metrics["loss"],
                            undiscounted_returns=metrics["undiscounted_returns"],
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


def plot_qvals(network_params, rng, config):
    """Performs a forward pass with final params and visualizes Q-values for all hallway locations."""
    #TODO Add logic to select between easy and hard env here and in train with jax.lax.cond
    basic_env = TwoRoomsMultiTask()
    env_params = basic_env.default_params
    network = MultiTaskMazeQNetwork(
        action_dim=basic_env.action_space(env_params).n, 
        n_tasks=env_params.n_tasks, # type: ignore
        n_features_per_task=config["N_FEATURES_PER_TASK"],
        n_shared_expand=config["N_SHARED_EXPAND"],
        n_shared_bottleneck=config["N_SHARED_BOTTLENECK"],
        n_features_conv1=config["N_FEATURES_CONV1"],
        n_features_conv2=config["N_FEATURES_CONV2"],
        )
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1,) + basic_env.observation_space(env_params).shape) # Conv layers need extra dimension for batch size
    init_task = jnp.zeros((1,), dtype=jnp.int32)
    _ = network.init(_rng, init_x, init_task)
    # Get grid dimensions
    N = basic_env.N
    num_hallways = env_params.hallway_locs.shape[0]
    
    
    # Store Q-values for all (hallway, agent_location) pairs
    all_q_values = {}
    
    # Outer loop: iterate over all hallway locations
    for hallway_idx in range(num_hallways):
        task = jnp.array([hallway_idx]) # TODO Check to make sure this is correct (might need to make shape different)
        hallway_loc = env_params.hallway_locs[hallway_idx]
        q_values_grid = jnp.zeros((N, N, 4))  # 4 actions: up, right, down, left
        
        # Inner loop: iterate over each location in the grid
        for row in range(N):
            for col in range(N):
                agent_loc = jnp.array([row, col])
                
                # Check if this is a valid location for the agent
                # 1. Not a wall location
                # 2. Not the goal location
                is_wall = (col == hallway_loc[1] and row != hallway_loc[0])
                is_goal = jnp.array_equal(agent_loc, env_params.goal_loc)
                
                if not is_wall and not is_goal:
                    # Create new state with current agent location
                    current_state = TwoRoomsEnvStateMultiTask(
                        time=0,
                        task=task,
                        hallway_loc=hallway_loc,
                        agent_loc=agent_loc
                    )
                    
                    # Generate observation for this state
                    obs = basic_env.get_obs(current_state, params=env_params)
                    obs_batch = jnp.expand_dims(obs, 0) # Add batch dimension
                    
                    # Forward pass through network
                    q_vals = network.apply(network_params, obs_batch, task)
                    q_values_grid = q_values_grid.at[row, col].set(q_vals[0])
        
        all_q_values[hallway_idx] = q_values_grid
    
    # Create visualization with dynamic sizing
    # Scale figure size and font sizes based on grid size
    base_fig_size = max(8, N * 0.5)  # Minimum 8, grows with grid size
    fig_size = min(base_fig_size, 20)  # Cap at 20 to avoid huge figures
    
    # Dynamic font sizes based on grid size
    q_value_fontsize = max(4, min(12, 60 / N))  # Scale inversely with grid size
    label_fontsize = max(8, min(24, 120 / N))   # For S and G labels
    edge_linewidth = max(0.3, min(1.0, 8 / N)) # Thinner lines for larger grids
    
    fig, axes = plt.subplots(1, num_hallways, figsize=(fig_size * num_hallways, fig_size))
    if num_hallways == 1:
        axes = [axes]
    
    for hallway_idx in range(num_hallways):
        ax = axes[hallway_idx]
        hallway_loc = env_params.hallway_locs[hallway_idx]
        q_values_grid = all_q_values[hallway_idx]
        
        # Normalize Q-values for color mapping
        valid_q_values = []
        for row in range(N):
            for col in range(N):
                agent_loc = jnp.array([row, col])
                is_wall = (col == hallway_loc[1] and row != hallway_loc[0])
                is_goal = jnp.array_equal(agent_loc, env_params.goal_loc)
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
                is_goal = jnp.array_equal(agent_loc, env_params.goal_loc)
                
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
                        
                        # Add Q-value text in triangle center
                        center_x = sum(v[0] for v in triangle_verts) / 3
                        center_y = sum(v[1] for v in triangle_verts) / 3
                        
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
                
                if jnp.array_equal(agent_loc, env_params.start_loc):
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
        ax.set_title(f'Action Values For Task Head {hallway_idx} with Hallway at ({hallway_loc[0]}, {hallway_loc[1]})', 
                     fontsize=title_fontsize)
        
        # # Adjust tick spacing for larger grids to avoid clutter
        # tick_spacing = max(1, N // 10)  # Show fewer ticks for large grids
        # ax.set_xticks(range(0, N + 1, tick_spacing))
        # ax.set_yticks(range(0, N + 1, tick_spacing))
        ax.grid(True, alpha=0.3, linewidth=edge_linewidth * 0.5)
    
    plt.tight_layout()
    plt.savefig(f'purejaxrl/plots/q_values_N={N}.png', dpi=300, bbox_inches='tight')
    
    print(f"Q-values visualization saved as 'q_values_N={N}.png'")
    print(f"Analyzed {num_hallways} hallway configuration(s) for {N}x{N} grid")


def main():

    config = {
        "NUM_ENVS": 1, # Must be set to 1, handling of truncation fails otherwise (entire batch is currently ignored if any transitions result in truncation)
        "BUFFER_SIZE": 10000,
        "BUFFER_BATCH_SIZE": 32,
        "TOTAL_TIMESTEPS": 10000, # 15e4, #5e5,
        "EPSILON_START": 0.1, # EPSILON_START==EPSILON_FINISH -> no annealing
        "EPSILON_FINISH": 0.1,
        "EPSILON_ANNEAL_TIME": 1e4,
        "TARGET_UPDATE_INTERVAL": 64,
        "LR": 2.5e-4,
        "LEARNING_STARTS": 1000,
        "TRAINING_INTERVAL": 1,
        "N_FEATURES_PER_TASK": 32,
        "N_SHARED_EXPAND": 128,
        "N_SHARED_BOTTLENECK": 8,
        "N_FEATURES_CONV1": 32,
        "N_FEATURES_CONV2": 16,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": "TwoRoomsMultiTask",
        "SEED": [0],
        "NUM_SEEDS": 1, # Leave as 1 otherwise wandb logging will not work properly
        "WANDB_MODE": "disabled",  # set to online to activate wandb
        "ENTITY": "odiamond-personal",
        "PROJECT": "feature-attainment-purejaxrl",
        "PRINT_METRICS": True,  # set to False to disable printing metrics
    }

    seeds = copy.deepcopy(config["SEED"])

    for seed in seeds:
        config["SEED"] = seed
        current_time = datetime.datetime.now().strftime("%y-%d-%H-%M-%S")
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["DQN_MULTITASK", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
            name=f'purejaxrl_dqn_multitask_{config["ENV_NAME"]}_{current_time}',
            config=config,
            mode=config["WANDB_MODE"],
        )

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        runner_state = outs["runner_state"]
        # This is a quick fix for now but eventually we might want to vmap the plotting over all of the seeds. 
        # NOTE This will currently fail if we try to squeeze after runnning with more than 1 random seed
        params = jax.tree_util.tree_map(lambda x: x.squeeze(axis=0), runner_state[0].params)
        plot_qvals(params, rng, config)

if __name__ == "__main__":
    main()
