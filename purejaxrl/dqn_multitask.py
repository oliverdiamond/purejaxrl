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

from wrappers import LogWrapper
from environments.rooms_multitask import TwoRoomsMultiTask

class TaskNet(nn.Module):
    action_dim: int
    n_features: int

    @nn.compact
    def __call__(self, common_input: jnp.ndarray, shared_rep: jnp.ndarray) -> jnp.ndarray:
        # Task-specific representation layer
        task_rep = nn.Dense(self.n_features, name="task_rep")(common_input)
        task_rep = nn.relu(task_rep)

        # Concatenate with the shared representation
        combined_rep = jnp.concatenate([shared_rep, task_rep], axis=-1)

        # Task-specific output head
        q_values = nn.Dense(self.action_dim, name="task_head")(combined_rep)
        return q_values

class MultiTaskMazeQNetwork(nn.Module):
    action_dim: int
    n_tasks: int
    n_features_per_task: int
    n_shared_expand: int
    n_shared_bottleneck: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, task: jnp.ndarray):
        w_conv_init = variance_scaling(scale=math.sqrt(5), mode='fan_avg', distribution='uniform')
        w_init = variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')

        # Conv Backbone
        x = nn.Conv(
            features=32, kernel_size=(4, 4), strides=(1, 1), padding=[(1, 1), (1, 1)],
            kernel_init=w_conv_init, name="conv1"
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=16, kernel_size=(4, 4), strides=(2, 2), padding=[(2, 2), (2, 2)],
            kernel_init=w_conv_init, name="conv2"
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        # Shared representation layer
        shared_rep = nn.Sequential([
            nn.Dense(self.n_shared_expand, kernel_init=w_init, name="shared_rep_expand"),
            nn.Dense(self.n_shared_bottleneck, kernel_init=w_init, name="shared_rep_bottleneck"),
        ], name="shared_rep")(x)
        shared_rep = nn.relu(shared_rep)

        # Task-specific representations and heads
        TaskNets = nn.vmap(
            TaskNet,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=(None, None), # type: ignore
            out_axes=0,
            axis_size=self.n_tasks
        )
        outputs = TaskNets(
            self.action_dim, 
            self.n_features_per_task, 
            name="TaskNets")(x, shared_rep)
        batch_indices = jnp.arange(x.shape[0])
        selected_outputs = outputs[task, batch_indices]

        return selected_outputs


@chex.dataclass(frozen=True)
class MultiTaskTimeStep:
    obs: chex.Array
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
    env = LogWrapper(basic_env) # type: ignore

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
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=config["NUM_ENVS"],
        )
        buffer = buffer.replace( # type: ignore
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        rng = jax.random.PRNGKey(0)  # use a dummy rng here
        _action = basic_env.action_space().sample(rng)
        _, _env_state = env.reset(rng, env_params)
        _obs, _env_state, _reward, _done, _ = env.step(rng, _env_state, _action, env_params)
        _timestep = MultiTaskTimeStep(obs=_obs, task=_env_state.task, action=_action, reward=_reward, done=_done) # type: ignore
        buffer_state = buffer.init(_timestep)

        # INIT NETWORK AND OPTIMIZER
        network = MultiTaskMazeQNetwork(
            action_dim=env.action_space(env_params).n, 
            n_tasks=env_params.n_tasks, # type: ignore
            n_features_per_task=config["N_FEATURES_PER_TASK"],
            n_shared_expand=config["N_SHARED_EXPAND"],
            n_shared_bottleneck=config["N_SHARED_BOTTLENECK"],
            )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1,) + env.observation_space(env_params).shape) # Conv layers need extra dimension for batch size
        init_task = jnp.zeros((1,))
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
            q_vals = network.apply(train_state.params, last_obs)
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
            timestep = MultiTaskTimeStep(obs=last_obs, task=last_env_state.task , action=action, reward=reward, done=done) # type: ignore
            buffer_state = buffer.add(buffer_state, timestep)

            # NETWORKS UPDATE
            def _learn_phase(train_state, rng):

                learn_batch = buffer.sample(buffer_state, rng).experience

                q_next_target = network.apply(
                    train_state.target_network_params, 
                    learn_batch.second.obs, 
                    learn_batch.second.task # Will only differ from learn_batch.first.task if transition was terminal, thus q_next_target is not used
                )  # (batch_size, num_actions)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,) # type: ignore
                target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target
                )

                def _loss_fn(params):
                    q_vals = network.apply(
                        params, 
                        learn_batch.first.obs, 
                        learn_batch.first.task
                    )  # (batch_size, num_actions)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals, # type: ignore
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
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
                "returns": info["returned_episode_returns"].mean(),
            }

            # report on wandb if required
            if config.get("WANDB_MODE", "disabled") == "online":

                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)

                jax.debug.callback(callback, metrics)

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


def main():

    config = {
        "NUM_ENVS": 1,
        "BUFFER_SIZE": 10000,
        "BUFFER_BATCH_SIZE": 32,
        "TOTAL_TIMESTEPS": 3e5, #5e5,
        "EPSILON_START": 0.1, # EPSILON_START==EPSILON_FINISH -> no annealing
        "EPSILON_FINISH": 0.1,
        "EPSILON_ANNEAL_TIME": 1e4,
        "TARGET_UPDATE_INTERVAL": 64,
        "LR": 2.5e-4,
        "LEARNING_STARTS": 1000,
        "TRAINING_INTERVAL": 10,
        "N_FEATURES_PER_TASK": 32,
        "N_SHARED_EXPAND": 128,
        "N_SHARED_BOTTLENECK": 8,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": "CartPole-v1",
        "SEED": [0],
        "NUM_ENVS_PER_SEED": 1,
        "WANDB_MODE": "disabled",  # set to online to activate wandb
        "ENTITY": "odiamond-personal",
        "PROJECT": "feature-attainment-purejaxrl",
    }

    seeds = copy.deepcopy(config["SEED"])

    for seed in seeds:
        config["SEED"] = seed
        current_time = datetime.datetime.now().strftime("%y-%d-%H-%M")
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["DQN_MULTITASK", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
            name=f'purejaxrl_dqn_multitask_{config["ENV_NAME"]}_{current_time}',
            config=config,
            mode=config["WANDB_MODE"],
        )

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_ENVS_PER_SEED"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))


if __name__ == "__main__":
    main()
