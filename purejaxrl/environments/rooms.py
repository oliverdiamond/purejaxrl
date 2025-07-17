"""JAX implementation of Four Rooms environment (Sutton et al., 1999).


Source: Comparable to https://github.com/howardh/gym-fourrooms Since gymnax
automatically resets env at done, we abstract different resets
"""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    time: int
    task: int
    hallway_loc: jax.Array
    agent_loc: jax.Array

@struct.dataclass
class EnvParams(environment.EnvParams):
    n_tasks: int = 3
    max_steps_in_episode: int = 500
    N: int = 5
    # TODO Set these to correct values
    goal_loc: jax.Array = jnp.array([0, 4])
    start_loc: jax.Array = jnp.array([0, 0])
    hallway_locs: jax.Array = jnp.array([[0, 2], [2, 2], [4, 2]])


class TwoRoomsMultiTask(environment.Environment[EnvState, EnvParams]):
    def __init__(
        self,
    ):
        super().__init__()
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition."""
        # Get new agent location based on action
        agent_loc_new = jnp.clip(
            state.agent_loc + self.directions[action], 0, params.N - 1
        )
        on_obstacle = (agent_loc_new[1] == state.hallway_loc[1]) and (
            agent_loc_new[0] != state.hallway_loc[0]
        )
        agent_loc_new = jax.lax.select(
            on_obstacle, state.agent_loc, agent_loc_new
        )

        state = EnvState(
            time=state.time + 1,
            task=state.task,
            hallway_loc=state.hallway_loc,
            agent_loc=agent_loc_new,
        )

        reward = jnp.asarray(
            jnp.array_equal(state.agent_loc, params.goal_loc),
            dtype=jnp.float32,
        )

        done = self.is_terminal(state, params)
        discount = self.discount(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state, params=params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"discount": discount}
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling hallway and start loc."""
        # Reset both the agents position (deterministic) and the hallway location (random)
        hallway_idx = jax.random.randint(
            key, (), 0, params.hallway_locs.shape[0]
        )
        hallway_loc = params.hallway_locs[hallway_idx]

        state = EnvState(
            time=0,
            task=hallway_idx.item(),
            hallway_loc=hallway_loc,
            agent_loc=params.start_loc,
    )

        return self.get_obs(state, params), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        # Check if agent has found the goal
        done_goal = jnp.array_equal(state.agent_loc, params.goal_loc)
        
        done = jnp.logical_or(done_goal, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRoomsMultiTask"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (params.N, params.N, 3), jnp.float32)


    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "hallway_loc": spaces.Box(
                    0,
                    params.N,
                    (2,),
                    jnp.float32,
                ),
                "agent_loc": spaces.Box(
                    0,
                    params.N,
                    (2,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "task": spaces.Discrete(params.n_tasks),
            }
        )
