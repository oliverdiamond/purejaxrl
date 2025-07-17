"""
Based on:
https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/rooms.py
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces as spaces
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, PRNGKeyArray, Scalar, Int

import feature_attainment.environment as environment


@struct.dataclass
class EnvState(environment.MultiTaskEnvState):
    time: int
    task: int
    hallway_loc: jax.Array
    agent_loc: jax.Array
    prev_agent_loc: jax.Array


@struct.dataclass
class EnvParams(environment.MultiTaskEnvParams):
    n_tasks: int = 3
    switch_task_every: int = 1
    max_steps_in_episode: int = 500
    # TODO Set these to correct values
    N: int = 5
    goal_loc: Array = jnp.array([0, 4])
    start_locs: Array = jnp.array([[0, 0], [2, 0], [4, 0]])
    hallway_locs: Array = jnp.array([[0, 2], [2, 2], [4, 2]])


class TwoRoomsMultiTask(environment.MultiTaskEnvironment[EnvState, EnvParams]):
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
        key: PRNGKeyArray,
        state: EnvState,
        action: Float[ArrayLike, "2"],
        params: EnvParams,
    ) -> Tuple[
        Array,
        EnvState,
        Int[Scalar, ""],
        Float[Array, ""],
        Bool[Array, ""],
        Bool[Array, ""],
        Float[Array, ""],
        Dict[Any, Any],
    ]:
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
            prev_agent_loc=state.agent_loc,
        )

        reward = jnp.asarray(
            jnp.array_equal(state.agent_loc, params.goal_loc),
            dtype=jnp.float32,
        )

        task = jnp.asarray(state.task, dtype=jnp.int32)
        term = self.is_terminal(state, params)
        done = self.is_done(state, params)
        discount = self.discount(state, params)

        return (
            lax.stop_gradient(self.get_obs(state, params=params)),
            lax.stop_gradient(state),
            task,
            reward,
            done,
            term,
            discount,
            {},
        )

    def reset_env(
        self, key: PRNGKeyArray, params: EnvParams,
    ) -> tuple[Array, EnvState]:
        """Reset environment state by sampling hallway and start loc."""
        # Reset both the agents position and the goal location
        key_hallway, key_start = jax.random.split(key, 2)

        start_index = jax.random.randint(
            key_start, (), 0, params.start_locs.shape[0]
        )
        start_loc = params.start_locs[start_index]

        hallway_idx = jax.random.randint(
            key_hallway, (), 0, params.hallway_locs.shape[0]
        )
        hallway_loc = params.hallway_locs[hallway_idx]

        state = EnvState(
            time=0,
            task=hallway_idx.item(),
            hallway_loc=hallway_loc,
            agent_loc=start_loc,
            prev_agent_loc=start_loc,
    )

        return self.get_obs(state, params), state

    def get_obs(
        self,
        state: EnvState,
        params: EnvParams,
        key: Optional[PRNGKeyArray] = None,
    ) -> Array:
        """Return observation from raw state info."""
        obs = jnp.zeros((params.N, params.N, 3), dtype=jnp.float32)
        wall_x = params.N % 2
        obs = obs.at[
            wall_x, jnp.arange(params.N) != state.hallway_loc[0], 0
        ].set(1)
        obs = obs.at[state.agent_loc[0], state.agent_loc[1], 2].set(1)
        task = jnp.where(params.hallway_locs == state.hallway_loc)[0]
        return obs

    def is_terminal(
        self, state: EnvState, params: EnvParams
    ) -> Bool[Scalar, ""]:
        """Check whether env is terminal for **any** reason."""
        cutoff = jnp.array(state.time >= params.max_steps_in_episode)
        return jnp.logical_or(cutoff, self.is_done(state, params))

    def is_done(self, state: EnvState, params: EnvParams) -> Bool[Scalar, ""]:
        """
        Check whether the goal state was reached.

        Since we use transition-based discounting, the episode is done after
        any action was taken in the goal state.
        """
        # TODO Check to make sure this is correct. Still seems strange
        return jnp.array_equal(state.prev_agent_loc, params.goal_loc)

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRooms"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        # TODO Make multitask-env superclass that allows a dict to be returned here so we can supply task idx as well
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
            }
        )
