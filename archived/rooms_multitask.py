"""JAX implementation of Four Rooms environment (Sutton et al., 1999).


Source: Comparable to https://github.com/howardh/gym-fourrooms Since gymnax
automatically resets env at done, we abstract different resets
"""

from typing import Any
from dataclasses import field

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    time: int
    start_loc: jax.Array
    hallway_loc: jax.Array
    goal_loc: jax.Array
    task: jax.Array
    agent_loc: jax.Array


@struct.dataclass
class EnvParams(environment.EnvParams):
    n_tasks: int = 3
    max_steps_in_episode: int = 500
    start_locs: jax.Array = field(default_factory=lambda: jnp.array([[0, 0], [4, 0], [8, 0]]))
    hallway_locs: jax.Array = field(default_factory=lambda: jnp.array([[0, 4], [4, 4], [8, 4]]))
    goal_locs: jax.Array = field(default_factory=lambda: jnp.array([[0, 8], [4, 8], [8, 8]]))


class TwoRoomsMT(environment.Environment[EnvState, EnvParams]):
    """ 
    Multitask TwoRooms environment where the task is the goal state and the start and hallway states are both random each episode.
    Grid is 9x9. See TwoRoomsMT5 and TwoRoomsMT15 for identical envs with different sized grids.
    """
    def __init__(
        self,
    ):
        super().__init__()
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.directions_str = ["up", "right", "down", "left"]
        self.N: int = 9  # Hardcoded for now, can put in config and pass as an argument to make so you can then set the envparams accordingly in default params.
        # TODO set start loc, goal loc and initalize hallway_locs based on N


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
            state.agent_loc + self.directions[action], 0, self.N - 1
        )
        on_obstacle = jnp.logical_and(
            agent_loc_new[1] == state.hallway_loc[1],
            agent_loc_new[0] != state.hallway_loc[0]
        )
        agent_loc_new = jax.lax.select(
            on_obstacle, state.agent_loc, agent_loc_new
        )

        state = EnvState(
            time=state.time + 1,
            task=state.task,
            start_loc=state.start_loc,
            hallway_loc=state.hallway_loc,
            goal_loc=state.goal_loc,
            agent_loc=agent_loc_new,
        )

        reward = jnp.asarray(
            jnp.array_equal(state.agent_loc, state.goal_loc),
            dtype=jnp.float32,
        )

        done = self.is_terminal(state, params)
        truncated = self.is_truncated(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state, params=params)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"truncated": truncated}
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling hallway and start loc."""
        # Reset agent, hallway and goal location
        start_idx = jax.random.randint(
            key, (), 0, params.start_locs.shape[0]
        )
        start_loc = params.start_locs[start_idx]
        
        hallway_idx = jax.random.randint(
            key, (), 0, params.hallway_locs.shape[0]
        )
        hallway_loc = params.hallway_locs[hallway_idx]

        goal_idx = jax.random.randint(
            key, (), 0, params.goal_locs.shape[0]
        )
        goal_loc = params.goal_locs[goal_idx]

        state = EnvState(
            time=0,
            task=goal_idx,
            start_loc=start_loc,
            hallway_loc=hallway_loc,
            agent_loc=start_loc,
            goal_loc=goal_loc,
    )

        return self.get_obs(state, params), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        done_steps = self.is_truncated(state, params)
        # Check if agent has found the goal
        done_goal = jnp.array_equal(state.agent_loc, state.goal_loc)
        
        done = jnp.logical_or(done_goal, done_steps)
        return done

    def is_truncated(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether episode timeout is reached."""
        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode

        return jnp.array(done_steps)

    def get_obs(
        self,
        state: EnvState,
        params: EnvParams,
        key = None,
    ) -> jax.Array:
        """Return observation from raw state info."""
        # N x N image with 3 Channels: [Wall, Empty, Agent]
        obs = jnp.zeros((self.N, self.N, 3), dtype=jnp.float32)
        obs = obs.at[:, :, 1].set(1) # Set all cells to empty
        
        wall_y = state.hallway_loc[1] # y-coordinate of the hallway
        
        # Set all non-hallway cells in the dividing column to walls
        wall_mask = jnp.arange(self.N) != state.hallway_loc[0]

        # Update empty channel (set to 0 where there are walls)
        obs = obs.at[:, wall_y, 1].set(
            jnp.where(wall_mask, 0, obs[:, wall_y, 1])
        )
        # Update wall channel (set to 1 where there are walls)
        obs = obs.at[:, wall_y, 0].set(
            jnp.where(wall_mask, 1, obs[:, wall_y, 0])
        )
        
        # Set agent location
        obs = obs.at[state.agent_loc[0], state.agent_loc[1], 1].set(0)
        obs = obs.at[state.agent_loc[0], state.agent_loc[1], 2].set(1)
        return obs

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRoomsMT"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self.N, self.N, 3), jnp.float32)


    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "start_loc": spaces.Box(
                    0,
                    self.N,
                    (2,),
                    jnp.float32,
                ),
                "hallway_loc": spaces.Box(
                    0,
                    self.N,
                    (2,),
                    jnp.float32,
                ),
                "goal_loc": spaces.Box(
                    0,
                    self.N,
                    (2,),
                    jnp.float32,
                ),
                "agent_loc": spaces.Box(
                    0,
                    self.N,
                    (2,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "task": spaces.Discrete(params.n_tasks),
            }
        )


class TwoRoomsMT5(TwoRoomsMT):
    """Two Rooms environment with 5x5 grid."""
    
    def __init__(self):
        super().__init__()
        self.N = 5

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(
            n_tasks=3,
            max_steps_in_episode=500,
            start_locs=jnp.array([[0, 0], [2, 0], [4, 0]]),
            hallway_locs=jnp.array([[0, 2], [2, 2], [4, 2]]),
            goal_locs=jnp.array([[0, 4], [2, 4], [4, 4]])
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRoomsMT5"

class TwoRoomsMT15(TwoRoomsMT):
    """Two Rooms environment with 15x15 grid."""

    def __init__(self):
        super().__init__()
        self.N = 15

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(
            n_tasks=3,
            max_steps_in_episode=500,
            start_locs=jnp.array([[0, 0], [7, 0], [14, 0]]),
            hallway_locs=jnp.array([[0, 7], [7, 7], [14, 7]]),
            goal_locs=jnp.array([[0, 14], [7, 14], [14, 14]])
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRoomsMT15"


class TwoRoomsMTFixedStart5(TwoRoomsMT):
    """
    Two Rooms environment with:
    - Fixed start location
    - Random hallway location each episode
    - 5x5 grid.
    """

    def __init__(self):
        super().__init__()
        self.N = 5

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(
            n_tasks=3,
            max_steps_in_episode=500,
            start_locs=jnp.array([[2, 0]]),
            hallway_locs=jnp.array([[0, 2], [2, 2], [4, 2]]),
            goal_locs=jnp.array([[0, 4], [2, 4], [4, 4]])
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRoomsMTFixedStart5"


class TwoRoomsMTFixedHallway5(TwoRoomsMT):
    """
    Two Rooms environment with:
    - Fixed hallway location
    - Random start location each episode
    - 5x5 grid.
    """

    def __init__(self):
        super().__init__()
        self.N = 5

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(
            n_tasks=3,
            max_steps_in_episode=500,
            start_locs=jnp.array([[0, 0], [2, 0], [4, 0]]),
            hallway_locs=jnp.array([[2, 2]]),
            goal_locs=jnp.array([[0, 4], [2, 4], [4, 4]])
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRoomsMTFixedHallway5"



class TwoRoomsMTHallwayAsTask(TwoRoomsMT):
    """Multitask TwoRooms environment where the task is the hallway state and the goal and start states are fixed."""
    def __init__(
        self,
    ):
        super().__init__()
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.directions_str = ["up", "right", "down", "left"]
        self.N: int = 9  # Hardcoded for now, could be parameterized later
        # TODO Hardcode start loc, goal loc and initalize hallway_locs based on N


    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(
            n_tasks=3,
            max_steps_in_episode=500,
            start_locs=jnp.array([[4, 0]]),
            hallway_locs=jnp.array([[0, 4], [4, 4], [8, 4]]),
            goal_locs=jnp.array([[0, 8]])
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling hallway and start loc."""
        # Reset agent, hallway and goal location
        start_idx = jax.random.randint(
            key, (), 0, params.start_locs.shape[0]
        )
        start_loc = params.start_locs[start_idx]
        
        hallway_idx = jax.random.randint(
            key, (), 0, params.hallway_locs.shape[0]
        )
        hallway_loc = params.hallway_locs[hallway_idx]

        goal_idx = jax.random.randint(
            key, (), 0, params.goal_locs.shape[0]
        )
        goal_loc = params.goal_locs[goal_idx]

        state = EnvState(
            time=0,
            task=hallway_idx, # The task is the hallway state
            start_loc=start_loc,
            hallway_loc=hallway_loc,
            agent_loc=start_loc,
            goal_loc=goal_loc,
        )
        return self.get_obs(state, params), state

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRoomsMTHallwayAsTask"


class TwoRoomsMTHallwayAsTask5(TwoRoomsMTHallwayAsTask):
    """Multitask TwoRooms environment where the task is the hallway state and the goal and start states are fixed - 5x5 grid."""

    def __init__(self):
        super().__init__()
        self.N = 5

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(
            n_tasks=3,
            max_steps_in_episode=500,
            start_locs=jnp.array([[2, 0]]),
            hallway_locs=jnp.array([[0, 2], [2, 2], [4, 2]]),
            goal_locs=jnp.array([[0, 4]])
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRoomsMTHallwayAsTask5"

class TwoRoomsMTHallwayAsTask15(TwoRoomsMTHallwayAsTask):
    """Multitask TwoRooms environment where the task is the hallway state and the goal and start states are fixed - 15x15 grid."""
    
    def __init__(self):
        super().__init__()
        self.N = 15

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(
            n_tasks=3,
            max_steps_in_episode=500,
            start_locs=jnp.array([[7, 0]]),
            hallway_locs=jnp.array([[0, 7], [7, 7], [14, 7]]),
            goal_locs=jnp.array([[0, 14]])
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRoomsMTHallwayAsTask15"
    

class TwoRoomsMTHallwayAsTaskRandomStart5(TwoRoomsMTHallwayAsTask):
    """Multitask TwoRooms environment where the task is the hallway state and the goal and start states are fixed - 5x5 grid."""

    def __init__(self):
        super().__init__()
        self.N = 5

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(
            n_tasks=3,
            max_steps_in_episode=500,
            start_locs=jnp.array([[0, 0], [2, 0], [4, 0]]),
            hallway_locs=jnp.array([[0, 2], [2, 2], [4, 2]]),
            goal_locs=jnp.array([[0, 4]])
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "TwoRoomsMTHallwayAsTaskRandomStart5"