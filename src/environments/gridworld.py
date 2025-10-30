# Modified from: https://github.com/erfanMhi/LTA-Representation-Properties/blob/main/core/environment/gridworlds_goal.py

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    time: int
    agent_loc: jax.Array

@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 500

class Gridworld(environment.Environment[EnvState, EnvParams]):
    def __init__(
        self,
        N=15,
        goal_loc=jnp.array([9, 9])
    ):
        super().__init__()
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.directions_str = ["up", "right", "down", "left"]
        self.N = N
        self.goal_loc = goal_loc
        self._obstacles_map = self._get_obstacles_map()
        self._start_locs = self._get_start_locs()

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def _get_obstacles_map(self) -> jax.Array:
        raise NotImplementedError

    def _get_start_locs(self):
        # Get all valid starting locations (not an obstacle and not the goal)
        valid_locs_mask = self._obstacles_map == 0.0
        valid_locs_mask = valid_locs_mask.at[self.goal_loc[0], self.goal_loc[1]].set(False)
        valid_locs = jnp.argwhere(valid_locs_mask)
        return valid_locs
        
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
        on_obstacle = self._obstacles_map[agent_loc_new[0], agent_loc_new[1]] == 1.0
        agent_loc_new = jax.lax.select(
            on_obstacle, state.agent_loc, agent_loc_new
        )

        state = EnvState(
            time=state.time + 1,
            agent_loc=agent_loc_new,
        )

        reward = jnp.asarray(
            jnp.array_equal(state.agent_loc, self.goal_loc),
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
        """Reset environment state by sampling start loc."""
        
        # Sample a random starting location
        random_idx = jax.random.randint(key, (), 0, self._start_locs.shape[0])
        start_loc = self._start_locs[random_idx]

        state = EnvState(
            time=0,
            agent_loc=start_loc,
        )

        return self.get_obs(state, params), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether goal state or episode timeout is reached."""
        # Check number of steps in episode termination condition
        done_steps = self.is_truncated(state, params)
        # Check if agent has found the goal
        done_goal = jnp.array_equal(state.agent_loc, self.goal_loc)
        
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
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Environment name."""
        raise NotImplementedError

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "agent_loc": spaces.Box(
                    0,
                    self.N,
                    (2,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

class GridworldOneHot(Gridworld):
    def __init__(
        self,
        N=15,
        goal_loc=jnp.array([9, 9])
    ):
        super().__init__(N=N, goal_loc=goal_loc)

    def get_obs(
        self,
        state: EnvState,
        params: EnvParams,
        key = None,
    ) -> jax.Array:
        """Return observation from raw state info."""
        # Create a one-hot encoding of the agent location
        agent_idx = state.agent_loc[0] * self.N + state.agent_loc[1]
        obs = jnp.zeros(self.N * self.N)
        obs = obs.at[agent_idx].set(1.0)
        
        return obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self.N * self.N,), jnp.float32)

    @property
    def name(self) -> str:
        """Environment name."""
        raise NotImplementedError

class GridworldRGB(Gridworld):
    def __init__(
        self,
        N=15,
        goal_loc=jnp.array([9, 9])
    ):
        super().__init__(N=N, goal_loc=goal_loc)
        self._rgb_template = self._get_rgb_template()

    def _get_rgb_template(self):
        # Create a (N, N, 1) boolean mask for obstacles
        obstacle_mask = jnp.expand_dims(self._obstacles_map, axis=-1)

        # Create templates for wall and empty space colors
        wall_color = jnp.array([1.0, 0.0, 0.0])
        empty_color = jnp.array([0.0, 1.0, 0.0])

        # Use jnp.where to select colors based on the obstacle mask
        rgb_template = jnp.where(obstacle_mask, wall_color, empty_color)
        return rgb_template

    def get_obs(
        self,
        state: EnvState,
        params: EnvParams,
        key = None,
    ) -> jax.Array:
        """Return observation from raw state info."""
        # N x N image with 3 Channels: [Wall, Empty, Agent]
        obs = self._rgb_template
        # Set agent location
        obs = obs.at[state.agent_loc[0], state.agent_loc[1]].set(jnp.array([0.0, 0.0, 1.0]))

        return obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self.N, self.N, 3), jnp.float32)

    @property
    def name(self) -> str:
        """Environment name."""
        raise NotImplementedError

# Mixin classes for different environment layouts
class MazeMixin:
    """Mixin for Maze layout."""
    N: int  # Type hint for mixin - provided by Gridworld base class
    
    def _get_obstacles_map(self):
        _map = jnp.zeros([self.N, self.N])
        _map = _map.at[2, 0:6].set(1.0)
        _map = _map.at[2, 8:].set(1.0)
        _map = _map.at[3, 5].set(1.0)
        _map = _map.at[4, 5].set(1.0)
        _map = _map.at[5, 2:7].set(1.0)
        _map = _map.at[5, 9:].set(1.0)
        _map = _map.at[8, 2].set(1.0)
        _map = _map.at[8, 5].set(1.0)
        _map = _map.at[8, 8:].set(1.0)
        _map = _map.at[9, 2].set(1.0)
        _map = _map.at[9, 5].set(1.0)
        _map = _map.at[9, 8].set(1.0)
        _map = _map.at[10, 2].set(1.0)
        _map = _map.at[10, 5].set(1.0)
        _map = _map.at[10, 8].set(1.0)
        _map = _map.at[11, 2:6].set(1.0)
        _map = _map.at[11, 8:12].set(1.0)
        _map = _map.at[12, 5].set(1.0)
        _map = _map.at[13, 5].set(1.0)
        _map = _map.at[14, 5].set(1.0)
        return _map

class TwoRoomsMixin:
    """Mixin for TwoRooms layout."""
    N: int  # Type hint for mixin - provided by Gridworld base class
    _obstacles_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    
    def _get_obstacles_map(self):
        _map = jnp.zeros([self.N, self.N])
        _map = _map.at[:2, 2].set(1.0)
        _map = _map.at[3:, 2].set(1.0)
        return _map
    
    def _get_start_locs(self):
        # Get all valid starting locations (first room)
        valid_locs_mask = self._obstacles_map == 0.0
        valid_locs_mask = valid_locs_mask.at[:, 2:].set(False)
        valid_locs = jnp.argwhere(valid_locs_mask)
        return valid_locs

# Concrete environment classes using mixins
class MazeRGB(MazeMixin, GridworldRGB):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "MazeRGB"

class MazeOneHot(MazeMixin, GridworldOneHot):
    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "MazeOneHot"

class TwoRoomsOneHot(TwoRoomsMixin, GridworldOneHot):
    def __init__(self):
        super().__init__(N=5, goal_loc=jnp.array([0, 4]))

    @property
    def name(self) -> str:
        return "TwoRoomsOneHot"

class TwoRoomsRGB(TwoRoomsMixin, GridworldRGB):
    def __init__(self):
        super().__init__(N=5, goal_loc=jnp.array([0, 4]))

    @property
    def name(self) -> str:
        return "TwoRoomsRGB"