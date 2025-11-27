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
    has_key: jax.Array = struct.field(default_factory=lambda: jnp.array(False))
    key_loc: jax.Array = struct.field(default_factory=lambda: jnp.array([0, 0]))

@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 500

class Gridworld(environment.Environment[EnvState, EnvParams]):
    def __init__(
        self,
        H: int,
        W: int,
        goal_loc: jax.Array
    ):
        super().__init__()
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.directions_str = ["up", "right", "down", "left"]
        self.H = H
        self.W = W
        self.goal_loc = goal_loc
        self._obstacles_map = self._get_obstacles_map()
        self._penalty_map = self._get_penalty_map()
        self._start_locs = self._get_start_locs()
        self.goal_reward = 1.0

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def _get_obstacles_map(self) -> jax.Array:
        raise NotImplementedError

    def _get_penalty_map(self) -> jax.Array:
        """Get map of penalty regions. Default is no penalty regions."""
        return jnp.zeros([self.H, self.W])

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
        agent_loc_new = state.agent_loc + self.directions[action]
        agent_loc_new = jnp.array([
            jnp.clip(agent_loc_new[0], 0, self.H - 1),
            jnp.clip(agent_loc_new[1], 0, self.W - 1)
        ])
        on_obstacle = self._obstacles_map[agent_loc_new[0], agent_loc_new[1]] == 1.0
        agent_loc_new = jax.lax.select(
            on_obstacle, state.agent_loc, agent_loc_new
        )

        state = EnvState(
            time=state.time + 1,
            agent_loc=agent_loc_new,
        )

        # Calculate reward
        goal_reward = jnp.asarray(
            jnp.array_equal(state.agent_loc, self.goal_loc),
            dtype=jnp.float32,
        ) * self.goal_reward
        penalty = self._penalty_map[state.agent_loc[0], state.agent_loc[1]]
        reward = goal_reward + penalty

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
                    jnp.array([self.H, self.W]),
                    (2,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

class GridworldOneHot(Gridworld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create mapping from grid positions to valid indices (excluding obstacles and goal)
        valid_locs_mask = self._obstacles_map == 0.0
        valid_locs_mask = valid_locs_mask.at[self.goal_loc[0], self.goal_loc[1]].set(False)
        self._valid_locs = jnp.argwhere(valid_locs_mask)
        self._num_valid_locs = self._valid_locs.shape[0]
        
        # Create a lookup table: position -> index in one-hot encoding
        # Initialize with -1 (invalid)
        self._pos_to_idx = jnp.full((self.H, self.W), -1, dtype=jnp.int32)
        # Fill in valid positions with their indices
        for i in range(self._num_valid_locs):
            loc = self._valid_locs[i]
            self._pos_to_idx = self._pos_to_idx.at[loc[0], loc[1]].set(i)

    def get_obs(
        self,
        state: EnvState,
        params: EnvParams,
        key = None,
    ) -> jax.Array:
        """Return observation from raw state info."""
        # Get the index of the agent's location in the valid locations
        agent_idx = self._pos_to_idx[state.agent_loc[0], state.agent_loc[1]]
        obs = jnp.zeros(self._num_valid_locs)
        obs = obs.at[agent_idx].set(1.0)
        
        return obs

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (self._num_valid_locs,), jnp.float32)

    @property
    def name(self) -> str:
        """Environment name."""
        raise NotImplementedError

class GridworldRGB(Gridworld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        return spaces.Box(0, 1, (self.H, self.W, 3), jnp.float32)

    @property
    def name(self) -> str:
        """Environment name."""
        raise NotImplementedError

# Mixin classes for different environment layouts
class MazeMixin:
    """Mixin for Maze layout."""
    H: int  # Type hint for mixin - provided by Gridworld base class
    W: int  # Type hint for mixin - provided by Gridworld base class
    
    def _get_obstacles_map(self):
        _map = jnp.zeros([self.H, self.W])
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
    H: int  # Type hint for mixin - provided by Gridworld base class
    W: int  # Type hint for mixin - provided by Gridworld base class
    _obstacles_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    
    def _get_obstacles_map(self):
        _map = jnp.zeros([self.H, self.W])
        _map = _map.at[:2, 2].set(1.0)
        _map = _map.at[3:, 2].set(1.0)
        return _map

    def _get_start_locs(self):
        # Get all valid starting locations (first room)
        valid_locs_mask = self._obstacles_map == 0.0
        valid_locs_mask = valid_locs_mask.at[self.goal_loc[0], self.goal_loc[1]].set(False) # type: ignore
        valid_locs_mask = valid_locs_mask.at[:, 2:].set(False)
        valid_locs = jnp.argwhere(valid_locs_mask)
        return valid_locs

class TwoRoomsPaperMixin:
    """Mixin for TwoRooms layout from STOMP paper."""
    H: int  # Type hint for mixin - provided by Gridworld base class
    W: int  # Type hint for mixin - provided by Gridworld base class
    _obstacles_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    
    def _get_obstacles_map(self):
        _map = jnp.zeros([self.H, self.W])
        _map = _map.at[:2, 6].set(1.0)
        _map = _map.at[3:, 6].set(1.0)
        return _map
    
    def _get_penalty_map(self):
        _map = jnp.zeros([self.H, self.W])
        _map = _map.at[:5, 1:5].set(-1.0)
        return _map
    
    def _get_start_locs(self):
        return jnp.array([[2, 0]])

class RandomTransitionsMixin:
    """Mixin that adds stochastic transitions (1/3 probability of random action)."""
    H: int  # Type hint for mixin - provided by Gridworld base class
    W: int  # Type hint for mixin - provided by Gridworld base class
    _obstacles_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    directions: jax.Array  # Type hint for mixin - provided by Gridworld base class
    goal_loc: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _penalty_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    goal_reward: float  # Type hint for mixin - provided by Gridworld base class
    
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition with stochastic action selection."""
        # Split key for randomness
        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        # With probability 1/3, choose a random action different from the selected one
        should_randomize = jax.random.uniform(subkey1) < (1.0 / 3.0)
        
        # Generate a random action index from {0, 1, 2} (excluding the selected action)
        # We do this by sampling from 0-2 and shifting if >= action
        random_offset = jax.random.randint(subkey2, (), 0, 3)
        random_action = jnp.where(random_offset >= action, random_offset + 1, random_offset)
        
        # Select actual action: use random_action if should_randomize, else use original action
        actual_action = jnp.where(should_randomize, random_action, action)
        
        # Get new agent location based on actual action
        agent_loc_new = state.agent_loc + self.directions[actual_action]
        agent_loc_new = jnp.array([
            jnp.clip(agent_loc_new[0], 0, self.H - 1),
            jnp.clip(agent_loc_new[1], 0, self.W - 1)
        ])
        on_obstacle = self._obstacles_map[agent_loc_new[0], agent_loc_new[1]] == 1.0
        agent_loc_new = jax.lax.select(
            on_obstacle, state.agent_loc, agent_loc_new
        )

        state = EnvState(
            time=state.time + 1,
            agent_loc=agent_loc_new,
        )

        # Calculate reward
        goal_reward = jnp.asarray(
            jnp.array_equal(state.agent_loc, self.goal_loc),
            dtype=jnp.float32,
        ) * self.goal_reward
        penalty = self._penalty_map[state.agent_loc[0], state.agent_loc[1]]
        reward = goal_reward + penalty

        done = self.is_terminal(state, params)  # type: ignore
        truncated = self.is_truncated(state, params)  # type: ignore

        return (
            jax.lax.stop_gradient(self.get_obs(state, params=params)),  # type: ignore
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"truncated": truncated}
        )

class KeyCollectionMixin:
    """Mixin that requires collecting a key before reaching the goal."""
    H: int  # Type hint for mixin - provided by Gridworld base class
    W: int  # Type hint for mixin - provided by Gridworld base class
    _obstacles_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _rgb_template: jax.Array  # Type hint for mixin - provided by GridworldRGB base class
    goal_loc: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _penalty_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    goal_reward: float  # Type hint for mixin - provided by Gridworld base class
    directions: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _start_locs: jax.Array  # Type hint for mixin - provided by Gridworld base class
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # type: ignore
        self._key_locs = self._get_key_locs()
    
    def _get_key_locs(self):
        """Get all valid key locations (not an obstacle, not the goal, not a start location)."""
        valid_locs_mask = self._obstacles_map == 0.0
        valid_locs_mask = valid_locs_mask.at[self.goal_loc[0], self.goal_loc[1]].set(False)
        # Exclude start locations
        for i in range(self._start_locs.shape[0]):
            valid_locs_mask = valid_locs_mask.at[self._start_locs[i, 0], self._start_locs[i, 1]].set(False)
        valid_locs = jnp.argwhere(valid_locs_mask)
        return valid_locs
    
    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling start loc and key loc."""
        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        # Sample a random starting location
        random_idx = jax.random.randint(subkey1, (), 0, self._start_locs.shape[0])
        start_loc = self._start_locs[random_idx]
        
        # Sample a random key location
        random_key_idx = jax.random.randint(subkey2, (), 0, self._key_locs.shape[0])
        key_loc = self._key_locs[random_key_idx]

        state = EnvState(
            time=0,
            agent_loc=start_loc,
            has_key=jnp.array(False),
            key_loc=key_loc,
        )

        return self.get_obs(state, params), state  # type: ignore
    
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition with key collection logic."""
        # Get new agent location based on action
        agent_loc_new = state.agent_loc + self.directions[action]
        agent_loc_new = jnp.array([
            jnp.clip(agent_loc_new[0], 0, self.H - 1),
            jnp.clip(agent_loc_new[1], 0, self.W - 1)
        ])
        on_obstacle = self._obstacles_map[agent_loc_new[0], agent_loc_new[1]] == 1.0
        agent_loc_new = jax.lax.select(
            on_obstacle, state.agent_loc, agent_loc_new
        )
        
        # Check if agent collected the key
        collected_key = jnp.array_equal(agent_loc_new, state.key_loc)
        has_key = jnp.logical_or(state.has_key, collected_key)

        state = EnvState(
            time=state.time + 1,
            agent_loc=agent_loc_new,
            has_key=has_key,
            key_loc=state.key_loc,
        )

        # Calculate reward - only give goal reward if agent has key
        at_goal = jnp.array_equal(state.agent_loc, self.goal_loc)
        goal_reward = jnp.asarray(
            jnp.logical_and(at_goal, state.has_key),
            dtype=jnp.float32,
        ) * self.goal_reward
        penalty = self._penalty_map[state.agent_loc[0], state.agent_loc[1]]
        reward = goal_reward + penalty

        done = self.is_terminal(state, params)  # type: ignore
        truncated = self.is_truncated(state, params)  # type: ignore

        return (
            jax.lax.stop_gradient(self.get_obs(state, params=params)),  # type: ignore
            jax.lax.stop_gradient(state),
            reward,
            done,
            {"truncated": truncated}
        )
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether goal state (with key) or episode timeout is reached."""
        # Check number of steps in episode termination condition
        done_steps = self.is_truncated(state, params)  # type: ignore
        # Check if agent has found the goal AND has the key
        at_goal = jnp.array_equal(state.agent_loc, self.goal_loc)
        done_goal = jnp.logical_and(at_goal, state.has_key)
        
        done = jnp.logical_or(done_goal, done_steps)
        return done
    
    def get_obs(
        self,
        state: EnvState,
        params: EnvParams,
        key = None,
    ) -> jax.Array:
        """Return observation with key location and agent color based on key possession."""
        # N x N image with 3 Channels: [Wall/Key, Empty, Agent]
        obs = self._rgb_template.copy()
        
        # Set key location (use red/wall channel if key not collected)
        key_visible = jnp.logical_not(state.has_key)
        key_color = jnp.where(
            key_visible,
            jnp.array([0.5, 0.0, 0.0]),  # Dark red (0.5) if key not collected
            jnp.array([0.0, 1.0, 0.0])   # Green if key collected (like empty space)
        )
        obs = obs.at[state.key_loc[0], state.key_loc[1]].set(key_color)
        
        # Set agent location - color depends on whether they have the key
        agent_color = jnp.where(
            state.has_key,
            jnp.array([0.0, 0.0, 0.5]),  # Dark blue if has key
            jnp.array([0.0, 0.0, 1.0])   # Bright blue if no key
        )
        obs = obs.at[state.agent_loc[0], state.agent_loc[1]].set(agent_color)

        return obs

class MazeRGB(MazeMixin, GridworldRGB):
    def __init__(self):
        super().__init__(H=15, W=15, goal_loc=jnp.array([9, 9]))

    @property
    def name(self) -> str:
        return "MazeRGB"

class MazeOneHot(MazeMixin, GridworldOneHot):
    def __init__(self):
        super().__init__(H=15, W=15, goal_loc=jnp.array([9, 9]))

    @property
    def name(self) -> str:
        return "MazeOneHot"

class TwoRoomsOneHot(TwoRoomsMixin, GridworldOneHot):
    def __init__(self):
        super().__init__(H=5, W=5, goal_loc=jnp.array([0, 4]))

    @property
    def name(self) -> str:
        return "TwoRoomsOneHot"

class TwoRoomsRGB(TwoRoomsMixin, GridworldRGB):
    def __init__(self):
        super().__init__(H=5, W=5, goal_loc=jnp.array([0, 4]))

    @property
    def name(self) -> str:
        return "TwoRoomsRGB"
    
class TwoRoomsPaperOneHot(TwoRoomsPaperMixin, GridworldOneHot):
    def __init__(self):
        super().__init__(H=6, W=13, goal_loc=jnp.array([5, 9]))

    @property
    def name(self) -> str:
        return "TwoRoomsPaperOneHot"


class TwoRoomsPaperRGB(TwoRoomsPaperMixin, GridworldRGB):
    def __init__(self):
        super().__init__(H=6, W=13, goal_loc=jnp.array([5, 9]))

    @property
    def name(self) -> str:
        return "TwoRoomsPaperRGB"
    
class TwoRoomsPaperRandom(RandomTransitionsMixin, TwoRoomsPaperMixin, GridworldOneHot):
    def __init__(self):
        super().__init__(H=6, W=13, goal_loc=jnp.array([5, 9]))

    @property
    def name(self) -> str:
        return "TwoRoomsPaperRandom"

class MazeRGBWithKey(KeyCollectionMixin, MazeMixin, GridworldRGB):
    def __init__(self):
        super().__init__(H=15, W=15, goal_loc=jnp.array([9, 9]))

    @property
    def name(self) -> str:
        return "MazeRGBWithKey"