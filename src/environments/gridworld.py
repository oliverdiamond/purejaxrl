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
    has_key2: jax.Array = struct.field(default_factory=lambda: jnp.array(False))
    key_loc2: jax.Array = struct.field(default_factory=lambda: jnp.array([0, 0]))

@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 500

class Gridworld(environment.Environment[EnvState, EnvParams]):
    def __init__(
        self,
        H: int,
        W: int,
        goal_loc: jax.Array,
        goal_reward: float = 1.0,
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
    
class Maze9x9Mixin:
    """Mixin for a 9x9 Maze layout."""
    H: int  # Type hint for mixin - provided by Gridworld base class
    W: int  # Type hint for mixin - provided by Gridworld base class
    
    def _get_obstacles_map(self):
        _map = jnp.zeros([self.H, self.W])
        _map = _map.at[1, 0:4].set(1.0)
        _map = _map.at[1, 5:].set(1.0)
        _map = _map.at[2, 4].set(1.0)
        _map = _map.at[3, 4].set(1.0)
        _map = _map.at[4, 1:5].set(1.0)
        _map = _map.at[4, 6:].set(1.0)
        _map = _map.at[5, 1].set(1.0)
        _map = _map.at[5, 4].set(1.0)
        _map = _map.at[5, 6].set(1.0)
        _map = _map.at[6, 1].set(1.0)
        _map = _map.at[6, 4].set(1.0)
        _map = _map.at[6, 6].set(1.0)
        _map = _map.at[7, 1:5].set(1.0)
        _map = _map.at[7, 6:].set(1.0)
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
    
class EmptyRoomMixin:
    """Mixin for an empty room with no obstacles."""
    H: int  # Type hint for mixin - provided by Gridworld base class
    W: int  # Type hint for mixin - provided by Gridworld base class
    
    def _get_obstacles_map(self):
        return jnp.zeros([self.H, self.W])

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
    fixed_key_loc: jax.Array  # Type hint for mixin - optional fixed key location
    use_fixed_key_loc: bool  # Type hint for mixin - whether to use fixed key location
    def __init__(self, fixed_key_loc: jax.Array, use_fixed_key_loc: bool = False, **kwargs):
        # 1. Initialize Superclass (Gridworld)
        super().__init__(**kwargs) 
        
        # 2. Set Mixin variables
        self.fixed_key_loc = fixed_key_loc
        self.use_fixed_key_loc = use_fixed_key_loc
        
        # 3. RE-CALCULATE start locations. 
        # We must do this here because the superclass __init__ called _get_start_locs 
        # BEFORE we set self.use_fixed_key_loc.
        self._start_locs = self._get_start_locs()

    def _get_start_locs(self):
        """
        Get all valid starting locations.
        Logic: Valid = (Not Obstacle) AND (Not Goal) AND (Not Fixed Key Loc).
        """
        # 1. Basic mask: Not a wall
        valid_locs_mask = self._obstacles_map == 0.0
        
        # 2. Exclude Goal
        valid_locs_mask = valid_locs_mask.at[self.goal_loc[0], self.goal_loc[1]].set(False)
        
        # 3. Exclude Fixed Key Location (if using fixed key)
        # We use Python 'if' because this runs during initialization (not JIT compiled)
        if hasattr(self, 'use_fixed_key_loc') and self.use_fixed_key_loc:
            valid_locs_mask = valid_locs_mask.at[self.fixed_key_loc[0], self.fixed_key_loc[1]].set(False)
        valid_locs = jnp.argwhere(valid_locs_mask)
        return valid_locs

    def reset_env(
        self, key: jax.Array, params: EnvParams
        ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling distinct start loc and key loc."""
        key, subkey = jax.random.split(key)
        
        num_start_locs = self._start_locs.shape[0]

        # Branch A: Use Fixed Key
        # Agent spawns anywhere in _start_locs (which already excludes the key loc)
        def reset_fixed_key():
            idx = jax.random.randint(subkey, (), 0, num_start_locs)
            a_loc = self._start_locs[idx]
            k_loc = self.fixed_key_loc
            return a_loc, k_loc

        # Branch B: Use Random Key
        # We need 2 distinct locations from _start_locs.
        # We sample 2 indices without replacement.
        def reset_random_key():
            # shape=(2,) ensures we get two indices. replace=False ensures they are unique.
            indices = jax.random.choice(subkey, num_start_locs, shape=(2,), replace=False)
            a_loc = self._start_locs[indices[0]] # First index for Agent
            k_loc = self._start_locs[indices[1]] # Second index for Key
            return a_loc, k_loc

        # Execute logic
        start_loc, key_loc = jax.lax.cond(
            self.use_fixed_key_loc,
            reset_fixed_key,
            reset_random_key
        )

        state = EnvState(
            time=0,
            agent_loc=start_loc,
            has_key=jnp.array(False),
            key_loc=key_loc,
        )

        return self.get_obs(state, params), state
    
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
        # at_goal = jnp.array_equal(state.agent_loc, self.goal_loc)
        # goal_reward = jnp.asarray(
        #     jnp.logical_and(at_goal, state.has_key),
        #     dtype=jnp.float32,
        # ) * self.goal_reward
        # penalty = self._penalty_map[state.agent_loc[0], state.agent_loc[1]]
        # reward = goal_reward + penalty

        reward = jnp.array(-1.0) # Changed to constant step cost

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

class KeyCollectionMixinSparseReward:
    """Mixin that requires collecting a key before reaching the goal with sparse rewards (1 at goal, 0 otherwise)."""
    H: int  # Type hint for mixin - provided by Gridworld base class
    W: int  # Type hint for mixin - provided by Gridworld base class
    _obstacles_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _rgb_template: jax.Array  # Type hint for mixin - provided by GridworldRGB base class
    goal_loc: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _penalty_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    goal_reward: float  # Type hint for mixin - provided by Gridworld base class
    directions: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _start_locs: jax.Array  # Type hint for mixin - provided by Gridworld base class
    fixed_key_loc: jax.Array  # Type hint for mixin - optional fixed key location
    use_fixed_key_loc: bool  # Type hint for mixin - whether to use fixed key location
    def __init__(self, fixed_key_loc: jax.Array, use_fixed_key_loc: bool = False, **kwargs):
        # 1. Initialize Superclass (Gridworld)
        super().__init__(**kwargs) 
        
        # 2. Set Mixin variables
        self.fixed_key_loc = fixed_key_loc
        self.use_fixed_key_loc = use_fixed_key_loc
        
        # 3. RE-CALCULATE start locations. 
        # We must do this here because the superclass __init__ called _get_start_locs 
        # BEFORE we set self.use_fixed_key_loc.
        self._start_locs = self._get_start_locs()

    def _get_start_locs(self):
        """
        Get all valid starting locations.
        Logic: Valid = (Not Obstacle) AND (Not Goal) AND (Not Fixed Key Loc).
        """
        # 1. Basic mask: Not a wall
        valid_locs_mask = self._obstacles_map == 0.0
        
        # 2. Exclude Goal
        valid_locs_mask = valid_locs_mask.at[self.goal_loc[0], self.goal_loc[1]].set(False)
        
        # 3. Exclude Fixed Key Location (if using fixed key)
        # We use Python 'if' because this runs during initialization (not JIT compiled)
        if hasattr(self, 'use_fixed_key_loc') and self.use_fixed_key_loc:
            valid_locs_mask = valid_locs_mask.at[self.fixed_key_loc[0], self.fixed_key_loc[1]].set(False)
        valid_locs = jnp.argwhere(valid_locs_mask)
        return valid_locs

    def reset_env(
        self, key: jax.Array, params: EnvParams
        ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling distinct start loc and key loc."""
        key, subkey = jax.random.split(key)
        
        num_start_locs = self._start_locs.shape[0]

        # Branch A: Use Fixed Key
        # Agent spawns anywhere in _start_locs (which already excludes the key loc)
        def reset_fixed_key():
            idx = jax.random.randint(subkey, (), 0, num_start_locs)
            a_loc = self._start_locs[idx]
            k_loc = self.fixed_key_loc
            return a_loc, k_loc

        # Branch B: Use Random Key
        # We need 2 distinct locations from _start_locs.
        # We sample 2 indices without replacement.
        def reset_random_key():
            # shape=(2,) ensures we get two indices. replace=False ensures they are unique.
            indices = jax.random.choice(subkey, num_start_locs, shape=(2,), replace=False)
            a_loc = self._start_locs[indices[0]] # First index for Agent
            k_loc = self._start_locs[indices[1]] # Second index for Key
            return a_loc, k_loc

        # Execute logic
        start_loc, key_loc = jax.lax.cond(
            self.use_fixed_key_loc,
            reset_fixed_key,
            reset_random_key
        )

        state = EnvState(
            time=0,
            agent_loc=start_loc,
            has_key=jnp.array(False),
            key_loc=key_loc,
        )

        return self.get_obs(state, params), state
    
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition with key collection logic and sparse rewards."""
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

        # Calculate reward - only give reward of 1 when reaching goal with key, 0 otherwise
        at_goal = jnp.array_equal(state.agent_loc, self.goal_loc)
        goal_reward = jnp.asarray(
            jnp.logical_and(at_goal, state.has_key),
            dtype=jnp.float32,
        )
        reward = goal_reward

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

class TwoKeyCollectionMixin:
    """Mixin that requires collecting two keys before reaching the goal."""
    H: int  # Type hint for mixin - provided by Gridworld base class
    W: int  # Type hint for mixin - provided by Gridworld base class
    _obstacles_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _rgb_template: jax.Array  # Type hint for mixin - provided by GridworldRGB base class
    goal_loc: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _penalty_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    goal_reward: float  # Type hint for mixin - provided by Gridworld base class
    directions: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _start_locs: jax.Array  # Type hint for mixin - provided by Gridworld base class
    fixed_key_loc: jax.Array  # Type hint for mixin - optional fixed key location
    fixed_key_loc2: jax.Array  # Type hint for mixin - optional fixed second key location
    use_fixed_key_loc: bool  # Type hint for mixin - whether to use fixed key locations
    
    def __init__(self, fixed_key_loc: jax.Array, fixed_key_loc2: jax.Array, use_fixed_key_loc: bool = False, **kwargs):
        # 1. Initialize Superclass (Gridworld)
        super().__init__(**kwargs) 
        
        # 2. Set Mixin variables
        self.fixed_key_loc = fixed_key_loc
        self.fixed_key_loc2 = fixed_key_loc2
        self.use_fixed_key_loc = use_fixed_key_loc
        
        # 3. RE-CALCULATE start locations. 
        self._start_locs = self._get_start_locs()

    def _get_start_locs(self):
        """
        Get all valid starting locations.
        Logic: Valid = (Not Obstacle) AND (Not Goal) AND (Not Fixed Key Locs).
        """
        # 1. Basic mask: Not a wall
        valid_locs_mask = self._obstacles_map == 0.0
        
        # 2. Exclude Goal
        valid_locs_mask = valid_locs_mask.at[self.goal_loc[0], self.goal_loc[1]].set(False)
        
        # 3. Exclude Fixed Key Locations (if using fixed keys)
        if hasattr(self, 'use_fixed_key_loc') and self.use_fixed_key_loc:
            valid_locs_mask = valid_locs_mask.at[self.fixed_key_loc[0], self.fixed_key_loc[1]].set(False)
            valid_locs_mask = valid_locs_mask.at[self.fixed_key_loc2[0], self.fixed_key_loc2[1]].set(False)
        valid_locs = jnp.argwhere(valid_locs_mask)
        return valid_locs

    def reset_env(
        self, key: jax.Array, params: EnvParams
        ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling distinct start loc and two key locs."""
        key, subkey = jax.random.split(key)
        
        num_start_locs = self._start_locs.shape[0]

        # Branch A: Use Fixed Keys
        # Agent spawns anywhere in _start_locs (which already excludes the key locs)
        def reset_fixed_keys():
            idx = jax.random.randint(subkey, (), 0, num_start_locs)
            a_loc = self._start_locs[idx]
            k_loc = self.fixed_key_loc
            k_loc2 = self.fixed_key_loc2
            return a_loc, k_loc, k_loc2

        # Branch B: Use Random Keys
        # We need 3 distinct locations from _start_locs.
        def reset_random_keys():
            # shape=(3,) ensures we get three indices. replace=False ensures they are unique.
            indices = jax.random.choice(subkey, num_start_locs, shape=(3,), replace=False)
            a_loc = self._start_locs[indices[0]]  # First index for Agent
            k_loc = self._start_locs[indices[1]]  # Second index for Key 1
            k_loc2 = self._start_locs[indices[2]] # Third index for Key 2
            return a_loc, k_loc, k_loc2

        # Execute logic
        start_loc, key_loc, key_loc2 = jax.lax.cond(
            self.use_fixed_key_loc,
            reset_fixed_keys,
            reset_random_keys
        )

        state = EnvState(
            time=0,
            agent_loc=start_loc,
            has_key=jnp.array(False),
            key_loc=key_loc,
            has_key2=jnp.array(False),
            key_loc2=key_loc2,
        )

        return self.get_obs(state, params), state
    
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition with two key collection logic."""
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
        
        # Check if agent collected either key
        collected_key = jnp.array_equal(agent_loc_new, state.key_loc)
        has_key = jnp.logical_or(state.has_key, collected_key)
        
        collected_key2 = jnp.array_equal(agent_loc_new, state.key_loc2)
        has_key2 = jnp.logical_or(state.has_key2, collected_key2)

        state = EnvState(
            time=state.time + 1,
            agent_loc=agent_loc_new,
            has_key=has_key,
            key_loc=state.key_loc,
            has_key2=has_key2,
            key_loc2=state.key_loc2,
        )

        reward = jnp.array(-1.0)  # Constant step cost

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
        """Check whether goal state (with both keys) or episode timeout is reached."""
        # Check number of steps in episode termination condition
        done_steps = self.is_truncated(state, params)  # type: ignore
        # Check if agent has found the goal AND has both keys
        at_goal = jnp.array_equal(state.agent_loc, self.goal_loc)
        has_both_keys = jnp.logical_and(state.has_key, state.has_key2)
        done_goal = jnp.logical_and(at_goal, has_both_keys)
        
        done = jnp.logical_or(done_goal, done_steps)
        return done
    
    def get_obs(
        self,
        state: EnvState,
        params: EnvParams,
        key = None,
    ) -> jax.Array:
        """Return observation with two key locations and agent always bright blue."""
        # N x N image with 3 Channels: [Wall/Key1, Empty, Agent/Key2]
        obs = self._rgb_template.copy()
        
        # Set first key location (dark red if not collected, green if collected)
        key1_visible = jnp.logical_not(state.has_key)
        key1_color = jnp.where(
            key1_visible,
            jnp.array([0.5, 0.0, 0.0]),  # Dark red (0.5) if key not collected
            jnp.array([0.0, 1.0, 0.0])   # Green if key collected (like empty space)
        )
        obs = obs.at[state.key_loc[0], state.key_loc[1]].set(key1_color)
        
        # Set second key location (dark blue if not collected, green if collected)
        key2_visible = jnp.logical_not(state.has_key2)
        key2_color = jnp.where(
            key2_visible,
            jnp.array([0.0, 0.0, 0.5]),  # Dark blue if key not collected
            jnp.array([0.0, 1.0, 0.0])   # Green if key collected (like empty space)
        )
        obs = obs.at[state.key_loc2[0], state.key_loc2[1]].set(key2_color)
        
        # Set agent location - always bright blue
        agent_color = jnp.array([0.0, 0.0, 1.0])  # Bright blue always
        obs = obs.at[state.agent_loc[0], state.agent_loc[1]].set(agent_color)

        return obs

class TwoKeyCollectionMixinSparseReward:
    """Mixin that requires collecting two keys before reaching the goal with sparse rewards (1 at goal, 0 otherwise)."""
    H: int  # Type hint for mixin - provided by Gridworld base class
    W: int  # Type hint for mixin - provided by Gridworld base class
    _obstacles_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _rgb_template: jax.Array  # Type hint for mixin - provided by GridworldRGB base class
    goal_loc: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _penalty_map: jax.Array  # Type hint for mixin - provided by Gridworld base class
    goal_reward: float  # Type hint for mixin - provided by Gridworld base class
    directions: jax.Array  # Type hint for mixin - provided by Gridworld base class
    _start_locs: jax.Array  # Type hint for mixin - provided by Gridworld base class
    fixed_key_loc: jax.Array  # Type hint for mixin - optional fixed key location
    fixed_key_loc2: jax.Array  # Type hint for mixin - optional fixed second key location
    use_fixed_key_loc: bool  # Type hint for mixin - whether to use fixed key locations
    
    def __init__(self, fixed_key_loc: jax.Array, fixed_key_loc2: jax.Array, use_fixed_key_loc: bool = False, **kwargs):
        # 1. Initialize Superclass (Gridworld)
        super().__init__(**kwargs) 
        
        # 2. Set Mixin variables
        self.fixed_key_loc = fixed_key_loc
        self.fixed_key_loc2 = fixed_key_loc2
        self.use_fixed_key_loc = use_fixed_key_loc
        
        # 3. RE-CALCULATE start locations. 
        self._start_locs = self._get_start_locs()

    def _get_start_locs(self):
        """
        Get all valid starting locations.
        Logic: Valid = (Not Obstacle) AND (Not Goal) AND (Not Fixed Key Locs).
        """
        # 1. Basic mask: Not a wall
        valid_locs_mask = self._obstacles_map == 0.0
        
        # 2. Exclude Goal
        valid_locs_mask = valid_locs_mask.at[self.goal_loc[0], self.goal_loc[1]].set(False)
        
        # 3. Exclude Fixed Key Locations (if using fixed keys)
        if hasattr(self, 'use_fixed_key_loc') and self.use_fixed_key_loc:
            valid_locs_mask = valid_locs_mask.at[self.fixed_key_loc[0], self.fixed_key_loc[1]].set(False)
            valid_locs_mask = valid_locs_mask.at[self.fixed_key_loc2[0], self.fixed_key_loc2[1]].set(False)
        valid_locs = jnp.argwhere(valid_locs_mask)
        return valid_locs

    def reset_env(
        self, key: jax.Array, params: EnvParams
        ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling distinct start loc and two key locs."""
        key, subkey = jax.random.split(key)
        
        num_start_locs = self._start_locs.shape[0]

        # Branch A: Use Fixed Keys
        # Agent spawns anywhere in _start_locs (which already excludes the key locs)
        def reset_fixed_keys():
            idx = jax.random.randint(subkey, (), 0, num_start_locs)
            a_loc = self._start_locs[idx]
            k_loc = self.fixed_key_loc
            k_loc2 = self.fixed_key_loc2
            return a_loc, k_loc, k_loc2

        # Branch B: Use Random Keys
        # We need 3 distinct locations from _start_locs.
        def reset_random_keys():
            # shape=(3,) ensures we get three indices. replace=False ensures they are unique.
            indices = jax.random.choice(subkey, num_start_locs, shape=(3,), replace=False)
            a_loc = self._start_locs[indices[0]]  # First index for Agent
            k_loc = self._start_locs[indices[1]]  # Second index for Key 1
            k_loc2 = self._start_locs[indices[2]] # Third index for Key 2
            return a_loc, k_loc, k_loc2

        # Execute logic
        start_loc, key_loc, key_loc2 = jax.lax.cond(
            self.use_fixed_key_loc,
            reset_fixed_keys,
            reset_random_keys
        )

        state = EnvState(
            time=0,
            agent_loc=start_loc,
            has_key=jnp.array(False),
            key_loc=key_loc,
            has_key2=jnp.array(False),
            key_loc2=key_loc2,
        )

        return self.get_obs(state, params), state
    
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition with two key collection logic and sparse rewards."""
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
        
        # Check if agent collected either key
        collected_key = jnp.array_equal(agent_loc_new, state.key_loc)
        has_key = jnp.logical_or(state.has_key, collected_key)
        
        collected_key2 = jnp.array_equal(agent_loc_new, state.key_loc2)
        has_key2 = jnp.logical_or(state.has_key2, collected_key2)

        state = EnvState(
            time=state.time + 1,
            agent_loc=agent_loc_new,
            has_key=has_key,
            key_loc=state.key_loc,
            has_key2=has_key2,
            key_loc2=state.key_loc2,
        )

        # Calculate reward - only give reward of 1 when reaching goal with both keys, 0 otherwise
        at_goal = jnp.array_equal(state.agent_loc, self.goal_loc)
        has_both_keys = jnp.logical_and(state.has_key, state.has_key2)
        goal_reward = jnp.asarray(
            jnp.logical_and(at_goal, has_both_keys),
            dtype=jnp.float32,
        )
        reward = goal_reward

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
        """Check whether goal state (with both keys) or episode timeout is reached."""
        # Check number of steps in episode termination condition
        done_steps = self.is_truncated(state, params)  # type: ignore
        # Check if agent has found the goal AND has both keys
        at_goal = jnp.array_equal(state.agent_loc, self.goal_loc)
        has_both_keys = jnp.logical_and(state.has_key, state.has_key2)
        done_goal = jnp.logical_and(at_goal, has_both_keys)
        
        done = jnp.logical_or(done_goal, done_steps)
        return done
    
    def get_obs(
        self,
        state: EnvState,
        params: EnvParams,
        key = None,
    ) -> jax.Array:
        """Return observation with two key locations and agent always bright blue."""
        # N x N image with 3 Channels: [Wall/Key1, Empty, Agent/Key2]
        obs = self._rgb_template.copy()
        
        # Set first key location (dark red if not collected, green if collected)
        key1_visible = jnp.logical_not(state.has_key)
        key1_color = jnp.where(
            key1_visible,
            jnp.array([0.5, 0.0, 0.0]),  # Dark red (0.5) if key not collected
            jnp.array([0.0, 1.0, 0.0])   # Green if key collected (like empty space)
        )
        obs = obs.at[state.key_loc[0], state.key_loc[1]].set(key1_color)
        
        # Set second key location (dark blue if not collected, green if collected)
        key2_visible = jnp.logical_not(state.has_key2)
        key2_color = jnp.where(
            key2_visible,
            jnp.array([0.0, 0.0, 0.5]),  # Dark blue if key not collected
            jnp.array([0.0, 1.0, 0.0])   # Green if key collected (like empty space)
        )
        obs = obs.at[state.key_loc2[0], state.key_loc2[1]].set(key2_color)
        
        # Set agent location - always bright blue
        agent_color = jnp.array([0.0, 0.0, 1.0])  # Bright blue always
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
    def __init__(self, fixed_key_loc=jnp.array([3, 3])):
        super().__init__(H=15, W=15, goal_loc=jnp.array([9, 9]), fixed_key_loc=fixed_key_loc, use_fixed_key_loc=True, goal_reward=0.0)

    @property
    def name(self) -> str:
        return "MazeRGBWithKey"

class Maze9x9RGBWithKey(KeyCollectionMixin, Maze9x9Mixin, GridworldRGB):
    def __init__(self, fixed_key_loc=jnp.array([1, 1])):
        super().__init__(H=9, W=9, goal_loc=jnp.array([5, 5]), fixed_key_loc=fixed_key_loc, use_fixed_key_loc=True, goal_reward=0.0)

    @property
    def name(self) -> str:
        return "Maze9x9RGBWithKey"

class EmptyRoom7x7RGBWithKey(KeyCollectionMixin, EmptyRoomMixin, GridworldRGB):
    def __init__(self, fixed_key_loc=jnp.array([0, 0])):
        super().__init__(H=7, W=7, goal_loc=jnp.array([6, 6]), fixed_key_loc=fixed_key_loc, use_fixed_key_loc=True, goal_reward=0.0)

    @property
    def name(self) -> str:
        return "EmptyRoom7x7RGBWithKey"

class EmptyRoom7x7RGBWithKeySparseReward(KeyCollectionMixinSparseReward, EmptyRoomMixin, GridworldRGB):
    def __init__(self, fixed_key_loc=jnp.array([0, 0])):
        super().__init__(H=7, W=7, goal_loc=jnp.array([6, 6]), fixed_key_loc=fixed_key_loc, use_fixed_key_loc=True)

    @property
    def name(self) -> str:
        return "EmptyRoom7x7RGBWithKeySparseReward"

class EmptyRoom7x7RGBWithTwoKeys(TwoKeyCollectionMixin, EmptyRoomMixin, GridworldRGB):
    def __init__(self, fixed_key_loc=jnp.array([6, 0]), fixed_key_loc2=jnp.array([0, 6])):
        super().__init__(H=7, W=7, goal_loc=jnp.array([6, 6]), fixed_key_loc=fixed_key_loc, fixed_key_loc2=fixed_key_loc2, use_fixed_key_loc=True, goal_reward=0.0)

    @property
    def name(self) -> str:
        return "EmptyRoom7x7RGBWithTwoKeys"

class EmptyRoom7x7RGBWithTwoKeysSparseReward(TwoKeyCollectionMixinSparseReward, EmptyRoomMixin, GridworldRGB):
    def __init__(self, fixed_key_loc=jnp.array([6, 0]), fixed_key_loc2=jnp.array([0, 6])):
        super().__init__(H=7, W=7, goal_loc=jnp.array([6, 6]), fixed_key_loc=fixed_key_loc, fixed_key_loc2=fixed_key_loc2, use_fixed_key_loc=True)

    @property
    def name(self) -> str:
        return "EmptyRoom7x7RGBWithTwoKeysSparseReward"

class EmptyRoom7x7RGBWithTwoKeysRandomLoc(TwoKeyCollectionMixin, EmptyRoomMixin, GridworldRGB):
    def __init__(self, fixed_key_loc=jnp.array([6, 0]), fixed_key_loc2=jnp.array([0, 6])):
        super().__init__(H=7, W=7, goal_loc=jnp.array([6, 6]), fixed_key_loc=fixed_key_loc, fixed_key_loc2=fixed_key_loc2, use_fixed_key_loc=False, goal_reward=0.0)

    @property
    def name(self) -> str:
        return "EmptyRoom7x7RGBWithTwoKeysRandomLoc"

class EmptyRoom7x7RGBWithTwoKeysSparseRewardRandomLoc(TwoKeyCollectionMixinSparseReward, EmptyRoomMixin, GridworldRGB):
    def __init__(self, fixed_key_loc=jnp.array([6, 0]), fixed_key_loc2=jnp.array([0, 6])):
        super().__init__(H=7, W=7, goal_loc=jnp.array([6, 6]), fixed_key_loc=fixed_key_loc, fixed_key_loc2=fixed_key_loc2, use_fixed_key_loc=False)

    @property
    def name(self) -> str:
        return "EmptyRoom7x7RGBWithTwoKeysSparseRewardRandomLoc"
