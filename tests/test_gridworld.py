import jax
import jax.numpy as jnp
import pytest
from src.environments.gridworld import (
    MazeRGB, MazeOneHot, TwoRoomsRGB, TwoRoomsOneHot,
    EnvState, EnvParams
)

@pytest.fixture
def maze_rgb_env():
    return MazeRGB()

@pytest.fixture
def maze_onehot_env():
    return MazeOneHot()

@pytest.fixture
def tworooms_rgb_env():
    return TwoRoomsRGB()

@pytest.fixture
def tworooms_onehot_env():
    return TwoRoomsOneHot()

# Test MazeRGB
def test_maze_rgb_init(maze_rgb_env):
    assert maze_rgb_env.N == 15
    assert maze_rgb_env.num_actions == 4
    assert maze_rgb_env.goal_loc.shape == (2,)
    assert maze_rgb_env._obstacles_map.shape == (15, 15)
    assert maze_rgb_env._start_locs.shape[0] > 0
    assert maze_rgb_env._rgb_template.shape == (15, 15, 3)
    assert maze_rgb_env.name == "MazeRGB"

def test_maze_rgb_reset(maze_rgb_env):
    key = jax.random.PRNGKey(0)
    params = maze_rgb_env.default_params
    obs, state = maze_rgb_env.reset_env(key, params)
    assert obs.shape == maze_rgb_env.observation_space(params).shape
    assert obs.shape == (15, 15, 3)
    assert state.time == 0
    assert state.agent_loc.shape == (2,)
    # Check if start location is valid
    assert maze_rgb_env._obstacles_map[state.agent_loc[0], state.agent_loc[1]] == 0.0
    assert not jnp.array_equal(state.agent_loc, maze_rgb_env.goal_loc)

def test_maze_rgb_step(maze_rgb_env):
    key = jax.random.PRNGKey(0)
    params = maze_rgb_env.default_params
    _, state = maze_rgb_env.reset_env(key, params)

    # Test a valid move
    action = 0  # Move up
    start_loc = state.agent_loc
    new_loc_expected = jnp.clip(start_loc + maze_rgb_env.directions[action], 0, maze_rgb_env.N - 1)
    
    # Manually check if the new location is an obstacle
    is_obstacle = maze_rgb_env._obstacles_map[new_loc_expected[0], new_loc_expected[1]] == 1.0
    if is_obstacle:
        new_loc_expected = start_loc

    obs, new_state, reward, done, info = maze_rgb_env.step_env(key, state, action, params)

    assert obs.shape == maze_rgb_env.observation_space(params).shape
    assert new_state.time == state.time + 1
    assert jnp.array_equal(new_state.agent_loc, new_loc_expected)
    
    # Test reaching the goal
    state_at_goal = EnvState(time=0, agent_loc=maze_rgb_env.goal_loc)
    _, _, reward_goal, done_goal, _ = maze_rgb_env.step_env(key, state_at_goal, 0, params)
    assert reward_goal == 1.0
    assert done_goal

def test_maze_rgb_step_obstacle(maze_rgb_env):
    key = jax.random.PRNGKey(0)
    params = maze_rgb_env.default_params
    
    # Find a location next to an obstacle
    # From the map, (3, 0) is valid, and moving up to (2,0) is an obstacle
    start_loc = jnp.array([3, 0])
    state = EnvState(time=0, agent_loc=start_loc)
    
    action = 0  # up
    
    obs, new_state, reward, done, info = maze_rgb_env.step_env(key, state, action, params)
    
    # Agent should not move
    assert jnp.array_equal(new_state.agent_loc, start_loc)
    assert reward == 0.0
    assert not done

def test_maze_rgb_get_obs(maze_rgb_env):
    params = maze_rgb_env.default_params
    state = EnvState(time=0, agent_loc=jnp.array([1, 1]))
    obs = maze_rgb_env.get_obs(state, params)
    assert obs.shape == (15, 15, 3)
    # Check agent location is marked correctly
    assert jnp.array_equal(obs[1, 1], jnp.array([0.0, 0.0, 1.0]))
    # Check wall
    assert jnp.array_equal(obs[2, 0], jnp.array([1.0, 0.0, 0.0]))
    # Check empty space
    assert jnp.array_equal(obs[0, 0], jnp.array([0.0, 1.0, 0.0]))

# Test MazeOneHot
def test_maze_onehot_init(maze_onehot_env):
    assert maze_onehot_env.N == 15
    assert maze_onehot_env.num_actions == 4
    assert maze_onehot_env.goal_loc.shape == (2,)
    assert maze_onehot_env._obstacles_map.shape == (15, 15)
    assert maze_onehot_env._start_locs.shape[0] > 0
    assert maze_onehot_env.name == "MazeOneHot"

def test_maze_onehot_reset(maze_onehot_env):
    key = jax.random.PRNGKey(0)
    params = maze_onehot_env.default_params
    obs, state = maze_onehot_env.reset_env(key, params)
    assert obs.shape == maze_onehot_env.observation_space(params).shape
    assert obs.shape == (225,)  # 15 * 15
    assert state.time == 0
    assert state.agent_loc.shape == (2,)
    # Check if start location is valid
    assert maze_onehot_env._obstacles_map[state.agent_loc[0], state.agent_loc[1]] == 0.0
    assert not jnp.array_equal(state.agent_loc, maze_onehot_env.goal_loc)
    # Check one-hot encoding
    assert jnp.sum(obs) == 1.0  # Only one position should be 1
    assert jnp.max(obs) == 1.0
    assert jnp.min(obs) == 0.0

def test_maze_onehot_get_obs(maze_onehot_env):
    params = maze_onehot_env.default_params
    state = EnvState(time=0, agent_loc=jnp.array([1, 1]))
    obs = maze_onehot_env.get_obs(state, params)
    assert obs.shape == (225,)  # 15 * 15
    # Check one-hot encoding: position (1,1) should be at index 1*15 + 1 = 16
    expected_idx = 1 * 15 + 1
    assert obs[expected_idx] == 1.0
    assert jnp.sum(obs) == 1.0

def test_maze_onehot_step(maze_onehot_env):
    key = jax.random.PRNGKey(0)
    params = maze_onehot_env.default_params
    _, state = maze_onehot_env.reset_env(key, params)

    action = 1  # Move right
    obs, new_state, reward, done, info = maze_onehot_env.step_env(key, state, action, params)

    assert obs.shape == (225,)
    assert new_state.time == state.time + 1
    assert jnp.sum(obs) == 1.0  # Still one-hot

# Test TwoRoomsRGB
def test_tworooms_rgb_init(tworooms_rgb_env):
    assert tworooms_rgb_env.N == 5
    assert tworooms_rgb_env.num_actions == 4
    assert jnp.array_equal(tworooms_rgb_env.goal_loc, jnp.array([0, 4]))
    assert tworooms_rgb_env._obstacles_map.shape == (5, 5)
    assert tworooms_rgb_env._start_locs.shape[0] > 0
    assert tworooms_rgb_env._rgb_template.shape == (5, 5, 3)
    assert tworooms_rgb_env.name == "TwoRoomsRGB"
    # Check that wall is at column 2
    assert tworooms_rgb_env._obstacles_map[0, 2] == 1.0
    assert tworooms_rgb_env._obstacles_map[1, 2] == 1.0
    assert tworooms_rgb_env._obstacles_map[3, 2] == 1.0
    assert tworooms_rgb_env._obstacles_map[4, 2] == 1.0
    # Gap at row 2, col 2
    assert tworooms_rgb_env._obstacles_map[2, 2] == 0.0

def test_tworooms_rgb_start_locations(tworooms_rgb_env):
    # All start locations should be in the first room (columns 0-1)
    for loc in tworooms_rgb_env._start_locs:
        assert loc[1] < 2  # Column should be 0 or 1

def test_tworooms_rgb_reset(tworooms_rgb_env):
    key = jax.random.PRNGKey(0)
    params = tworooms_rgb_env.default_params
    obs, state = tworooms_rgb_env.reset_env(key, params)
    assert obs.shape == (5, 5, 3)
    assert state.time == 0
    # Check agent starts in first room
    assert state.agent_loc[1] < 2

def test_tworooms_rgb_get_obs(tworooms_rgb_env):
    params = tworooms_rgb_env.default_params
    state = EnvState(time=0, agent_loc=jnp.array([0, 0]))
    obs = tworooms_rgb_env.get_obs(state, params)
    assert obs.shape == (5, 5, 3)
    # Check agent location
    assert jnp.array_equal(obs[0, 0], jnp.array([0.0, 0.0, 1.0]))
    # Check wall at column 2 (except row 2)
    assert jnp.array_equal(obs[0, 2], jnp.array([1.0, 0.0, 0.0]))

# Test TwoRoomsOneHot
def test_tworooms_onehot_init(tworooms_onehot_env):
    assert tworooms_onehot_env.N == 5
    assert tworooms_onehot_env.num_actions == 4
    assert jnp.array_equal(tworooms_onehot_env.goal_loc, jnp.array([0, 4]))
    assert tworooms_onehot_env._obstacles_map.shape == (5, 5)
    assert tworooms_onehot_env._start_locs.shape[0] > 0
    assert tworooms_onehot_env.name == "TwoRoomsOneHot"

def test_tworooms_onehot_reset(tworooms_onehot_env):
    key = jax.random.PRNGKey(0)
    params = tworooms_onehot_env.default_params
    obs, state = tworooms_onehot_env.reset_env(key, params)
    assert obs.shape == (25,)  # 5 * 5
    assert state.time == 0
    # Check agent starts in first room
    assert state.agent_loc[1] < 2
    # Check one-hot encoding
    assert jnp.sum(obs) == 1.0

def test_tworooms_onehot_get_obs(tworooms_onehot_env):
    params = tworooms_onehot_env.default_params
    state = EnvState(time=0, agent_loc=jnp.array([2, 1]))
    obs = tworooms_onehot_env.get_obs(state, params)
    assert obs.shape == (25,)
    # Check one-hot encoding: position (2,1) should be at index 2*5 + 1 = 11
    expected_idx = 2 * 5 + 1
    assert obs[expected_idx] == 1.0
    assert jnp.sum(obs) == 1.0

def test_tworooms_onehot_step(tworooms_onehot_env):
    key = jax.random.PRNGKey(0)
    params = tworooms_onehot_env.default_params
    _, state = tworooms_onehot_env.reset_env(key, params)

    action = 1  # Move right
    obs, new_state, reward, done, info = tworooms_onehot_env.step_env(key, state, action, params)

    assert obs.shape == (25,)
    assert new_state.time == state.time + 1
    assert jnp.sum(obs) == 1.0

# General tests for all environments
@pytest.mark.parametrize("env_fixture", [
    "maze_rgb_env", "maze_onehot_env", "tworooms_rgb_env", "tworooms_onehot_env"
])
def test_is_terminal(env_fixture, request):
    env = request.getfixturevalue(env_fixture)
    params = env.default_params
    
    # Not terminal
    state = EnvState(time=0, agent_loc=jnp.array([0, 0]))
    assert not env.is_terminal(state, params)

    # Terminal by reaching goal
    state_goal = EnvState(time=10, agent_loc=env.goal_loc)
    assert env.is_terminal(state_goal, params)

    # Terminal by timeout
    state_timeout = EnvState(time=params.max_steps_in_episode, agent_loc=jnp.array([0, 0]))
    assert env.is_terminal(state_timeout, params)

@pytest.mark.parametrize("env_fixture", [
    "maze_rgb_env", "maze_onehot_env", "tworooms_rgb_env", "tworooms_onehot_env"
])
def test_is_truncated(env_fixture, request):
    env = request.getfixturevalue(env_fixture)
    params = env.default_params
    
    # Not truncated
    state = EnvState(time=0, agent_loc=jnp.array([0, 0]))
    assert not env.is_truncated(state, params)

    # Truncated
    state_truncated = EnvState(time=params.max_steps_in_episode, agent_loc=jnp.array([0, 0]))
    assert env.is_truncated(state_truncated, params)

@pytest.mark.parametrize("env_fixture", [
    "maze_rgb_env", "maze_onehot_env", "tworooms_rgb_env", "tworooms_onehot_env"
])
def test_action_space(env_fixture, request):
    env = request.getfixturevalue(env_fixture)
    params = env.default_params
    action_space = env.action_space(params)
    assert action_space.n == 4

@pytest.mark.parametrize("env_fixture", [
    "maze_rgb_env", "maze_onehot_env", "tworooms_rgb_env", "tworooms_onehot_env"
])
def test_observation_space(env_fixture, request):
    env = request.getfixturevalue(env_fixture)
    params = env.default_params
    obs_space = env.observation_space(params)
    key = jax.random.PRNGKey(42)
    obs, _ = env.reset_env(key, params)
    assert obs.shape == obs_space.shape

# Test obstacle maps are correctly defined
def test_maze_obstacles_correct(maze_rgb_env):
    # Test a few known obstacle positions from the maze
    assert maze_rgb_env._obstacles_map[2, 0] == 1.0
    assert maze_rgb_env._obstacles_map[2, 5] == 1.0
    assert maze_rgb_env._obstacles_map[5, 5] == 1.0
    # Test some non-obstacle positions
    assert maze_rgb_env._obstacles_map[0, 0] == 0.0
    assert maze_rgb_env._obstacles_map[1, 1] == 0.0

def test_tworooms_obstacles_correct(tworooms_rgb_env):
    # Test wall positions (column 2, except row 2)
    assert tworooms_rgb_env._obstacles_map[0, 2] == 1.0
    assert tworooms_rgb_env._obstacles_map[1, 2] == 1.0
    assert tworooms_rgb_env._obstacles_map[3, 2] == 1.0
    assert tworooms_rgb_env._obstacles_map[4, 2] == 1.0
    # Test gap
    assert tworooms_rgb_env._obstacles_map[2, 2] == 0.0
    # Test non-obstacle positions
    assert tworooms_rgb_env._obstacles_map[0, 0] == 0.0
    assert tworooms_rgb_env._obstacles_map[0, 4] == 0.0
