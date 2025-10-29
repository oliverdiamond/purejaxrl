import jax
import jax.numpy as jnp
import pytest
from archived.maze import Maze, EnvState, EnvParams

@pytest.fixture
def env():
    return Maze()

def test_init(env):
    assert env.N == 15
    assert env.num_actions == 4
    assert env.goal_loc.shape == (2,)
    assert env._obstacles_map.shape == (15, 15)
    assert env._start_locs.shape[0] > 0
    assert env._rgb_template.shape == (15, 15, 3)

def test_reset(env):
    key = jax.random.PRNGKey(0)
    params = env.default_params
    obs, state = env.reset_env(key, params)
    assert obs.shape == env.observation_space(params).shape
    assert state.time == 0
    assert state.agent_loc.shape == (2,)
    # Check if start location is valid
    assert env._obstacles_map[state.agent_loc[0], state.agent_loc[1]] == 0.0
    assert not jnp.array_equal(state.agent_loc, env.goal_loc)

def test_step(env):
    key = jax.random.PRNGKey(0)
    params = env.default_params
    _, state = env.reset_env(key, params)

    # Test a valid move
    action = 0  # Move up
    start_loc = state.agent_loc
    new_loc_expected = jnp.clip(start_loc + env.directions[action], 0, env.N - 1)
    
    # Manually check if the new location is an obstacle
    is_obstacle = env._obstacles_map[new_loc_expected[0], new_loc_expected[1]] == 1.0
    if is_obstacle:
        new_loc_expected = start_loc

    obs, new_state, reward, done, info = env.step_env(key, state, action, params)

    assert obs.shape == env.observation_space(params).shape
    assert new_state.time == state.time + 1
    assert jnp.array_equal(new_state.agent_loc, new_loc_expected)
    
    # Test reaching the goal
    state_at_goal = EnvState(time=0, agent_loc=env.goal_loc)
    _, _, reward_goal, done_goal, _ = env.step_env(key, state_at_goal, 0, params)
    assert reward_goal == 1.0
    assert done_goal

def test_step_obstacle(env):
    key = jax.random.PRNGKey(0)
    params = env.default_params
    
    # Find a location next to an obstacle
    # From the map, (1, 0) is a valid spot, and moving up to (2,0) is an obstacle
    start_loc = jnp.array([3, 0])
    state = EnvState(time=0, agent_loc=start_loc)
    
    action = 0 # up
    
    obs, new_state, reward, done, info = env.step_env(key, state, action, params)
    
    # Agent should not move
    assert jnp.array_equal(new_state.agent_loc, start_loc)
    assert reward == 0.0
    assert not done

def test_is_terminal(env):
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

def test_is_truncated(env):
    params = env.default_params
    # Not truncated
    state = EnvState(time=0, agent_loc=jnp.array([0, 0]))
    assert not env.is_truncated(state, params)

    # Truncated
    state_truncated = EnvState(time=params.max_steps_in_episode, agent_loc=jnp.array([0, 0]))
    assert env.is_truncated(state_truncated, params)

def test_get_obs(env):
    params = env.default_params
    state = EnvState(time=0, agent_loc=jnp.array([1, 1]))
    obs = env.get_obs(state, params)
    assert obs.shape == (15, 15, 3)
    # Check agent location is marked correctly
    assert jnp.array_equal(obs[1, 1], jnp.array([0.0, 0.0, 1.0]))
    # Check wall
    assert jnp.array_equal(obs[2, 0], jnp.array([1.0, 0.0, 0.0]))
    # Check empty space
    assert jnp.array_equal(obs[0, 0], jnp.array([0.0, 1.0, 0.0]))
