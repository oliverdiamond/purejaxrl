import numpy as np
import jax.numpy as jnp

data = np.load("results/experiments/tests/collect.json/0/dataset.npz")
obs = jnp.array(data["obs"])
action = jnp.array(data["action"])
next_obs = jnp.array(data["next_obs"])
reward = jnp.array(data["reward"])
done = jnp.array(data["done"])
truncated = jnp.array(data["truncated"])
dataset = {
    "obs": obs,
    "action": action,
    "next_obs": next_obs,
    "reward": reward,
    "done": done,
    "truncated": truncated
}
print(f"Dataset shapes - obs: {obs.shape}, action: {action.shape}, next_obs: {next_obs.shape}, reward: {reward.shape}, done: {done.shape}, truncated: {truncated.shape}")
print(f"First 5 observations:\n{obs[:5]}")
print(f"First 5 next observations:\n{next_obs[:5]}")
print(f"First 5 actions:\n{action[:5]}")
print(f"First 5 rewards:\n{reward[:5]}")
print(f"First 5 dones:\n{done[:5]}")
print(f"First 5 truncateds:\n{truncated[:5]}")
