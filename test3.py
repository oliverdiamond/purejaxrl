import jax.numpy as jnp
obs = jnp.zeros((4, 4, 3), dtype=jnp.float32)
obs = obs.at[
    :, 2, 0
].set(1)
print(obs)