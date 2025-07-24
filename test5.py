import jax.numpy as jnp
import jax

@jax.jit
def test_function(x):
    """A simple test function to demonstrate JIT compilation."""
    j = jnp.sin(x)
    return jnp.sin(x) + jnp.cos(x)

test_function(jnp.array([2]))