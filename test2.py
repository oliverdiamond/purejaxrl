import jax
import jax.numpy as jnp

# Enable 64-bit precision for consistency if needed, though not required for this example.
# from jax import config
# config.update("jax_enable_x64", True)

@jax.jit
def select_from_3d(data_array, indices_array, tasks_array):
    i = jnp.arange(data_array.shape[0])
    result = data_array[i, tasks_array, indices_array]
    jax.debug.print("Selected values: {}", result)
    return result

# 2. Create some sample data
key = jax.random.PRNGKey(0)
data = jnp.array([
    [
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33]
    ], 
    [
    [40, 41, 42, 43],
    [50, 51, 52, 53],
    [60, 61, 62, 63]
    ],
    [
    [70, 71, 72, 73],
    [80, 81, 82, 83],
    [90, 91, 92, 93]
    ]
    ])

# Indices to select:
# - from row 0, select item at index 1
# - from row 1, select item at index 3
# - from row 2, select item at index 0
indices = jnp.array([1, 3, 0])
tasks = jnp.array([0, 2, 2])

# 3. Call the jitted function and print the result
selected_items = select_from_3d(data, indices, tasks)