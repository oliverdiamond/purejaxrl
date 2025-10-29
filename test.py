# import jax
# import flax.linen as nn

# class TestModel(nn.Module):
#     """A simple model to demonstrate sequential layer naming."""
    
#     def setup(self):
#         self.start = nn.Dense(2)
#         self.head = nn.Sequential([
#             nn.Dense(2),
#             nn.relu,
#             nn.Dense(2),
#             nn.relu,
#             nn.Dense(2)
#         ])
        
    
#     def __call__(self, x):
#         # This sequential block is similar to the 'head' in your QNet
#         x = self.start(x)
#         return self.head(x)

# def run_test():
#     """Initializes the model and prints its parameter structure."""
#     print("--- Testing Flax Sequential Layer Naming ---")
    
#     # 1. Create a PRNG key and dummy input
#     key = jax.random.PRNGKey(0)
#     dummy_input = jax.numpy.ones((1, 2)) # (Batch size, features)

#     # 2. Instantiate and initialize the model
#     model = TestModel()
#     params = model.init(key, dummy_input)['params']

#     # 3. Print the parameter structure
#     # The pretty_repr function gives a nice, readable tree
#     print("Initialized model parameter structure:")
#     print(params)

# if __name__ == "__main__":
#     run_test()

import jax
import jax.numpy as jnp

# --- 1. Setup (from your previous steps) ---
n_features = 4
batch_size = 32
n_actions = 2

# A dummy output array with shape (n_features, batch_size, n_actions)
# E.g., (4, 32, 2)
all_outputs = jnp.arange(n_features * batch_size * n_actions).reshape(
    (n_features, batch_size, n_actions)
)

# A dummy array of action indices with shape (batch_size)
# E.g., (32)
# Contains integers from 0 to n_actions-1
key = jax.random.PRNGKey(42)
action_indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=n_actions)

print(f"Shape of all_outputs: {all_outputs.shape}")
print(f"Shape of action_indices: {action_indices.shape}")

# --- 2. Perform the Indexing (The Solution) ---

# Create an array [0, 1, ..., batch_size-1]
batch_indices = jnp.arange(batch_size)

# Indexing `all_outputs` with:
# [
#   :                  (all features),
#   [0, 1, ..., 31]    (all batch items),
#   [a_0, a_1, ..., a_31] (the corresponding action for each batch item)
# ]
selected_outputs = all_outputs[:, batch_indices, action_indices]

# --- 3. Verification ---
print(f"\nShape of selected_outputs: {selected_outputs.shape}")

# Let's check the logic for the first feature (feature_idx=0)
# and the first batch item (batch_idx=0)
f = 0
b = 0

# Get the action that was chosen for batch item 0
action_to_get = action_indices[b]

# Get the value we *expect* to be selected
expected_value = all_outputs[f, b, action_to_get]

# Get the value that *was* selected
actual_value = selected_outputs[f, b]

print(f"\n--- Verification (f=0, b=0) ---")
print(f"Action chosen for batch item 0: {action_to_get}")
print(f"Original values at [0, 0, :]: {all_outputs[f, b, :]}")
print(f"Expected selected value: {expected_value}")
print(f"Actual selected value:   {actual_value}")

assert expected_value == actual_value