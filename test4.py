import jax
import jax.numpy as jnp
import flax.linen as nn
import math
from flax.linen.initializers import variance_scaling
from typing import Sequence

class TaskNet(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, common_input: jnp.ndarray, shared_rep: jnp.ndarray) -> jnp.ndarray:
        # Task-specific representation layer
        task_rep = nn.Dense(32, name="task_rep")(common_input)
        task_rep = nn.relu(task_rep)

        # Concatenate with the shared representation
        combined_rep = jnp.concatenate([shared_rep, task_rep], axis=-1)

        # Task-specific output head
        q_values = nn.Dense(self.action_dim, name="task_head")(combined_rep)
        return q_values

class MultiTaskMazeQNetwork(nn.Module):
    action_dim: int
    n_tasks: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, tasks: jnp.ndarray):
        w_conv_init = variance_scaling(scale=math.sqrt(5), mode='fan_avg', distribution='uniform')
        w_init = variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')

        # Conv Backbone
        x = nn.Conv(
            features=32, kernel_size=(4, 4), strides=(1, 1), padding=[(1, 1), (1, 1)],
            kernel_init=w_conv_init, name="conv1"
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=16, kernel_size=(4, 4), strides=(2, 2), padding=[(2, 2), (2, 2)],
            kernel_init=w_conv_init, name="conv2"
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        # Shared representation layer
        shared_rep = nn.Sequential([
            nn.Dense(128, kernel_init=w_init, name="shared_rep_expand"),
            nn.Dense(8, kernel_init=w_init, name="shared_rep_bottleneck"),
        ], name="shared_rep")(x)
        shared_rep = nn.relu(shared_rep)

        # Task-specific representations and heads
        TaskNets = nn.vmap(
            TaskNet,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=(None, None),
            out_axes=0,
            axis_size=self.n_tasks
        )
        outputs = TaskNets(self.action_dim, name="TaskNets")(x, shared_rep)
        jax.debug.print("Task-specific outputs shape: {}", outputs.shape)

        batch_indices = jnp.arange(x.shape[0])
        selected_outputs = outputs[tasks, batch_indices]
        
        return selected_outputs

def print_params_tree(params, indent=0):
    """Recursively print parameter tree with proper indentation."""
    for key, value in params.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_params_tree(value, indent + 1)
        else:
            # For arrays, show shape, dtype, and first 5 values
            if hasattr(value, 'shape'):
                print("  " * indent + f"{key}: shape={value.shape}, dtype={value.dtype}, first_5={value.flatten()[:5]}")
            else:
                print("  " * indent + f"{key}: {value}")

# --- Example Usage ---
if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    batch_size = 2
    n_tasks = 3
    action_dim = 4
    img_size = 5

    # 1. Dummy input data
    dummy_input = jnp.ones((1, img_size, img_size, 3))
    dummy_batch = jnp.ones((batch_size, img_size, img_size, 3))
    dummy_tasks = jnp.array([0])
    dummy_tasks_batch = jnp.array([0, 1])
    
    # 2. Instantiate the network
    model = MultiTaskMazeQNetwork(action_dim=action_dim, n_tasks=n_tasks)

    # 3. Initialize parameters
    # The 'vmapped' dimension of size `n_tasks` will be created automatically.
    params = model.init(key, dummy_input, dummy_tasks)
    print("\nParameters structure:")
    print_params_tree(params)
    outputs = model.apply(params, dummy_batch, dummy_tasks_batch)
    print("Output shape:", outputs.shape)


    # # 1. Dummy input data
    # dummy_input2 = jnp.ones((batch_size, img_size, img_size, 3))
    # dummy_tasks = jnp.array([0, 1])
    
    # # 2. Instantiate the network
    # model = MultiTaskMazeQNetwork(action_dim=action_dim, n_tasks=n_tasks)

    # # 3. Initialize parameters
    # # The 'vmapped' dimension of size `n_tasks` will be created automatically.
    # params = model.init(key, dummy_input, dummy_tasks)
    # print("\nParameters structure with batch:")
    # print_params_tree(params)


    
    # assert 1==2
    
    # # 4. Define and jit-compile the apply function
    # @jax.jit
    # def forward_pass(params, x, tasks):
    #     return model.apply(params, x, tasks)

    # # 5. Run the forward pass
    # keys = jax.random.split(key, num=5)
    # task_lists = [jax.random.randint(k, shape=(2,), minval=1, maxval=4) for k in keys]
    # for tasks in task_lists:
    #     output = forward_pass(params, dummy_input, tasks)
    #     print("Output shape:", output.shape)

    # assert 1 == 2
    # # 6. Verify the output shape
    # # Expected shape: (n_tasks, batch_size, action_dim)
    # print(f"Input shape: {dummy_input.shape}")
    # print(f"Output shape: {output.shape}")
    # print(f"Expected shape: ({n_tasks}, {batch_size}, {action_dim})")
    
    # print("\nParameters structure:")
    # print_params_tree(params)
    
    # # Print detailed view of task-specific parameters
    # print("\nTask-specific parameters (first 5 values):")
    # vmapped_params = params['params']['TaskNets']
    
    # for task_id in range(n_tasks):
    #     print(f"\nTask {task_id}:")
    #     # Task representation layer
    #     task_rep_kernel = vmapped_params['task_rep']['kernel'][task_id]
    #     task_rep_bias = vmapped_params['task_rep']['bias'][task_id]
    #     print(f"  task_rep kernel: {task_rep_kernel.flatten()[:5]}")
    #     print(f"  task_rep bias: {task_rep_bias.flatten()[:5]}")
        
    # for task_id in range(n_tasks):
    #     print(f"\nTask {task_id}:")
    #     # Task head layer
    #     task_head_kernel = vmapped_params['task_head']['kernel'][task_id]
    #     task_head_bias = vmapped_params['task_head']['bias'][task_id]
    #     print(f"  task_head kernel: {task_head_kernel.flatten()[:5]}")
    #     print(f"  task_head bias: {task_head_bias.flatten()[:5]}")
