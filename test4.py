import math 

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import variance_scaling

class TaskNet(nn.Module):
    action_dim: int
    n_features: int

    def setup(self):
        self.task_rep = nn.Dense(self.n_features, name="task_rep")
        self.task_head = nn.Dense(self.action_dim, name="task_head")
    
    def __call__(self, common_input: jnp.ndarray, shared_rep: jnp.ndarray) -> jnp.ndarray:
        # Task-specific representation layer
        task_rep = self.task_rep(common_input)
        task_rep = nn.relu(task_rep)

        # Concatenate with the shared representation
        combined_rep = jnp.concatenate([shared_rep, task_rep], axis=-1)

        # Task-specific output head
        q_values = self.task_head(combined_rep)
        return q_values
    
    def get_activations(self, common_input: jnp.ndarray, shared_rep: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        task_rep = self.task_rep(common_input)
        task_rep = nn.relu(task_rep)

        combined_rep = jnp.concatenate([shared_rep, task_rep], axis=-1)

        q_values = self.task_head(combined_rep)

        return {
            "task_rep": task_rep,
            "q_values": q_values
        }

class MultiTaskMazeQNetwork(nn.Module):
    action_dim: int
    n_tasks: int
    n_features_per_task: int
    n_shared_expand: int
    n_shared_bottleneck: int
    n_features_conv1: int
    n_features_conv2: int

    def setup(self):
        w_conv_init = variance_scaling(scale=math.sqrt(5), mode='fan_avg', distribution='uniform')
        w_init = variance_scaling(scale=1.0, mode='fan_avg', distribution='uniform')

        # Conv Backbone
        self.conv1 = nn.Conv(
            features=self.n_features_conv1, kernel_size=(4, 4), strides=(1, 1), padding=[(1, 1), (1, 1)],
            kernel_init=w_conv_init, name="conv1"
        )
        self.conv2 = nn.Conv(
            features=self.n_features_conv2, kernel_size=(4, 4), strides=(2, 2), padding=[(2, 2), (2, 2)],
            kernel_init=w_conv_init, name="conv2"
        )
        self.conv_backbone = nn.Sequential([
            self.conv1,
            nn.relu,
            self.conv2,
            nn.relu
        ], name="conv_backbone")

        self.shared_rep = nn.Sequential([
            nn.Dense(self.n_shared_expand, kernel_init=w_init, name="shared_rep_expand"),
            nn.Dense(self.n_shared_bottleneck, kernel_init=w_init, name="shared_rep_bottleneck"),
            nn.relu
        ], name="shared_rep")

        # Task-specific networks
        self.TaskNets = nn.vmap(
            TaskNet,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=(None, None),
            out_axes=0,
            axis_size=self.n_tasks,
            methods=['__call__', 'get_activations']
        )(self.action_dim, self.n_features_per_task, name="TaskNets")

    def __call__(self, x: jnp.ndarray, task: jnp.ndarray):
        # Apply the convolutional backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output

        # Shared representation layer
        shared_rep = self.shared_rep(x)

        # Task-specific outputs
        q_vals = self.TaskNets(x, shared_rep)
        batch_indices = jnp.arange(x.shape[0])
        selected_outputs = q_vals[task, batch_indices]

        return selected_outputs

    def get_activations(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Returns intermediate activations."""
        # Conv Backbone
        x = self.conv_backbone(x)
        x = x.reshape((x.shape[0], -1))  # Flatten the output
        # Shared representation layer
        shared_rep = self.shared_rep(x)
        # Task-specific representations and heads
        task_data = self.TaskNets.get_activations(x, shared_rep)

        task_reps = task_data["task_rep"]
        q_vals = task_data["q_values"]
        
        return {
            "shared_rep": shared_rep,
            "task_rep": task_reps,
            "q_values": q_vals
        }


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
    model = MultiTaskMazeQNetwork(
        action_dim=action_dim, 
        n_tasks=n_tasks,
        n_features_per_task=32,
        n_shared_expand=128,
        n_shared_bottleneck=8,
        n_features_conv1=32,
        n_features_conv2=16
        )

    # 3. Initialize parameters
    print("\nInitializing model parameters:")
    params = model.init(key, dummy_input, dummy_tasks)
    print("\nParameters structure:")
    print_params_tree(params)
    print("\nTesting model with dummy input:")
    outputs = model.apply(params, dummy_batch, dummy_tasks_batch)
    print("Output shape:", outputs.shape)
    
    # 4. Define and jit-compile the apply function
    print("\n Testing JIT-compiled forward pass function:")
    @jax.jit
    def forward_pass(params, x, tasks):
        return model.apply(params, x, tasks)
    
    # 5. Run a forward pass
    output = forward_pass(params, dummy_batch, dummy_tasks_batch)

    # 6. Verify the output shape
    # Expected shape: (n_tasks, batch_size, action_dim)
    print(f"Input shape: {dummy_batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {action_dim})")
    
    # Print detailed view of task-specific parameters
    print("\nTask-specific parameters (first 5 values):")
    vmapped_params = params['params']['TaskNets']
    
    for task_id in range(n_tasks):
        print(f"\nTask {task_id}:")
        # Task representation layer
        task_rep_kernel = vmapped_params['task_rep']['kernel'][task_id]
        task_rep_bias = vmapped_params['task_rep']['bias'][task_id]
        print(f"  task_rep kernel: {task_rep_kernel.flatten()[:5]}")
        print(f"  task_rep bias: {task_rep_bias.flatten()[:5]}")
        
    for task_id in range(n_tasks):
        print(f"\nTask {task_id}:")
        # Task head layer
        task_head_kernel = vmapped_params['task_head']['kernel'][task_id]
        task_head_bias = vmapped_params['task_head']['bias'][task_id]
        print(f"  task_head kernel: {task_head_kernel.flatten()[:5]}")
        print(f"  task_head bias: {task_head_bias.flatten()[:5]}")

    # --- Test get_activations function ---
    print("\n" + "="*60)
    print("TESTING GET_ACTIVATIONS FUNCTIONS")
    print("="*60)
    
    # Test MultiTaskMazeQNetwork get_activations
    print("\n1. Testing MultiTaskMazeQNetwork.get_activations:")
    activations = model.apply(params, dummy_batch, method=model.get_activations)
    
    print(f"Activations keys: {list(activations.keys())}")
    for _key, value in activations.items():
        print(f"  {_key}: shape={value.shape}, dtype={value.dtype}")
        print(f"    first 3 values: {value.flatten()[:3]}")
    
    # Test individual TaskNet get_activations (need to create a single TaskNet for this)
    print("\n2. Testing individual TaskNet.get_activations:")
    
    # Create a single TaskNet for testing
    single_task_net = TaskNet(action_dim=action_dim, n_features=32)
    
    # Create dummy inputs for the TaskNet
    dummy_common_input = jnp.ones((batch_size, 80))  # Flattened conv output size
    dummy_shared_rep = jnp.ones((batch_size, 8))     # Shared bottleneck size
    
    # Initialize the single TaskNet
    single_task_params = single_task_net.init(key, dummy_common_input, dummy_shared_rep)
    
    # Test TaskNet's get_activations
    task_activations = single_task_net.apply(
        single_task_params, 
        dummy_common_input, 
        dummy_shared_rep, 
        method=single_task_net.get_activations
    )
    
    print(f"TaskNet activations keys: {list(task_activations.keys())}")
    for _key, value in task_activations.items():
        print(f"  {_key}: shape={value.shape}, dtype={value.dtype}")
        print(f"    first 3 values: {value.flatten()[:3]}")
    
    # Test consistency between regular call and get_activations
    print("\n3. Testing consistency between __call__ and get_activations:")
    
    # Regular forward pass
    regular_output = single_task_net.apply(single_task_params, dummy_common_input, dummy_shared_rep)
    
    # Output from get_activations
    activations_output = task_activations["q_values"]
    
    # Check if they match
    outputs_match = jnp.allclose(regular_output, activations_output)
    print(f"Regular __call__ output shape: {regular_output.shape}")
    print(f"get_activations q_values shape: {activations_output.shape}")
    print(f"Outputs match: {outputs_match}")
    print(f"Max difference: {jnp.max(jnp.abs(regular_output - activations_output))}")
    
    # Test with different batch sizes
    print("\n4. Testing get_activations with different batch sizes:")
    
    for test_batch_size in [1, 3, 5]:
        test_input = jnp.ones((test_batch_size, img_size, img_size, 3))
        test_activations = model.apply(params, test_input, method=model.get_activations)
        
        print(f"  Batch size {test_batch_size}:")
        for _key, value in test_activations.items():
            print(f"    {_key}: shape={value.shape}")

    # Test JIT compilation of get_activations
    print("\n5. Testing JIT compilation of get_activations:")
    @jax.jit
    def get_activations_jit(params, x):
        return model.apply(params, x, method=model.get_activations)
    
    jit_activations = get_activations_jit(params, dummy_batch)
    for _key, value in jit_activations.items():
        print(f"  {_key}: shape={value.shape}, dtype={value.dtype}")