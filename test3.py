import jax
import jax.numpy as jnp
import flax.linen as nn

# Module that creates a batch of MLPs using vmap
class BatchedMLPs(nn.Module):
    num_models: int = 3
    features: int = 64
    
    def setup(self):
        # Create a batch of MLP modules using vmap
        self.mlp_batch = nn.vmap(
            nn.Dense,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=0,  
            out_axes=0,
            axis_size=self.num_models
        )(features=self.features)
    
    def __call__(self, x):
        # x should be shape (input_dim,) and will be broadcast to all models
        return self.mlp_batch(x)


def test_batched_mlps():
    print("Testing Flax vmap with batch of modules...")
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    model1 = BatchedMLPs(num_models=3, features=32)
    
    # Single input broadcasted to all models
    x_single = jnp.ones((10,))  # Input dimension: 10

    # Initialize parameters
    params1 = model1.init(key, x_single)
    
    # Forward pass
    output = model1.apply(params1, x_single)
    jax.debug.print(output)
    # print(f"Input shape: {x_single.shape}")
    # print(f"Output shape: {output.shape}")  # Should be (3, 1)
    # print(f"Output values: {output.flatten()}")

    # JIT compile
    @jax.jit
    def apply_model():
        return model1.apply(params1, x_single)
    
    out = apply_model()
    jax.debug.print(out)


if __name__ == "__main__":
    test_batched_mlps()
