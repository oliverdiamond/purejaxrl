import flax.linen as nn
import jax
import jax.numpy as jnp

def test_compact_submodule_params():
    """Tests if parameters of submodules defined inside a @nn.compact call method are accessible after init."""

    class MyModule(nn.Module):
        def setup(self):
            self.dense = [nn.Dense(features=10, name=f"Dense_{i}") for i in range(2)]
        def __call__(self, x, task):
            def use_dense0(_):
                return self.dense[0](x)
            def use_dense1(_):
                return self.dense[1](x)
            # Use jax.lax.cond for control flow
            return jax.lax.cond(
                task == 0,
                use_dense0,
                use_dense1,
                operand=None
            )

    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 5))  # Example input
    module = MyModule()
    params = module.init(key, x, 0)
    print("Parameters initialized:", params)
    output = module.apply(params, x, 0)
    
    params_dict = params['params']
    # Check if the parameters of the 'dense' submodule are accessible
    assert 'Dense_0' in params_dict, "Dense_0 submodule parameters not found in params"
    print("Test passed: Parameters of submodules defined inside @nn.compact call method are accessible.")

def test_jit_with_control_flow():
    """Tests if MyModule can be used inside a jitted function with static control flow for 'task'."""
    class MyModule(nn.Module):
        @nn.compact
        def __call__(self, x, task):
            return jax.lax.cond(
                task == 0,
                lambda x: nn.Dense(features=10, name=f"Dense_{0}")(x),
                lambda x: nn.Dense(features=10, name=f"Dense_{1}")(x),
                x
            )

    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 5))
    module = MyModule()
    params = module.init(key, x, 0)
    print(params)

    # Jitted function with static task value
    @jax.jit
    def apply_module(params, x, task):
        return module.apply(params, x, task)

    out0 = apply_module(params, x, 0)
    out1 = apply_module(params, x, 1)
    print("JIT output for task 0:", out0)
    print("JIT output for task 1:", out1)
    print("Test passed: MyModule works inside jitted function with static control flow.")

# Run the test
if __name__ == "__main__":
    #test_compact_submodule_params()

    test_jit_with_control_flow()
