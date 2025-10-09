
from functools import partial

import jax
import jax.numpy as jnp
from jax._src.typing import Array, ArrayLike
from jax.nn import relu


@partial(jax.jit, static_argnums=(2))
def fta(x: ArrayLike, eta: float = 2, tiles: int = 20, lower_bound: float = -20, upper_bound: float = 20) -> Array:
    r"""Fuzzy Tiling Activation

    Computes the element-wise function:

    .. math::
        I_{\eta(,+}(x) = I_+(x - \eta) x + I_+(x - \eta)
        \mathrm{fta}(x) = 1 - I_{\eta(,+}\max(c - x, 0) + \max(x - c - \delta, 0))

    For more information, see
    `Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online
    <https://arxiv.org/abs/1911.08068>`_.

    Args:
        x : input array
        eta : sparsity control parameter
        tiles : number of tiles
        lower_bound : lower bound for the input
        upper_bound : upper bound for the input

    Returns:
        An array.
    """
    delta = (upper_bound - lower_bound) / tiles
    c = lower_bound + jnp.arange(tiles) * delta
    c = c[None, :]
    x = x[..., None]
    z = 1 - fuzzy_indicator_function(relu(c - x) + relu(x - delta - c), eta)
    z = z.reshape(x.shape[0], -1)
    return z

@jax.jit
def fuzzy_indicator_function(x: ArrayLike, eta: float):
    return jnp.greater(eta, x).astype(x.dtype) * x + jnp.greater(x, eta).astype(x.dtype)
