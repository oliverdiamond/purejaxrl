import jax
import jax.numpy as jnp
import pytest
from src.util.fta import fta, fuzzy_indicator_function


class TestFuzzyIndicatorFunction:
    """Tests for the fuzzy_indicator_function helper."""
    
    def test_below_threshold(self):
        """Test values below eta threshold."""
        x = jnp.array([0.5, 1.0, 1.5])
        eta = 2.0
        result = fuzzy_indicator_function(x, eta)
        # When x < eta, result should be x
        assert jnp.allclose(result, x)
    
    def test_above_threshold(self):
        """Test values above eta threshold."""
        x = jnp.array([3.0, 4.0, 5.0])
        eta = 2.0
        result = fuzzy_indicator_function(x, eta)
        # When x > eta, result should be 1
        assert jnp.allclose(result, jnp.ones_like(x))
    
    def test_at_threshold(self):
        """Test values at eta threshold."""
        x = jnp.array([2.0, 2.0, 2.0])
        eta = 2.0
        result = fuzzy_indicator_function(x, eta)
        # At threshold, should return 0
        assert jnp.allclose(result, jnp.zeros_like(x))
    
    def test_mixed_values(self):
        """Test mixed values around threshold."""
        x = jnp.array([0.5, 2.0, 3.5])
        eta = 2.0
        result = fuzzy_indicator_function(x, eta)
        expected = jnp.array([0.5, 0.0, 1.0])
        assert jnp.allclose(result, expected)


class TestFTA:
    """Tests for the Fuzzy Tiling Activation function."""
    
    def test_basic_shape(self):
        """Test that output shape is correct."""
        x = jnp.array([0.0, 1.0, 2.0])
        tiles = 10
        result = fta(x, tiles=tiles)
        # Output should be (batch_size, tiles)
        assert result.shape == (3, tiles)
    
    def test_single_value(self):
        """Test with a single value."""
        x = jnp.array([0.0])
        tiles = 20
        result = fta(x, tiles=tiles)
        assert result.shape == (1, tiles)
        # All values should be between 0 and 1
        assert jnp.all(result >= 0) and jnp.all(result <= 1)
    
    def test_output_range(self):
        """Test that output values are in [0, 1] range."""
        x = jnp.linspace(-40, 40, 100)
        result = fta(x)
        assert jnp.all(result >= 0) and jnp.all(result <= 1)
    
    def test_different_eta_values(self):
        """Test with different eta (sparsity) parameters."""
        x = jnp.array([0.0, 5.0, 10.0])
        
        # Smaller eta should produce sparser representations
        result_sparse = fta(x, eta=0.5)
        result_dense = fta(x, eta=5.0)
        
        # Count non-zero activations (above threshold)
        threshold = 0.1
        sparse_active = jnp.sum(result_sparse > threshold)
        dense_active = jnp.sum(result_dense > threshold)
        
        # Dense should have more active tiles
        assert dense_active >= sparse_active
    
    def test_different_tile_counts(self):
        """Test with different numbers of tiles."""
        x = jnp.array([0.0, 5.0, 10.0])
        
        result_10 = fta(x, tiles=10)
        result_20 = fta(x, tiles=20)
        result_40 = fta(x, tiles=40)
        
        assert result_10.shape == (3, 10)
        assert result_20.shape == (3, 20)
        assert result_40.shape == (3, 40)
    
    def test_custom_bounds(self):
        """Test with custom lower and upper bounds."""
        x = jnp.array([0.0, 5.0, 10.0])
        
        # Test with bounds [0, 10]
        result = fta(x, tiles=10, lower_bound=0, upper_bound=10)
        assert result.shape == (3, 10)
        assert jnp.all(result >= 0) and jnp.all(result <= 1)
        
        # Test with bounds [-5, 5]
        result = fta(x, tiles=10, lower_bound=-5, upper_bound=5)
        assert result.shape == (3, 10)
        assert jnp.all(result >= 0) and jnp.all(result <= 1)
    
    def test_values_outside_bounds(self):
        """Test behavior with values outside specified bounds."""
        # Values outside bounds should still produce valid output
        x = jnp.array([-30.0, 30.0])
        result = fta(x, lower_bound=-20, upper_bound=20)
        
        assert result.shape == (2, 20)
        assert jnp.all(result >= 0) and jnp.all(result <= 1)
    
    def test_center_of_tile(self):
        """Test that values at tile centers have high activation."""
        tiles = 10
        lower_bound = 0
        upper_bound = 10
        delta = (upper_bound - lower_bound) / tiles
        
        # Create input at center of first tile
        tile_center = lower_bound + 0.5 * delta
        x = jnp.array([tile_center])
        
        result = fta(x, tiles=tiles, lower_bound=lower_bound, upper_bound=upper_bound)
        
        # The activation at the first tile should 1
        assert result[0, 0] == 1
    
    def test_batch_processing(self):
        """Test with a batch of inputs."""
        batch_size = 32
        x = jnp.linspace(-10, 10, batch_size)
        tiles = 20
        
        result = fta(x, tiles=tiles)
        
        assert result.shape == (batch_size, tiles)
        assert jnp.all(result >= 0) and jnp.all(result <= 1)
    
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        def loss_fn(x):
            return jnp.sum(fta(x, tiles=10))
        
        x = jnp.array([0.0, 1.0, 2.0])
        grad_fn = jax.grad(loss_fn)
        
        # Should compute gradients without error
        grads = grad_fn(x)
        assert grads.shape == x.shape
        assert not jnp.any(jnp.isnan(grads))
    
    def test_symmetry(self):
        """Test that symmetric inputs around center produce similar patterns."""
        center = 0.0
        offset = 5.0
        
        x_pos = jnp.array([center + offset])
        x_neg = jnp.array([center - offset])
        
        result_pos = fta(x_pos, lower_bound=-20, upper_bound=20)
        result_neg = fta(x_neg, lower_bound=-20, upper_bound=20)
        
        # Results should be flipped versions
        # Just check they have same sum of activations
        assert jnp.abs(jnp.sum(result_pos) - jnp.sum(result_neg)) < 0.000001
    
    def test_edge_values_at_bounds(self):
        """Test behavior at exact bound values."""
        lower_bound = -20
        upper_bound = 20
        
        x = jnp.array([lower_bound, upper_bound])
        result = fta(x)
        
        assert result.shape == (2, 20)
        assert jnp.all(result >= 0) and jnp.all(result <= 1)
        
        # First value (lower_bound) should activate first 2 tiles (close to 1)
        assert jnp.allclose(result[0, :2], 1.0, atol=0.001)
        # Middle tiles should be 0
        assert jnp.allclose(result[0, 2:-2], 0.0, atol=0.001)
        
        # Last value (upper_bound) should activate last 2 tiles (close to 1)
        assert jnp.allclose(result[1, -2:], 1.0, atol=0.001)
        # Middle tiles should be 0
        assert jnp.allclose(result[1, 2:-2], 0.0, atol=0.001)
