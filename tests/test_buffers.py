"""Tests for the buffers module."""

import numpy as np
import pytest

from pyForwardFolding.buffers import BufferManager


class TestBufferManager:
    """Test the BufferManager class."""
    
    def test_initialization(self):
        """Test BufferManager initialization."""
        datasets = {
            "comp1": (100,),
            "comp2": (200,)
        }
        
        buffer_manager = BufferManager(datasets)
        
        # Check that buffers were created
        assert "comp1_weight" in buffer_manager.buffers
        assert "comp1_weight_sq" in buffer_manager.buffers
        assert "comp1_component" in buffer_manager.buffers
        assert "comp2_weight" in buffer_manager.buffers
        assert "comp2_weight_sq" in buffer_manager.buffers
        assert "comp2_component" in buffer_manager.buffers
        
        # Check buffer shapes
        assert buffer_manager.buffers["comp1_weight"].shape == (100,)
        assert buffer_manager.buffers["comp2_weight"].shape == (200,)
    
    def test_get_buffer(self):
        """Test getting buffers by name."""
        datasets = {"comp1": (50,)}
        buffer_manager = BufferManager(datasets)
        
        # Test getting existing buffer
        weight_buffer = buffer_manager.get_buffer("comp1_weight")
        assert weight_buffer.shape == (50,)
        
        # Test getting non-existent buffer
        with pytest.raises(KeyError):
            buffer_manager.get_buffer("nonexistent_buffer")
    
    def test_get_weight_buffer(self):
        """Test getting weight buffer for a component."""
        datasets = {"comp1": (75,)}
        buffer_manager = BufferManager(datasets)
        
        weight_buffer = buffer_manager.get_weight_buffer("comp1")
        assert weight_buffer.shape == (75,)
        
        # Verify it's the same as getting buffer directly
        direct_buffer = buffer_manager.get_buffer("comp1_weight")
        assert np.array_equal(weight_buffer, direct_buffer)
    
    def test_get_weight_sq_buffer(self):
        """Test getting weight squared buffer for a component."""
        datasets = {"comp1": (60,)}
        buffer_manager = BufferManager(datasets)
        
        weight_sq_buffer = buffer_manager.get_weight_sq_buffer("comp1")
        assert weight_sq_buffer.shape == (60,)
        
        # Verify it's the same as getting buffer directly
        direct_buffer = buffer_manager.get_buffer("comp1_weight_sq")
        assert np.array_equal(weight_sq_buffer, direct_buffer)
    
    def test_get_component_buffer(self):
        """Test getting component buffer for a component."""
        datasets = {"comp1": (40,)}
        buffer_manager = BufferManager(datasets)
        
        component_buffer = buffer_manager.get_component_buffer("comp1")
        assert component_buffer.shape == (40,)
        
        # Verify it's the same as getting buffer directly
        direct_buffer = buffer_manager.get_buffer("comp1_component")
        assert np.array_equal(component_buffer, direct_buffer)
    
    def test_from_datasets_classmethod(self):
        """Test creating BufferManager from datasets."""
        datasets = {
            "detector1": {
                "energy": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                "zenith": np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            },
            "detector2": {
                "energy": np.array([1.5, 2.5]),
                "zenith": np.array([0.15, 0.25])
            }
        }
        
        buffer_manager = BufferManager.from_datasets(datasets)
        
        # Check that buffers were created with correct sizes
        assert buffer_manager.get_weight_buffer("detector1").shape == (5,)
        assert buffer_manager.get_weight_buffer("detector2").shape == (2,)
        
        # Verify all expected buffers exist
        expected_buffers = [
            "detector1_weight", "detector1_weight_sq", "detector1_component",
            "detector2_weight", "detector2_weight_sq", "detector2_component"
        ]
        for buffer_name in expected_buffers:
            assert buffer_name in buffer_manager.buffers
    
    def test_multiple_components_different_sizes(self):
        """Test BufferManager with multiple components of different sizes."""
        datasets = {
            "small_comp": (10,),
            "medium_comp": (100,),
            "large_comp": (1000,)
        }
        
        buffer_manager = BufferManager(datasets)
        
        # Verify each component has correct buffer sizes
        assert buffer_manager.get_weight_buffer("small_comp").shape == (10,)
        assert buffer_manager.get_weight_buffer("medium_comp").shape == (100,)
        assert buffer_manager.get_weight_buffer("large_comp").shape == (1000,)
        
        # Verify all buffers are initialized to zeros
        for comp_name in datasets.keys():
            weight_buffer = buffer_manager.get_weight_buffer(comp_name)
            weight_sq_buffer = buffer_manager.get_weight_sq_buffer(comp_name)
            component_buffer = buffer_manager.get_component_buffer(comp_name)
            
            assert np.allclose(weight_buffer, 0.0)
            assert np.allclose(weight_sq_buffer, 0.0)
            assert np.allclose(component_buffer, 0.0)
    
    def test_buffer_independence(self):
        """Test that buffers for different components are independent."""
        datasets = {
            "comp1": (50,),
            "comp2": (50,)
        }
        
        buffer_manager = BufferManager(datasets)
        
        # Get buffers for both components
        buffer1 = buffer_manager.get_weight_buffer("comp1")
        buffer2 = buffer_manager.get_weight_buffer("comp2")
        
        # Modify one buffer
        buffer1[0] = 1.0
        
        # Check that the other buffer is unchanged
        assert buffer2[0] == 0.0
        assert buffer1[0] == 1.0
    
    def test_empty_datasets(self):
        """Test BufferManager with empty datasets."""
        datasets = {}
        buffer_manager = BufferManager(datasets)
        
        # Should have no buffers
        assert len(buffer_manager.buffers) == 0
    
    def test_single_element_datasets(self):
        """Test BufferManager with single-element datasets."""
        datasets = {
            "single_element": (1,)
        }
        
        buffer_manager = BufferManager(datasets)
        
        # Check that buffers have correct shape
        weight_buffer = buffer_manager.get_weight_buffer("single_element")
        assert weight_buffer.shape == (1,)
        assert weight_buffer[0] == 0.0


class TestBufferManagerIntegration:
    """Test BufferManager integration with other components."""
    
    def test_buffer_usage_pattern(self):
        """Test typical buffer usage pattern."""
        # Simulate typical usage with datasets
        datasets = {
            "astro": {
                "log10_true_energy": np.array([4.0, 5.0, 6.0]),
                "cos_zenith": np.array([0.1, 0.5, 0.9])
            },
            "atmo": {
                "log10_true_energy": np.array([4.5, 5.5]),
                "cos_zenith": np.array([0.3, 0.7])
            }
        }
        
        buffer_manager = BufferManager.from_datasets(datasets)
        
        # Simulate filling buffers with computed weights
        astro_weights = np.array([1.0, 2.0, 3.0])
        atmo_weights = np.array([0.5, 1.5])
        
        # Get buffers and fill them
        astro_weight_buffer = buffer_manager.get_weight_buffer("astro")
        atmo_weight_buffer = buffer_manager.get_weight_buffer("atmo")
        
        # In practice, these would be filled by the model evaluation
        astro_weight_buffer[:] = astro_weights
        atmo_weight_buffer[:] = atmo_weights
        
        # Verify buffers contain expected values
        np.testing.assert_array_equal(
            buffer_manager.get_weight_buffer("astro"), 
            astro_weights
        )
        np.testing.assert_array_equal(
            buffer_manager.get_weight_buffer("atmo"), 
            atmo_weights
        )
    
    def test_buffer_reuse(self):
        """Test that buffers can be reused across evaluations."""
        datasets = {"test_comp": (100,)}
        buffer_manager = BufferManager(datasets)
        
        # First evaluation
        buffer = buffer_manager.get_weight_buffer("test_comp")
        buffer[:10] = 1.0
        
        # Second evaluation (buffer should be reusable)
        buffer = buffer_manager.get_weight_buffer("test_comp")
        buffer[:10] = 2.0
        
        # Verify buffer was updated
        assert np.all(buffer[:10] == 2.0)
        assert np.all(buffer[10:] == 0.0)
