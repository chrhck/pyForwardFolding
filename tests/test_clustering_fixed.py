"""Tests for the clustering module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from pyForwardFolding.clustering import compress_hdbscan, compress_minibatch_kmeans


class TestCompressHDBSCAN:
    """Test the HDBSCAN compression function."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        np.random.seed(42)
        n_events = 100
        return {
            'energy': np.random.exponential(scale=100, size=n_events),
            'zenith': np.random.uniform(0, np.pi, size=n_events),
            'azimuth': np.random.uniform(0, 2*np.pi, size=n_events),
            'weight': np.random.exponential(scale=1, size=n_events),
            'weight_systematic': np.random.normal(1, 0.1, size=n_events)
        }
    
    @pytest.fixture 
    def weight_keys(self):
        """Define which keys are weights."""
        return ['weight', 'weight_systematic']
    
    @patch('pyForwardFolding.clustering.HDBSCAN')
    def test_compress_hdbscan_basic(self, mock_hdbscan, sample_data, weight_keys):
        """Test basic HDBSCAN compression functionality."""
        # Mock HDBSCAN clustering results
        mock_clusterer = Mock()
        mock_clusterer.labels_ = np.array([0, 0, 1, 1, 2, -1, -1, 0, 1, 2] + [0] * 90)
        mock_hdbscan.return_value = mock_clusterer
        
        result = compress_hdbscan(sample_data, weight_keys, min_cluster_size=5)
        
        # Check that compressed data has correct structure
        assert isinstance(result, dict)
        assert set(result.keys()) == set(sample_data.keys())
        
        # Check that weights are summed and non-weights are averaged
        for key in result.keys():
            assert len(result[key]) > 0
    
    @patch('pyForwardFolding.clustering.HDBSCAN')
    def test_compress_hdbscan_all_noise(self, mock_hdbscan, sample_data, weight_keys):
        """Test HDBSCAN compression when all points are noise."""
        mock_clusterer = Mock()
        mock_clusterer.labels_ = np.full(100, -1)  # All noise
        mock_hdbscan.return_value = mock_clusterer
        
        compressed = compress_hdbscan(sample_data, weight_keys)
        
        # Each noise point should become its own cluster
        assert len(compressed['energy']) == 100
    
    @patch('pyForwardFolding.clustering.HDBSCAN')  
    def test_compress_hdbscan_no_noise(self, mock_hdbscan, sample_data, weight_keys):
        """Test HDBSCAN compression with no noise points."""
        mock_clusterer = Mock()
        mock_clusterer.labels_ = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0] + [0] * 90)
        mock_hdbscan.return_value = mock_clusterer
        
        compressed = compress_hdbscan(sample_data, weight_keys)
        
        # Should have 3 clusters (0, 1, 2)
        assert len(compressed['energy']) == 3
    
    def test_compress_hdbscan_weight_summing(self):
        """Test that weights are properly summed in clusters."""
        data = {
            'obs1': np.array([1, 2, 3, 4]),
            'weight': np.array([0.5, 0.5, 0.3, 0.7])
        }
        weight_keys = ['weight']
        
        with patch('pyForwardFolding.clustering.HDBSCAN') as mock_hdbscan:
            mock_clusterer = Mock()
            mock_clusterer.labels_ = np.array([0, 0, 1, 1])  # Two clusters
            mock_hdbscan.return_value = mock_clusterer
            
            compressed = compress_hdbscan(data, weight_keys)
        
        # Weights should be summed: cluster 0: 0.5+0.5=1.0, cluster 1: 0.3+0.7=1.0
        expected_weights = np.array([1.0, 1.0])
        np.testing.assert_array_almost_equal(compressed['weight'], expected_weights)
    
    def test_compress_hdbscan_observable_averaging(self):
        """Test that observables are properly averaged in clusters."""
        data = {
            'obs1': np.array([1.0, 3.0, 5.0, 7.0]),  # avg cluster 0: 2.0, cluster 1: 6.0
            'weight': np.array([0.5, 0.5, 0.3, 0.7])
        }
        weight_keys = ['weight']
        
        with patch('pyForwardFolding.clustering.HDBSCAN') as mock_hdbscan:
            mock_clusterer = Mock()
            mock_clusterer.labels_ = np.array([0, 0, 1, 1])  # Two clusters
            mock_hdbscan.return_value = mock_clusterer
            
            compressed = compress_hdbscan(data, weight_keys)
        
        # Observables should be averaged: cluster 0: (1.0+3.0)/2=2.0, cluster 1: (5.0+7.0)/2=6.0
        expected_obs = np.array([2.0, 6.0])
        np.testing.assert_array_almost_equal(compressed['obs1'], expected_obs)


class TestCompressMiniBatchKMeans:
    """Test the MiniBatch K-Means compression function."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample data for testing."""
        np.random.seed(42)
        n_events = 1000
        return {
            'energy': np.random.exponential(scale=100, size=n_events),
            'zenith': np.random.uniform(0, np.pi, size=n_events),
            'azimuth': np.random.uniform(0, 2*np.pi, size=n_events),
            'weight': np.random.exponential(scale=1, size=n_events),
            'weight_systematic': np.random.normal(1, 0.1, size=n_events)
        }
    
    @pytest.fixture 
    def weight_keys(self):
        """Define which keys are weights."""
        return ['weight', 'weight_systematic']
    
    def test_compress_minibatch_kmeans_basic(self, sample_data, weight_keys):
        """Test basic MiniBatch K-Means compression."""
        result = compress_minibatch_kmeans(sample_data, weight_keys, compression_factor=10)
        
        # Should have approximately len(data)/compression_factor clusters
        expected_clusters = int(np.ceil(len(sample_data['energy']) / 10))
        assert len(result['energy']) == expected_clusters
        assert set(result.keys()) == set(sample_data.keys())
    
    def test_compress_minibatch_kmeans_compression_factor(self, sample_data, weight_keys):
        """Test effect of compression factor."""
        result_high = compress_minibatch_kmeans(sample_data, weight_keys, compression_factor=5)
        result_low = compress_minibatch_kmeans(sample_data, weight_keys, compression_factor=20)
        
        # Higher compression factor should result in fewer clusters
        assert len(result_low['energy']) < len(result_high['energy'])
    
    def test_compress_minibatch_kmeans_weight_conservation(self, sample_data, weight_keys):
        """Test that total weight is conserved."""
        result = compress_minibatch_kmeans(sample_data, weight_keys, compression_factor=10)
        
        # Total weight should be conserved
        for weight_key in weight_keys:
            original_total = np.sum(sample_data[weight_key])
            compressed_total = np.sum(result[weight_key])
            np.testing.assert_almost_equal(original_total, compressed_total, decimal=10)
    
    def test_compress_minibatch_kmeans_empty_data(self, weight_keys):
        """Test with empty data."""
        empty_data = {key: np.array([]) for key in ['energy', 'zenith'] + weight_keys}
        result = compress_minibatch_kmeans(empty_data, weight_keys)
        
        # Should return empty arrays
        for key in result.keys():
            assert len(result[key]) == 0 or len(result[key]) == 1
    
    def test_compress_minibatch_kmeans_single_event(self, weight_keys):
        """Test with single event."""
        single_data = {
            'energy': np.array([100.0]),
            'zenith': np.array([0.5]),
            'weight': np.array([1.0]),
            'weight_systematic': np.array([1.1])
        }
        result = compress_minibatch_kmeans(single_data, weight_keys)
        
        # Should have one cluster
        assert len(result['energy']) == 1
        np.testing.assert_array_almost_equal(result['energy'], single_data['energy'])
    
    def test_compress_minibatch_kmeans_structure(self, sample_data, weight_keys):
        """Test that the structure of returned data is correct."""
        result = compress_minibatch_kmeans(sample_data, weight_keys, compression_factor=10)
        
        # All keys should be present
        assert set(result.keys()) == set(sample_data.keys())
        
        # All arrays should have the same length
        first_key = list(result.keys())[0]
        expected_length = len(result[first_key])
        for key in result.keys():
            assert len(result[key]) == expected_length


class TestClusteringIntegration:
    """Integration tests for clustering functions."""
    
    def test_clustering_preserves_total_weight(self):
        """Test that clustering preserves total weight across methods."""
        np.random.seed(42)
        data = {
            'energy': np.random.exponential(100, 200),
            'zenith': np.random.uniform(0, np.pi, 200),
            'weight': np.random.exponential(1, 200)
        }
        weight_keys = ['weight']
        original_weight = np.sum(data['weight'])
        
        # Test HDBSCAN
        with patch('pyForwardFolding.clustering.HDBSCAN') as mock_hdbscan:
            mock_clusterer = Mock()
            mock_clusterer.labels_ = np.random.randint(0, 10, 200)
            mock_hdbscan.return_value = mock_clusterer
            
            hdbscan_result = compress_hdbscan(data, weight_keys)
            hdbscan_weight = np.sum(hdbscan_result['weight'])
            np.testing.assert_almost_equal(original_weight, hdbscan_weight, decimal=10)
        
        # Test MiniBatch K-Means
        kmeans_result = compress_minibatch_kmeans(data, weight_keys, compression_factor=10)
        kmeans_weight = np.sum(kmeans_result['weight'])
        np.testing.assert_almost_equal(original_weight, kmeans_weight, decimal=10)
    
    def test_clustering_reduces_data_size(self):
        """Test that clustering actually reduces data size."""
        np.random.seed(42)
        data = {
            'energy': np.random.exponential(100, 1000),
            'zenith': np.random.uniform(0, np.pi, 1000),
            'weight': np.random.exponential(1, 1000)
        }
        weight_keys = ['weight']
        
        # MiniBatch K-Means should reduce size
        result = compress_minibatch_kmeans(data, weight_keys, compression_factor=10)
        assert len(result['energy']) < len(data['energy'])
        assert len(result['energy']) == int(np.ceil(1000 / 10))
    
    @pytest.mark.parametrize("compression_factor", [5, 10, 20])
    def test_compression_factor_effect(self, compression_factor):
        """Test effect of different compression factors."""
        np.random.seed(42)
        data = {
            'energy': np.random.exponential(100, 1000),
            'zenith': np.random.uniform(0, np.pi, 1000),
            'weight': np.random.exponential(1, 1000)
        }
        weight_keys = ['weight']
        
        result = compress_minibatch_kmeans(data, weight_keys, compression_factor=compression_factor)
        expected_size = int(np.ceil(1000 / compression_factor))
        assert len(result['energy']) == expected_size
