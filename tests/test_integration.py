"""Test cases for utility functions."""

import pytest
import numpy as np
from unified_xai.utils.preprocessing import (
    normalize_attribution,
    resize_attribution,
    smooth_attribution,
    threshold_attribution,
    aggregate_attributions
)


class TestNormalizeAttribution:
    """Test normalize_attribution function."""
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        attribution = np.array([1, 2, 3, 4, 5])
        normalized = normalize_attribution(attribution, method='minmax')
        
        assert normalized.min() == 0
        assert normalized.max() == 1
        assert np.allclose(normalized, [0, 0.25, 0.5, 0.75, 1.0])
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        attribution = np.array([1, 2, 3, 4, 5])
        normalized = normalize_attribution(attribution, method='zscore')
        
        assert np.allclose(normalized.mean(), 0)
        assert np.allclose(normalized.std(), 1)
    
    def test_abs_max_normalization(self):
        """Test absolute max normalization."""
        attribution = np.array([-2, -1, 0, 1, 2])
        normalized = normalize_attribution(attribution, method='abs_max')
        
        assert normalized.max() <= 1
        assert normalized.min() >= -1
        assert np.allclose(normalized, [-1, -0.5, 0, 0.5, 1])
    
    def test_zero_variance(self):
        """Test normalization with zero variance."""
        attribution = np.ones(5)
        
        # Min-max should handle this
        normalized = normalize_attribution(attribution, method='minmax')
        assert np.allclose(normalized, attribution)
        
        # Z-score should handle this
        normalized = normalize_attribution(attribution, method='zscore')
        assert np.allclose(normalized, np.zeros(5))
    
    def test_invalid_method(self):
        """Test error for invalid normalization method."""
        attribution = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_attribution(attribution, method='invalid')


class TestResizeAttribution:
    """Test resize_attribution function."""
    
    def test_resize_upscale(self):
        """Test upscaling attribution."""
        attribution = np.ones((10, 10))
        resized = resize_attribution(attribution, (20, 20))
        
        assert resized.shape == (20, 20)
    
    def test_resize_downscale(self):
        """Test downscaling attribution."""
        attribution = np.ones((20, 20))
        resized = resize_attribution(attribution, (10, 10))
        
        assert resized.shape == (10, 10)
    
    def test_resize_same_size(self):
        """Test resizing to same size."""
        attribution = np.random.randn(10, 10)
        resized = resize_attribution(attribution, (10, 10))
        
        assert resized.shape == (10, 10)
        assert np.allclose(resized, attribution)


class TestSmoothAttribution:
    """Test smooth_attribution function."""
    
    def test_smooth_default_kernel(self):
        """Test smoothing with default kernel size."""
        attribution = np.random.randn(10, 10)
        smoothed = smooth_attribution(attribution)
        
        assert smoothed.shape == attribution.shape
        # Smoothing should reduce variance
        assert smoothed.std() < attribution.std()
    
    def test_smooth_custom_kernel(self):
        """Test smoothing with custom kernel size."""
        attribution = np.random.randn(10, 10)
        smoothed = smooth_attribution(attribution, kernel_size=5)
        
        assert smoothed.shape == attribution.shape


class TestThresholdAttribution:
    """Test threshold_attribution function."""
    
    def test_threshold_percentile(self):
        """Test thresholding with percentile."""
        attribution = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        thresholded = threshold_attribution(attribution, threshold=0.6, percentile=True)
        
        # 60th percentile of |attribution| should keep top 40%
        assert np.sum(thresholded != 0) <= 2
    
    def test_threshold_absolute(self):
        """Test thresholding with absolute value."""
        attribution = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        thresholded = threshold_attribution(attribution, threshold=0.3, percentile=False)
        
        # Values below 0.3 should be zero
        assert thresholded[0] == 0
        assert thresholded[1] == 0
        assert thresholded[2] != 0
        assert thresholded[3] != 0
        assert thresholded[4] != 0


class TestAggregateAttributions:
    """Test aggregate_attributions function."""
    
    def test_aggregate_mean(self):
        """Test mean aggregation."""
        attributions = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9])
        ]
        
        aggregated = aggregate_attributions(attributions, method='mean')
        assert np.allclose(aggregated, [4, 5, 6])
    
    def test_aggregate_median(self):
        """Test median aggregation."""
        attributions = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9])
        ]
        
        aggregated = aggregate_attributions(attributions, method='median')
        assert np.allclose(aggregated, [4, 5, 6])
    
    def test_aggregate_max(self):
        """Test max aggregation."""
        attributions = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9])
        ]
        
        aggregated = aggregate_attributions(attributions, method='max')
        assert np.allclose(aggregated, [7, 8, 9])
    
    def test_aggregate_min(self):
        """Test min aggregation."""
        attributions = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9])
        ]
        
        aggregated = aggregate_attributions(attributions, method='min')
        assert np.allclose(aggregated, [1, 2, 3])
    
    def test_aggregate_std(self):
        """Test standard deviation aggregation."""
        attributions = [
            np.array([1, 1, 1]),
            np.array([2, 2, 2]),
            np.array([3, 3, 3])
        ]
        
        aggregated = aggregate_attributions(attributions, method='std')
        expected_std = np.std([1, 2, 3])
        assert np.allclose(aggregated, [expected_std, expected_std, expected_std])
    
    def test_aggregate_empty_list(self):
        """Test error for empty list."""
        with pytest.raises(ValueError, match="No attributions to aggregate"):
            aggregate_attributions([], method='mean')
    
    def test_aggregate_invalid_method(self):
        """Test error for invalid aggregation method."""
        attributions = [np.array([1, 2, 3])]
        
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_attributions(attributions, method='invalid')