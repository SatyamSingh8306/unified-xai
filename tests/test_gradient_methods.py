"""Test cases for gradient-based explanation methods."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
from unified_xai.methods.gradient_based import (
    VanillaGradient, IntegratedGradients, SmoothGrad, GradCAM
)
from unified_xai.config import ExplanationType


class TestVanillaGradient:
    """Test VanillaGradient explainer."""
    
    @pytest.fixture
    def mock_wrapper(self):
        """Create mock model wrapper."""
        wrapper = Mock()
        wrapper.get_gradients = Mock(return_value=np.random.randn(1, 3, 224, 224))
        return wrapper
    
    @pytest.fixture
    def explainer(self, mock_wrapper):
        """Create VanillaGradient explainer."""
        config = {'abs_value': True, 'normalize': True}
        return VanillaGradient(mock_wrapper, config)
    
    def test_explain(self, explainer, mock_wrapper):
        """Test explanation generation."""
        inputs = np.random.randn(1, 3, 224, 224)
        result = explainer.explain(inputs, targets=0)
        
        assert result is not None
        assert result.method == "vanilla_gradient"
        assert result.explanation_type == ExplanationType.GRADIENT
        assert result.attribution.shape == (1, 3, 224, 224)
        mock_wrapper.get_gradients.assert_called_once()
    
    def test_explain_no_normalization(self, mock_wrapper):
        """Test explanation without normalization."""
        config = {'abs_value': False, 'normalize': False}
        explainer = VanillaGradient(mock_wrapper, config)
        
        inputs = np.random.randn(1, 3, 224, 224)
        result = explainer.explain(inputs)
        
        assert result is not None
        # Check that values are not normalized (can be negative)
        assert result.attribution.min() < 0 or result.attribution.max() > 1
    
    def test_validate_inputs(self, explainer):
        """Test input validation."""
        assert explainer.validate_inputs(np.array([1, 2, 3])) == True
        assert explainer.validate_inputs(torch.tensor([1, 2, 3])) == True
        assert explainer.validate_inputs("invalid") == False
    
    def test_invalid_input(self, explainer):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="Invalid input format"):
            explainer.explain("invalid_input")


class TestIntegratedGradients:
    """Test IntegratedGradients explainer."""
    
    @pytest.fixture
    def mock_wrapper(self):
        """Create mock model wrapper."""
        wrapper = Mock()
        wrapper.get_gradients = Mock(side_effect=lambda x, t: np.random.randn(*x.shape))
        return wrapper
    
    @pytest.fixture
    def explainer(self, mock_wrapper):
        """Create IntegratedGradients explainer."""
        config = {'n_steps': 10, 'normalize': True}
        return IntegratedGradients(mock_wrapper, config)
    
    def test_explain(self, explainer, mock_wrapper):
        """Test explanation generation."""
        inputs = np.random.randn(1, 3, 224, 224)
        result = explainer.explain(inputs, targets=0)
        
        assert result is not None
        assert result.method == "integrated_gradients"
        assert result.explanation_type == ExplanationType.GRADIENT
        assert result.attribution.shape == inputs.shape
        assert mock_wrapper.get_gradients.call_count == 10  # n_steps
    
    def test_custom_baseline(self, explainer):
        """Test with custom baseline."""
        inputs = np.random.randn(1, 3, 224, 224)
        baseline = np.ones_like(inputs) * 0.5
        
        result = explainer.explain(inputs, targets=0, baseline=baseline)
        
        assert result is not None
        assert result.metadata['baseline_type'] == 'custom'
    
    def test_zero_baseline(self, explainer):
        """Test with zero baseline."""
        inputs = np.random.randn(1, 3, 224, 224)
        baseline = np.zeros_like(inputs)
        
        result = explainer.explain(inputs, targets=0, baseline=baseline)
        
        assert result.metadata['baseline_type'] == 'zero'
    
    def test_different_n_steps(self, mock_wrapper):
        """Test with different number of integration steps."""
        config = {'n_steps': 50, 'normalize': True}
        explainer = IntegratedGradients(mock_wrapper, config)
        
        inputs = np.random.randn(1, 10)
        result = explainer.explain(inputs)
        
        assert mock_wrapper.get_gradients.call_count == 50
        assert result.metadata['n_steps'] == 50


class TestSmoothGrad:
    """Test SmoothGrad explainer."""
    
    @pytest.fixture
    def mock_wrapper(self):
        """Create mock model wrapper."""
        wrapper = Mock()
        wrapper.get_gradients = Mock(return_value=np.random.randn(1, 3, 224, 224))
        return wrapper
    
    @pytest.fixture
    def explainer(self, mock_wrapper):
        """Create SmoothGrad explainer."""
        config = {
            'smooth_samples': 5,
            'noise_scale': 0.1,
            'abs_value': True,
            'normalize': True
        }
        return SmoothGrad(mock_wrapper, config)
    
    def test_explain(self, explainer, mock_wrapper):
        """Test explanation generation."""
        inputs = np.random.randn(1, 3, 224, 224)
        result = explainer.explain(inputs, targets=0)
        
        assert result is not None
        assert result.method == "smoothgrad"
        assert result.explanation_type == ExplanationType.GRADIENT
        assert result.attribution.shape == inputs.shape
        assert mock_wrapper.get_gradients.call_count == 5  # smooth_samples
    
    def test_noise_application(self, explainer, mock_wrapper):
        """Test that noise is properly applied."""
        inputs = np.ones((1, 10))
        
        # Track the inputs passed to get_gradients
        call_inputs = []
        mock_wrapper.get_gradients.side_effect = lambda x, t: (
            call_inputs.append(x), np.random.randn(*x.shape)
        )[1]
        
        result = explainer.explain(inputs, targets=0)
        
        # Check that inputs are different (noise was added)
        assert len(call_inputs) == 5
        for noisy_input in call_inputs:
            assert not np.allclose(noisy_input, inputs)
    
    def test_metadata(self, explainer):
        """Test metadata in result."""
        inputs = np.random.randn(1, 10)
        result = explainer.explain(inputs)
        
        assert result.metadata['n_samples'] == 5
        assert result.metadata['noise_scale'] == 0.1


class TestGradCAM:
    """Test GradCAM explainer."""
    
    @pytest.fixture
    def mock_wrapper(self):
        """Create mock model wrapper."""
        wrapper = Mock()
        wrapper.get_gradients = Mock(return_value=np.random.randn(1, 64, 14, 14))
        wrapper.get_activations = Mock(return_value=np.random.randn(1, 64, 14, 14))
        return wrapper
    
    @pytest.fixture
    def explainer(self, mock_wrapper):
        """Create GradCAM explainer."""
        config = {'normalize': True}
        return GradCAM(mock_wrapper, config)
    
    def test_explain(self, explainer, mock_wrapper):
        """Test explanation generation."""
        inputs = np.random.randn(1, 3, 224, 224)
        result = explainer.explain(inputs, targets=0, layer_name='conv5')
        
        assert result is not None
        assert result.method == "gradcam"
        assert result.explanation_type == ExplanationType.GRADIENT
        assert result.metadata['layer_name'] == 'conv5'
        
        mock_wrapper.get_activations.assert_called_once()
        mock_wrapper.get_gradients.assert_called_once()
    
    def test_validate_inputs(self, explainer):
        """Test input validation for GradCAM."""
        # Valid 4D input (batch of images)
        assert explainer.validate_inputs(np.random.randn(1, 3, 224, 224)) == True
        
        # Invalid inputs
        assert explainer.validate_inputs(np.random.randn(10)) == False
        assert explainer.validate_inputs(np.random.randn(224, 224)) == False
        assert explainer.validate_inputs("invalid") == False
    
    def test_relu_application(self, mock_wrapper):
        """Test that ReLU is applied to CAM."""
        # Set up to return negative values
        mock_wrapper.get_gradients.return_value = np.ones((1, 2, 2, 2)) * -1
        mock_wrapper.get_activations.return_value = np.ones((1, 2, 2, 2))
        
        explainer = GradCAM(mock_wrapper, {'normalize': False})
        inputs = np.random.randn(1, 3, 224, 224)
        result = explainer.explain(inputs, targets=0)
        
        # After ReLU, all values should be non-negative
        assert np.all(result.attribution >= 0)