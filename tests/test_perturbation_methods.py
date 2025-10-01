"""Test cases for perturbation-based methods."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from unified_xai.methods.perturbation_based import (
    LIMEExplainer, SHAPExplainer, OcclusionExplainer
)
from unified_xai.config import ExplanationType


class TestLIMEExplainer:
    """Test LIME explainer."""
    
    @pytest.fixture
    def mock_wrapper(self):
        """Create mock model wrapper."""
        wrapper = Mock()
        wrapper.forward = Mock(return_value=np.array([[0.2, 0.8]]))
        return wrapper
    
    @pytest.fixture
    def explainer(self, mock_wrapper):
        """Create LIME explainer."""
        config = {
            'modality': 'image',
            'num_samples': 100,
            'num_features': 5
        }
        return LIMEExplainer(mock_wrapper, config)
    
    @patch('unified_xai.methods.perturbation_based.lime.lime_image.LimeImageExplainer')
    def test_initialization(self, mock_lime_class, mock_wrapper):
        """Test LIME explainer initialization."""
        explainer = LIMEExplainer(mock_wrapper, {'modality': 'image'})
        
        assert explainer.model_wrapper == mock_wrapper
        mock_lime_class.assert_called_once()
    
    def test_validate_inputs(self, explainer):
        """Test input validation."""
        assert explainer.validate_inputs(np.array([1, 2, 3])) == True
        assert explainer.validate_inputs("text input") == True
        assert explainer.validate_inputs([1, 2, 3]) == True
        assert explainer.validate_inputs(None) == False
    
    @patch('unified_xai.methods.perturbation_based.lime.lime_image.LimeImageExplainer')
    def test_explain_image(self, mock_lime_class, mock_wrapper):
        """Test image explanation."""
        # Setup mock LIME explainer
        mock_lime_instance = MagicMock()
        mock_explanation = MagicMock()
        mock_explanation.top_labels = [0]
        mock_explanation.get_image_and_mask.return_value = (
            np.zeros((224, 224, 3)),
            np.ones((224, 224))
        )
        mock_lime_instance.explain_instance.return_value = mock_explanation
        mock_lime_class.return_value = mock_lime_instance
        
        config = {'modality': 'image', 'num_samples': 100, 'num_features': 5}
        explainer = LIMEExplainer(mock_wrapper, config)
        
        image = np.random.randn(224, 224, 3)
        result = explainer.explain(image, targets=0)
        
        assert result is not None
        assert result.method == "lime"
        assert result.explanation_type == ExplanationType.PERTURBATION
        assert result.attribution.shape == (224, 224)
    
    def test_unsupported_modality(self, mock_wrapper):
        """Test error for unsupported modality."""
        config = {'modality': 'unsupported'}
        explainer = LIMEExplainer(mock_wrapper, config)
        
        with pytest.raises(ValueError, match="Unsupported modality"):
            explainer.explain(np.random.randn(10))


class TestSHAPExplainer:
    """Test SHAP explainer."""
    
    @pytest.fixture
    def mock_wrapper(self):
        """Create mock model wrapper."""
        wrapper = Mock()
        wrapper.forward = Mock(return_value=np.array([[0.3, 0.7]]))
        return wrapper
    
    @pytest.fixture
    def explainer(self, mock_wrapper):
        """Create SHAP explainer."""
        config = {
            'explainer_type': 'kernel',
            'background_data': np.random.randn(10, 5),
            'normalize': True
        }
        with patch('unified_xai.methods.perturbation_based.shap.KernelExplainer'):
            return SHAPExplainer(mock_wrapper, config)
    
    def test_validate_inputs(self, explainer):
        """Test input validation."""
        assert explainer.validate_inputs(np.array([1, 2, 3])) == True
        assert explainer.validate_inputs([[1, 2], [3, 4]]) == True
        assert explainer.validate_inputs("invalid") == False
    
    @patch('unified_xai.methods.perturbation_based.shap.KernelExplainer')
    def test_kernel_explainer_creation(self, mock_shap_kernel, mock_wrapper):
        """Test KernelExplainer creation."""
        background_data = np.random.randn(10, 5)
        config = {
            'explainer_type': 'kernel',
            'background_data': background_data
        }
        
        explainer = SHAPExplainer(mock_wrapper, config)
        mock_shap_kernel.assert_called_once()
    
    def test_no_explainer_initialized(self, mock_wrapper):
        """Test error when explainer is not initialized."""
        config = {'explainer_type': 'kernel'}  # No background data
        explainer = SHAPExplainer(mock_wrapper, config)
        
        with pytest.raises(RuntimeError, match="SHAP explainer not properly initialized"):
            explainer.explain(np.random.randn(1, 5))


class TestOcclusionExplainer:
    """Test Occlusion explainer."""
    
    @pytest.fixture
    def mock_wrapper(self):
        """Create mock model wrapper."""
        wrapper = Mock()
        # Return different scores for occluded regions
        wrapper.forward = Mock(side_effect=[
            np.array([[0.2, 0.8]]),  # Baseline
            np.array([[0.3, 0.7]]),  # First occlusion
            np.array([[0.4, 0.6]]),  # Second occlusion
        ])
        return wrapper
    
    @pytest.fixture
    def explainer(self, mock_wrapper):
        """Create Occlusion explainer."""
        config = {
            'window_size': (2, 2),
            'stride': 2,
            'normalize': False
        }
        return OcclusionExplainer(mock_wrapper, config)
    
    def test_explain(self, explainer, mock_wrapper):
        """Test occlusion explanation."""
        # Reset mock to control behavior
        mock_wrapper.forward.side_effect = None
        mock_wrapper.forward.return_value = np.array([[0.2, 0.8]])
        
        image = np.random.randn(1, 4, 4, 3)
        result = explainer.explain(image, targets=1)
        
        assert result is not None
        assert result.method == "occlusion"
        assert result.explanation_type == ExplanationType.PERTURBATION
        assert result.metadata['window_size'] == (2, 2)
        assert result.metadata['stride'] == 2
    
    def test_validate_inputs(self, explainer):
        """Test input validation for occlusion."""
        # Valid inputs
        assert explainer.validate_inputs(np.random.randn(1, 224, 224, 3)) == True
        assert explainer.validate_inputs(np.random.randn(224, 224, 3)) == True
        
        # Invalid inputs
        assert explainer.validate_inputs(np.random.randn(10)) == False
        assert explainer.validate_inputs("invalid") == False
    
    def test_occlusion_importance_calculation(self, mock_wrapper):
        """Test that occlusion correctly calculates importance."""
        # Setup controlled responses
        baseline_score = 0.8
        occluded_score = 0.3
        mock_wrapper.forward.side_effect = [
            np.array([[0.2, baseline_score]]),  # Baseline
            np.array([[0.7, occluded_score]]),  # Occluded
        ]
        
        config = {'window_size': (10, 10), 'stride': 10, 'normalize': False}
        explainer = OcclusionExplainer(mock_wrapper, config)
        
        image = np.ones((1, 10, 10, 3))
        result = explainer.explain(image, targets=1)
        
        # Importance should be baseline - occluded
        expected_importance = baseline_score - occluded_score
        assert np.allclose(result.attribution[0:10, 0:10], expected_importance)