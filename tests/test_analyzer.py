"""Test cases for main XAIAnalyzer class."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
from unified_xai import XAIAnalyzer, XAIConfig
from unified_xai.config import Framework, Modality
from unified_xai.core.base import ExplanationResult, ExplanationType


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestXAIAnalyzer:
    """Test XAIAnalyzer class."""
    
    @pytest.fixture
    def model(self):
        """Create a simple model."""
        return SimpleModel()
    
    @pytest.fixture
    def config(self):
        """Create configuration."""
        return XAIConfig(
            framework=Framework.PYTORCH,
            modality=Modality.TABULAR
        )
    
    @pytest.fixture
    def analyzer(self, model, config):
        """Create XAIAnalyzer instance."""
        with patch('unified_xai.core.analyzer.ModelWrapperFactory'):
            return XAIAnalyzer(model, config)
    
    def test_initialization(self, model):
        """Test analyzer initialization."""
        analyzer = XAIAnalyzer(model)
        
        assert analyzer.config is not None
        assert analyzer.model_wrapper is not None
        assert analyzer.explainers is not None
        assert analyzer.visualizer is not None
        assert analyzer.evaluator is not None
        assert analyzer.comparator is not None
    
    def test_initialization_with_config(self, model, config):
        """Test analyzer initialization with custom config."""
        analyzer = XAIAnalyzer(model, config)
        
        assert analyzer.config == config
        assert analyzer.config.framework == Framework.PYTORCH
        assert analyzer.config.modality == Modality.TABULAR
    
    @patch('unified_xai.core.analyzer.VanillaGradient')
    def test_explain_single_method(self, mock_gradient_class, analyzer):
        """Test single explanation generation."""
        # Setup mock explainer
        mock_explainer = MagicMock()
        mock_result = ExplanationResult(
            attribution=np.random.randn(10),
            method="vanilla_gradient",
            explanation_type=ExplanationType.GRADIENT,
            metadata={}
        )
        mock_explainer.explain.return_value = mock_result
        analyzer.explainers['vanilla_gradient'] = mock_explainer
        
        # Generate explanation
        inputs = np.random.randn(1, 10)
        result = analyzer.explain(inputs, method='vanilla_gradient')
        
        assert result is not None
        assert result.method == "vanilla_gradient"
        mock_explainer.explain.assert_called_once()
    
    def test_explain_auto_method(self, analyzer):
        """Test automatic method selection."""
        # Mock auto selection
        analyzer._auto_select_method = Mock(return_value='vanilla_gradient')
        
        # Setup mock explainer
        mock_explainer = MagicMock()
        mock_result = ExplanationResult(
            attribution=np.random.randn(10),
            method="vanilla_gradient",
            explanation_type=ExplanationType.GRADIENT,
            metadata={}
        )
        mock_explainer.explain.return_value = mock_result
        analyzer.explainers['vanilla_gradient'] = mock_explainer
        
        inputs = np.random.randn(1, 10)
        result = analyzer.explain(inputs, method='auto')
        
        assert result is not None
        analyzer._auto_select_method.assert_called_once()
    
    def test_explain_with_cache(self, analyzer):
        """Test explanation caching."""
        analyzer.config.cache_enabled = True
        
        # Setup mock explainer
        mock_explainer = MagicMock()
        mock_result = ExplanationResult(
            attribution=np.random.randn(10),
            method="vanilla_gradient",
            explanation_type=ExplanationType.GRADIENT,
            metadata={}
        )
        mock_explainer.explain.return_value = mock_result
        analyzer.explainers['vanilla_gradient'] = mock_explainer
        
        inputs = np.random.randn(1, 10)
        
        # First call
        result1 = analyzer.explain(inputs, method='vanilla_gradient', target=0)
        
        # Second call (should use cache)
        result2 = analyzer.explain(inputs, method='vanilla_gradient', target=0)
        
        # Explainer should only be called once
        mock_explainer.explain.assert_called_once()
        assert result1 == result2
    
    def test_explain_unknown_method(self, analyzer):
        """Test error for unknown method."""
        inputs = np.random.randn(1, 10)
        
        with pytest.raises(ValueError, match="Unknown explanation method"):
            analyzer.explain(inputs, method='unknown_method')
    
    def test_explain_multiple(self, analyzer):
        """Test multiple explanation generation."""
        # Setup mock explainers
        methods = ['vanilla_gradient', 'integrated_gradients']
        for method in methods:
            mock_explainer = MagicMock()
            mock_result = ExplanationResult(
                attribution=np.random.randn(10),
                method=method,
                explanation_type=ExplanationType.GRADIENT,
                metadata={}
            )
            mock_explainer.explain.return_value = mock_result
            analyzer.explainers[method] = mock_explainer
        
        inputs = np.random.randn(1, 10)
        results = analyzer.explain_multiple(inputs, methods=methods)
        
        assert len(results) == 2
        assert 'vanilla_gradient' in results
        assert 'integrated_gradients' in results
    
    def test_compare_methods(self, analyzer):
        """Test method comparison."""
        # Setup mock components
        mock_explainer = MagicMock()
        mock_result = ExplanationResult(
            attribution=np.random.randn(10),
            method="vanilla_gradient",
            explanation_type=ExplanationType.GRADIENT,
            metadata={}
        )
        mock_explainer.explain.return_value = mock_result
        analyzer.explainers['vanilla_gradient'] = mock_explainer
        
        analyzer.comparator.compare = Mock(return_value={
            'vanilla_gradient': {'faithfulness': 0.8, 'complexity': 0.3}
        })
        analyzer.comparator.rank_explanations = Mock(return_value=[
            ('vanilla_gradient', 0.55)
        ])
        analyzer.visualizer.compare_explanations = Mock(return_value=None)
        
        inputs = np.random.randn(1, 10)
        comparison = analyzer.compare_methods(
            inputs,
            methods=['vanilla_gradient'],
            visualize=True
        )
        
        assert 'explanations' in comparison
        assert 'scores' in comparison
        assert 'rankings' in comparison
        assert 'visualization' in comparison
    
    def test_visualize(self, analyzer):
        """Test visualization."""
        explanation = ExplanationResult(
            attribution=np.random.randn(224, 224),
            method="gradcam",
            explanation_type=ExplanationType.GRADIENT,
            metadata={}
        )
        
        analyzer.visualizer.visualize = Mock(return_value="figure")
        
        result = analyzer.visualize(explanation)
        
        assert result == "figure"
        analyzer.visualizer.visualize.assert_called_once_with(
            explanation, None
        )
    
    def test_aggregate_explanations(self, analyzer):
        """Test explanation aggregation."""
        explanations = [
            ExplanationResult(
                attribution=np.ones((10,)) * i,
                method=f"method_{i}",
                explanation_type=ExplanationType.GRADIENT,
                metadata={}
            )
            for i in range(3)
        ]
        
        result = analyzer.aggregate_explanations(explanations, method='mean')
        
        assert result is not None
        assert result.method == "aggregated_mean"
        assert np.allclose(result.attribution, np.ones(10))  # Mean of 0, 1, 2
    
    def test_aggregate_empty_list(self, analyzer):
        """Test aggregation with empty list."""
        with pytest.raises(ValueError, match="No explanations to aggregate"):
            analyzer.aggregate_explanations([])
    
    def test_save_load_explanation(self, analyzer):
        """Test saving and loading explanations."""
        explanation = ExplanationResult(
            attribution=np.random.randn(10),
            method="test_method",
            explanation_type=ExplanationType.GRADIENT,
            metadata={'test': 'data'}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_explanation.pkl"
            
            # Save
            analyzer.save_explanation(explanation, path)
            assert path.exists()
            
            # Load
            loaded = analyzer.load_explanation(path)
            
            assert loaded.method == explanation.method
            assert loaded.explanation_type == explanation.explanation_type
            assert np.allclose(loaded.attribution, explanation.attribution)
            assert loaded.metadata == explanation.metadata
    
    def test_auto_select_method(self, analyzer):
        """Test automatic method selection logic."""
        # Image modality
        analyzer.config.modality = Modality.IMAGE
        analyzer._has_conv_layers = Mock(return_value=True)
        assert analyzer._auto_select_method() == 'gradcam'
        
        analyzer._has_conv_layers = Mock(return_value=False)
        assert analyzer._auto_select_method() == 'integrated_gradients'
        
        # Text modality
        analyzer.config.modality = Modality.TEXT
        assert analyzer._auto_select_method() == 'integrated_gradients'
        
        # Tabular modality
        analyzer.config.modality = Modality.TABULAR
        assert analyzer._auto_select_method() == 'shap'
        
        # Default
        analyzer.config.modality = Modality.TIMESERIES
        assert analyzer._auto_select_method() == 'vanilla_gradient'