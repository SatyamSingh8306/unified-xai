"""Test cases for visualization module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
from unified_xai.visualization.visualizers import ExplanationVisualizer
from unified_xai.core.base import ExplanationResult, ExplanationType


class TestExplanationVisualizer:
    """Test ExplanationVisualizer class."""
    
    @pytest.fixture
    def config(self):
        """Create visualization config."""
        return {
            'cmap': 'RdBu_r',
            'alpha': 0.7,
            'save_path': None
        }
    
    @pytest.fixture
    def visualizer(self, config):
        """Create visualizer instance."""
        return ExplanationVisualizer(config)
    
    @pytest.fixture
    def explanation_2d(self):
        """Create 2D explanation result."""
        return ExplanationResult(
            attribution=np.random.randn(224, 224),
            method="gradcam",
            explanation_type=ExplanationType.GRADIENT,
            metadata={'method': 'gradcam'}
        )
    
    @pytest.fixture
    def explanation_1d(self):
        """Create 1D explanation result."""
        return ExplanationResult(
            attribution=np.random.randn(10),
            method="shap",
            explanation_type=ExplanationType.PERTURBATION,
            metadata={'method': 'shap'}
        )
    
    def test_initialization(self, visualizer, config):
        """Test visualizer initialization."""
        assert visualizer.config == config
        assert visualizer.cmap == 'RdBu_r'
        assert visualizer.alpha == 0.7
        assert visualizer.save_path is None
    
    @patch('matplotlib.pyplot.show')
    def test_visualize_2d(self, mock_show, visualizer, explanation_2d):
        """Test 2D visualization."""
        fig = visualizer.visualize(explanation_2d)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_visualize_2d_with_original(self, mock_show, visualizer, explanation_2d):
        """Test 2D visualization with original image."""
        original = np.random.randn(224, 224, 3)
        fig = visualizer.visualize(explanation_2d, original)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # Original, attribution, overlay
        plt.close(fig)
    
    def test_visualize_1d(self, visualizer, explanation_1d):
        """Test 1D visualization with plotly."""
        fig = visualizer.visualize(explanation_1d)
        
        assert fig is not None
        # Check if it's a plotly figure
        assert hasattr(fig, 'add_trace')
    
    def test_visualize_unsupported_shape(self, visualizer):
        """Test error for unsupported attribution shape."""
        explanation = ExplanationResult(
            attribution=np.random.randn(10, 10, 10, 10),  # 4D
            method="test",
            explanation_type=ExplanationType.GRADIENT,
            metadata={}
        )
        
        with pytest.raises(ValueError, match="Unsupported attribution shape"):
            visualizer.visualize(explanation)
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_visualization(self, mock_savefig, visualizer, explanation_2d):
        """Test saving visualization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer.save_path = tmpdir
            fig = visualizer.visualize(explanation_2d)
            
            # Check if save was attempted
            expected_path = Path(tmpdir) / 'explanation_gradcam.png'
            plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_compare_explanations(self, mock_show, visualizer):
        """Test comparing multiple explanations."""
        explanations = [
            ExplanationResult(
                attribution=np.random.randn(224, 224),
                method=f"method_{i}",
                explanation_type=ExplanationType.GRADIENT,
                metadata={}
            )
            for i in range(3)
        ]
        
        fig = visualizer.compare_explanations(explanations)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_compare_with_original(self, mock_show, visualizer):
        """Test comparison with original input."""
        explanations = [
            ExplanationResult(
                attribution=np.random.randn(224, 224),
                method="method",
                explanation_type=ExplanationType.GRADIENT,
                metadata={}
            )
        ]
        original = np.random.randn(224, 224, 3)
        
        fig = visualizer.compare_explanations(explanations, original)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Original + 1 explanation
        plt.close(fig)
    
    def test_create_overlay(self, visualizer):
        """Test overlay creation."""
        attribution = np.random.randn(100, 100)
        target_shape = (224, 224)
        
        overlay = visualizer._create_overlay(attribution, target_shape)
        
        assert overlay.shape == target_shape
    
    def test_visualize_multichannel(self, visualizer):
        """Test multi-channel visualization."""
        explanation = ExplanationResult(
            attribution=np.random.randn(224, 224, 3),
            method="test",
            explanation_type=ExplanationType.GRADIENT,
            metadata={'method': 'test'}
        )
        
        fig = visualizer.visualize(explanation)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # One for each channel
        plt.close(fig)