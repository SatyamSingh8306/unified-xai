"""Test cases for evaluation metrics."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from unified_xai.metrics.evaluation import ExplanationEvaluator, ExplanationComparator
from unified_xai.core.base import ExplanationResult, ExplanationType


class TestExplanationEvaluator:
    """Test ExplanationEvaluator class."""
    
    @pytest.fixture
    def mock_wrapper(self):
        """Create mock model wrapper."""
        wrapper = Mock()
        wrapper.forward = Mock(return_value=np.array([[0.3, 0.7]]))
        return wrapper
    
    @pytest.fixture
    def evaluator(self, mock_wrapper):
        """Create evaluator instance."""
        return ExplanationEvaluator(mock_wrapper)
    
    @pytest.fixture
    def explanation(self):
        """Create sample explanation."""
        return ExplanationResult(
            attribution=np.random.randn(10),
            method="test_method",
            explanation_type=ExplanationType.GRADIENT,
            metadata={}
        )
    
    def test_evaluate_all_metrics(self, evaluator, explanation):
        """Test evaluation with all metrics."""
        input_data = np.random.randn(1, 10)
        
        # Mock internal methods
        evaluator.compute_faithfulness = Mock(return_value=0.8)
        evaluator.compute_complexity = Mock(return_value=0.3)
        evaluator.compute_stability = Mock(return_value=0.9)
        
        results = evaluator.evaluate(explanation, input_data)
        
        assert 'faithfulness' in results
        assert 'complexity' in results
        assert 'stability' in results
        assert results['faithfulness'] == 0.8
        assert results['complexity'] == 0.3
        assert results['stability'] == 0.9
    
    def test_evaluate_specific_metrics(self, evaluator, explanation):
        """Test evaluation with specific metrics."""
        input_data = np.random.randn(1, 10)
        
        evaluator.compute_faithfulness = Mock(return_value=0.8)
        evaluator.compute_sensitivity = Mock(return_value=0.5)
        
        results = evaluator.evaluate(
            explanation,
            input_data,
            metrics=['faithfulness', 'sensitivity']
        )
        
        assert 'faithfulness' in results
        assert 'sensitivity' in results
        assert 'complexity' not in results
        assert 'stability' not in results
    
    def test_compute_faithfulness(self, evaluator, explanation, mock_wrapper):
        """Test faithfulness computation."""
        input_data = np.random.randn(1, 10)
        
        # Setup mock responses
        mock_wrapper.forward.side_effect = [
            np.array([[0.2, 0.8]]),  # Original
            np.array([[0.3, 0.7]]),  # First deletion
            np.array([[0.4, 0.6]]),  # Second deletion
        ]
        
        faithfulness = evaluator.compute_faithfulness(explanation, input_data)
        
        assert isinstance(faithfulness, float)
        assert mock_wrapper.forward.called
    
    def test_compute_complexity(self, evaluator, explanation):
        """Test complexity computation."""
        # Test with uniform attribution (low complexity)
        explanation.attribution = np.ones(10) / 10
        complexity = evaluator.compute_complexity(explanation)
        
        assert isinstance(complexity, float)
        assert 0 <= complexity <= 1
        
        # Test with sparse attribution (high complexity)
        explanation.attribution = np.zeros(10)
        explanation.attribution[0] = 1.0
        complexity_sparse = evaluator.compute_complexity(explanation)
        
        assert complexity_sparse > complexity  # Sparse should be less complex
    
    def test_compute_complexity_zero_attribution(self, evaluator):
        """Test complexity with zero attribution."""
        explanation = ExplanationResult(
            attribution=np.zeros(10),
            method="test",
            explanation_type=ExplanationType.GRADIENT,
            metadata={}
        )
        
        complexity = evaluator.compute_complexity(explanation)
        assert complexity == 0.0
    
    @patch('unified_xai.metrics.evaluation.spearmanr')
    def test_compute_stability(self, mock_spearmanr, evaluator, explanation):
        """Test stability computation."""
        mock_spearmanr.return_value = (0.9, 0.01)
        
        # Mock the method for getting explanations
        evaluator._get_explanation_for_input = Mock(return_value=explanation)
        
        input_data = np.random.randn(1, 10)
        stability = evaluator.compute_stability(
            explanation,
            input_data,
            n_samples=5,
            noise_level=0.01
        )
        
        assert isinstance(stability, float)
        assert evaluator._get_explanation_for_input.call_count == 5
    
    def test_compute_sensitivity(self, evaluator, explanation):
        """Test sensitivity computation."""
        evaluator._get_explanation_for_input = Mock(return_value=explanation)
        
        input_data = np.random.randn(1, 10)
        sensitivity = evaluator.compute_sensitivity(explanation, input_data)
        
        assert isinstance(sensitivity, float)
        assert evaluator._get_explanation_for_input.called


class TestExplanationComparator:
    """Test ExplanationComparator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create mock evaluator."""
        evaluator = Mock()
        return evaluator
    
    @pytest.fixture
    def comparator(self, evaluator):
        """Create comparator instance."""
        return ExplanationComparator(evaluator)
    
    @pytest.fixture
    def explanations(self):
        """Create sample explanations."""
        return [
            ExplanationResult(
                attribution=np.random.randn(10),
                method=f"method_{i}",
                explanation_type=ExplanationType.GRADIENT,
                metadata={}
            )
            for i in range(3)
        ]
    
    def test_compare(self, comparator, evaluator, explanations):
        """Test comparison of explanations."""
        input_data = np.random.randn(1, 10)
        
        # Mock evaluator responses
        evaluator.evaluate.side_effect = [
            {'faithfulness': 0.8, 'complexity': 0.3},
            {'faithfulness': 0.7, 'complexity': 0.4},
            {'faithfulness': 0.9, 'complexity': 0.2},
        ]
        
        results = comparator.compare(explanations, input_data)
        
        assert len(results) == 3
        assert 'method_0' in results
        assert 'method_1' in results
        assert 'method_2' in results
        assert results['method_0']['faithfulness'] == 0.8
    
    def test_rank_explanations(self, comparator):
        """Test ranking of explanations."""
        comparison_results = {
            'method_1': {'faithfulness': 0.8, 'complexity': 0.3},
            'method_2': {'faithfulness': 0.7, 'complexity': 0.4},
            'method_3': {'faithfulness': 0.9, 'complexity': 0.2},
        }
        
        rankings = comparator.rank_explanations(comparison_results)
        
        assert len(rankings) == 3
        assert rankings[0][0] == 'method_3'  # Highest average score
        assert rankings[0][1] == 0.55  # (0.9 + 0.2) / 2
        assert rankings[-1][0] == 'method_2'  # Lowest average score