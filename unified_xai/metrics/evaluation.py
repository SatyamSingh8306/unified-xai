"""Evaluation metrics for explanations."""

import numpy as np
import torch
import tensorflow as tf
from typing import Any, Dict, List, Optional, Tuple, Union
from unified_xai.core.base import ExplanationResult, ModelWrapper
from scipy.stats import spearmanr, pearsonr


class ExplanationEvaluator:
    """Evaluate quality of explanations using various metrics."""
    
    def __init__(self, model_wrapper: ModelWrapper):
        self.model_wrapper = model_wrapper
    
    def evaluate(self, explanation: ExplanationResult, input_data: Any,
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate explanation using specified metrics."""
        if metrics is None:
            metrics = ['faithfulness', 'complexity', 'stability']
        
        results = {}
        
        if 'faithfulness' in metrics:
            results['faithfulness'] = self.compute_faithfulness(explanation, input_data)
        
        if 'complexity' in metrics:
            results['complexity'] = self.compute_complexity(explanation)
        
        if 'stability' in metrics:
            results['stability'] = self.compute_stability(explanation, input_data)
        
        if 'sensitivity' in metrics:
            results['sensitivity'] = self.compute_sensitivity(explanation, input_data)
        
        return results
    
    def _convert_to_numpy(self, data: Any) -> np.ndarray:
        """Convert input data to numpy array."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, tf.Tensor):
            return data.numpy()
        else:
            return np.array(data)
    
    def _convert_to_original_type(self, data: np.ndarray, original: Any) -> Any:
        """Convert numpy array back to original tensor type."""
        if isinstance(original, torch.Tensor):
            return torch.tensor(data, dtype=original.dtype, device=original.device)
        elif isinstance(original, tf.Tensor):
            return tf.convert_to_tensor(data, dtype=original.dtype)
        else:
            return data
    
    def compute_faithfulness(self, explanation: ExplanationResult, input_data: Any) -> float:
        """Compute faithfulness using insertion/deletion metric."""
        # Convert to numpy for manipulation
        input_np = self._convert_to_numpy(input_data)
        
        attribution = explanation.attribution.flatten()
        n_features = len(attribution)
        
        # Get indices sorted by importance
        important_indices = np.argsort(np.abs(attribution))[::-1]
        
        # Original prediction
        original_pred = self.model_wrapper.forward(input_data)
        if not isinstance(original_pred, np.ndarray):
            original_pred = self._convert_to_numpy(original_pred)
        
        if len(original_pred.shape) == 1:
            original_class = np.argmax(original_pred)
            original_prob = original_pred[original_class]
        else:
            original_class = np.argmax(original_pred)
            original_prob = original_pred.flat[original_class]
        
        # Deletion metric: progressively remove important features
        deletion_scores = []
        masked_input_np = input_np.copy()
        
        for i in range(0, min(n_features, 100), 10):
            # Mask top-i important features
            masked_input_temp = masked_input_np.copy()
            for idx in important_indices[:i]:
                if idx < masked_input_temp.size:
                    masked_input_temp.flat[idx] = 0
            
            # Convert back to original type for model forward
            masked_input = self._convert_to_original_type(masked_input_temp, input_data)
            masked_pred = self.model_wrapper.forward(masked_input)
            
            if not isinstance(masked_pred, np.ndarray):
                masked_pred = self._convert_to_numpy(masked_pred)
            
            if len(masked_pred.shape) == 1:
                masked_prob = masked_pred[original_class]
            else:
                masked_prob = masked_pred.flat[original_class]
            
            deletion_scores.append(float(original_prob - masked_prob))
        
        # Area under deletion curve (higher is better)
        faithfulness = np.mean(deletion_scores) if deletion_scores else 0.0
        
        return float(faithfulness)
    
    def compute_complexity(self, explanation: ExplanationResult) -> float:
        """Compute complexity of explanation (entropy-based)."""
        attribution = np.abs(explanation.attribution.flatten())
        
        # Normalize to probability distribution
        if attribution.sum() > 0:
            attribution = attribution / attribution.sum()
        else:
            return 0.0
        
        # Compute entropy
        entropy = -np.sum(attribution * np.log(attribution + 1e-10))
        
        # Normalize by maximum entropy
        max_entropy = np.log(len(attribution))
        complexity = entropy / max_entropy if max_entropy > 0 else 0
        
        return float(1 - complexity)  # Lower entropy = simpler explanation = higher score
    
    def compute_stability(self, explanation: ExplanationResult, input_data: Any,
                         n_samples: int = 10, noise_level: float = 0.01) -> float:
        """Compute stability of explanations to input perturbations."""
        # Convert to numpy for manipulation
        input_np = self._convert_to_numpy(input_data)
        
        correlations = []
        
        for _ in range(n_samples):
            # Add small noise to input
            noise = np.random.normal(0, noise_level, input_np.shape)
            noisy_input_np = input_np + noise
            
            # Convert back to original type
            noisy_input = self._convert_to_original_type(noisy_input_np, input_data)
            
            # Get explanation for noisy input
            noisy_explanation = self._get_explanation_for_input(noisy_input, explanation.method)
            
            # Compute correlation between original and noisy explanations
            try:
                corr, _ = spearmanr(
                    explanation.attribution.flatten(),
                    noisy_explanation.attribution.flatten()
                )
                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                # Handle cases where correlation cannot be computed
                pass
        
        stability = np.mean(correlations) if correlations else 0.5
        return float(stability)
    
    def compute_sensitivity(self, explanation: ExplanationResult, input_data: Any) -> float:
        """Compute sensitivity - how much attribution changes with input changes."""
        # Convert to numpy for manipulation
        input_np = self._convert_to_numpy(input_data)
        
        attribution = explanation.attribution.flatten()
        
        # Compute gradient of attribution with respect to input
        sensitivity_scores = []
        
        for i in range(min(len(attribution), 100)):
            # Perturb single feature
            perturbed_np = input_np.copy()
            if i < perturbed_np.size:
                perturbed_np.flat[i] += 0.1
            
            # Convert back to original type
            perturbed = self._convert_to_original_type(perturbed_np, input_data)
            
            # Get new explanation
            new_explanation = self._get_explanation_for_input(perturbed, explanation.method)
            
            # Measure change in attribution
            if i < len(new_explanation.attribution.flat):
                attr_change = np.abs(new_explanation.attribution.flat[i] - attribution[i])
                sensitivity_scores.append(float(attr_change))
        
        return float(np.mean(sensitivity_scores)) if sensitivity_scores else 0.0
    
    def _get_explanation_for_input(self, input_data: Any, method: str) -> ExplanationResult:
        """Helper to get explanation for given input."""
        # This would need to call the appropriate explainer
        # For now, return a dummy result
        input_np = self._convert_to_numpy(input_data)
        return ExplanationResult(
            attribution=np.random.randn(*input_np.shape),
            method=method,
            explanation_type=None,
            metadata={}
        )


class ExplanationComparator:
    """Compare different explanation methods."""
    
    def __init__(self, evaluator: ExplanationEvaluator):
        self.evaluator = evaluator
    
    def compare(self, explanations: List[ExplanationResult], input_data: Any,
               metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Compare multiple explanations using evaluation metrics."""
        results = {}
        
        for explanation in explanations:
            scores = self.evaluator.evaluate(explanation, input_data, metrics)
            results[explanation.method] = scores
        
        return results
    
    def rank_explanations(self, comparison_results: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """Rank explanations based on overall scores."""
        rankings = []
        
        for method, scores in comparison_results.items():
            # Simple average of all metrics
            avg_score = np.mean(list(scores.values()))
            rankings.append((method, float(avg_score)))
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings