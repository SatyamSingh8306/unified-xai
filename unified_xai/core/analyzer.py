"""Main analyzer class that orchestrates all explanation methods."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from pathlib import Path
from unified_xai.config import XAIConfig, ExplanationType
from unified_xai.core.base import ExplanationResult, ModelWrapper
from unified_xai.core.model_wrappers import ModelWrapperFactory
from unified_xai.methods.gradient_based import (
    VanillaGradient, IntegratedGradients, SmoothGrad, GradCAM
)
from unified_xai.methods.perturbation_based import (
    LIMEExplainer, SHAPExplainer, OcclusionExplainer
)
from unified_xai.visualization.visualizers import ExplanationVisualizer
from unified_xai.metrics.evaluation import ExplanationEvaluator, ExplanationComparator
from unified_xai.utils.preprocessing import aggregate_attributions
import logging


class XAIAnalyzer:
    """Main class for unified XAI analysis."""
    
    def __init__(self, model: Any, config: Optional[XAIConfig] = None):
        """Initialize XAI Analyzer.
        
        Args:
            model: The model to explain (PyTorch, TensorFlow, or Keras)
            config: Configuration object for XAI analysis
        """
        self.config = config or XAIConfig()
        self.logger = self._setup_logger()
        
        # Create model wrapper
        self.model_wrapper = ModelWrapperFactory.create(model, self.config.framework)
        
        # Initialize components
        self.explainers = self._initialize_explainers()
        self.visualizer = ExplanationVisualizer(self.config.visualization_config)
        self.evaluator = ExplanationEvaluator(self.model_wrapper)
        self.comparator = ExplanationComparator(self.evaluator)
        
        # Cache for explanations
        self._explanation_cache = {}
        
        self.logger.info(f"XAIAnalyzer initialized for {self.config.framework.value} model")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_explainers(self) -> Dict[str, Any]:
        """Initialize all available explainers."""
        explainers = {}
        
        # Gradient-based explainers
        explainers['vanilla_gradient'] = VanillaGradient(
            self.model_wrapper, self.config.gradient_config
        )
        explainers['integrated_gradients'] = IntegratedGradients(
            self.model_wrapper, self.config.gradient_config
        )
        explainers['smoothgrad'] = SmoothGrad(
            self.model_wrapper, self.config.gradient_config
        )
        explainers['gradcam'] = GradCAM(
            self.model_wrapper, self.config.gradient_config
        )
        
        # Perturbation-based explainers
        lime_config = {**self.config.lime_config, 'modality': self.config.modality.value}
        explainers['lime'] = LIMEExplainer(self.model_wrapper, lime_config)
        
        shap_config = {**self.config.shap_config}
        explainers['shap'] = SHAPExplainer(self.model_wrapper, shap_config)
        
        explainers['occlusion'] = OcclusionExplainer(
            self.model_wrapper, {'window_size': (10, 10), 'stride': 5}
        )
        
        return explainers
    
    def explain(self, inputs: Any, method: str = 'auto', target: Optional[int] = None,
               **kwargs) -> ExplanationResult:
        """Generate explanation for given inputs.
        
        Args:
            inputs: Input data to explain
            method: Explanation method to use ('auto' for automatic selection)
            target: Target class to explain (None for predicted class)
            **kwargs: Additional arguments for specific methods
        
        Returns:
            ExplanationResult object containing attribution and metadata
        """
        self.logger.info(f"Generating explanation using method: {method}")
        
        # Auto-select method based on modality and model type
        if method == 'auto':
            method = self._auto_select_method()
        
        # Check cache
        cache_key = self._get_cache_key(inputs, method, target)
        if self.config.cache_enabled and cache_key in self._explanation_cache:
            self.logger.info("Returning cached explanation")
            return self._explanation_cache[cache_key]
        
        # Get explainer
        if method not in self.explainers:
            raise ValueError(f"Unknown explanation method: {method}")
        
        explainer = self.explainers[method]
        
        # Generate explanation
        try:
            explanation = explainer.explain(inputs, target, **kwargs)
            
            # Cache result
            if self.config.cache_enabled:
                self._explanation_cache[cache_key] = explanation
            
            # Evaluate if configured
            if self.config.metrics_config['compute_faithfulness']:
                metrics = self.evaluator.evaluate(explanation, inputs, ['faithfulness'])
                explanation.metadata['metrics'] = metrics
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {str(e)}")
            raise
    
    def explain_multiple(self, inputs: Any, methods: Optional[List[str]] = None,
                        target: Optional[int] = None, **kwargs) -> Dict[str, ExplanationResult]:
        """Generate explanations using multiple methods.
        
        Args:
            inputs: Input data to explain
            methods: List of methods to use (None for all available)
            target: Target class to explain
            **kwargs: Additional arguments
        
        Returns:
            Dictionary mapping method names to ExplanationResult objects
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        results = {}
        for method in methods:
            try:
                self.logger.info(f"Generating explanation with {method}")
                results[method] = self.explain(inputs, method, target, **kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to generate {method} explanation: {str(e)}")
        
        return results
    
    def compare_methods(self, inputs: Any, methods: Optional[List[str]] = None,
                       target: Optional[int] = None, metrics: Optional[List[str]] = None,
                       visualize: bool = True) -> Dict[str, Any]:
        """Compare different explanation methods.
        
        Args:
            inputs: Input data to explain
            methods: Methods to compare
            target: Target class
            metrics: Evaluation metrics to use
            visualize: Whether to create comparison visualization
        
        Returns:
            Comparison results including scores and rankings
        """
        # Generate explanations
        explanations = self.explain_multiple(inputs, methods, target)
        
        # Evaluate explanations
        comparison_results = self.comparator.compare(
            list(explanations.values()), inputs, metrics
        )
        
        # Rank methods
        rankings = self.comparator.rank_explanations(comparison_results)
        
        # Visualize if requested
        visualization = None
        if visualize:
            visualization = self.visualizer.compare_explanations(
                list(explanations.values()), inputs
            )
        
        return {
            'explanations': explanations,
            'scores': comparison_results,
            'rankings': rankings,
            'visualization': visualization
        }
    
    def visualize(self, explanation: ExplanationResult, original_input: Optional[Any] = None,
                 **kwargs) -> Any:
        """Visualize explanation.
        
        Args:
            explanation: ExplanationResult to visualize
            original_input: Original input for overlay
            **kwargs: Additional visualization parameters
        
        Returns:
            Visualization figure
        """
        return self.visualizer.visualize(explanation, original_input, **kwargs)
    
    def aggregate_explanations(self, explanations: List[ExplanationResult],
                              method: str = 'mean') -> ExplanationResult:
        """Aggregate multiple explanations into a single one.
        
        Args:
            explanations: List of explanations to aggregate
            method: Aggregation method ('mean', 'median', 'max', etc.)
        
        Returns:
            Aggregated ExplanationResult
        """
        if not explanations:
            raise ValueError("No explanations to aggregate")
        
        attributions = [exp.attribution for exp in explanations]
        aggregated = aggregate_attributions(attributions, method)
        
        return ExplanationResult(
            attribution=aggregated,
            method=f"aggregated_{method}",
            explanation_type=explanations[0].explanation_type,
            metadata={
                'aggregation_method': method,
                'n_explanations': len(explanations),
                'methods': [exp.method for exp in explanations]
            }
        )
    
    def _auto_select_method(self) -> str:
        """Automatically select best explanation method based on context."""
        if self.config.modality.value == 'image':
            return 'gradcam' if self._has_conv_layers() else 'integrated_gradients'
        elif self.config.modality.value == 'text':
            return 'integrated_gradients'
        elif self.config.modality.value == 'tabular':
            return 'shap'
        else:
            return 'vanilla_gradient'
    
    def _has_conv_layers(self) -> bool:
        """Check if model has convolutional layers."""
        # Simplified check - would need proper implementation
        return True
    
    def _get_cache_key(self, inputs: Any, method: str, target: Optional[int]) -> str:
        """Generate cache key for explanation."""
        # Simple hash-based key - would need better implementation
        import hashlib
        input_hash = hashlib.md5(str(inputs).encode()).hexdigest()[:8]
        return f"{method}_{target}_{input_hash}"
    
    def save_explanation(self, explanation: ExplanationResult, path: Path) -> None:
        """Save explanation to file.
        
        Args:
            explanation: Explanation to save
            path: Path to save location
        """
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(explanation, f)
        
        self.logger.info(f"Saved explanation to {path}")
    
    def load_explanation(self, path: Path) -> ExplanationResult:
        """Load explanation from file.
        
        Args:
            path: Path to saved explanation
        
        Returns:
            Loaded ExplanationResult
        """
        import pickle
        with open(path, 'rb') as f:
            explanation = pickle.load(f)
        
        self.logger.info(f"Loaded explanation from {path}")
        return explanation