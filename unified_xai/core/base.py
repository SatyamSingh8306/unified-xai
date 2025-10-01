"""Base classes and interfaces for XAI methods."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
import numpy as np
from dataclasses import dataclass
import torch
import tensorflow as tf
from unified_xai.config import Framework, ExplanationType, Modality


@dataclass
class ExplanationResult:
    """Container for explanation results."""
    
    attribution: np.ndarray
    method: str
    explanation_type: ExplanationType
    metadata: Dict[str, Any]
    raw_output: Optional[Any] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "attribution": self.attribution.tolist() if isinstance(self.attribution, np.ndarray) else self.attribution,
            "method": self.method,
            "type": self.explanation_type.value,
            "metadata": self.metadata,
            "confidence": self.confidence
        }


class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    def __init__(self, model: Any, framework: Framework):
        self.model = model
        self.framework = framework
        self._setup()
    
    @abstractmethod
    def _setup(self) -> None:
        """Setup model-specific configurations."""
        pass
    
    @abstractmethod
    def forward(self, inputs: Any, **kwargs) -> Any:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def get_gradients(self, inputs: Any, target: Optional[Any] = None) -> np.ndarray:
        """Get gradients with respect to inputs."""
        pass
    
    @abstractmethod
    def get_activations(self, inputs: Any, layer_name: str) -> np.ndarray:
        """Get activations from specific layer."""
        pass
    
    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape."""
        pass


class ExplainerBase(ABC):
    """Abstract base class for all explainers."""
    
    def __init__(self, model_wrapper: ModelWrapper, config: Dict[str, Any]):
        self.model_wrapper = model_wrapper
        self.config = config
        self._cache = {}
    
    @abstractmethod
    def explain(self, inputs: Any, targets: Optional[Any] = None, **kwargs) -> ExplanationResult:
        """Generate explanation for given inputs."""
        pass
    
    @abstractmethod
    def validate_inputs(self, inputs: Any) -> bool:
        """Validate input format and shape."""
        pass
    
    def clear_cache(self) -> None:
        """Clear internal cache."""
        self._cache.clear()