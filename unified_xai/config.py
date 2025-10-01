"""Configuration management for XAI library."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json
import yaml
from pathlib import Path


class Framework(Enum):
    """Supported deep learning frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    ONNX = "onnx"


class ExplanationType(Enum):
    """Types of explanations available."""
    GRADIENT = "gradient"
    ATTENTION = "attention"
    PERTURBATION = "perturbation"
    EXAMPLE_BASED = "example_based"
    CONCEPT = "concept"


class Modality(Enum):
    """Data modalities supported."""
    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"
    TIMESERIES = "timeseries"
    MULTIMODAL = "multimodal"


@dataclass
class XAIConfig:
    """Main configuration class for XAI analysis."""
    
    framework: Framework = Framework.PYTORCH
    modality: Modality = Modality.IMAGE
    batch_size: int = 32
    device: str = "auto"  # auto, cpu, cuda, mps
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".xai_cache"))
    
    # Method-specific configs
    gradient_config: Dict[str, Any] = field(default_factory=lambda: {
        "normalize": True,
        "abs_value": True,
        "smooth_samples": 50,
        "noise_scale": 0.1
    })
    
    lime_config: Dict[str, Any] = field(default_factory=lambda: {
        "num_samples": 1000,
        "num_features": 10,
        "kernel_width": 0.25
    })
    
    shap_config: Dict[str, Any] = field(default_factory=lambda: {
        "n_samples": 100,
        "max_evals": 500,
        "link": "identity"
    })
    
    visualization_config: Dict[str, Any] = field(default_factory=lambda: {
        "cmap": "RdBu_r",
        "alpha": 0.7,
        "overlay": True,
        "save_path": None
    })
    
    metrics_config: Dict[str, Any] = field(default_factory=lambda: {
        "compute_faithfulness": True,
        "compute_complexity": True,
        "compute_stability": True
    })
    
    @classmethod
    def from_file(cls, path: Path) -> "XAIConfig":
        """Load configuration from JSON or YAML file."""
        path = Path(path)
        with open(path, 'r') as f:
            if path.suffix == '.json':
                data = json.load(f)
            elif path.suffix in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        # Convert string enums to actual enums
        if 'framework' in data:
            data['framework'] = Framework(data['framework'])
        if 'modality' in data:
            data['modality'] = Modality(data['modality'])
            
        return cls(**data)
    
    def to_file(self, path: Path) -> None:
        """Save configuration to file."""
        path = Path(path)
        data = self.__dict__.copy()
        
        # Convert enums to strings for serialization
        data['framework'] = self.framework.value
        data['modality'] = self.modality.value
        data['cache_dir'] = str(self.cache_dir)
        
        with open(path, 'w') as f:
            if path.suffix == '.json':
                json.dump(data, f, indent=2)
            elif path.suffix in ['.yml', '.yaml']:
                yaml.dump(data, f, default_flow_style=False)