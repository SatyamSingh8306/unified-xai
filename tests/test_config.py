"""Test cases for configuration module."""

import pytest
import tempfile
from pathlib import Path
import json
import yaml
from unified_xai.config import XAIConfig, Framework, Modality, ExplanationType


class TestXAIConfig:
    """Test XAIConfig class."""
    
    def test_default_initialization(self):
        """Test default configuration values."""
        config = XAIConfig()
        
        assert config.framework == Framework.PYTORCH
        assert config.modality == Modality.IMAGE
        assert config.batch_size == 32
        assert config.device == "auto"
        assert config.cache_enabled == True
        assert isinstance(config.cache_dir, Path)
        
    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = XAIConfig(
            framework=Framework.TENSORFLOW,
            modality=Modality.TEXT,
            batch_size=64,
            device="cuda"
        )
        
        assert config.framework == Framework.TENSORFLOW
        assert config.modality == Modality.TEXT
        assert config.batch_size == 64
        assert config.device == "cuda"
    
    def test_gradient_config(self):
        """Test gradient-specific configuration."""
        config = XAIConfig(
            gradient_config={
                'normalize': False,
                'abs_value': False,
                'smooth_samples': 100
            }
        )
        
        assert config.gradient_config['normalize'] == False
        assert config.gradient_config['abs_value'] == False
        assert config.gradient_config['smooth_samples'] == 100
    
    def test_save_load_json(self):
        """Test saving and loading configuration from JSON."""
        config = XAIConfig(
            framework=Framework.PYTORCH,
            modality=Modality.IMAGE,
            batch_size=16
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config.to_file(Path(f.name))
            loaded_config = XAIConfig.from_file(Path(f.name))
        
        assert loaded_config.framework == config.framework
        assert loaded_config.modality == config.modality
        assert loaded_config.batch_size == config.batch_size
    
    def test_save_load_yaml(self):
        """Test saving and loading configuration from YAML."""
        config = XAIConfig(
            framework=Framework.TENSORFLOW,
            modality=Modality.TEXT
        )
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            config.to_file(Path(f.name))
            loaded_config = XAIConfig.from_file(Path(f.name))
        
        assert loaded_config.framework == config.framework
        assert loaded_config.modality == config.modality
    
    def test_invalid_file_format(self):
        """Test loading from unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                XAIConfig.from_file(Path(f.name))


class TestEnums:
    """Test enumeration classes."""
    
    def test_framework_enum(self):
        """Test Framework enumeration."""
        assert Framework.PYTORCH.value == "pytorch"
        assert Framework.TENSORFLOW.value == "tensorflow"
        assert Framework.KERAS.value == "keras"
        assert Framework.ONNX.value == "onnx"
    
    def test_modality_enum(self):
        """Test Modality enumeration."""
        assert Modality.IMAGE.value == "image"
        assert Modality.TEXT.value == "text"
        assert Modality.TABULAR.value == "tabular"
        assert Modality.TIMESERIES.value == "timeseries"
        assert Modality.MULTIMODAL.value == "multimodal"
    
    def test_explanation_type_enum(self):
        """Test ExplanationType enumeration."""
        assert ExplanationType.GRADIENT.value == "gradient"
        assert ExplanationType.ATTENTION.value == "attention"
        assert ExplanationType.PERTURBATION.value == "perturbation"