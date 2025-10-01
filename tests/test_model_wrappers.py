"""Test cases for model wrappers."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from unified_xai.core.model_wrappers import (
    PyTorchWrapper, TensorFlowWrapper, ModelWrapperFactory
)
from unified_xai.config import Framework


class SimplePyTorchModel(nn.Module):
    """Simple PyTorch model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 224 * 224, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestPyTorchWrapper:
    """Test PyTorchWrapper class."""
    
    @pytest.fixture
    def model(self):
        """Create a simple PyTorch model."""
        return SimplePyTorchModel()
    
    @pytest.fixture
    def wrapper(self, model):
        """Create PyTorchWrapper instance."""
        return PyTorchWrapper(model, Framework.PYTORCH)
    
    def test_initialization(self, wrapper, model):
        """Test wrapper initialization."""
        assert wrapper.model == model
        assert wrapper.framework == Framework.PYTORCH
        assert wrapper.device is not None
    
    def test_forward_pass(self, wrapper):
        """Test forward pass through wrapper."""
        inputs = torch.randn(1, 3, 224, 224)
        outputs = wrapper.forward(inputs)
        
        assert outputs is not None
        assert outputs.shape == (1, 10)
    
    def test_forward_with_numpy(self, wrapper):
        """Test forward pass with numpy input."""
        inputs = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = wrapper.forward(inputs)
        
        assert outputs is not None
        assert outputs.shape == (1, 10)
    
    def test_get_gradients(self, wrapper):
        """Test gradient computation."""
        inputs = torch.randn(1, 3, 224, 224)
        gradients = wrapper.get_gradients(inputs, target=0)
        
        assert gradients is not None
        assert gradients.shape == (1, 3, 224, 224)
        assert isinstance(gradients, np.ndarray)
    
    def test_get_gradients_no_target(self, wrapper):
        """Test gradient computation without target."""
        inputs = torch.randn(1, 3, 224, 224)
        gradients = wrapper.get_gradients(inputs)
        
        assert gradients is not None
        assert gradients.shape == (1, 3, 224, 224)
    
    def test_get_activations(self, wrapper):
        """Test activation extraction."""
        inputs = torch.randn(1, 3, 224, 224)
        activations = wrapper.get_activations(inputs, 'conv1')
        
        assert activations is not None
        assert isinstance(activations, np.ndarray)
    
    def test_get_activations_invalid_layer(self, wrapper):
        """Test activation extraction with invalid layer name."""
        inputs = torch.randn(1, 3, 224, 224)
        
        with pytest.raises(ValueError, match="Layer .* not found"):
            wrapper.get_activations(inputs, 'nonexistent_layer')
    
    def test_input_shape_property(self, wrapper):
        """Test input shape property."""
        shape = wrapper.input_shape
        assert isinstance(shape, tuple)


class TestTensorFlowWrapper:
    """Test TensorFlowWrapper class."""
    
    @pytest.fixture
    def model(self):
        """Create a simple Keras model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 3, padding='same', input_shape=(224, 224, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])
        return model
    
    @pytest.fixture
    def wrapper(self, model):
        """Create TensorFlowWrapper instance."""
        return TensorFlowWrapper(model, Framework.TENSORFLOW)
    
    def test_initialization(self, wrapper, model):
        """Test wrapper initialization."""
        assert wrapper.model == model
        assert wrapper.framework == Framework.TENSORFLOW
        assert wrapper.is_keras == True
    
    def test_forward_pass(self, wrapper):
        """Test forward pass through wrapper."""
        inputs = np.random.randn(1, 224, 224, 3).astype(np.float32)
        outputs = wrapper.forward(inputs)
        
        assert outputs is not None
        assert outputs.shape == (1, 10)
        assert isinstance(outputs, np.ndarray)
    
    def test_get_gradients(self, wrapper):
        """Test gradient computation."""
        inputs = np.random.randn(1, 224, 224, 3).astype(np.float32)
        gradients = wrapper.get_gradients(inputs, target=0)
        
        assert gradients is not None
        assert gradients.shape == (1, 224, 224, 3)
        assert isinstance(gradients, np.ndarray)
    
    def test_get_activations(self, wrapper):
        """Test activation extraction."""
        inputs = np.random.randn(1, 224, 224, 3).astype(np.float32)
        activations = wrapper.get_activations(inputs, 'conv2d')
        
        assert activations is not None
        assert isinstance(activations, np.ndarray)
    
    def test_input_shape_property(self, wrapper):
        """Test input shape property."""
        shape = wrapper.input_shape
        assert shape == (224, 224, 3)


class TestModelWrapperFactory:
    """Test ModelWrapperFactory class."""
    
    def test_create_pytorch_wrapper(self):
        """Test creating PyTorch wrapper."""
        model = SimplePyTorchModel()
        wrapper = ModelWrapperFactory.create(model, Framework.PYTORCH)
        
        assert isinstance(wrapper, PyTorchWrapper)
        assert wrapper.model == model
    
    def test_create_tensorflow_wrapper(self):
        """Test creating TensorFlow wrapper."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=(100,))
        ])
        wrapper = ModelWrapperFactory.create(model, Framework.TENSORFLOW)
        
        assert isinstance(wrapper, TensorFlowWrapper)
        assert wrapper.model == model
    
    def test_unsupported_framework(self):
        """Test error for unsupported framework."""
        model = SimplePyTorchModel()
        
        with pytest.raises(ValueError, match="Unsupported framework"):
            ModelWrapperFactory.create(model, "unsupported")