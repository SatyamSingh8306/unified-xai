"""Model wrappers for different frameworks."""

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from typing import Any, Optional, Tuple, Dict
from unified_xai.core.base import ModelWrapper
from unified_xai.config import Framework
import warnings


class PyTorchWrapper(ModelWrapper):
    """Wrapper for PyTorch models."""
    
    def _setup(self) -> None:
        """Setup PyTorch model for explanation."""
        self.model.eval()
        
        # Detect device
        self.device = next(self.model.parameters()).device
        
        # Store hooks for gradient/activation extraction
        self.hooks = []
        self.gradients = {}
        self.activations = {}
    
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through PyTorch model."""
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        inputs = inputs.to(self.device)
        
        with torch.set_grad_enabled(kwargs.get('requires_grad', False)):
            outputs = self.model(inputs)
        
        return outputs
    
    def get_gradients(self, inputs: torch.Tensor, target: Optional[int] = None) -> np.ndarray:
        """Get gradients using backpropagation."""
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
        else:
            inputs = inputs.requires_grad_(True)
        
        inputs = inputs.to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Select target class
        if target is not None:
            if len(outputs.shape) > 1:
                outputs = outputs[:, target]
            else:
                outputs = outputs[target]
        else:
            outputs = outputs.max(dim=-1)[0] if len(outputs.shape) > 1 else outputs
        
        # Backward pass
        self.model.zero_grad()
        outputs.backward(torch.ones_like(outputs))
        
        gradients = inputs.grad.detach().cpu().numpy()
        return gradients
    
    def get_activations(self, inputs: torch.Tensor, layer_name: str) -> np.ndarray:
        """Extract activations from specific layer."""
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu().numpy())
        
        # Find layer by name
        target_layer = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # Register hook
        hook = target_layer.register_forward_hook(hook_fn)
        
        try:
            # Forward pass
            _ = self.forward(inputs)
            return activations[0] if activations else None
        finally:
            hook.remove()
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape from first layer."""
        first_layer = next(self.model.modules())
        if hasattr(first_layer, 'in_features'):
            return (first_layer.in_features,)
        elif hasattr(first_layer, 'in_channels'):
            # Assuming standard image input
            return (first_layer.in_channels, 224, 224)
        else:
            warnings.warn("Could not determine input shape automatically")
            return ()


class TensorFlowWrapper(ModelWrapper):
    """Wrapper for TensorFlow/Keras models."""
    
    def _setup(self) -> None:
        """Setup TensorFlow model for explanation."""
        self.is_keras = hasattr(self.model, 'layers')
        
        if self.is_keras:
            self._input_tensor = self.model.input
            self._output_tensor = self.model.output
        else:
            # For pure TF models, assume they have input/output attributes
            self._input_tensor = self.model.inputs[0] if hasattr(self.model, 'inputs') else None
            self._output_tensor = self.model.outputs[0] if hasattr(self.model, 'outputs') else None
    
    def forward(self, inputs: np.ndarray, **kwargs) -> np.ndarray:
        """Forward pass through TensorFlow model."""
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        
        if self.is_keras:
            outputs = self.model.predict(inputs, batch_size=kwargs.get('batch_size', 32))
        else:
            outputs = self.model(inputs)
            if isinstance(outputs, tf.Tensor):
                outputs = outputs.numpy()
        
        return outputs
    
    @tf.function
    def _compute_gradients(self, inputs: tf.Tensor, target: Optional[int] = None):
        """Compute gradients using GradientTape."""
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self.model(inputs)
            
            if target is not None:
                if len(outputs.shape) > 1:
                    outputs = outputs[:, target]
                else:
                    outputs = outputs[target]
            else:
                outputs = tf.reduce_max(outputs, axis=-1) if len(outputs.shape) > 1 else outputs
        
        gradients = tape.gradient(outputs, inputs)
        return gradients
    
    def get_gradients(self, inputs: np.ndarray, target: Optional[int] = None) -> np.ndarray:
        """Get gradients with respect to inputs."""
        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        gradients = self._compute_gradients(inputs_tensor, target)
        return gradients.numpy()
    
    def get_activations(self, inputs: np.ndarray, layer_name: str) -> np.ndarray:
        """Extract activations from specific layer."""
        if not self.is_keras:
            raise NotImplementedError("Activation extraction only supported for Keras models")
        
        # Find layer
        target_layer = None
        for layer in self.model.layers:
            if layer.name == layer_name:
                target_layer = layer
                break
        
        if target_layer is None:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # Create intermediate model
        intermediate_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=target_layer.output
        )
        
        activations = intermediate_model.predict(inputs)
        return activations
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape."""
        if self.is_keras:
            return tuple(self.model.input_shape[1:])
        elif self._input_tensor is not None:
            return tuple(self._input_tensor.shape[1:])
        else:
            warnings.warn("Could not determine input shape automatically")
            return ()


class ModelWrapperFactory:
    """Factory for creating appropriate model wrappers."""
    
    @staticmethod
    def create(model: Any, framework: Framework) -> ModelWrapper:
        """Create model wrapper based on framework."""
        if framework == Framework.PYTORCH:
            return PyTorchWrapper(model, framework)
        elif framework in [Framework.TENSORFLOW, Framework.KERAS]:
            return TensorFlowWrapper(model, framework)
        else:
            raise ValueError(f"Unsupported framework: {framework}")