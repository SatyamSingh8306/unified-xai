"""Gradient-based explanation methods."""

import numpy as np
import torch
import tensorflow as tf
from typing import Any, Optional, Dict
from unified_xai.core.base import ExplainerBase, ExplanationResult
from unified_xai.config import ExplanationType
from unified_xai.utils.preprocessing import normalize_attribution


class VanillaGradient(ExplainerBase):
    """Vanilla gradient explanation method."""
    
    def explain(self, inputs: Any, targets: Optional[Any] = None, **kwargs) -> ExplanationResult:
        """Generate vanilla gradient explanation."""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid input format")
        
        # Get gradients
        gradients = self.model_wrapper.get_gradients(inputs, targets)
        
        # Post-process
        if self.config.get('abs_value', True):
            gradients = np.abs(gradients)
        
        if self.config.get('normalize', True):
            gradients = normalize_attribution(gradients)
        
        return ExplanationResult(
            attribution=gradients,
            method="vanilla_gradient",
            explanation_type=ExplanationType.GRADIENT,
            metadata={"target": targets}
        )
    
    def validate_inputs(self, inputs: Any) -> bool:
        """Validate input format."""
        if isinstance(inputs, (np.ndarray, torch.Tensor, tf.Tensor)):
            return True
        return False


class IntegratedGradients(ExplainerBase):
    """Integrated Gradients explanation method."""
    
    def explain(self, inputs: Any, targets: Optional[Any] = None, **kwargs) -> ExplanationResult:
        """Generate integrated gradients explanation."""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid input format")
        
        # Convert inputs to numpy if needed
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.detach().cpu().numpy()
        elif isinstance(inputs, tf.Tensor):
            inputs_np = inputs.numpy()
        else:
            inputs_np = inputs
        
        # Get baseline
        baseline = kwargs.get('baseline', None)
        if baseline is None:
            baseline = np.zeros_like(inputs_np)
        elif isinstance(baseline, torch.Tensor):
            baseline = baseline.detach().cpu().numpy()
        elif isinstance(baseline, tf.Tensor):
            baseline = baseline.numpy()
        
        n_steps = self.config.get('n_steps', 50)
        
        # Generate interpolated inputs
        alphas = np.linspace(0, 1, n_steps)
        interpolated_inputs = []
        
        for alpha in alphas:
            # Ensure numpy array operations
            interpolated = baseline + alpha * (inputs_np - baseline)
            interpolated_inputs.append(interpolated)
        
        # Calculate gradients for each interpolated input
        gradients = []
        for interp_input in interpolated_inputs:
            # Convert back to original tensor type if needed
            if isinstance(inputs, torch.Tensor):
                interp_tensor = torch.tensor(interp_input, dtype=inputs.dtype, device=inputs.device)
                grad = self.model_wrapper.get_gradients(interp_tensor, targets)
            elif isinstance(inputs, tf.Tensor):
                interp_tensor = tf.convert_to_tensor(interp_input, dtype=inputs.dtype)
                grad = self.model_wrapper.get_gradients(interp_tensor, targets)
            else:
                grad = self.model_wrapper.get_gradients(interp_input, targets)
            
            gradients.append(grad)
        
        # Integrate gradients
        avg_gradients = np.mean(gradients, axis=0)
        integrated_grads = (inputs_np - baseline) * avg_gradients
        
        if self.config.get('normalize', True):
            integrated_grads = normalize_attribution(integrated_grads)
        
        return ExplanationResult(
            attribution=integrated_grads,
            method="integrated_gradients",
            explanation_type=ExplanationType.GRADIENT,
            metadata={
                "target": targets,
                "n_steps": n_steps,
                "baseline_type": "zero" if np.allclose(baseline, 0) else "custom"
            }
        )
    
    def validate_inputs(self, inputs: Any) -> bool:
        """Validate input format."""
        if isinstance(inputs, (np.ndarray, torch.Tensor, tf.Tensor)):
            return True
        return False


class SmoothGrad(ExplainerBase):
    """SmoothGrad explanation method."""
    
    def explain(self, inputs: Any, targets: Optional[Any] = None, **kwargs) -> ExplanationResult:
        """Generate SmoothGrad explanation."""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid input format")
        
        n_samples = self.config.get('smooth_samples', 50)
        noise_scale = self.config.get('noise_scale', 0.1)
        
        # Convert inputs to numpy for consistent operations
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.detach().cpu().numpy()
            is_torch = True
        elif isinstance(inputs, tf.Tensor):
            inputs_np = inputs.numpy()
            is_torch = False
            is_tf = True
        else:
            inputs_np = inputs
            is_torch = False
            is_tf = False
        
        # Calculate standard deviation for noise
        input_range = inputs_np.max() - inputs_np.min()
        std_dev = noise_scale * input_range
        
        # Generate noisy samples and compute gradients
        gradients = []
        for _ in range(n_samples):
            noise = np.random.normal(0, std_dev, inputs_np.shape).astype(inputs_np.dtype)
            noisy_input_np = inputs_np + noise
            
            # Convert back to original tensor type
            if is_torch:
                noisy_input = torch.tensor(noisy_input_np, dtype=inputs.dtype, device=inputs.device)
            elif is_tf:
                noisy_input = tf.convert_to_tensor(noisy_input_np, dtype=inputs.dtype)
            else:
                noisy_input = noisy_input_np
            
            grad = self.model_wrapper.get_gradients(noisy_input, targets)
            gradients.append(grad)
        
        # Average gradients
        smooth_grad = np.mean(gradients, axis=0)
        
        if self.config.get('abs_value', True):
            smooth_grad = np.abs(smooth_grad)
        
        if self.config.get('normalize', True):
            smooth_grad = normalize_attribution(smooth_grad)
        
        return ExplanationResult(
            attribution=smooth_grad,
            method="smoothgrad",
            explanation_type=ExplanationType.GRADIENT,
            metadata={
                "target": targets,
                "n_samples": n_samples,
                "noise_scale": noise_scale
            }
        )
    
    def validate_inputs(self, inputs: Any) -> bool:
        """Validate input format."""
        if isinstance(inputs, (np.ndarray, torch.Tensor, tf.Tensor)):
            return True
        return False


class GradCAM(ExplainerBase):
    """Grad-CAM explanation method for convolutional networks."""
    
    def explain(self, inputs: Any, targets: Optional[Any] = None, **kwargs) -> ExplanationResult:
        """Generate Grad-CAM explanation."""
        if not self.validate_inputs(inputs):
            raise ValueError("Invalid input format")
        
        layer_name = kwargs.get('layer_name', self._get_last_conv_layer())
        
        # Get activations from target layer
        activations = self.model_wrapper.get_activations(inputs, layer_name)
        
        # Get gradients with respect to activations
        gradients = self.model_wrapper.get_gradients(inputs, targets)
        
        # Ensure numpy arrays
        if not isinstance(activations, np.ndarray):
            if hasattr(activations, 'numpy'):
                activations = activations.numpy()
            elif hasattr(activations, 'detach'):
                activations = activations.detach().cpu().numpy()
        
        if not isinstance(gradients, np.ndarray):
            if hasattr(gradients, 'numpy'):
                gradients = gradients.numpy()
            elif hasattr(gradients, 'detach'):
                gradients = gradients.detach().cpu().numpy()
        
        # Global average pooling of gradients
        if len(gradients.shape) == 4:  # Batch x Channels x Height x Width
            weights = np.mean(gradients, axis=(2, 3))
        else:
            weights = gradients
        
        # Weighted combination of activation maps
        if len(activations.shape) >= 3:
            cam = np.zeros(activations.shape[-2:])  # Use last two dimensions
            # Handle different channel positions
            if len(activations.shape) == 4:
                # Batch dimension present
                for i in range(activations.shape[1]):
                    if i < len(weights[0]):
                        cam += weights[0, i] * activations[0, i]
            else:
                # No batch dimension
                for i in range(activations.shape[0]):
                    if i < len(weights):
                        cam += weights[i] * activations[i]
        else:
            cam = activations * weights
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        if self.config.get('normalize', True):
            cam = normalize_attribution(cam)
        
        return ExplanationResult(
            attribution=cam,
            method="gradcam",
            explanation_type=ExplanationType.GRADIENT,
            metadata={
                "target": targets,
                "layer_name": layer_name
            }
        )
    
    def validate_inputs(self, inputs: Any) -> bool:
        """Validate input format."""
        if isinstance(inputs, (np.ndarray, torch.Tensor, tf.Tensor)):
            shape = inputs.shape if isinstance(inputs, np.ndarray) else list(inputs.shape)
            if len(shape) == 4:  # Image data
                return True
        return False
    
    def _get_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model."""
        # This is framework-specific and would need proper implementation
        # For PyTorch models
        if hasattr(self.model_wrapper, 'model'):
            model = self.model_wrapper.model
            if hasattr(model, 'named_modules'):
                conv_layers = []
                for name, module in model.named_modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
                        conv_layers.append(name)
                if conv_layers:
                    return conv_layers[-1]
        
        # Default fallback
        return "conv1"  # or another sensible default