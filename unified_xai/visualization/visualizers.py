"""Visualization utilities for explanations."""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Tuple, Union
from unified_xai.core.base import ExplanationResult
import cv2
from pathlib import Path
import torch
import tensorflow as tf


class ExplanationVisualizer:
    """Main visualization class for explanations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cmap = config.get('cmap', 'RdBu_r')
        self.alpha = config.get('alpha', 0.7)
        self.save_path = config.get('save_path')
    
    def _convert_to_numpy(self, data: Any) -> np.ndarray:
        """Convert various tensor types to numpy array."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, tf.Tensor):
            return data.numpy()
        elif hasattr(data, '__array__'):
            return np.array(data)
        else:
            raise ValueError(f"Cannot convert {type(data)} to numpy array")
    
    def visualize(self, explanation: ExplanationResult, original_input: Optional[Any] = None,
                  **kwargs) -> Union[plt.Figure, go.Figure]:
        """Visualize explanation based on data type."""
        attribution = explanation.attribution
        
        # Convert original input to numpy if provided
        if original_input is not None:
            original_input = self._convert_to_numpy(original_input)
            
            # Handle different input formats
            if len(original_input.shape) == 4:  # Batch dimension
                original_input = original_input[0]
            
            # Convert CHW to HWC for images
            if len(original_input.shape) == 3 and original_input.shape[0] in [1, 3]:
                original_input = np.transpose(original_input, (1, 2, 0))
            
            # Normalize to [0, 1] for visualization
            if original_input.min() < 0 or original_input.max() > 1:
                original_input = (original_input - original_input.min()) / (original_input.max() - original_input.min() + 1e-8)
        
        if len(attribution.shape) == 2:
            # 2D attribution (image)
            return self._visualize_image(attribution, original_input, explanation.metadata)
        elif len(attribution.shape) == 1:
            # 1D attribution (tabular/text)
            return self._visualize_1d(attribution, explanation.metadata)
        elif len(attribution.shape) == 3:
            # 3D attribution (could be multi-channel image)
            return self._visualize_multichannel(attribution, original_input, explanation.metadata)
        elif len(attribution.shape) == 4:
            # 4D attribution (batch of images) - visualize first one
            return self._visualize_image(attribution[0], original_input, explanation.metadata)
        else:
            raise ValueError(f"Unsupported attribution shape: {attribution.shape}")
    
    def _visualize_image(self, attribution: np.ndarray, original: Optional[np.ndarray],
                        metadata: Dict) -> plt.Figure:
        """Visualize image attribution."""
        # Handle multi-channel attribution
        if len(attribution.shape) == 3:
            # If CHW format, take mean across channels or select first channel
            if attribution.shape[0] in [1, 3]:
                attribution = np.mean(attribution, axis=0)
            else:
                attribution = np.mean(attribution, axis=2)
        
        fig, axes = plt.subplots(1, 3 if original is not None else 1, figsize=(15, 5))
        
        if original is not None:
            # Original image
            ax = axes[0] if isinstance(axes, np.ndarray) else axes
            if len(original.shape) == 3:
                if original.shape[2] == 1:
                    ax.imshow(original[:, :, 0], cmap='gray')
                else:
                    ax.imshow(original)
            else:
                ax.imshow(original, cmap='gray')
            ax.set_title('Original')
            ax.axis('off')
            
            # Attribution heatmap
            ax = axes[1] if isinstance(axes, np.ndarray) else axes
            im = ax.imshow(attribution, cmap=self.cmap)
            ax.set_title(f'Attribution ({metadata.get("method", "unknown")})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Overlay
            if isinstance(axes, np.ndarray) and len(axes) > 2:
                ax = axes[2]
                if len(original.shape) == 3:
                    ax.imshow(original)
                else:
                    ax.imshow(original, cmap='gray')
                
                # Overlay attribution
                overlay = self._create_overlay(attribution, original.shape[:2])
                ax.imshow(overlay, cmap=self.cmap, alpha=self.alpha)
                ax.set_title('Overlay')
                ax.axis('off')
        else:
            # Just attribution
            ax = axes if not isinstance(axes, np.ndarray) else axes[0]
            im = ax.imshow(attribution, cmap=self.cmap)
            ax.set_title(f'Attribution ({metadata.get("method", "unknown")})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if self.save_path:
            save_path = Path(self.save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / f'explanation_{metadata.get("method", "unknown")}.png', dpi=100, bbox_inches='tight')
        
        return fig
    
    def _visualize_1d(self, attribution: np.ndarray, metadata: Dict) -> go.Figure:
        """Visualize 1D attribution using Plotly."""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(len(attribution))),
            y=attribution,
            marker_color=attribution,
            marker_colorscale=self.cmap,
            text=[f'{val:.3f}' for val in attribution],
            textposition='auto',
        ))
        
        fig.update_layout(
            title=f'Feature Attribution ({metadata.get("method", "unknown")})',
            xaxis_title='Feature Index',
            yaxis_title='Attribution Score',
            showlegend=False,
            height=500,
            template='plotly_white'
        )
        
        if self.save_path:
            save_path = Path(self.save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path / f'explanation_{metadata.get("method", "unknown")}.html')
        
        return fig
    
    def _visualize_multichannel(self, attribution: np.ndarray, original: Optional[np.ndarray],
                               metadata: Dict) -> plt.Figure:
        """Visualize multi-channel attribution."""
        # If CHW format, transpose to HWC
        if attribution.shape[0] in [1, 3] and len(attribution.shape) == 3:
            attribution = np.transpose(attribution, (1, 2, 0))
        
        n_channels = attribution.shape[-1]
        n_cols = min(n_channels, 3)
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        
        if n_channels == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_channels):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[col]
            
            im = ax.imshow(attribution[:, :, i], cmap=self.cmap)
            ax.set_title(f'Channel {i}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for i in range(n_channels, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[col]
            ax.axis('off')
        
        plt.suptitle(f'Attribution ({metadata.get("method", "unknown")})')
        plt.tight_layout()
        
        if self.save_path:
            save_path = Path(self.save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / f'explanation_{metadata.get("method", "unknown")}.png', dpi=100, bbox_inches='tight')
        
        return fig
    
    def _create_overlay(self, attribution: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Create overlay by resizing attribution to match target shape."""
        if attribution.shape != target_shape:
            attribution = cv2.resize(attribution, (target_shape[1], target_shape[0]))
        return attribution
    
    def compare_explanations(self, explanations: List[ExplanationResult],
                            original_input: Optional[Any] = None) -> plt.Figure:
        """Compare multiple explanations side by side."""
        if original_input is not None:
            original_input = self._convert_to_numpy(original_input)
            if len(original_input.shape) == 4:
                original_input = original_input[0]
            if len(original_input.shape) == 3 and original_input.shape[0] in [1, 3]:
                original_input = np.transpose(original_input, (1, 2, 0))
            # Normalize
            if original_input.min() < 0 or original_input.max() > 1:
                original_input = (original_input - original_input.min()) / (original_input.max() - original_input.min() + 1e-8)
        
        n_explanations = len(explanations)
        fig, axes = plt.subplots(1, n_explanations + (1 if original_input is not None else 0),
                                 figsize=(5*(n_explanations+1), 5))
        
        if n_explanations + (1 if original_input is not None else 0) == 1:
            axes = [axes]
        
        if original_input is not None:
            ax = axes[0]
            if len(original_input.shape) == 3:
                ax.imshow(original_input)
            else:
                ax.imshow(original_input, cmap='gray')
            ax.set_title('Original')
            ax.axis('off')
            start_idx = 1
        else:
            start_idx = 0
        
        for i, explanation in enumerate(explanations):
            ax = axes[start_idx + i]
            attribution = explanation.attribution
            
            # Handle multi-channel attribution
            if len(attribution.shape) == 3:
                if attribution.shape[0] in [1, 3]:
                    attribution = np.mean(attribution, axis=0)
            elif len(attribution.shape) == 4:
                attribution = np.mean(attribution[0], axis=0)
            
            im = ax.imshow(attribution, cmap=self.cmap)
            ax.set_title(explanation.method)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if self.save_path:
            save_path = Path(self.save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / 'comparison.png', dpi=100, bbox_inches='tight')
        
        return fig