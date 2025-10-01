"""Preprocessing utilities for inputs and attributions."""

import numpy as np
from typing import Union, Tuple, Optional
import cv2


def normalize_attribution(attribution: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize attribution values."""
    if method == 'minmax':
        min_val = attribution.min()
        max_val = attribution.max()
        if max_val - min_val > 0:
            return (attribution - min_val) / (max_val - min_val)
        return attribution
    
    elif method == 'zscore':
        mean = attribution.mean()
        std = attribution.std()
        if std > 0:
            return (attribution - mean) / std
        return attribution - mean
    
    elif method == 'abs_max':
        max_abs = np.abs(attribution).max()
        if max_abs > 0:
            return attribution / max_abs
        return attribution
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resize_attribution(attribution: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize attribution map to target shape."""
    if attribution.shape[:2] == target_shape:
        return attribution
    
    return cv2.resize(attribution, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def smooth_attribution(attribution: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Gaussian smoothing to attribution."""
    return cv2.GaussianBlur(attribution, (kernel_size, kernel_size), 0)


def threshold_attribution(attribution: np.ndarray, threshold: float = 0.5,
                         percentile: bool = True) -> np.ndarray:
    """Apply threshold to attribution values."""
    if percentile:
        threshold_value = np.percentile(np.abs(attribution), threshold * 100)
    else:
        threshold_value = threshold
    
    mask = np.abs(attribution) >= threshold_value
    return attribution * mask


def aggregate_attributions(attributions: list, method: str = 'mean') -> np.ndarray:
    """Aggregate multiple attribution maps."""
    if not attributions:
        raise ValueError("No attributions to aggregate")
    
    attributions = np.array(attributions)
    
    if method == 'mean':
        return np.mean(attributions, axis=0)
    elif method == 'median':
        return np.median(attributions, axis=0)
    elif method == 'max':
        return np.max(attributions, axis=0)
    elif method == 'min':
        return np.min(attributions, axis=0)
    elif method == 'std':
        return np.std(attributions, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")