"""Example usage of the Unified XAI library."""

import torch
import torch.nn as nn
import numpy as np
from unified_xai import XAIAnalyzer, XAIConfig
from unified_xai.config import Framework, Modality
import matplotlib.pyplot as plt


def create_example_model():
    """Create a simple CNN model for demonstration."""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(64 * 56 * 56, 128)
            self.fc2 = nn.Linear(128, 10)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    return SimpleCNN()


def main():
    """Main example function."""
    # Create model
    model = create_example_model()
    model.eval()
    
    # Configure XAI
    config = XAIConfig(
        framework=Framework.PYTORCH,
        modality=Modality.IMAGE,
        batch_size=1,
        gradient_config={
            'normalize': True,
            'smooth_samples': 50
        },
        metrics_config={
            'compute_faithfulness': False,  # Disable for faster demo
            'compute_complexity': True,
            'compute_stability': False
        }
    )
    
    # Initialize analyzer
    analyzer = XAIAnalyzer(model, config)
    
    # Create dummy input (without requiring gradients)
    input_image = torch.randn(1, 3, 224, 224)
    
    # Single explanation
    print("Generating single explanation...")
    explanation = analyzer.explain(input_image, method='integrated_gradients', target=0)
    print(f"Explanation shape: {explanation.attribution.shape}")
    print(f"Method: {explanation.method}")
    
    # Multiple explanations
    print("\nGenerating multiple explanations...")
    explanations = analyzer.explain_multiple(
        input_image,
        methods=['vanilla_gradient', 'integrated_gradients', 'smoothgrad'],
        target=0
    )
    print(f"Generated {len(explanations)} explanations")
    
    # Compare methods
    print("\nComparing explanation methods...")
    comparison = analyzer.compare_methods(
        input_image,
        methods=['vanilla_gradient', 'integrated_gradients'],
        metrics=['complexity'],  # Only use complexity for faster demo
        visualize=False
    )
    
    print("Rankings:")
    for method, score in comparison['rankings']:
        print(f"  {method}: {score:.4f}")
    
    # Aggregate explanations
    print("\nAggregating explanations...")
    aggregated = analyzer.aggregate_explanations(
        list(explanations.values()),
        method='mean'
    )
    print(f"Aggregated explanation shape: {aggregated.attribution.shape}")
    
    # Visualize (would display in notebook or save to file)
    # Convert input image to numpy properly
    if isinstance(input_image, torch.Tensor):
        # Detach from computation graph and convert to numpy
        input_np = input_image.detach().cpu().numpy()
        # Remove batch dimension and transpose to HWC format for visualization
        if len(input_np.shape) == 4:
            input_np = input_np[0]  # Remove batch dimension
        if input_np.shape[0] in [1, 3]:  # CHW format
            input_np = np.transpose(input_np, (1, 2, 0))  # Convert to HWC
    else:
        input_np = input_image
    
    try:
        fig = analyzer.visualize(explanation, input_np)
        
        # Save the figure instead of showing (for non-interactive environments)
        if fig is not None:
            if hasattr(fig, 'savefig'):  # Matplotlib figure
                fig.savefig('explanation_visualization.png')
                print("\nVisualization saved to 'explanation_visualization.png'")
                plt.close(fig)
            elif hasattr(fig, 'write_html'):  # Plotly figure
                fig.write_html('explanation_visualization.html')
                print("\nVisualization saved to 'explanation_visualization.html'")
    except Exception as e:
        print(f"\nVisualization skipped due to: {e}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()