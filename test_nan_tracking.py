#!/usr/bin/env python3
"""
Test script to demonstrate NaN gradient tracking functionality.
This will show you exactly what the NaN percentage tracking looks like.
"""

import torch
import torch.nn as nn
from training.trainer import TrainingState
from configs.adaptive_moe_config import AdaptiveMoEModelConfig

def create_test_model():
    """Create a simple test model that can generate NaN gradients."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    return model

def simulate_nan_gradients(model, optimizer):
    """Simulate NaN gradients by manually setting some gradients to NaN."""
    # Forward pass
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, y)
    
    # Backward pass
    loss.backward()
    
    # Manually introduce NaN gradients in some parameters
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            if param_count % 3 == 0:  # Every 3rd parameter gets NaN
                param.grad.data.fill_(float('nan'))
            elif param_count % 5 == 0:  # Every 5th parameter gets Inf
                param.grad.data.fill_(float('inf'))
            param_count += 1
    
    return model

def test_nan_tracking():
    """Test the NaN gradient tracking functionality."""
    print("🧪 Testing NaN Gradient Tracking")
    print("=" * 50)
    
    # Create test model and optimizer
    model = create_test_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create training state
    config = AdaptiveMoEModelConfig()
    training_state = TrainingState(
        model=model,
        optimizers=[optimizer],
        schedulers=[],
        scaler=None,
        config=config,
        device=torch.device('cpu'),
        start_step=0
    )
    
    print("📊 Simulating NaN gradients...")
    
    # Simulate multiple gradient steps with NaN/Inf
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        # Simulate NaN gradients
        model = simulate_nan_gradients(model, optimizer)
        
        # Check gradients (this is what happens in the trainer)
        total_norm = 0.0
        param_count = 0
        nan_count = 0
        inf_count = 0
        total_grad_elements = 0
        nan_grad_elements = 0
        inf_grad_elements = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_count += 1
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Count gradient elements
                total_grad_elements += param.grad.numel()
                
                # Check for NaN/inf gradients
                has_nan = torch.isnan(param.grad).any()
                has_inf = torch.isinf(param.grad).any()
                
                if has_nan:
                    nan_count += 1
                    nan_grad_elements += torch.isnan(param.grad).sum().item()
                    print(f"⚠️  WARNING: Invalid gradient detected in parameter")
                    print(f"   Param shape: {param.shape}")
                    print(f"   Grad contains NaN: {has_nan}")
                    print(f"   Grad contains Inf: {has_inf}")
                    # Zero out invalid gradients
                    param.grad.data.zero_()
                elif has_inf:
                    inf_count += 1
                    inf_grad_elements += torch.isinf(param.grad).sum().item()
                    print(f"⚠️  WARNING: Invalid gradient detected in parameter")
                    print(f"   Param shape: {param.shape}")
                    print(f"   Grad contains NaN: {has_nan}")
                    print(f"   Grad contains Inf: {has_inf}")
                    # Zero out invalid gradients
                    param.grad.data.zero_()
        
        # Calculate and display NaN/Inf statistics
        if param_count > 0:
            nan_param_percentage = (nan_count / param_count) * 100
            inf_param_percentage = (inf_count / param_count) * 100
            
            if total_grad_elements > 0:
                nan_element_percentage = (nan_grad_elements / total_grad_elements) * 100
                inf_element_percentage = (inf_grad_elements / total_grad_elements) * 100
                
                print(f"📊 Gradient Statistics:")
                print(f"   Parameters with NaN: {nan_count}/{param_count} ({nan_param_percentage:.2f}%)")
                print(f"   Parameters with Inf: {inf_count}/{param_count} ({inf_param_percentage:.2f}%)")
                print(f"   Gradient elements NaN: {nan_grad_elements:,}/{total_grad_elements:,} ({nan_element_percentage:.4f}%)")
                print(f"   Gradient elements Inf: {inf_grad_elements:,}/{total_grad_elements:,} ({inf_element_percentage:.4f}%)")
                
                # Accumulate statistics
                training_state.total_nan_params += nan_count
                training_state.total_inf_params += inf_count
                training_state.total_nan_elements += nan_grad_elements
                training_state.total_inf_elements += inf_grad_elements
                training_state.total_gradient_steps += 1
            else:
                print(f"📊 Gradient Statistics:")
                print(f"   Parameters with NaN: {nan_count}/{param_count} ({nan_param_percentage:.2f}%)")
                print(f"   Parameters with Inf: {inf_count}/{param_count} ({inf_param_percentage:.2f}%)")
                
                # Accumulate statistics
                training_state.total_nan_params += nan_count
                training_state.total_inf_params += inf_count
                training_state.total_gradient_steps += 1
        
        # Clear gradients for next iteration
        optimizer.zero_grad()
    
    # Display final cumulative statistics
    print(f"\n🎯 Test completed!")
    
    if training_state.total_gradient_steps > 0:
        print(f"\n🔍 Cumulative NaN Gradient Statistics:")
        print(f"   Total gradient steps: {training_state.total_gradient_steps}")
        print(f"   Total parameters with NaN: {training_state.total_nan_params}")
        print(f"   Total parameters with Inf: {training_state.total_inf_params}")
        print(f"   Total gradient elements NaN: {training_state.total_nan_elements:,}")
        print(f"   Total gradient elements Inf: {training_state.total_inf_elements:,}")
        
        avg_nan_params_per_step = training_state.total_nan_params / training_state.total_gradient_steps
        avg_inf_params_per_step = training_state.total_inf_params / training_state.total_gradient_steps
        avg_nan_elements_per_step = training_state.total_nan_elements / training_state.total_gradient_steps
        avg_inf_elements_per_step = training_state.total_inf_elements / training_state.total_gradient_steps
        
        print(f"   Average NaN params per step: {avg_nan_params_per_step:.2f}")
        print(f"   Average Inf params per step: {avg_inf_params_per_step:.2f}")
        print(f"   Average NaN elements per step: {avg_nan_elements_per_step:.0f}")
        print(f"   Average Inf elements per step: {avg_inf_elements_per_step:.0f}")
    else:
        print(f"\n🔍 No gradient statistics available (no gradient steps performed)")

if __name__ == "__main__":
    test_nan_tracking()
