#!/usr/bin/env python3
"""
Integration test for the GPU-adaptive matmul system.

This script tests the system configuration detection and matmul dispatching
to ensure everything works correctly across different hardware configurations.
"""

import torch
import time
import numpy as np
from ..system import SYSTEM_CONFIG, print_system_info
from ..ops.matmul import matmul, matmul_with_info, get_available_kernels, print_kernel_info


def test_system_detection():
    """Test that system detection works correctly."""
    print("🧪 Testing System Detection...")
    
    config = SYSTEM_CONFIG
    
    # Basic checks
    assert hasattr(config, 'device_count')
    assert hasattr(config, 'capability')
    assert hasattr(config, 'architecture')
    assert hasattr(config, 'has_fp8_support')
    assert hasattr(config, 'has_tensor_cores')
    
    print(f"   ✅ Device count: {config.device_count}")
    print(f"   ✅ Architecture: {config.architecture}")
    print(f"   ✅ Compute capability: {config.capability}")
    print(f"   ✅ FP8 support: {config.has_fp8_support}")
    print(f"   ✅ Tensor cores: {config.has_tensor_cores}")
    
    return True


def test_matmul_functionality():
    """Test that matmul operations work correctly."""
    print("\n🧪 Testing Matmul Functionality...")
    
    # Create test tensors
    batch_size, seq_len, d_model = 2, 128, 512
    d_out = 256
    
    x = torch.randn(batch_size, seq_len, d_model, device='cuda' if torch.cuda.is_available() else 'cpu')
    w = torch.randn(d_out, d_model, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test basic matmul
    try:
        result = matmul(x, w)
        expected_shape = (batch_size, seq_len, d_out)
        assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"
        print(f"   ✅ Basic matmul: {result.shape}")
    except Exception as e:
        print(f"   ❌ Basic matmul failed: {e}")
        return False
    
    # Test matmul with info
    try:
        result, kernel_name = matmul_with_info(x, w)
        assert result.shape == expected_shape
        print(f"   ✅ Matmul with info: {kernel_name}")
    except Exception as e:
        print(f"   ❌ Matmul with info failed: {e}")
        return False
    
    return True


def test_kernel_selection():
    """Test that the correct kernels are selected."""
    print("\n🧪 Testing Kernel Selection...")
    
    available_kernels = get_available_kernels()
    assert len(available_kernels) > 0, "No kernels available"
    
    print(f"   ✅ Available kernels: {available_kernels}")
    
    # Test that we get different kernels for different architectures
    config = SYSTEM_CONFIG
    if config.device_count > 0:
        if config.has_fp8_support:
            assert any('fp8' in kernel.lower() for kernel in available_kernels), "FP8 kernel not found"
            print("   ✅ FP8 kernel detected")
        elif config.has_tensor_cores:
            assert any('tensor' in kernel.lower() or 'bf16' in kernel.lower() or 'fp16' in kernel.lower() for kernel in available_kernels), "Tensor core kernel not found"
            print("   ✅ Tensor core kernel detected")
        else:
            assert 'fallback' in available_kernels or 'generic' in available_kernels, "No fallback kernel"
            print("   ✅ Fallback kernel detected")
    
    return True


def test_performance():
    """Test performance of different kernels."""
    print("\n🧪 Testing Performance...")
    
    if not torch.cuda.is_available():
        print("   ⏭️ Skipping performance test (no CUDA)")
        return True
    
    # Create larger tensors for performance testing
    batch_size, seq_len, d_model = 8, 1024, 1024
    d_out = 1024
    
    x = torch.randn(batch_size, seq_len, d_model, device='cuda')
    w = torch.randn(d_out, d_model, device='cuda')
    
    # Warmup
    for _ in range(5):
        _ = matmul(x, w)
    
    torch.cuda.synchronize()
    
    # Time the operation
    num_iterations = 10
    start_time = time.time()
    
    for _ in range(num_iterations):
        result = matmul(x, w)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"   ✅ Average matmul time: {avg_time*1000:.2f}ms")
    
    return True


def test_numerical_stability():
    """Test numerical stability across different kernels."""
    print("\n🧪 Testing Numerical Stability...")
    
    # Create test case that might cause numerical issues
    batch_size, seq_len, d_model = 4, 64, 256
    d_out = 128
    
    # Use values that might cause overflow/underflow
    x = torch.randn(batch_size, seq_len, d_model, device='cuda' if torch.cuda.is_available() else 'cpu') * 100
    w = torch.randn(d_out, d_model, device='cuda' if torch.cuda.is_available() else 'cpu') * 0.01
    
    try:
        result = matmul(x, w)
        
        # Check for NaN or Inf
        assert not torch.isnan(result).any(), "Result contains NaN"
        assert not torch.isinf(result).any(), "Result contains Inf"
        
        print("   ✅ No numerical issues detected")
        return True
    except Exception as e:
        print(f"   ❌ Numerical stability test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 GPU-Adaptive Matmul System Test Suite")
    print("=" * 50)
    
    # Print system info
    print_system_info()
    print()
    
    # Run tests
    tests = [
        test_system_detection,
        test_matmul_functionality,
        test_kernel_selection,
        test_performance,
        test_numerical_stability,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The GPU-adaptive system is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    # Print kernel info
    print("\n🔧 Kernel Information:")
    print_kernel_info()


if __name__ == "__main__":
    main()
