#!/usr/bin/env python3
"""
Quick comparison between Megatron and Native PyTorch.
"""

import subprocess
import sys
import time

def run_quick_test(backend_name, megatron_flag):
    """Run a quick test and capture key output."""
    print(f"\n{'='*50}")
    print(f"🚀 Testing {backend_name}")
    print(f"{'='*50}")
    
    cmd = [sys.executable, "core/train_auto.py", megatron_flag]
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        # Run for 30 seconds max to get initial output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        end_time = time.time()
        
        print(f"⏱️ Time: {end_time - start_time:.1f}s")
        print(f"📊 Exit code: {result.returncode}")
        
        # Look for key indicators
        output = result.stdout + result.stderr
        
        if "Data Parallel" in output:
            print("🔍 Backend: PyTorch DataParallel")
        if "MegatronWrapper" in output:
            print("🔍 Backend: Megatron")
        if "Starting Megatron-LM" in output:
            print("🔍 Backend: Megatron Training")
        if "Training MoE model" in output:
            print("🔍 Backend: Native PyTorch")
            
        # Show first few lines of output
        lines = output.split('\n')[:10]
        print("\n📋 First 10 lines of output:")
        for line in lines:
            if line.strip():
                print(f"   {line}")
        
        return {
            'backend': backend_name,
            'time': end_time - start_time,
            'success': result.returncode == 0,
            'megatron_detected': 'MegatronWrapper' in output or 'Starting Megatron-LM' in output,
            'native_detected': 'Data Parallel' in output and 'MegatronWrapper' not in output
        }
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 30 seconds (this is normal)")
        return {
            'backend': backend_name,
            'time': 30,
            'success': True,  # Timeout is expected
            'megatron_detected': False,
            'native_detected': False
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'backend': backend_name,
            'time': 0,
            'success': False,
            'megatron_detected': False,
            'native_detected': False
        }

def main():
    print("🫐 Quick Megatron vs Native Comparison")
    print("=" * 50)
    print("This will run each backend for 30 seconds to show the differences")
    
    # Test both backends
    results = []
    
    # Test Native PyTorch
    result1 = run_quick_test("Native PyTorch", "--no-megatron")
    results.append(result1)
    
    # Test Megatron
    result2 = run_quick_test("Megatron-LM", "--use-megatron")
    results.append(result2)
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*50}")
    
    for result in results:
        backend_type = "Megatron" if result['megatron_detected'] else "Native" if result['native_detected'] else "Unknown"
        print(f"{result['backend']:20} | {backend_type:10} | {result['time']:5.1f}s")
    
    print(f"\n{'='*50}")
    print("💡 Key Differences:")
    print("   Native: 'Data Parallel' + 'Training MoE model'")
    print("   Megatron: 'MegatronWrapper' + 'Starting Megatron-LM'")
    print("   Both should show 'Auto-launching with 2 GPUs'")

if __name__ == "__main__":
    main()
