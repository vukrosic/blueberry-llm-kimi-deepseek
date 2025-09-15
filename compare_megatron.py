#!/usr/bin/env python3
"""
Compare Megatron vs Native PyTorch training performance.
"""

import subprocess
import sys
import time
import os

def run_training(backend_name, megatron_flag, max_steps=50):
    """Run training with specified backend."""
    print(f"\n{'='*60}")
    print(f"🚀 Testing {backend_name}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "train.py",
        megatron_flag,
        "--max-steps", str(max_steps),
        "--batch-size", "8",  # Smaller batch for faster testing
        "--num-documents", "500"  # Smaller dataset for faster testing
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        print(f"⏱️ Training time: {end_time - start_time:.1f} seconds")
        print(f"📊 Exit code: {result.returncode}")
        
        # Extract key metrics from output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'it/s' in line or 'loss=' in line or 'Training:' in line:
                print(f"📈 {line.strip()}")
        
        return {
            'backend': backend_name,
            'time': end_time - start_time,
            'success': result.returncode == 0,
            'output': result.stdout
        }
        
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 5 minutes")
        return {
            'backend': backend_name,
            'time': 300,
            'success': False,
            'output': "Timeout"
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'backend': backend_name,
            'time': 0,
            'success': False,
            'output': str(e)
        }

def main():
    print("🫐 Blueberry LLM: Megatron vs Native Comparison")
    print("=" * 60)
    
    # Test configurations
    configs = [
        ("Native PyTorch", "--no-megatron"),
        ("Megatron-LM", "--use-megatron")
    ]
    
    results = []
    
    for backend_name, flag in configs:
        result = run_training(backend_name, flag)
        results.append(result)
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{result['backend']:20} | {result['time']:6.1f}s | {status}")
    
    # Performance comparison
    if len(results) == 2 and all(r['success'] for r in results):
        native_time = results[0]['time']
        megatron_time = results[1]['time']
        
        if megatron_time < native_time:
            speedup = native_time / megatron_time
            print(f"\n🚀 Megatron is {speedup:.1f}x faster!")
        else:
            slowdown = megatron_time / native_time
            print(f"\n🐌 Megatron is {slowdown:.1f}x slower")
    
    print(f"\n{'='*60}")
    print("💡 Tips:")
    print("   - Check GPU utilization: nvidia-smi")
    print("   - Monitor memory usage during training")
    print("   - Look for 'Data Parallel' vs 'Megatron' in output")
    print("   - Compare training speed (it/s)")

if __name__ == "__main__":
    main()
