#!/usr/bin/env python3
"""
Quick Scale Test - Test just Tiny and Default models
"""

import subprocess
import sys
import time

def run_quick_test():
    """Run a quick test with just Tiny and Default models."""
    print("🫐 Quick Scale Test")
    print("=" * 40)
    print("Testing Tiny and Default models only")
    print("Hardware: 2x RTX 4090")
    print()
    
    experiments = [
        ("Tiny", "Native", ["--config", "dev", "--no-megatron"]),
        ("Tiny", "Megatron", ["--config", "dev", "--use-megatron"]),
        ("Default", "Native", ["--no-megatron"]),
        ("Default", "Megatron", ["--use-megatron"])
    ]
    
    results = []
    
    for i, (model, backend, args) in enumerate(experiments, 1):
        print(f"\n{'='*40}")
        print(f"🧪 Test {i}/4: {model} Model with {backend}")
        print(f"{'='*40}")
        
        cmd = [sys.executable, "core/train_auto.py"] + args
        print(f"🚀 Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Run with 2 minute timeout for quick test
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            end_time = time.time()
            
            duration = end_time - start_time
            success = result.returncode == 0
            
            print(f"⏱️ Duration: {duration:.1f}s")
            print(f"📊 Success: {'✅' if success else '❌'}")
            
            if success:
                # Extract speed from output
                output = result.stdout
                lines = output.split('\n')
                for line in lines:
                    if 'it/s' in line and ('Training' in line or 'Training MoE' in line):
                        try:
                            parts = line.split('it/s')[0].split()
                            for part in reversed(parts):
                                if ',' in part:
                                    speed = float(part.split(',')[-1])
                                    print(f"⚡ Speed: {speed:.2f} it/s")
                                    break
                        except:
                            pass
                        break
            
            results.append({
                "model": model,
                "backend": backend,
                "duration": duration,
                "success": success
            })
            
        except subprocess.TimeoutExpired:
            print("⏰ Test timed out after 2 minutes")
            results.append({
                "model": model,
                "backend": backend,
                "duration": 120,
                "success": False
            })
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "model": model,
                "backend": backend,
                "duration": 0,
                "success": False
            })
        
        time.sleep(2)  # Brief pause between tests
    
    # Print summary
    print(f"\n{'='*40}")
    print("📊 QUICK TEST SUMMARY")
    print(f"{'='*40}")
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{result['model']:8} | {result['backend']:8} | {result['duration']:6.1f}s | {status}")
    
    successful = sum(1 for r in results if r["success"])
    print(f"\n✅ Successful: {successful}/4 ({successful/4*100:.0f}%)")
    
    print(f"\n💡 To run full experiment:")
    print(f"   python scale_experiment.py")

if __name__ == "__main__":
    run_quick_test()
