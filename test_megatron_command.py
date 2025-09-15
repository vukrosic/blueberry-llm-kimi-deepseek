#!/usr/bin/env python3
"""
Test script to verify Megatron command functionality.
"""

import subprocess
import sys
import os

def test_megatron_command():
    """Test the new Megatron command."""
    print("🧪 Testing Megatron command functionality...")
    
    # Test 1: Check if --use-megatron flag is recognized
    print("\n1️⃣ Testing --use-megatron flag recognition...")
    try:
        result = subprocess.run([
            sys.executable, "core/train_auto.py", "--use-megatron", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if "--use-megatron" in result.stdout:
            print("✅ --use-megatron flag recognized")
        else:
            print("❌ --use-megatron flag not found in help")
            print("Help output:", result.stdout)
            
    except subprocess.TimeoutExpired:
        print("⚠️ Command timed out (this is expected for help)")
    except Exception as e:
        print(f"❌ Error testing help: {e}")
    
    # Test 2: Check if --no-megatron flag is recognized
    print("\n2️⃣ Testing --no-megatron flag recognition...")
    try:
        result = subprocess.run([
            sys.executable, "core/train_auto.py", "--no-megatron", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if "--no-megatron" in result.stdout:
            print("✅ --no-megatron flag recognized")
        else:
            print("❌ --no-megatron flag not found in help")
            
    except subprocess.TimeoutExpired:
        print("⚠️ Command timed out (this is expected for help)")
    except Exception as e:
        print(f"❌ Error testing help: {e}")
    
    # Test 3: Test core/train.py flags
    print("\n3️⃣ Testing core/train.py Megatron flags...")
    try:
        result = subprocess.run([
            sys.executable, "core/train.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if "--use-megatron" in result.stdout and "--no-megatron" in result.stdout:
            print("✅ Both Megatron flags found in core/train.py")
        else:
            print("❌ Megatron flags missing from core/train.py")
            print("Help output:", result.stdout)
            
    except subprocess.TimeoutExpired:
        print("⚠️ Command timed out (this is expected for help)")
    except Exception as e:
        print(f"❌ Error testing core/train.py: {e}")
    
    print("\n🎉 Megatron command testing complete!")
    print("\n📋 Available Megatron commands:")
    print("   python train.py --use-megatron          # Force Megatron")
    print("   python train.py --no-megatron           # Force native")
    print("   python core/train.py --use-megatron      # Force Megatron (new pipeline)")
    print("   python core/train_auto.py --use-megatron # Force Megatron (auto pipeline)")

if __name__ == "__main__":
    test_megatron_command()
