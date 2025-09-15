#!/usr/bin/env python3
"""
Comprehensive Scale Experiment for Blueberry LLM
Tests different model sizes and configurations on 2x RTX 4090 setup.
"""

import subprocess
import sys
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import argparse

class ScaleExperiment:
    """Comprehensive experiment testing different model scales."""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        self.experiment_id = f"scale_exp_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create results directory
        os.makedirs("experiment_results", exist_ok=True)
        
    def get_model_configs(self) -> List[Dict[str, Any]]:
        """Define model configurations to test."""
        return [
            # Small models (should fit easily)
            {
                "name": "Tiny",
                "d_model": 256,
                "n_layers": 4,
                "n_heads": 4,
                "num_experts": 4,
                "batch_size": 32,
                "max_steps": 200,
                "expected_memory": "~8GB",
                "description": "Tiny model for rapid testing"
            },
            {
                "name": "Small",
                "d_model": 384,
                "n_layers": 6,
                "n_heads": 8,
                "num_experts": 8,
                "batch_size": 24,
                "max_steps": 200,
                "expected_memory": "~12GB",
                "description": "Small model (current default)"
            },
            {
                "name": "Medium",
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "num_experts": 12,
                "batch_size": 16,
                "max_steps": 200,
                "expected_memory": "~18GB",
                "description": "Medium model pushing memory limits"
            },
            {
                "name": "Large",
                "d_model": 768,
                "n_layers": 12,
                "n_heads": 12,
                "num_experts": 16,
                "batch_size": 8,
                "max_steps": 200,
                "expected_memory": "~25GB",
                "description": "Large model near memory limit"
            },
            {
                "name": "XL",
                "d_model": 1024,
                "n_layers": 16,
                "n_heads": 16,
                "num_experts": 20,
                "batch_size": 4,
                "max_steps": 200,
                "expected_memory": "~30GB",
                "description": "Extra large model (may hit memory limits)"
            }
        ]
    
    def estimate_model_size(self, config: Dict[str, Any]) -> int:
        """Estimate model size in parameters."""
        d_model = config["d_model"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]
        num_experts = config["num_experts"]
        
        # Rough estimation: embedding + transformer blocks + output head
        embedding_params = d_model * 50000  # vocab size ~50k
        attention_params = n_layers * (4 * d_model * d_model)  # Q, K, V, O projections
        expert_params = n_layers * num_experts * (2 * d_model * d_model * 4)  # 4 FFN layers per expert
        output_params = d_model * 50000  # output head
        
        total_params = embedding_params + attention_params + expert_params + output_params
        return int(total_params)
    
    def create_config_file(self, config: Dict[str, Any]) -> str:
        """Create a temporary config file for the experiment."""
        config_content = f"""
from configs import AdaptiveMoEModelConfig

def get_experiment_config():
    return AdaptiveMoEModelConfig(
        d_model={config['d_model']},
        n_layers={config['n_layers']},
        n_heads={config['n_heads']},
        num_experts={config['num_experts']},
        batch_size={config['batch_size']},
        max_steps={config['max_steps']},
        max_seq_len=512,  # Reduced for memory efficiency
        num_documents=1000,  # Reduced for faster experiments
        max_tokens=100000,  # Reduced for faster experiments
        use_megatron=False,  # Will be overridden by command line
        use_amp=True,
        vocab_size=50000
    )
"""
        
        config_file = f"temp_config_{config['name'].lower()}.py"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        return config_file
    
    def run_experiment(self, config: Dict[str, Any], backend: str) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        print(f"\n{'='*60}")
        print(f"🧪 Testing {config['name']} Model with {backend}")
        print(f"{'='*60}")
        
        estimated_params = self.estimate_model_size(config)
        print(f"📊 Estimated parameters: {estimated_params:,} ({estimated_params/1e6:.1f}M)")
        print(f"💾 Expected memory: {config['expected_memory']}")
        print(f"📝 Description: {config['description']}")
        
        # Create config file
        config_file = self.create_config_file(config)
        
        # Prepare command
        cmd = [
            sys.executable, "core/train_auto.py",
            "--use-megatron" if backend == "Megatron" else "--no-megatron"
        ]
        
        print(f"🚀 Command: {' '.join(cmd)}")
        
        start_time = time.time()
        result = {
            "config": config,
            "backend": backend,
            "estimated_params": estimated_params,
            "start_time": start_time,
            "success": False,
            "error": None,
            "metrics": {}
        }
        
        try:
            # Run the experiment with timeout
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=600  # 10 minute timeout
            )
            
            end_time = time.time()
            result["end_time"] = end_time
            result["duration"] = end_time - start_time
            result["success"] = process.returncode == 0
            
            # Parse output for metrics
            output = process.stdout + process.stderr
            result["raw_output"] = output
            
            # Extract key metrics
            self.extract_metrics(result, output)
            
            if result["success"]:
                print(f"✅ Success! Duration: {result['duration']:.1f}s")
                if "training_speed" in result["metrics"]:
                    print(f"⚡ Training speed: {result['metrics']['training_speed']:.2f} it/s")
                if "final_loss" in result["metrics"]:
                    print(f"🎯 Final loss: {result['metrics']['final_loss']:.4f}")
            else:
                print(f"❌ Failed! Exit code: {process.returncode}")
                result["error"] = f"Exit code {process.returncode}"
                
        except subprocess.TimeoutExpired:
            result["duration"] = 600
            result["error"] = "Timeout (10 minutes)"
            print("⏰ Experiment timed out after 10 minutes")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ Error: {e}")
        
        # Clean up config file
        try:
            os.remove(config_file)
        except:
            pass
            
        return result
    
    def extract_metrics(self, result: Dict[str, Any], output: str):
        """Extract performance metrics from training output."""
        lines = output.split('\n')
        
        for line in lines:
            # Extract training speed
            if 'it/s' in line and 'Training' in line:
                try:
                    speed_str = line.split('it/s')[0].split()[-1]
                    result["metrics"]["training_speed"] = float(speed_str)
                except:
                    pass
            
            # Extract final loss
            if 'Final validation loss:' in line:
                try:
                    loss_str = line.split('Final validation loss:')[1].strip()
                    result["metrics"]["final_loss"] = float(loss_str)
                except:
                    pass
            
            # Extract final accuracy
            if 'Final validation accuracy:' in line:
                try:
                    acc_str = line.split('Final validation accuracy:')[1].strip()
                    result["metrics"]["final_accuracy"] = float(acc_str)
                except:
                    pass
            
            # Extract memory usage (if available)
            if 'Memory:' in line and 'GB' in line:
                try:
                    mem_str = line.split('Memory:')[1].split('GB')[0].strip()
                    result["metrics"]["memory_usage"] = float(mem_str)
                except:
                    pass
    
    def run_comprehensive_experiment(self):
        """Run the complete scale experiment."""
        print("🫐 Blueberry LLM Comprehensive Scale Experiment")
        print("=" * 60)
        print(f"🕐 Started: {self.start_time}")
        print(f"🖥️ Hardware: 2x RTX 4090 (23.5GB each)")
        print(f"📁 Results will be saved to: experiment_results/{self.experiment_id}/")
        
        configs = self.get_model_configs()
        backends = ["Native", "Megatron"]
        
        total_experiments = len(configs) * len(backends)
        current_experiment = 0
        
        for config in configs:
            for backend in backends:
                current_experiment += 1
                print(f"\n📊 Progress: {current_experiment}/{total_experiments}")
                
                result = self.run_experiment(config, backend)
                self.results.append(result)
                
                # Save intermediate results
                self.save_results()
                
                # Brief pause between experiments
                time.sleep(5)
        
        # Generate final report
        self.generate_report()
    
    def save_results(self):
        """Save current results to file."""
        results_file = f"experiment_results/{self.experiment_id}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "experiment_id": self.experiment_id,
                "start_time": self.start_time.isoformat(),
                "hardware": "2x RTX 4090",
                "results": self.results
            }, f, indent=2)
    
    def generate_report(self):
        """Generate comprehensive experiment report."""
        report_file = f"experiment_results/{self.experiment_id}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Comprehensive Scale Experiment Report\n\n")
            f.write(f"**Experiment ID**: {self.experiment_id}\n")
            f.write(f"**Start Time**: {self.start_time}\n")
            f.write(f"**Hardware**: 2x RTX 4090 (23.5GB each)\n")
            f.write(f"**Total Experiments**: {len(self.results)}\n\n")
            
            # Summary table
            f.write("## 📊 Experiment Summary\n\n")
            f.write("| Model | Backend | Params | Duration | Success | Speed | Loss |\n")
            f.write("|-------|---------|--------|----------|---------|-------|------|\n")
            
            for result in self.results:
                config = result["config"]
                params = result["estimated_params"]
                duration = result.get("duration", 0)
                success = "✅" if result["success"] else "❌"
                speed = result["metrics"].get("training_speed", 0)
                loss = result["metrics"].get("final_loss", 0)
                
                f.write(f"| {config['name']} | {result['backend']} | {params/1e6:.1f}M | {duration:.1f}s | {success} | {speed:.2f} | {loss:.4f} |\n")
            
            # Analysis
            f.write("\n## 🔍 Analysis\n\n")
            
            # Success rate
            successful = sum(1 for r in self.results if r["success"])
            f.write(f"- **Success Rate**: {successful}/{len(self.results)} ({successful/len(self.results)*100:.1f}%)\n")
            
            # Performance comparison
            native_results = [r for r in self.results if r["backend"] == "Native" and r["success"]]
            megatron_results = [r for r in self.results if r["backend"] == "Megatron" and r["success"]]
            
            if native_results and megatron_results:
                f.write("\n### Performance Comparison\n\n")
                f.write("| Model | Native Speed | Megatron Speed | Winner |\n")
                f.write("|-------|--------------|----------------|--------|\n")
                
                for native in native_results:
                    config_name = native["config"]["name"]
                    native_speed = native["metrics"].get("training_speed", 0)
                    
                    megatron = next((r for r in megatron_results if r["config"]["name"] == config_name), None)
                    if megatron:
                        megatron_speed = megatron["metrics"].get("training_speed", 0)
                        winner = "Native" if native_speed > megatron_speed else "Megatron"
                        f.write(f"| {config_name} | {native_speed:.2f} | {megatron_speed:.2f} | {winner} |\n")
            
            # Memory analysis
            f.write("\n### Memory Analysis\n\n")
            f.write("| Model | Estimated Params | Expected Memory | Status |\n")
            f.write("|-------|------------------|-----------------|--------|\n")
            
            for result in self.results:
                config = result["config"]
                params = result["estimated_params"]
                expected_mem = config["expected_memory"]
                status = "✅ Success" if result["success"] else "❌ Failed"
                f.write(f"| {config['name']} | {params/1e6:.1f}M | {expected_mem} | {status} |\n")
        
        print(f"\n📋 Report saved to: {report_file}")
        print(f"📊 Results saved to: experiment_results/{self.experiment_id}_results.json")

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive scale experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer configurations")
    parser.add_argument("--models", nargs="+", help="Specific models to test (Tiny, Small, Medium, Large, XL)")
    
    args = parser.parse_args()
    
    experiment = ScaleExperiment()
    
    if args.quick:
        # Quick test with just Tiny and Small models
        configs = experiment.get_model_configs()[:2]
        print("🚀 Running quick test with Tiny and Small models only")
    elif args.models:
        # Test specific models
        all_configs = experiment.get_model_configs()
        configs = [c for c in all_configs if c["name"] in args.models]
        print(f"🚀 Testing specific models: {args.models}")
    else:
        # Full experiment
        configs = experiment.get_model_configs()
        print("🚀 Running full comprehensive experiment")
    
    # Override configs if specified
    if args.quick or args.models:
        experiment.configs = configs
    
    experiment.run_comprehensive_experiment()

if __name__ == "__main__":
    main()
