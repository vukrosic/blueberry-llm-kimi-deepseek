#!/usr/bin/env python3
"""
Scale Experiment for Blueberry LLM
Tests different model sizes on 2x RTX 4090 setup.
"""

import subprocess
import sys
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any

class ScaleExperiment:
    """Scale experiment testing different model configurations."""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        self.experiment_id = f"scale_exp_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create results directory
        os.makedirs("experiment_results", exist_ok=True)
        
    def get_experiment_configs(self) -> List[Dict[str, Any]]:
        """Define experiment configurations."""
        return [
            {
                "name": "Tiny",
                "description": "Tiny model for rapid testing",
                "args": ["--config", "dev"],  # Use dev config (smallest)
                "expected_params": "~50M",
                "expected_memory": "~8GB"
            },
            {
                "name": "Default",
                "description": "Default model configuration",
                "args": [],  # Use default config
                "expected_params": "~79M",
                "expected_memory": "~12GB"
            },
            {
                "name": "RTX5090",
                "description": "RTX 5090 optimized configuration",
                "args": ["--config", "rtx5090"],
                "expected_params": "~150M",
                "expected_memory": "~20GB"
            }
        ]
    
    def run_single_experiment(self, config: Dict[str, Any], backend: str) -> Dict[str, Any]:
        """Run a single experiment."""
        print(f"\n{'='*50}")
        print(f"🧪 Testing {config['name']} Model with {backend}")
        print(f"{'='*50}")
        print(f"📝 Description: {config['description']}")
        print(f"📊 Expected params: {config['expected_params']}")
        print(f"💾 Expected memory: {config['expected_memory']}")
        
        # Prepare command
        cmd = [sys.executable, "core/train_auto.py"] + config["args"]
        
        if backend == "Megatron":
            cmd.append("--use-megatron")
        else:
            cmd.append("--no-megatron")
        
        print(f"🚀 Command: {' '.join(cmd)}")
        
        start_time = time.time()
        result = {
            "config": config,
            "backend": backend,
            "start_time": start_time,
            "success": False,
            "error": None,
            "metrics": {}
        }
        
        try:
            # Run with timeout
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for quick experiments
            )
            
            end_time = time.time()
            result["end_time"] = end_time
            result["duration"] = end_time - start_time
            result["success"] = process.returncode == 0
            
            # Parse output
            output = process.stdout + process.stderr
            result["raw_output"] = output
            
            # Extract metrics
            self.extract_metrics(result, output)
            
            if result["success"]:
                print(f"✅ Success! Duration: {result['duration']:.1f}s")
                if "training_speed" in result["metrics"]:
                    print(f"⚡ Speed: {result['metrics']['training_speed']:.2f} it/s")
                if "final_loss" in result["metrics"]:
                    print(f"🎯 Loss: {result['metrics']['final_loss']:.4f}")
            else:
                print(f"❌ Failed! Exit code: {process.returncode}")
                result["error"] = f"Exit code {process.returncode}"
                
        except subprocess.TimeoutExpired:
            result["duration"] = 300
            result["error"] = "Timeout (5 minutes)"
            print("⏰ Experiment timed out")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ Error: {e}")
        
        return result
    
    def extract_metrics(self, result: Dict[str, Any], output: str):
        """Extract performance metrics from output."""
        lines = output.split('\n')
        
        for line in lines:
            # Training speed
            if 'it/s' in line and ('Training' in line or 'Training MoE' in line):
                try:
                    # Extract speed from lines like "Training MoE: 100%|█| 1000/1000 [03:28<00:00, 4.80it/s"
                    parts = line.split('it/s')[0].split()
                    for part in reversed(parts):
                        if ',' in part:
                            speed_str = part.split(',')[-1]
                            result["metrics"]["training_speed"] = float(speed_str)
                            break
                except:
                    pass
            
            # Final loss
            if 'Final validation loss:' in line:
                try:
                    loss_str = line.split('Final validation loss:')[1].strip()
                    result["metrics"]["final_loss"] = float(loss_str)
                except:
                    pass
            
            # Final accuracy
            if 'Final validation accuracy:' in line:
                try:
                    acc_str = line.split('Final validation accuracy:')[1].strip()
                    result["metrics"]["final_accuracy"] = float(acc_str)
                except:
                    pass
            
            # Model parameters
            if 'Total parameters:' in line:
                try:
                    param_str = line.split('Total parameters:')[1].split()[0].replace(',', '')
                    result["metrics"]["total_params"] = int(param_str)
                except:
                    pass
    
    def run_experiment(self):
        """Run the complete scale experiment."""
        print("🫐 Blueberry LLM Scale Experiment")
        print("=" * 50)
        print(f"🕐 Started: {self.start_time}")
        print(f"🖥️ Hardware: 2x RTX 4090 (23.5GB each)")
        
        configs = self.get_experiment_configs()
        backends = ["Native", "Megatron"]
        
        total_experiments = len(configs) * len(backends)
        current_experiment = 0
        
        for config in configs:
            for backend in backends:
                current_experiment += 1
                print(f"\n📊 Progress: {current_experiment}/{total_experiments}")
                
                result = self.run_single_experiment(config, backend)
                self.results.append(result)
                
                # Save intermediate results
                self.save_results()
                
                # Brief pause
                time.sleep(3)
        
        # Generate report
        self.generate_report()
    
    def save_results(self):
        """Save results to JSON file."""
        results_file = f"experiment_results/{self.experiment_id}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "experiment_id": self.experiment_id,
                "start_time": self.start_time.isoformat(),
                "hardware": "2x RTX 4090",
                "results": self.results
            }, f, indent=2)
    
    def generate_report(self):
        """Generate experiment report."""
        report_file = f"experiment_results/{self.experiment_id}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Scale Experiment Report\n\n")
            f.write(f"**Experiment ID**: {self.experiment_id}\n")
            f.write(f"**Start Time**: {self.start_time}\n")
            f.write(f"**Hardware**: 2x RTX 4090 (23.5GB each)\n")
            f.write(f"**Total Experiments**: {len(self.results)}\n\n")
            
            # Summary table
            f.write("## 📊 Results Summary\n\n")
            f.write("| Model | Backend | Duration | Success | Speed | Loss | Accuracy |\n")
            f.write("|-------|---------|----------|---------|-------|------|----------|\n")
            
            for result in self.results:
                config = result["config"]
                duration = result.get("duration", 0)
                success = "✅" if result["success"] else "❌"
                speed = result["metrics"].get("training_speed", 0)
                loss = result["metrics"].get("final_loss", 0)
                accuracy = result["metrics"].get("final_accuracy", 0)
                
                f.write(f"| {config['name']} | {result['backend']} | {duration:.1f}s | {success} | {speed:.2f} | {loss:.4f} | {accuracy:.4f} |\n")
            
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
                f.write("| Model | Native Speed | Megatron Speed | Winner | Speedup |\n")
                f.write("|-------|--------------|----------------|--------|----------|\n")
                
                for native in native_results:
                    config_name = native["config"]["name"]
                    native_speed = native["metrics"].get("training_speed", 0)
                    
                    megatron = next((r for r in megatron_results if r["config"]["name"] == config_name), None)
                    if megatron:
                        megatron_speed = megatron["metrics"].get("training_speed", 0)
                        if native_speed > 0 and megatron_speed > 0:
                            speedup = native_speed / megatron_speed
                            winner = "Native" if native_speed > megatron_speed else "Megatron"
                            f.write(f"| {config_name} | {native_speed:.2f} | {megatron_speed:.2f} | {winner} | {speedup:.2f}x |\n")
            
            # Model scaling analysis
            f.write("\n### Model Scaling Analysis\n\n")
            f.write("| Model | Expected Params | Expected Memory | Success Rate |\n")
            f.write("|-------|------------------|-----------------|--------------|\n")
            
            for config in self.get_experiment_configs():
                config_name = config["name"]
                config_results = [r for r in self.results if r["config"]["name"] == config_name]
                success_rate = sum(1 for r in config_results if r["success"]) / len(config_results) * 100
                
                f.write(f"| {config_name} | {config['expected_params']} | {config['expected_memory']} | {success_rate:.0f}% |\n")
        
        print(f"\n📋 Report saved to: {report_file}")
        print(f"📊 Results saved to: experiment_results/{self.experiment_id}_results.json")
        
        # Print summary to console
        self.print_summary()
    
    def print_summary(self):
        """Print experiment summary to console."""
        print(f"\n{'='*50}")
        print("📊 EXPERIMENT SUMMARY")
        print(f"{'='*50}")
        
        successful = sum(1 for r in self.results if r["success"])
        print(f"✅ Successful experiments: {successful}/{len(self.results)}")
        
        # Show performance comparison
        native_results = [r for r in self.results if r["backend"] == "Native" and r["success"]]
        megatron_results = [r for r in self.results if r["backend"] == "Megatron" and r["success"]]
        
        if native_results and megatron_results:
            print(f"\n🏆 Performance Winners:")
            for native in native_results:
                config_name = native["config"]["name"]
                native_speed = native["metrics"].get("training_speed", 0)
                
                megatron = next((r for r in megatron_results if r["config"]["name"] == config_name), None)
                if megatron:
                    megatron_speed = megatron["metrics"].get("training_speed", 0)
                    if native_speed > megatron_speed:
                        speedup = native_speed / megatron_speed
                        print(f"   {config_name}: Native PyTorch ({speedup:.2f}x faster)")
                    else:
                        speedup = megatron_speed / native_speed
                        print(f"   {config_name}: Megatron ({speedup:.2f}x faster)")

def main():
    experiment = ScaleExperiment()
    experiment.run_experiment()

if __name__ == "__main__":
    main()
