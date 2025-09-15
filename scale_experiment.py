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
import threading
import queue
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
                "name": "Default",
                "description": "Default model configuration",
                "args": [],  # Use default config
                "expected_params": "~79M",
                "expected_memory": "~12GB"
            }
        ]
    
    def run_single_experiment(self, config: Dict[str, Any], backend: str) -> Dict[str, Any]:
        """Run a single experiment with real-time output streaming."""
        print(f"\n{'='*60}")
        print(f"🧪 Testing {config['name']} Model with {backend}")
        print(f"{'='*60}")
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
        print(f"\n📊 Training Progress:")
        print("-" * 60)
        
        start_time = time.time()
        result = {
            "config": config,
            "backend": backend,
            "start_time": start_time,
            "success": False,
            "error": None,
            "metrics": {},
            "raw_output": ""
        }
        
        try:
            # Start process with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            output_lines = []
            training_started = False
            current_step = 0
            total_steps = 0
            
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                
                output_lines.append(line.rstrip())
                result["raw_output"] += line
                
                # Print important lines with formatting
                if self._should_print_line(line):
                    self._print_training_line(line, training_started, current_step, total_steps)
                    
                    # Track training progress
                    if "Starting training" in line or "🚀 Starting training" in line:
                        training_started = True
                        print("🎯 Training started - monitoring progress...")
                    
                    # Extract step information
                    if "Training MoE:" in line or "Training:" in line:
                        step_info = self._extract_step_info(line)
                        if step_info:
                            current_step, total_steps = step_info
                
                # Check for timeout
                if time.time() - start_time > 300:  # 5 minute timeout
                    process.terminate()
                    result["duration"] = 300
                    result["error"] = "Timeout (5 minutes)"
                    print("\n⏰ Experiment timed out after 5 minutes")
                    break
            
            # Wait for process to complete
            return_code = process.wait()
            end_time = time.time()
            result["end_time"] = end_time
            result["duration"] = end_time - start_time
            result["success"] = return_code == 0
            
            # Extract metrics from all output
            self.extract_metrics(result, result["raw_output"])
            
            # Print final results
            print("\n" + "-" * 60)
            if result["success"]:
                print(f"✅ Training completed successfully!")
                print(f"⏱️  Duration: {result['duration']:.1f}s")
                if "training_speed" in result["metrics"]:
                    print(f"⚡ Average speed: {result['metrics']['training_speed']:.2f} it/s")
                if "final_loss" in result["metrics"]:
                    print(f"🎯 Final validation loss: {result['metrics']['final_loss']:.4f}")
                if "final_accuracy" in result["metrics"]:
                    print(f"🎯 Final validation accuracy: {result['metrics']['final_accuracy']:.4f}")
                if "total_params" in result["metrics"]:
                    print(f"📊 Total parameters: {result['metrics']['total_params']:,}")
            else:
                print(f"❌ Training failed with exit code: {return_code}")
                result["error"] = f"Exit code {return_code}"
                
        except Exception as e:
            result["error"] = str(e)
            print(f"\n❌ Error during experiment: {e}")
        
        return result
    
    def _should_print_line(self, line: str) -> bool:
        """Determine if a line should be printed to console."""
        important_keywords = [
            "Starting", "Training", "Validation", "Loss", "Accuracy", 
            "GPU", "Memory", "Parameters", "Speed", "Progress",
            "Error", "Warning", "Complete", "Saved", "Final"
        ]
        return any(keyword in line for keyword in important_keywords)
    
    def _print_training_line(self, line: str, training_started: bool, current_step: int, total_steps: int):
        """Print a training line with appropriate formatting."""
        line = line.rstrip()
        
        # Add progress indicator for training lines
        if "Training MoE:" in line or "Training:" in line:
            if current_step > 0 and total_steps > 0:
                progress = (current_step / total_steps) * 100
                print(f"🔄 [{progress:5.1f}%] {line}")
            else:
                print(f"🔄 {line}")
        elif "Validation" in line:
            print(f"📊 {line}")
        elif "Loss:" in line or "loss:" in line:
            print(f"📉 {line}")
        elif "Accuracy:" in line or "accuracy:" in line:
            print(f"🎯 {line}")
        elif "GPU" in line or "Memory" in line:
            print(f"💾 {line}")
        elif "Parameters:" in line:
            print(f"📊 {line}")
        elif "Error" in line or "error" in line:
            print(f"❌ {line}")
        elif "Complete" in line or "Saved" in line:
            print(f"✅ {line}")
        else:
            print(f"   {line}")
    
    def _extract_step_info(self, line: str) -> tuple:
        """Extract current step and total steps from progress line."""
        try:
            # Look for patterns like "1000/1000" or "500/1000"
            import re
            match = re.search(r'(\d+)/(\d+)', line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                return current, total
        except:
            pass
        return 0, 0
    
    def extract_metrics(self, result: Dict[str, Any], output: str):
        """Extract comprehensive performance metrics from output."""
        lines = output.split('\n')
        
        for line in lines:
            # Training speed (multiple patterns)
            if 'it/s' in line:
                try:
                    import re
                    # Pattern 1: "Training MoE: 100%|█| 1000/1000 [03:28<00:00, 4.80it/s"
                    speed_match = re.search(r'(\d+\.?\d*)it/s', line)
                    if speed_match:
                        result["metrics"]["training_speed"] = float(speed_match.group(1))
                except:
                    pass
            
            # Loss metrics (multiple patterns)
            if 'loss:' in line.lower() or 'Loss:' in line:
                try:
                    import re
                    # Pattern 1: "Final validation loss: 0.1234"
                    if 'Final validation loss:' in line:
                        loss_str = line.split('Final validation loss:')[1].strip()
                        result["metrics"]["final_loss"] = float(loss_str)
                    # Pattern 2: "Loss: 0.1234"
                    elif 'Loss:' in line:
                        loss_match = re.search(r'Loss:\s*(\d+\.?\d*)', line)
                        if loss_match:
                            result["metrics"]["current_loss"] = float(loss_match.group(1))
                except:
                    pass
            
            # Accuracy metrics
            if 'accuracy:' in line.lower() or 'Accuracy:' in line:
                try:
                    import re
                    # Pattern 1: "Final validation accuracy: 0.9876"
                    if 'Final validation accuracy:' in line:
                        acc_str = line.split('Final validation accuracy:')[1].strip()
                        result["metrics"]["final_accuracy"] = float(acc_str)
                    # Pattern 2: "Accuracy: 0.9876"
                    elif 'Accuracy:' in line:
                        acc_match = re.search(r'Accuracy:\s*(\d+\.?\d*)', line)
                        if acc_match:
                            result["metrics"]["current_accuracy"] = float(acc_match.group(1))
                except:
                    pass
            
            # Model parameters
            if 'parameters:' in line.lower() or 'Parameters:' in line:
                try:
                    import re
                    # Pattern 1: "Total parameters: 79,123,456"
                    if 'Total parameters:' in line:
                        param_str = line.split('Total parameters:')[1].split()[0].replace(',', '')
                        result["metrics"]["total_params"] = int(param_str)
                    # Pattern 2: "Parameters: 79,123,456"
                    elif 'Parameters:' in line:
                        param_match = re.search(r'Parameters:\s*([\d,]+)', line)
                        if param_match:
                            param_str = param_match.group(1).replace(',', '')
                            result["metrics"]["total_params"] = int(param_str)
                except:
                    pass
            
            # GPU Memory usage
            if 'GPU' in line and ('memory' in line.lower() or 'Memory' in line):
                try:
                    import re
                    # Pattern: "GPU Memory: 12.5GB / 24GB"
                    memory_match = re.search(r'(\d+\.?\d*)GB', line)
                    if memory_match:
                        result["metrics"]["gpu_memory_used"] = float(memory_match.group(1))
                except:
                    pass
            
            # Training steps
            if 'steps' in line.lower() or 'Steps' in line:
                try:
                    import re
                    # Pattern: "Training steps: 1000"
                    steps_match = re.search(r'(\d+)\s*steps?', line)
                    if steps_match:
                        result["metrics"]["total_steps"] = int(steps_match.group(1))
                except:
                    pass
            
            # Batch size
            if 'batch' in line.lower() and 'size' in line.lower():
                try:
                    import re
                    # Pattern: "Batch size: 32"
                    batch_match = re.search(r'batch.*?(\d+)', line, re.IGNORECASE)
                    if batch_match:
                        result["metrics"]["batch_size"] = int(batch_match.group(1))
                except:
                    pass
            
            # Learning rate
            if 'learning rate' in line.lower() or 'lr:' in line.lower():
                try:
                    import re
                    # Pattern: "Learning rate: 0.001" or "lr: 0.001"
                    lr_match = re.search(r'(?:learning rate|lr):\s*(\d+\.?\d*)', line, re.IGNORECASE)
                    if lr_match:
                        result["metrics"]["learning_rate"] = float(lr_match.group(1))
                except:
                    pass
            
            # Training time
            if 'time' in line.lower() and ('training' in line.lower() or 'duration' in line.lower()):
                try:
                    import re
                    # Pattern: "Training time: 03:28"
                    time_match = re.search(r'(\d+):(\d+)', line)
                    if time_match:
                        minutes = int(time_match.group(1))
                        seconds = int(time_match.group(2))
                        result["metrics"]["training_time_minutes"] = minutes + seconds / 60.0
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
        """Print comprehensive experiment summary to console."""
        print(f"\n{'='*70}")
        print("📊 COMPREHENSIVE EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        
        successful = sum(1 for r in self.results if r["success"])
        print(f"✅ Successful experiments: {successful}/{len(self.results)}")
        
        # Detailed results table
        print(f"\n📋 Detailed Results:")
        print("-" * 70)
        print(f"{'Model':<15} {'Backend':<10} {'Duration':<10} {'Speed':<8} {'Loss':<8} {'Accuracy':<10} {'Params':<12}")
        print("-" * 70)
        
        for result in self.results:
            config = result["config"]
            duration = result.get("duration", 0)
            speed = result["metrics"].get("training_speed", 0)
            loss = result["metrics"].get("final_loss", 0)
            accuracy = result["metrics"].get("final_accuracy", 0)
            params = result["metrics"].get("total_params", 0)
            
            status = "✅" if result["success"] else "❌"
            print(f"{config['name']:<15} {result['backend']:<10} {duration:>7.1f}s {speed:>6.2f} {loss:>6.4f} {accuracy:>8.4f} {params:>10,}")
        
        # Show performance comparison
        native_results = [r for r in self.results if r["backend"] == "Native" and r["success"]]
        megatron_results = [r for r in self.results if r["backend"] == "Megatron" and r["success"]]
        
        if native_results and megatron_results:
            print(f"\n🏆 Performance Comparison:")
            print("-" * 70)
            print(f"{'Model':<15} {'Native Speed':<12} {'Megatron Speed':<14} {'Winner':<10} {'Speedup':<8}")
            print("-" * 70)
            
            for native in native_results:
                config_name = native["config"]["name"]
                native_speed = native["metrics"].get("training_speed", 0)
                
                megatron = next((r for r in megatron_results if r["config"]["name"] == config_name), None)
                if megatron:
                    megatron_speed = megatron["metrics"].get("training_speed", 0)
                    if native_speed > 0 and megatron_speed > 0:
                        if native_speed > megatron_speed:
                            speedup = native_speed / megatron_speed
                            winner = "Native"
                        else:
                            speedup = megatron_speed / native_speed
                            winner = "Megatron"
                        print(f"{config_name:<15} {native_speed:>10.2f} {megatron_speed:>12.2f} {winner:<10} {speedup:>6.2f}x")
        
        # Resource usage analysis
        print(f"\n💾 Resource Usage Analysis:")
        print("-" * 70)
        for result in self.results:
            if result["success"]:
                config = result["config"]
                memory_used = result["metrics"].get("gpu_memory_used", 0)
                batch_size = result["metrics"].get("batch_size", 0)
                learning_rate = result["metrics"].get("learning_rate", 0)
                
                print(f"{config['name']} ({result['backend']}):")
                if memory_used > 0:
                    print(f"  💾 GPU Memory: {memory_used:.1f}GB")
                if batch_size > 0:
                    print(f"  📦 Batch Size: {batch_size}")
                if learning_rate > 0:
                    print(f"  📈 Learning Rate: {learning_rate}")
                print()
        
        # Training efficiency metrics
        print(f"⚡ Training Efficiency:")
        print("-" * 70)
        for result in self.results:
            if result["success"]:
                config = result["config"]
                duration = result.get("duration", 0)
                steps = result["metrics"].get("total_steps", 0)
                speed = result["metrics"].get("training_speed", 0)
                
                if duration > 0 and steps > 0:
                    efficiency = steps / duration  # steps per second
                    print(f"{config['name']} ({result['backend']}): {efficiency:.2f} steps/sec")
        
        print(f"\n{'='*70}")

def main():
    experiment = ScaleExperiment()
    experiment.run_experiment()

if __name__ == "__main__":
    main()
