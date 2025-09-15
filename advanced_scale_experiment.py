#!/usr/bin/env python3
"""
Advanced Scale Experiment - Tests different model sizes by creating custom configurations
"""

import subprocess
import sys
import time
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Any

class AdvancedScaleExperiment:
    """Advanced scale experiment with custom configurations."""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        self.experiment_id = f"advanced_scale_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create results directory
        os.makedirs("experiment_results", exist_ok=True)
    
    def get_model_configs(self) -> List[Dict[str, Any]]:
        """Define different model configurations to test."""
        return [
            {
                "name": "Tiny",
                "description": "Tiny model for rapid testing",
                "d_model": 256,
                "n_layers": 4,
                "n_heads": 4,
                "num_experts": 4,
                "batch_size": 16,
                "max_steps": 200,
                "expected_params": "~50M",
                "expected_memory": "~8GB"
            },
            {
                "name": "Small",
                "description": "Small model (current default)",
                "d_model": 384,
                "n_layers": 6,
                "n_heads": 8,
                "num_experts": 8,
                "batch_size": 24,
                "max_steps": 200,
                "expected_params": "~79M",
                "expected_memory": "~12GB"
            },
            {
                "name": "Medium",
                "description": "Medium model pushing memory limits",
                "d_model": 512,
                "n_layers": 8,
                "n_heads": 8,
                "num_experts": 12,
                "batch_size": 16,
                "max_steps": 200,
                "expected_params": "~150M",
                "expected_memory": "~18GB"
            },
            {
                "name": "Large",
                "description": "Large model near memory limit",
                "d_model": 768,
                "n_layers": 12,
                "n_heads": 12,
                "num_experts": 16,
                "batch_size": 8,
                "max_steps": 200,
                "expected_params": "~300M",
                "expected_memory": "~25GB"
            }
        ]
    
    def create_custom_config_file(self, config: Dict[str, Any]) -> str:
        """Create a temporary config file with custom settings."""
        config_content = f'''#!/usr/bin/env python3
"""
Custom configuration for {config['name']} model experiment
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from core.auto_config import auto_configure
from legacy.llm import train_moe_model, load_and_cache_data, TextTokenDataset
from models import create_model
from training import train_model
from configs import AdaptiveMoEModelConfig

def main():
    print("🫐 Starting Custom {config['name']} Model Training")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-megatron", action="store_true")
    parser.add_argument("--no-megatron", action="store_true")
    args = parser.parse_args()
    
    # Create custom configuration
    custom_config = AdaptiveMoEModelConfig(
        d_model={config['d_model']},
        n_layers={config['n_layers']},
        n_heads={config['n_heads']},
        d_ff={config['d_model'] * 4},  # Standard FFN ratio
        num_experts={config['num_experts']},
        batch_size={config['batch_size']},
        max_steps={config['max_steps']},
        max_seq_len=512,  # Reduced for memory efficiency
        num_documents=1000,  # Reduced for faster experiments
        max_tokens=100000,  # Reduced for faster experiments
        use_megatron=args.use_megatron,
        use_amp=True,
        vocab_size=50000
    )
    
    print(f"📋 Custom {config['name']} Configuration:")
    print(f"   Architecture: {custom_config.d_model}d-{custom_config.n_layers}L-{custom_config.n_heads}H")
    print(f"   MoE: {custom_config.num_experts}experts")
    print(f"   Training: {custom_config.max_steps}steps-bs{custom_config.batch_size}")
    
    # Load data
    print("\\n📚 Loading data...")
    train_data, val_data = load_and_cache_data(
        num_documents=custom_config.num_documents,
        max_tokens=custom_config.max_tokens,
        max_seq_len=custom_config.max_seq_len
    )
    
    train_dataset = TextTokenDataset(train_data, custom_config.max_seq_len)
    val_dataset = TextTokenDataset(val_data, custom_config.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=custom_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=custom_config.batch_size, shuffle=False)
    
    print(f"✅ Loaded {{len(train_dataset)}} training samples, {{len(val_dataset)}} validation samples")
    
    # Train the model
    print("\\n🚀 Starting training...")
    
    if custom_config.use_megatron:
        print("🚀 Using Megatron-enabled training pipeline...")
        
        # Create model with Megatron support
        model = create_model(custom_config, "moe")
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Train with new pipeline
        model, final_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=custom_config,
            device=device
        )
    else:
        # Use legacy training pipeline
        model, final_metrics = train_moe_model(custom_config, train_loader, val_loader)
    
    print("\\n✅ Training completed!")
    print(f"📊 Final metrics: {{final_metrics}}")

if __name__ == "__main__":
    main()
'''
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        temp_file.write(config_content)
        temp_file.close()
        
        return temp_file.name
    
    def run_single_experiment(self, config: Dict[str, Any], backend: str) -> Dict[str, Any]:
        """Run a single experiment with custom configuration."""
        print(f"\\n{'='*60}")
        print(f"🧪 Testing {config['name']} Model with {backend}")
        print(f"{'='*60}")
        print(f"📝 Description: {config['description']}")
        print(f"📊 Expected params: {config['expected_params']}")
        print(f"💾 Expected memory: {config['expected_memory']}")
        
        # Create custom config file
        config_file = self.create_custom_config_file(config)
        
        # Prepare command
        cmd = [sys.executable, config_file]
        
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
                timeout=600  # 10 minute timeout
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
            result["duration"] = 600
            result["error"] = "Timeout (10 minutes)"
            print("⏰ Experiment timed out")
            
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
        """Extract performance metrics from output."""
        lines = output.split('\\n')
        
        for line in lines:
            # Training speed
            if 'it/s' in line and ('Training' in line or 'Training MoE' in line):
                try:
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
    
    def run_experiment(self):
        """Run the complete advanced scale experiment."""
        print("🫐 Advanced Scale Experiment")
        print("=" * 60)
        print(f"🕐 Started: {self.start_time}")
        print(f"🖥️ Hardware: 2x RTX 4090 (23.5GB each)")
        
        configs = self.get_model_configs()
        backends = ["Native", "Megatron"]
        
        total_experiments = len(configs) * len(backends)
        current_experiment = 0
        
        for config in configs:
            for backend in backends:
                current_experiment += 1
                print(f"\\n📊 Progress: {current_experiment}/{total_experiments}")
                
                result = self.run_single_experiment(config, backend)
                self.results.append(result)
                
                # Save intermediate results
                self.save_results()
                
                # Brief pause
                time.sleep(5)
        
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
            f.write(f"# Advanced Scale Experiment Report\\n\\n")
            f.write(f"**Experiment ID**: {self.experiment_id}\\n")
            f.write(f"**Start Time**: {self.start_time}\\n")
            f.write(f"**Hardware**: 2x RTX 4090 (23.5GB each)\\n")
            f.write(f"**Total Experiments**: {len(self.results)}\\n\\n")
            
            # Summary table
            f.write("## 📊 Results Summary\\n\\n")
            f.write("| Model | Backend | Duration | Success | Speed | Loss |\\n")
            f.write("|-------|---------|----------|---------|-------|------|\\n")
            
            for result in self.results:
                config = result["config"]
                duration = result.get("duration", 0)
                success = "✅" if result["success"] else "❌"
                speed = result["metrics"].get("training_speed", 0)
                loss = result["metrics"].get("final_loss", 0)
                
                f.write(f"| {config['name']} | {result['backend']} | {duration:.1f}s | {success} | {speed:.2f} | {loss:.4f} |\\n")
            
            # Analysis
            f.write("\\n## 🔍 Analysis\\n\\n")
            
            # Success rate
            successful = sum(1 for r in self.results if r["success"])
            f.write(f"- **Success Rate**: {successful}/{len(self.results)} ({successful/len(self.results)*100:.1f}%)\\n")
            
            # Performance comparison
            native_results = [r for r in self.results if r["backend"] == "Native" and r["success"]]
            megatron_results = [r for r in self.results if r["backend"] == "Megatron" and r["success"]]
            
            if native_results and megatron_results:
                f.write("\\n### Performance Comparison\\n\\n")
                f.write("| Model | Native Speed | Megatron Speed | Winner | Speedup |\\n")
                f.write("|-------|--------------|----------------|--------|----------|\\n")
                
                for native in native_results:
                    config_name = native["config"]["name"]
                    native_speed = native["metrics"].get("training_speed", 0)
                    
                    megatron = next((r for r in megatron_results if r["config"]["name"] == config_name), None)
                    if megatron:
                        megatron_speed = megatron["metrics"].get("training_speed", 0)
                        if native_speed > 0 and megatron_speed > 0:
                            speedup = native_speed / megatron_speed
                            winner = "Native" if native_speed > megatron_speed else "Megatron"
                            f.write(f"| {config_name} | {native_speed:.2f} | {megatron_speed:.2f} | {winner} | {speedup:.2f}x |\\n")
        
        print(f"\\n📋 Report saved to: {report_file}")
        print(f"📊 Results saved to: experiment_results/{self.experiment_id}_results.json")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print experiment summary to console."""
        print(f"\\n{'='*60}")
        print("📊 EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(1 for r in self.results if r["success"])
        print(f"✅ Successful experiments: {successful}/{len(self.results)}")
        
        # Show performance comparison
        native_results = [r for r in self.results if r["backend"] == "Native" and r["success"]]
        megatron_results = [r for r in self.results if r["backend"] == "Megatron" and r["success"]]
        
        if native_results and megatron_results:
            print(f"\\n🏆 Performance Winners:")
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
    experiment = AdvancedScaleExperiment()
    experiment.run_experiment()

if __name__ == "__main__":
    main()
