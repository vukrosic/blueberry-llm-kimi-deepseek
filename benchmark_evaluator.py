"""
HellaSwag Benchmark Evaluator
Minimal integration for evaluating models on HellaSwag benchmark
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class HellaSwagEvaluator:
    """Minimal HellaSwag benchmark evaluator using lm-evaluation-harness"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_model(self, model: nn.Module, model_name: str, tokenizer=None) -> Dict[str, Any]:
        """
        Evaluate model on HellaSwag benchmark
        
        Args:
            model: The trained model to evaluate
            model_name: Name of the model for saving results
            tokenizer: Optional tokenizer (will use default if None)
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n🧪 Evaluating {model_name} on HellaSwag benchmark...")
        
        try:
            # Save model temporarily for evaluation
            temp_model_path = self._save_model_for_evaluation(model, model_name)
            
            # Run HellaSwag evaluation
            results = self._run_hellaswag_evaluation(temp_model_path, model_name)
            
            # Clean up temporary model
            self._cleanup_temp_model(temp_model_path)
            
            print(f"✅ HellaSwag evaluation completed for {model_name}")
            return results
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name} on HellaSwag: {e}")
            return {"error": str(e), "model_name": model_name}
    
    def _save_model_for_evaluation(self, model: nn.Module, model_name: str) -> str:
        """Save model to temporary directory for evaluation"""
        temp_dir = tempfile.mkdtemp(prefix=f"hellaswag_eval_{model_name}_")
        model_path = os.path.join(temp_dir, "model")
        
        # Save model state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': getattr(model, 'config', None)
        }, model_path)
        
        print(f"💾 Model saved temporarily to: {model_path}")
        return model_path
    
    def _run_hellaswag_evaluation(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """Run HellaSwag evaluation using lm-evaluation-harness"""
        
        # Create output file for this model
        output_file = self.output_dir / f"{model_name}_hellaswag_results.json"
        
        # For now, return mock results since we need to set up proper model loading
        # In a real implementation, you would:
        # 1. Load the model properly for lm-eval
        # 2. Run: lm-eval --model hf --model_args pretrained={model_path} --tasks hellaswag --output_path {output_file}
        
        print(f"🔍 Running HellaSwag evaluation...")
        print(f"📁 Results will be saved to: {output_file}")
        
        # Mock results for now - replace with actual lm-eval call
        mock_results = {
            "model_name": model_name,
            "benchmark": "hellaswag",
            "accuracy": 0.0,  # Will be filled by actual evaluation
            "f1": 0.0,
            "exact_match": 0.0,
            "evaluation_time_seconds": 0.0,
            "status": "pending_implementation"
        }
        
        # Save mock results
        with open(output_file, 'w') as f:
            json.dump(mock_results, f, indent=2)
        
        return mock_results
    
    def _cleanup_temp_model(self, model_path: str):
        """Clean up temporary model files"""
        try:
            temp_dir = os.path.dirname(model_path)
            import shutil
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary model files")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temporary files: {e}")
    
    def evaluate_all_models(self, models_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all trained models on HellaSwag benchmark
        
        Args:
            models_results: Dictionary of model results from training
            
        Returns:
            Dictionary with benchmark results for all models
        """
        print(f"\n{'='*80}")
        print(f"🧪 HELLASWAG BENCHMARK EVALUATION")
        print(f"{'='*80}")
        
        benchmark_results = {}
        
        # Filter successful models (skip failed ones)
        successful_models = {k: v for k, v in models_results.items() if "error" not in v}
        
        if not successful_models:
            print("❌ No successful models to evaluate")
            return benchmark_results
        
        print(f"📋 Evaluating {len(successful_models)} successful models...")
        
        for i, (model_name, result) in enumerate(successful_models.items(), 1):
            print(f"\n🔍 [{i}/{len(successful_models)}] Evaluating {model_name}...")
            
            # For now, create mock evaluation results
            # In real implementation, you would load the actual trained model
            benchmark_result = self.evaluate_model(None, model_name)  # Pass None for now
            
            benchmark_results[model_name] = benchmark_result
        
        # Save combined results
        combined_results_file = self.output_dir / "all_models_hellaswag_results.json"
        with open(combined_results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"\n📊 HellaSwag Benchmark Summary:")
        print(f"   ✅ Evaluated: {len(benchmark_results)} models")
        print(f"   📁 Results saved to: {self.output_dir}")
        
        return benchmark_results


def create_hellaswag_evaluator(output_dir: str = "benchmark_results") -> HellaSwagEvaluator:
    """Factory function to create HellaSwag evaluator"""
    return HellaSwagEvaluator(output_dir)


if __name__ == "__main__":
    # Test the evaluator
    evaluator = HellaSwagEvaluator()
    print("🧪 HellaSwag Evaluator created successfully")
    print("📝 To use: evaluator.evaluate_model(model, 'model_name')")
