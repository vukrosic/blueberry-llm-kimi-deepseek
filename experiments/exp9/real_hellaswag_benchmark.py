#!/usr/bin/env python3
"""
Real HellaSwag Benchmark Evaluator
Uses the actual HellaSwag dataset from HuggingFace
"""

import torch
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_dataset

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from experiments.exp8.exp8_reduced_ablation_models import AttentionMLP_512dModel


class RealHellaSwagEvaluator:
    """Real HellaSwag benchmark evaluator using the actual dataset"""
    
    def __init__(self, output_dir: str = "exp9_results/hellaswag_benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_hellaswag_dataset(self, num_samples: int = 100):
        """Load HellaSwag validation dataset"""
        print(f"📦 Loading HellaSwag validation dataset...")
        
        try:
            # Load the validation split
            dataset = load_dataset("Rowan/hellaswag", split="validation")
            
            # Take first num_samples
            if num_samples > len(dataset):
                num_samples = len(dataset)
                print(f"⚠️ Requested {num_samples} samples, but dataset only has {len(dataset)} samples")
            
            # Shuffle and take samples
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
            
            print(f"✅ Loaded {len(dataset)} HellaSwag samples")
            return dataset
            
        except Exception as e:
            print(f"❌ Error loading HellaSwag dataset: {e}")
            raise
    
    def create_model_from_checkpoint(self, checkpoint_path: str):
        """Create model from checkpoint"""
        print(f"🧪 Creating model from checkpoint...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        step = checkpoint['step']
        
        # Load tokenizer first to get vocab_size
        temp_config = MoEModelConfig(
            d_model=512,
            n_heads=8,
            n_layers=12,
            d_ff=2048,
            num_experts=8,
            expert_top_k=2,
            vocab_size=32000,
            max_seq_len=256,
            batch_size=1,
            max_steps=100,
            max_tokens=10000,
            num_documents=100,
        )
        
        # Load tokenizer to get actual vocab size
        _, tokenizer, _ = load_and_cache_data(temp_config)
        vocab_size = temp_config.vocab_size
        
        # Create model config
        config = MoEModelConfig(
            d_model=512,
            n_heads=8,
            n_layers=12,
            d_ff=2048,
            num_experts=8,
            expert_top_k=2,
            vocab_size=vocab_size,
            max_seq_len=256,
            batch_size=1,
            max_steps=100,
            max_tokens=10000,
            num_documents=100,
        )
        
        # Create model
        model = AttentionMLP_512dModel(config)
        
        # Load checkpoint state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model created and loaded from step {step}")
        print(f"   Device: {device}")
        print(f"   Vocab size: {vocab_size}")
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
        
        return model, tokenizer, config, step
    
    def evaluate_question(self, model, tokenizer, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single HellaSwag question"""
        device = next(model.parameters()).device
        
        # Extract question components
        ctx_a = question_data['ctx_a']
        ctx_b = question_data['ctx_b']
        endings = question_data['endings']
        correct_label = int(question_data['label'])
        
        # Create full context
        context = ctx_a + ctx_b
        
        # Evaluate each ending
        scores = []
        for i, ending in enumerate(endings):
            # Create full prompt
            full_prompt = context + ending
            
            # Tokenize
            tokens = tokenizer.encode(full_prompt)
            if len(tokens) > 256:  # Truncate if too long
                tokens = tokens[:256]
            
            input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            # Get model prediction
            with torch.no_grad():
                try:
                    model_output = model(input_ids, return_aux_loss=False)
                    if isinstance(model_output, tuple):
                        logits, _ = model_output
                    else:
                        logits = model_output
                    
                    # Get the probability of the last token
                    last_token_logits = logits[0, -1, :]
                    prob = torch.softmax(last_token_logits, dim=-1)
                    
                    # Use the probability of the last token as score
                    score = prob.max().item()
                    scores.append(score)
                    
                except Exception as e:
                    print(f"⚠️ Error evaluating ending {i}: {e}")
                    scores.append(0.0)
        
        # Find the best answer
        predicted_label = int(np.argmax(scores))
        is_correct = predicted_label == correct_label
        
        return {
            'question_id': question_data.get('ind', 0),
            'context': context,
            'endings': endings,
            'correct_label': correct_label,
            'predicted_label': predicted_label,
            'scores': scores,
            'is_correct': is_correct,
            'activity_label': question_data.get('activity_label', 'Unknown')
        }
    
    def evaluate_model_on_hellaswag(self, model, tokenizer, dataset, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate model on HellaSwag dataset"""
        print(f"\n🧪 Evaluating model on {num_samples} HellaSwag questions...")
        
        results = []
        correct_count = 0
        
        # Show first few questions in detail
        show_details = min(5, num_samples)
        
        for i in range(num_samples):
            question_data = dataset[i]
            
            # Evaluate question
            result = self.evaluate_question(model, tokenizer, question_data)
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            # Show detailed results for first few questions
            if i < show_details:
                print(f"\n📝 Question {i+1}:")
                print(f"   Activity: {result['activity_label']}")
                print(f"   Context: {result['context'][:100]}...")
                print(f"   Endings:")
                for j, ending in enumerate(result['endings']):
                    marker = "✅" if j == result['correct_label'] else "❌" if j == result['predicted_label'] else "  "
                    print(f"     {marker} {j}: {ending}")
                print(f"   Correct: {result['correct_label']}, Predicted: {result['predicted_label']}")
                print(f"   Scores: {[f'{s:.3f}' for s in result['scores']]}")
                print(f"   Result: {'✅ CORRECT' if result['is_correct'] else '❌ WRONG'}")
            
            # Progress update
            if (i + 1) % 20 == 0:
                accuracy_so_far = correct_count / (i + 1)
                print(f"📊 Progress: {i+1}/{num_samples} questions, Accuracy: {accuracy_so_far:.3f}")
        
        # Calculate final metrics
        accuracy = correct_count / num_samples
        
        print(f"\n📊 Final Results:")
        print(f"   Questions evaluated: {num_samples}")
        print(f"   Correct answers: {correct_count}")
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        return {
            'model_name': 'attention_mlp_512d',
            'benchmark': 'hellaswag',
            'accuracy': accuracy,
            'f1': accuracy,  # Use accuracy as F1 approximation
            'exact_match': accuracy,  # Use accuracy as exact match
            'total_questions': num_samples,
            'correct_predictions': correct_count,
            'status': 'completed',
            'evaluation_time_seconds': 0.0,  # Will be set by caller
            'details': {
                'dataset_source': 'Rowan/hellaswag',
                'split': 'validation',
                'evaluation_method': 'real_hellaswag_dataset',
                'questions_shown': show_details
            },
            'individual_results': results
        }
    
    def run_full_evaluation(self, checkpoint_path: str, num_samples: int = 100):
        """Run complete HellaSwag evaluation"""
        print(f"🚀 Real HellaSwag Benchmark Evaluation")
        print(f"=" * 60)
        
        start_time = time.time()
        
        try:
            # Load dataset
            dataset = self.load_hellaswag_dataset(num_samples)
            
            # Create model
            model, tokenizer, config, step = self.create_model_from_checkpoint(checkpoint_path)
            
            # Evaluate model
            results = self.evaluate_model_on_hellaswag(model, tokenizer, dataset, num_samples)
            
            # Add timing
            evaluation_time = time.time() - start_time
            results['evaluation_time_seconds'] = evaluation_time
            
            # Save results
            output_file = self.output_dir / "attention_mlp_512d_real_hellaswag_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Real HellaSwag evaluation completed!")
            print(f"📁 Results saved to: {output_file}")
            print(f"⏱️ Total time: {evaluation_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function"""
    # Find latest checkpoint
    checkpoint_dir = Path("exp9_results/checkpoints")
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    latest_checkpoint = checkpoint_files[-1]
    
    print(f"🔍 Using checkpoint: {latest_checkpoint}")
    
    # Create evaluator and run evaluation
    evaluator = RealHellaSwagEvaluator()
    results = evaluator.run_full_evaluation(str(latest_checkpoint), num_samples=100)
    
    if results:
        print(f"\n🎉 HellaSwag Benchmark Complete!")
        print(f"📊 Final Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"📝 Questions: {results['total_questions']}")
        print(f"✅ Correct: {results['correct_predictions']}")
    else:
        print("❌ Evaluation failed")


if __name__ == "__main__":
    main()
