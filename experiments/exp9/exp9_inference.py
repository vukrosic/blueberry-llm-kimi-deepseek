"""
Experiment 9: Inference Script for Attention+MLP 512d Model
Loads trained checkpoints and performs text generation inference
"""

import torch
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.moe_config import MoEModelConfig
from data.loader import load_and_cache_data
from experiments.exp8.exp8_reduced_ablation_models import AttentionMLP_512dModel


class Experiment9Inference:
    """Inference class for Experiment 9 trained models"""
    
    def __init__(self, results_dir: str = "exp9_results", checkpoint_dir: str = "exp9_results/checkpoints"):
        self.results_dir = Path(results_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Load tokenizer and config
        self._load_tokenizer_and_config()
        
        # Load available checkpoints
        self.checkpoints = self._load_checkpoints()
        
        print(f"🔍 Found {len(self.checkpoints)} checkpoints")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"💾 Checkpoint directory: {self.checkpoint_dir}")
    
    def _load_tokenizer_and_config(self):
        """Load tokenizer and create config"""
        # Create a minimal config for loading tokenizer
        temp_config = MoEModelConfig(
            d_model=512,
            n_heads=8,
            n_layers=3,
            d_ff=2048,
            num_experts=8,
            expert_top_k=2,
            vocab_size=32000,
            max_seq_len=256,
            batch_size=16,
            max_steps=100,
            max_tokens=10000,
            num_documents=100,
        )
        
        # Load tokenizer
        _, self.tokenizer, _ = load_and_cache_data(temp_config)
        self.vocab_size = temp_config.vocab_size
        self.max_seq_len = temp_config.max_seq_len
        
        print(f"✅ Loaded tokenizer with vocab size: {self.vocab_size}")
    
    def _load_checkpoints(self) -> Dict[int, Dict[str, Any]]:
        """Load all available checkpoints"""
        checkpoints = {}
        
        if not self.checkpoint_dir.exists():
            print(f"❌ Checkpoint directory not found: {self.checkpoint_dir}")
            return checkpoints
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_step_*.pt"):
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                step = checkpoint['step']
                checkpoints[step] = {
                    'path': str(checkpoint_file),
                    'val_loss': checkpoint['val_loss'],
                    'timestamp': checkpoint['timestamp'],
                    'model_state_dict': checkpoint['model_state_dict']
                }
                print(f"📁 Loaded checkpoint: step {step}, val_loss: {checkpoint['val_loss']:.6f}")
            except Exception as e:
                print(f"❌ Error loading {checkpoint_file}: {e}")
        
        return checkpoints
    
    def load_model(self, step: int = None) -> torch.nn.Module:
        """Load model from checkpoint"""
        if not self.checkpoints:
            raise ValueError("No checkpoints available")
        
        # Use latest checkpoint if step not specified
        if step is None:
            step = max(self.checkpoints.keys())
        
        if step not in self.checkpoints:
            available_steps = sorted(self.checkpoints.keys())
            raise ValueError(f"Checkpoint for step {step} not found. Available: {available_steps}")
        
        print(f"🔄 Loading model from step {step}...")
        
        # Create model
        config = MoEModelConfig(
            d_model=512,
            n_heads=8,
            n_layers=12,  # Deeper model (was 3, now 12)
            d_ff=2048,
            num_experts=8,
            expert_top_k=2,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            batch_size=1,  # Inference batch size
            max_steps=100,
            max_tokens=10000,
            num_documents=100,
        )
        
        model = AttentionMLP_512dModel(config)
        
        # Load checkpoint
        checkpoint = self.checkpoints[step]
        model.load_state_dict(checkpoint['model_state_dict'])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded from step {step}")
        print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
        print(f"   Device: {device}")
        
        return model
    
    def generate_text(self, model: torch.nn.Module, prompt: str, max_length: int = 100, 
                     temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> str:
        """Generate text from prompt"""
        device = next(model.parameters()).device
        
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.max_seq_len - max_length:
            tokens = tokens[-(self.max_seq_len - max_length):]
        
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        generated_ids = input_ids.clone()
        
        print(f"🎯 Generating text from prompt: '{prompt}'")
        print(f"📏 Max length: {max_length}, Temperature: {temperature}")
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model output - handle both single and tuple returns
                model_output = model(generated_ids, return_aux_loss=False)
                if isinstance(model_output, tuple):
                    logits, _ = model_output
                else:
                    logits = model_output
                
                # Debug: print model output type for troubleshooting
                if _ == 0:  # Only print on first iteration
                    print(f"🔍 Debug: Model output type: {type(model_output)}")
                    if isinstance(model_output, tuple):
                        print(f"🔍 Debug: Tuple length: {len(model_output)}")
                    else:
                        print(f"🔍 Debug: Single tensor shape: {logits.shape}")
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit max sequence length
                if generated_ids.size(1) >= self.max_seq_len:
                    break
        
        # Decode generated text
        generated_tokens = generated_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text
    
    def interactive_inference(self, step: int = None):
        """Interactive inference session"""
        print(f"\n🎮 Interactive Inference Session")
        print(f"{'='*50}")
        
        # Load model
        model = self.load_model(step)
        
        print(f"\n💡 Available commands:")
        print(f"   'quit' or 'exit' - End session")
        print(f"   'checkpoints' - Show available checkpoints")
        print(f"   'load <step>' - Load different checkpoint")
        print(f"   'params' - Show generation parameters")
        print(f"   Any other text - Generate continuation")
        
        # Default generation parameters
        params = {
            'max_length': 100,
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.9
        }
        
        while True:
            try:
                user_input = input(f"\n🎯 Prompt: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'checkpoints':
                    print(f"\n📁 Available checkpoints:")
                    for step_num, checkpoint in sorted(self.checkpoints.items()):
                        print(f"   Step {step_num}: val_loss={checkpoint['val_loss']:.6f}")
                
                elif user_input.lower().startswith('load '):
                    try:
                        new_step = int(user_input.split()[1])
                        model = self.load_model(new_step)
                    except (ValueError, IndexError):
                        print("❌ Invalid step number")
                
                elif user_input.lower() == 'params':
                    print(f"\n⚙️ Current generation parameters:")
                    for key, value in params.items():
                        print(f"   {key}: {value}")
                    print(f"\n💡 To change parameters, modify the code or restart with different defaults")
                
                elif user_input:
                    # Generate text
                    generated = self.generate_text(
                        model, 
                        user_input, 
                        max_length=params['max_length'],
                        temperature=params['temperature'],
                        top_k=params['top_k'],
                        top_p=params['top_p']
                    )
                    
                    print(f"\n🤖 Generated:")
                    print(f"{generated}")
                    print(f"{'='*50}")
                
            except KeyboardInterrupt:
                print(f"\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def batch_inference(self, prompts: List[str], step: int = None, 
                       max_length: int = 100, temperature: float = 1.0) -> Dict[str, str]:
        """Run inference on multiple prompts"""
        print(f"\n🔄 Batch Inference on {len(prompts)} prompts")
        
        # Load model
        model = self.load_model(step)
        
        results = {}
        
        for i, prompt in enumerate(prompts):
            print(f"\n📝 Processing prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")
            
            try:
                generated = self.generate_text(
                    model, 
                    prompt, 
                    max_length=max_length,
                    temperature=temperature
                )
                results[prompt] = generated
                print(f"✅ Generated {len(generated)} characters")
                
            except Exception as e:
                print(f"❌ Error processing prompt: {e}")
                results[prompt] = f"ERROR: {e}"
        
        return results


def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Experiment 9 Inference')
    parser.add_argument('--results_dir', default='exp9_results', help='Results directory')
    parser.add_argument('--checkpoint_dir', default='exp9_results/checkpoints', help='Checkpoint directory')
    parser.add_argument('--step', type=int, help='Specific checkpoint step to load')
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive', 
                       help='Inference mode')
    parser.add_argument('--prompts', nargs='+', help='Prompts for batch mode')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Generation temperature')
    
    args = parser.parse_args()
    
    # Create inference instance
    inference = Experiment9Inference(args.results_dir, args.checkpoint_dir)
    
    if args.mode == 'interactive':
        inference.interactive_inference(args.step)
    elif args.mode == 'batch':
        if not args.prompts:
            print("❌ No prompts provided for batch mode")
            return
        
        results = inference.batch_inference(
            args.prompts, 
            args.step, 
            args.max_length, 
            args.temperature
        )
        
        # Save results
        output_file = Path(args.results_dir) / "batch_inference_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Batch inference results saved to: {output_file}")


if __name__ == "__main__":
    main()
