"""
Experiment 7 Inference: Chat with the Best Architecture Model
Interactive inference script for testing conversation with the trained model
"""

import os
import sys
import torch
import json
import pickle
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from configs.moe_config import MoEModelConfig
from experiments.exp7.exp7_attention_mlp_model import create_exp7_model
# from utils.training_utils import load_model  # Not needed


class Exp7Inference:
    """Inference class for Experiment 7 model"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = None
        self.tokenizer = None
        
        print(f"🔍 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Load model and config
        self.load_model(model_path, config_path)
        
        # Load tokenizer
        self.load_tokenizer()
    
    def load_model(self, model_path: str, config_path: Optional[str] = None):
        """Load the trained model"""
        print(f"📦 Loading model from {model_path}...")
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                results = json.load(f)
            config_dict = results.get('config', {})
            # Remove any keys that aren't in MoEModelConfig
            valid_keys = {field.name for field in MoEModelConfig.__dataclass_fields__.values()}
            config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            self.config = MoEModelConfig(**config_dict)
        else:
            # Default config matching training
            self.config = MoEModelConfig(
                max_steps=1000,
                batch_size=16,
                max_tokens=100000,
                eval_every=100,
                num_documents=1000,
                max_seq_len=256,
                d_model=256,
                n_heads=4,
                n_layers=3,
                d_ff=1024,
                num_experts=8,
                expert_top_k=2,
                vocab_size=32000,
                learning_rate=1e-3,
                weight_decay=0.01
            )
        
        # Create model
        self.model = create_exp7_model(self.config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully")
    
    def load_tokenizer(self):
        """Load the tokenizer used during training"""
        print("📦 Loading tokenizer...")
        
        # Try to load from cache
        cache_file = f"data_cache/tokenized_data_{self.config.num_documents}_{self.config.max_tokens}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Use the actual tokenizer from training
            self.tokenizer = data.get('tokenizer')
            if self.tokenizer is not None:
                self.vocab_size = self.tokenizer.vocab_size
                print(f"✅ Loaded tokenizer with {self.vocab_size} tokens")
            else:
                # Fallback: create simple tokenizer
                print("⚠️ Tokenizer not found in cache, creating simple tokenizer...")
                self.tokenizer = None
                self.vocab = {f"<token_{i}>": i for i in range(self.config.vocab_size)}
                self.id_to_token = {i: f"<token_{i}>" for i in range(self.config.vocab_size)}
                self.vocab_size = self.config.vocab_size
        else:
            # Fallback: create simple tokenizer
            print("⚠️ Tokenizer cache not found, creating simple tokenizer...")
            self.tokenizer = None
            self.vocab = {f"<token_{i}>": i for i in range(self.config.vocab_size)}
            self.id_to_token = {i: f"<token_{i}>" for i in range(self.config.vocab_size)}
            self.vocab_size = self.config.vocab_size
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if self.tokenizer is not None:
            # Use the actual tokenizer
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # Simple word-based tokenization fallback
            words = text.lower().split()
            tokens = []
            for word in words:
                if word in self.vocab:
                    tokens.append(self.vocab[word])
                else:
                    # Use unknown token or random token
                    tokens.append(1)  # Assume 1 is unknown token
            return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        if self.tokenizer is not None:
            # Use the actual tokenizer
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            # Simple word-based decoding fallback
            words = []
            for token_id in token_ids:
                if token_id in self.id_to_token:
                    words.append(self.id_to_token[token_id])
                else:
                    words.append("<unk>")
            return " ".join(words)
    
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.8) -> str:
        """Generate text from a prompt"""
        # Encode prompt
        input_ids = self.encode(prompt)
        
        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Generate
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits
                output = self.model(input_tensor, return_aux_loss=False)
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to input
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we hit max sequence length
                if input_tensor.size(1) >= self.config.max_seq_len:
                    break
        
        # Decode generated tokens
        generated_ids = input_tensor[0].tolist()
        generated_text = self.decode(generated_ids)
        
        return generated_text
    
    def chat(self):
        """Interactive chat interface"""
        print("=" * 60)
        print("🤖 Exp7 Model Chat Interface")
        print("=" * 60)
        print("Type 'quit' to exit, 'clear' to clear conversation")
        print("=" * 60)
        
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    conversation_history = []
                    print("🧹 Conversation cleared!")
                    continue
                
                if not user_input:
                    continue
                
                # Add to conversation history
                conversation_history.append(f"Human: {user_input}")
                
                # Create prompt from conversation history
                prompt = " ".join(conversation_history[-5:])  # Last 5 exchanges
                
                # Generate response
                print("🤖 Bot: ", end="", flush=True)
                response = self.generate(prompt, max_length=30, temperature=0.7)
                
                # Extract just the bot's response
                if "Bot:" in response:
                    bot_response = response.split("Bot:")[-1].strip()
                else:
                    bot_response = response[len(prompt):].strip()
                
                print(bot_response)
                
                # Add to conversation history
                conversation_history.append(f"Bot: {bot_response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue


def main():
    """Main function"""
    # Model paths
    model_path = "experiments/exp7/exp7_results/exp7_model.pt"
    config_path = "experiments/exp7/exp7_results/exp7_training_results.json"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first by running: python experiments/exp7/exp7_trainer.py")
        return
    
    # Create inference instance
    try:
        inference = Exp7Inference(model_path, config_path)
        
        # Start chat
        inference.chat()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please make sure the model was trained successfully.")


if __name__ == "__main__":
    main()
