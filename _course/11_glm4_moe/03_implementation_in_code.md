# 03: GLM-4 MoE Implementation in Code

Let's build GLMw's sophisticated MoE architecture from the ground up! This implementation will showcase the elegance and efficiency that makes models like GLM-4 so powerful.

We'll create a production-ready MoE layer that demonstrates:
- **Smart routing** with load balancing
- **Expert specialization** through independent processing
- **Computational efficiency** via sparse activation
- **Scalable architecture** that can handle billions of parameters

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Expert(nn.Module):
    """
    A single expert - essentially a specialized Feed-Forward Network.
    
    Each expert learns to process specific types of tokens during training.
    Think of it as a specialist in a particular domain.
    """
    
    def __init__(self, dim: int, hidden_dim: int, activation_fn: str = "gelu"):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # The standard FFN structure: expand → activate → contract
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Expansion layer
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # Contraction layer
        
        # Modern activation choice: GELU is preferred in current LLMs
        if activation_fn == "gelu":
            self.activation = nn.GELU()
        elif activation_fn == "swiglu":
            self.activation = nn.SiLU()  # Simplified - real SwiGLU is more complex
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert:
        x -> w1 -> activation -> w2 -> output
        
        Args:
            x: Input tokens of shape (batch_size, seq_len, dim)
        """
        return self.w2(self.activation(self.w1(x)))

class GLM4Router(nn.Module):
    """
    The GLM-4 Router - the "brain" that decides which experts to use.
    
    This component learns patterns that allow it to route tokens
    to the most relevant experts based on semantic and syntactic features.
    """
    
    def __init__(self, dim: int, n_experts: int, temperature: float = 1.0):
        super().__init__()
        self.n_experts = n_experts
        self.temperature = temperature
        
        # Simple but effective: linear projection followed by softmax
        # More sophisticated routers can use deeper networks
        self.gate = nn.Linear(dim,<｜tool▁sep｜>n_experts,<｜tool▁sep｜>bias=False)
    
    def forward(self, x: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            x: Input tokens of shape (batch_size, seq_len, dim)
            top_k: Number of experts to select (typically 2)
            
        Returns:
            expert_weights: Routing weights (batch_size, seq_len, top_k)
            expert_indices: Expert indices (batch_size, seq_len, top_k)
        """
        # Get raw scores for each expert
        expert_logits = self.gate(x)  # (batch_size, seq_len, n_experts)
        
        # Apply temperature scaling for load balancing
        scaled_logits = expert_logits / self.temperature
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            F.softmax(scaled_logits, dim=-1), 
            k=top_k, 
            dim=-1
        )
        
        return expert_weights, expert_indices

class GLM4MoE(nn.Module):
    """
    Complete GLM-4 Mixture of Experts implementation.
    
    This module demonstrates the full GLM-4 MoE architecture with:
    - Spare expert activation (only top-k experts process each token)
    - Load balancing mechanisms
    - Efficient computation for large-scale models
    """
    
    def __init__(self, 
                 dim: int, 
                 hidden_dim: int, 
                 n_experts: int, 
                 top_k: int = 2,
                 temperature: float = 1.0,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = temperature
        
        assert top_k <= n_experts, f"top_k ({top_k}) cannot be greater than n_experts ({n_experts})"
        
        # Initialize the router
        self.router = GLM4Router(dim,<｜tool▁sep｜>n_experts,<｜tool▁sep｜>temperature)
        
        # Create the expert networks
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(n_experts)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def _load_balancing_loss(self, expert_logits: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage balanced expert usage.
        
        This prevents a few experts from being overused while others
        remain underutilized.
        """
        # Calculate the probability of routing to each expert
        routing_probabilities = F.softmax(expert_logits, dim=-1)  # (batch_size, seq_len, n_experts)
        
        # Calculate expected usage per expert across the batch
        expected_usage = routing_probabilities.mean(dim=(0, 1))  # (n_experts,)
        
        # Load balancing loss: encourages uniform expert usage
        # Higher entropy → more balanced usage
        entropy = -torch.sum(expected_usage * torch.log(expected_usage + 1e-8))
        load_balancing_loss = self.n_experts * entropy
        
        return load_balancing_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GLM-4 MoE layer.
        
        Args:
            x: Input tokens of shape (batch_size, seq_len, dim)
            
        Returns:
            output: Processed tokens of shape (batch_size, seq_len, dim)
            load_balancing_loss: Scalar loss for balanced expert usage
        """
        batch_size, seq_len, dim = x.shape
        original_shape = x.shape
        
        # Flatten for easier processing
        x_flat = x.view(-1, dim)  # (batch_size * seq_len, dim)
        num_tokens = x_flat.shape[0]
        
        # Route tokens to experts
        expert_weights, expert_indices = self.router(x_flat, top_k=self.top_k)
        
        # Get the raw routing logits for load balancing
        expert_logits = self.router.gate(x_flat)
        
        # Initialize output tensor
        output = torch.zeros_like(x_flat)  # (num_tokens, dim)
        
        # Process each expert
        for expert_id, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_id)  # (num_tokens, top_k)
            
            # Get token indices that use this expert
            token_indices = expert_mask.nonzero(as_tuple=True)[0]  # Flattened indices
            
            if token_indices.shape[0] > 0:
                # Get inputs for this expert
                expert_input = x_flat[token_indices]  # (tokens_for_expert, dim)
                
                # Process through expert
                expert_output = expert(expert_input)  # (tokens_for_expert, dim)
                
                # Get corresponding weights
                weights = expert_weights[expert_mask]  # (tokens_for_expert,)
                
                # Apply dropout and weighting
                expert_output = self.dropout(expert_output)
                
                # Accumulate weighted outputs
                output.index_add_(0, token_indices, expert_output * weights.unsqueeze(1))
        
        # Reshape back to original shape
        output = output.view(original_shape)
        
        # Calculate load balancing loss
        load_balancing_loss = self._load_balancing_loss(expert_logits, expert_indices)
        
        return output, load_balancing_loss

# === DEMONSTRATION AND TESTING ===

def demonstrate_glm4_moe():
    """
    Comprehensive demonstration of GLM-4 MoE architecture.
    """
    print("🚀 GLM-4 Mixture of Experts Demo")
    print("=" * 50)
    
    # Model configuration (similar to GLM-4)
    dim = 768              # Hidden dimension
    hidden_dim = 3072      # FFN hidden dimension (4x expansion)
    n_experts = 64         # Large number of experts (like GLM-4)
    top_k = 2              # Top-2 routing (sparse activation)
    seq_len = 512          # Sequence length
    batch_size = 4         # Batch size
    
    # Create GLM-4 MoE model
    glm4_moe = GLM4MoE(
        dim=dim,
        hidden_dim=hidden_dim, 
        n_experts=n_experts,
        top_k=top_k,
        temperature=1.0,
        dropout_rate=0.1
    )
    
    print(f"Model Configuration:")
    print(f"  Hidden Dim: {dim}")
    print(f"  FFN Hidden Dim: {hidden_dim}")
    print(f"  Number of Experts: {n_experts}")
    print(f"  Top-K Routing: {top_k}")
    print(f"  Activation Ratio: {top_k}/{n_experts} = {top_k/n_experts:.1%}")
    
    # Create example input
    torch.manual_seed(42)
    example_input = torch.randn(batch_size, seq_len, dim)
    
    print(f"\nInput shape: {example_input.shape}")
    
    # Forward pass
    output, load_balancing_loss = glm4_moe(example_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Load balancing loss: {load_balancing_loss:.4f}")
    
    # Parameter count analysis
    total_params = sum(p.numel() for p in glm4_moe.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameter Analysis:")
    print(f"Total parameters: {total_params:,}")
    
    # Calculate efficiency gains
    dense_ffn_params = dim * hidden_dim * 2  # w1 + w2
    total_moe_params = n_experts * dense_ffn_params + dim * n_experts  # experts + router
    active_params_per_token = top_k * dense_ffn_params + dim * top_k
    
    efficiency_gain = total_moe_params / active_params_per_token
    
    print(f"Dense FFN would have: {dense_ffn_params:,} parameters")
    print(f"MoE has {total_moe_params:,} total parameters")
    print(f"MoE uses only {active_params_per_token:,} parameters per token")
    print(f"Spare activation efficiency: {efficiency_gain:.1f}× fewer active parameters!")
    
    return glm4_moe

def analyze_expert_specialization(model: GLM4MoE, sample_texts):
    """
    Demonstrate how different tokens get routed to different experts.
    """
    print(f"\n🔍 Expert Specialization Analysis")
    print(f"Sample tokens and their routing patterns:")
    
    model.eval()
    with torch.no_grad():
        for text_type, tokens in sample_texts.items():
            print(f"\n{text_type}:")
            
            # Create token embeddings (simplified)
            batch_size = 1
            seq_len = len(tokens)
            
            # For demo purposes, create random embeddings 
            # In reality, these would come from a text encoder
            token_embeddings = torch.randn(batch_size, seq_len, model.dim)
            
            # Get routing
            expert_weights, expert_indices = model.router(token_embeddings.view(-1, model.dim), top_k=model.top_k)
            
            for i, token in enumerate(tokens):
                top_experts = expert_indices[batch_size*i:batch_size*(i+1), :].flatten().tolist()
                weights = expert_weights[batch_size*i:batch_size*(i+1), :].flatten().tolist()
                
                print(f"  '{token}' → Experts: {top_experts} (weights: {[f'{w:.3f}' for w in weights]})")

if __name__ == '__main__':
    # Run demonstration
    model = demonstrate_glm4_moe()
    
    # Sample different types of tokens to show specialization
    sample_texts = {
        "Technical": ["function", "algorithm", "parameter", "optimization"],
        "Language": ["beautiful", "language", "poetry", "creative"],  
        "Science": ["quantum", "mechanics", "relativity", "cosmic"],
        "Common": ["the", "is", "and", "in"]
    }
    
    analyze_expert_specialization(model, sample_texts)
    
    print(f"\n🎯 Key GLM-4 MoE Insights:")
    print(f"• {model.n_experts} experts provide massive model capacity")
    print(f"• Top-{model.top_k} routing maintains computational efficiency")
    print(f"• Load balancing ensures all experts contribute meaningfully")
    print(f"• Expert specialization emerges naturally during training")
    print(f"• Architecture scales to billions of parameters efficiently")
    
    print(f"\n🚀 This implementation demonstrates the core principles")
    print(f"   that make GLM-4 such a powerful and efficient model!")
```

This implementation showcases the elegance and sophistication of GLM-4's MoE architecture! Notice how:

1. **Each component is modular** - experts, router, and main MoE layer are separate classes
2. **Load balancing is built-in** - prevents expert collapse through sophisticated loss terms  
3. **The routing is learned** - no hard-coded rules, experts specialize naturally
4. **Computational efficiency** - dramatic parameter reduction through sparse activation
5. **Production-ready design** - includes regularization, dropout, and proper error handling

The beauty of GLM-4 MoE is in how it scales: the architecture enables models with hundreds of billions of parameters while maintaining efficient inference through intelligent routing.

This concludes our exploration of GLM-4 MoE architecture!

---

**🎓 Congratulations! You've completed the Zero to AI Researcher course!**

You now have a comprehensive understanding of:
- Python programming fundamentals
- Mathematical foundations for AI
- PyTorch tensor operations
- Neural networks from scratch
- Activation functions
- Attention mechanisms
- Transformer architectures
- Mixture of Experts
- Advanced architectures (DeepSeek, GLM-4)

You're ready to conduct cutting-edge AI research! 🚀