"""
Dataset classes for LLM training.

This module provides PyTorch Dataset classes for different types of
language modeling tasks.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import random


class TextTokenDataset(Dataset):
    """
    Dataset for causal language modeling with token sequences.
    
    This dataset takes a flat list of tokens and creates overlapping
    sequences for training next-token prediction.
    """
    
    def __init__(self, tokens: List[int], seq_len: int = 512):
        """
        Initialize the dataset.
        
        Args:
            tokens: Flat list of token IDs
            seq_len: Sequence length for each sample
        """
        self.tokens = tokens
        self.seq_len = seq_len
        
        # Ensure we have enough tokens
        if len(tokens) <= seq_len:
            raise ValueError(f"Not enough tokens ({len(tokens)}) for sequence length {seq_len}")

    def __len__(self) -> int:
        """Return the number of possible sequences."""
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (input_tokens, target_tokens) where targets are shifted by 1
        """
        # Input sequence: tokens[idx:idx+seq_len]
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        
        # Target sequence: tokens[idx+1:idx+seq_len+1] (shifted by 1)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        
        return x, y


class PackedTextDataset(Dataset):
    """
    Dataset that packs multiple documents into fixed-length sequences.
    
    This is more efficient than TextTokenDataset as it minimizes padding
    and ensures all tokens are used for training.
    """
    
    def __init__(self, tokens: List[int], seq_len: int = 512, num_sequences: Optional[int] = None):
        """
        Initialize the packed dataset.
        
        Args:
            tokens: Flat list of token IDs
            seq_len: Sequence length for each sample
            num_sequences: Number of sequences to create (if None, use all possible)
        """
        self.tokens = tokens
        self.seq_len = seq_len
        
        # Calculate number of sequences
        max_sequences = len(tokens) // (seq_len + 1)  # +1 for target shift
        if num_sequences is None:
            self.num_sequences = max_sequences
        else:
            self.num_sequences = min(num_sequences, max_sequences)
        
        if self.num_sequences == 0:
            raise ValueError(f"Not enough tokens ({len(tokens)}) for any sequences of length {seq_len}")

    def __len__(self) -> int:
        """Return the number of sequences."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample from packed sequences.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (input_tokens, target_tokens)
        """
        start_idx = idx * (self.seq_len + 1)
        
        # Input and target sequences
        x = torch.tensor(self.tokens[start_idx:start_idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[start_idx + 1:start_idx + self.seq_len + 1], dtype=torch.long)
        
        return x, y


class DocumentDataset(Dataset):
    """
    Dataset that treats each document as a separate sequence.
    
    This preserves document boundaries and can be useful for certain
    training scenarios where document structure matters.
    """
    
    def __init__(self, documents: List[List[int]], max_seq_len: int = 512, pad_token_id: int = 0):
        """
        Initialize the document dataset.
        
        Args:
            documents: List of tokenized documents (each is a list of token IDs)
            max_seq_len: Maximum sequence length (documents will be truncated/padded)
            pad_token_id: Token ID to use for padding
        """
        self.documents = documents
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        
        # Filter out empty documents
        self.documents = [doc for doc in documents if len(doc) > 0]
        
        if len(self.documents) == 0:
            raise ValueError("No valid documents provided")

    def __len__(self) -> int:
        """Return the number of documents."""
        return len(self.documents)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a document sample.
        
        Args:
            idx: Index of the document
            
        Returns:
            Tuple of (input_tokens, target_tokens, attention_mask)
        """
        doc = self.documents[idx]
        
        # Truncate if too long
        if len(doc) > self.max_seq_len:
            doc = doc[:self.max_seq_len]
        
        # Create input and target sequences
        if len(doc) == 1:
            # Handle single token documents
            x = torch.tensor([doc[0]] + [self.pad_token_id] * (self.max_seq_len - 1), dtype=torch.long)
            y = torch.tensor([self.pad_token_id] * self.max_seq_len, dtype=torch.long)
            attention_mask = torch.tensor([1] + [0] * (self.max_seq_len - 1), dtype=torch.bool)
        else:
            # Normal case: input is doc[:-1], target is doc[1:]
            x_tokens = doc[:-1]
            y_tokens = doc[1:]
            
            # Pad to max_seq_len
            seq_len = len(x_tokens)
            padding_len = self.max_seq_len - seq_len
            
            x = torch.tensor(x_tokens + [self.pad_token_id] * padding_len, dtype=torch.long)
            y = torch.tensor(y_tokens + [self.pad_token_id] * padding_len, dtype=torch.long)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.tensor([1] * seq_len + [0] * padding_len, dtype=torch.bool)
        
        return x, y, attention_mask


class RandomChunkDataset(Dataset):
    """
    Dataset that creates random chunks from the token stream.
    
    This can help with generalization by ensuring the model sees
    tokens in different contexts across epochs.
    """
    
    def __init__(self, tokens: List[int], seq_len: int = 512, num_samples: int = 10000, seed: int = 42):
        """
        Initialize the random chunk dataset.
        
        Args:
            tokens: Flat list of token IDs
            seq_len: Sequence length for each sample
            num_samples: Number of random samples to generate per epoch
            seed: Random seed for reproducibility
        """
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.rng = random.Random(seed)
        
        if len(tokens) <= seq_len:
            raise ValueError(f"Not enough tokens ({len(tokens)}) for sequence length {seq_len}")
        
        # Pre-generate random start indices for this epoch
        self._generate_indices()

    def _generate_indices(self):
        """Generate random start indices for the current epoch."""
        max_start = len(self.tokens) - self.seq_len - 1
        self.start_indices = [
            self.rng.randint(0, max_start) 
            for _ in range(self.num_samples)
        ]

    def __len__(self) -> int:
        """Return the number of samples per epoch."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random chunk sample.
        
        Args:
            idx: Index of the sample (used to index into pre-generated start positions)
            
        Returns:
            Tuple of (input_tokens, target_tokens)
        """
        start_idx = self.start_indices[idx]
        
        x = torch.tensor(self.tokens[start_idx:start_idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[start_idx + 1:start_idx + self.seq_len + 1], dtype=torch.long)
        
        return x, y

    def on_epoch_end(self):
        """Call this at the end of each epoch to generate new random indices."""
        self._generate_indices()


def create_dataset(
    tokens: List[int],
    seq_len: int = 512,
    dataset_type: str = "text_token",
    **kwargs
) -> Dataset:
    """
    Factory function to create different types of datasets.
    
    Args:
        tokens: Flat list of token IDs
        seq_len: Sequence length for each sample
        dataset_type: Type of dataset to create
        **kwargs: Additional arguments for specific dataset types
        
    Returns:
        Dataset instance
    """
    if dataset_type == "text_token":
        return TextTokenDataset(tokens, seq_len)
    elif dataset_type == "packed":
        return PackedTextDataset(tokens, seq_len, **kwargs)
    elif dataset_type == "random_chunk":
        return RandomChunkDataset(tokens, seq_len, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataset_info(dataset: Dataset) -> dict:
    """
    Get information about a dataset.
    
    Args:
        dataset: Dataset instance
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        "type": type(dataset).__name__,
        "length": len(dataset),
    }
    
    if hasattr(dataset, 'seq_len'):
        info["seq_len"] = dataset.seq_len
    if hasattr(dataset, 'tokens'):
        info["total_tokens"] = len(dataset.tokens)
    
    return info
