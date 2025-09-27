import torch
from torch.utils.data import Dataset
from typing import List
import os
import pickle
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from configs.moe_config import MoEModelConfig


class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512, window_indices: List[int] = None):
        self.tokens = tokens
        self.seq_len = seq_len
        # If window_indices is provided, use those specific windows
        # Otherwise, use all possible windows (original behavior)
        if window_indices is not None:
            self.window_indices = window_indices
        else:
            self.window_indices = list(range(max(0, len(tokens) - seq_len)))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        # Get the actual window start position
        window_start = self.window_indices[idx]
        x = torch.tensor(self.tokens[window_start:window_start + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[window_start + 1:window_start + self.seq_len + 1], dtype=torch.long)
        return x, y


def load_and_cache_data(config: MoEModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens
