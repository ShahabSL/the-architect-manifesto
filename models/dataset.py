"""
TinyStories Dataset for PyTorch training.

This module provides a professional PyTorch Dataset implementation that:
- Loads processed TinyStories data efficiently
- Integrates with our custom BPE tokenizer
- Supports both pre-tokenized and lazy tokenization modes
- Handles sequence length management for transformer training
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import sys
import os

# Add the project root to Python path to import our BPE tokenizer
sys.path.append(str(Path(__file__).parent.parent))


class TinyStoriesDataset(Dataset):
    """
    PyTorch Dataset for TinyStories text data.
    
    This dataset handles:
    - Loading processed stories from JSON
    - Tokenization (lazy or pre-computed)
    - Sequence length management
    - Tensor conversion for PyTorch training
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_length: int = 512,
        mode: str = "train",
        pre_tokenize: bool = True
    ):
        """
        Initialize the TinyStories dataset.
        
        Args:
            data_path (str): Path to the processed TinyStories JSON file
            tokenizer: BPE tokenizer instance (optional if pre_tokenize=False)
            max_length (int): Maximum sequence length for training
            mode (str): Dataset mode ('train', 'val', 'test')
            pre_tokenize (bool): Whether to tokenize all data at initialization
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.pre_tokenize = pre_tokenize
        
        # Load the processed data
        self.data = self._load_data()
        
        # Extract stories from the nested structure
        self.stories = self.data.get("stories", [])
        self.metadata = self.data.get("metadata", {})
        
        print(f"Loaded {len(self.stories)} stories for {mode} mode")
        print(f"Dataset metadata: {self.metadata}")
        
        # Pre-tokenize if requested
        if self.pre_tokenize and self.tokenizer is not None:
            print("Pre-tokenizing all stories...")
            self.tokenized_stories = self._tokenize_all_stories()
            print(f"Pre-tokenization complete. Average tokens per story: {self._avg_tokens():.1f}")
        else:
            self.tokenized_stories = None
    
    def _load_data(self) -> Dict[str, Any]:
        """Load the processed TinyStories data from JSON."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _tokenize_all_stories(self) -> List[List[int]]:
        """Pre-tokenize all stories for faster training."""
        tokenized = []
        for story in self.stories:
            tokens = self.tokenizer.encode(story)
            # Truncate if too long
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            tokenized.append(tokens)
        return tokenized
    
    def _avg_tokens(self) -> float:
        """Calculate average tokens per story (for pre-tokenized mode)."""
        if self.tokenized_stories is None:
            return 0.0
        total_tokens = sum(len(tokens) for tokens in self.tokenized_stories)
        return total_tokens / len(self.tokenized_stories)
    
    def __len__(self) -> int:
        """Return the total number of stories in the dataset."""
        return len(self.stories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx (int): Index of the story to retrieve
            
        Returns:
            Dict containing:
                - 'input_ids': Token IDs as tensor
                - 'attention_mask': Attention mask for padding
                - 'labels': Target tokens for language modeling (shifted input_ids)
        """
        if self.pre_tokenize and self.tokenized_stories is not None:
            # Use pre-tokenized data (fast)
            tokens = self.tokenized_stories[idx].copy()
        else:
            # Tokenize on-the-fly (flexible but slower)
            story = self.stories[idx]
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for lazy tokenization mode")
            tokens = self.tokenizer.encode(story)
            
            # Truncate if too long
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
        
        # Convert to tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(len(tokens), dtype=torch.long)
        
        # For language modeling, labels are the same as input_ids but shifted
        # The model predicts the next token for each position
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size from the tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available")
        return len(self.tokenizer.vocab)
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens back to text (useful for debugging)."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available for decoding")
        
        # Convert tensor to list if needed
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
            
        # Simple decoding (this would need to be implemented in the tokenizer)
        # For now, just return a placeholder
        return f"[Decoded text from {len(tokens)} tokens]"
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader to handle batching with padding.
        
        Args:
            batch: List of examples from __getitem__
            
        Returns:
            Batched and padded tensors
        """
        # Find the maximum length in this batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        # Initialize batch tensors
        batch_size = len(batch)
        batched_input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        batched_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        batched_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 is ignored in loss
        
        # Fill the batch tensors
        for i, item in enumerate(batch):
            seq_len = len(item['input_ids'])
            batched_input_ids[i, :seq_len] = item['input_ids']
            batched_attention_mask[i, :seq_len] = item['attention_mask']
            batched_labels[i, :seq_len] = item['labels']
        
        return {
            'input_ids': batched_input_ids,
            'attention_mask': batched_attention_mask,
            'labels': batched_labels
        }


def create_dataloaders(
    data_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 0,
    shuffle: bool = True
) -> tuple:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        data_path (str): Path to processed data
        tokenizer: BPE tokenizer instance
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create the full dataset first
    full_dataset = TinyStoriesDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="full",
        pre_tokenize=True
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=TinyStoriesDataset.collate_fn,
        pin_memory=torch.cuda.is_available()  # Speed up GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=TinyStoriesDataset.collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=TinyStoriesDataset.collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


# Example usage and testing
if __name__ == "__main__":
    # This is for testing the dataset implementation
    print("Testing TinyStoriesDataset...")
    
    # Test basic dataset loading (without tokenizer for now)
    try:
        dataset = TinyStoriesDataset(
            data_path="data/processed/tinystories_processed.json",
            tokenizer=None,
            pre_tokenize=False
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"First story preview: {dataset.stories[0][:100]}...")
        
        # Test without tokenization
        print("✅ Basic dataset loading works!")
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}") 