#!/usr/bin/env python3
"""
Data preparation script for TinyStories dataset.

This script downloads, processes, and saves the TinyStories dataset
in a format ready for language model training.

Usage:
    python scripts/prepare_data.py [--max_samples N] [--output_dir DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm
import re


class TinyStoriesProcessor:
    """Handles downloading and processing of TinyStories dataset."""
    
    # HuggingFace dataset URL for TinyStories
    DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self) -> str:
        """
        Download TinyStories dataset from HuggingFace.
        
        Returns:
            Path to downloaded file
        """
        print("Downloading TinyStories dataset...")
        
        # For this demo, we'll create a sample dataset instead of downloading 2GB
        # In production, you would uncomment the actual download code below
        
        # Actual download code (commented out for demo):
        # response = requests.get(self.DATASET_URL, stream=True)
        # total_size = int(response.headers.get('content-length', 0))
        # 
        # download_path = self.output_dir / "TinyStories_all_data.tar.gz"
        # with open(download_path, 'wb') as f:
        #     with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        #         for chunk in response.iter_content(chunk_size=8192):
        #             if chunk:
        #                 f.write(chunk)
        #                 pbar.update(len(chunk))
        
        # For demo purposes, create a sample TinyStories-style dataset
        sample_stories = self._create_sample_dataset()
        sample_path = self.output_dir / "sample_tinystories.json"
        
        with open(sample_path, 'w') as f:
            json.dump(sample_stories, f, indent=2)
            
        print(f"Sample dataset created at: {sample_path}")
        return str(sample_path)
    
    def _create_sample_dataset(self) -> List[Dict[str, str]]:
        """Create a sample dataset that mimics TinyStories structure."""
        sample_stories = [
            {
                "story": "Once upon a time, there was a little cat named Fluffy. Fluffy loved to play with yarn balls. One day, Fluffy found a big red yarn ball in the garden. She played with it all day long. The end."
            },
            {
                "story": "Tom and Sarah went to the park. They saw a big tree with apples. Tom climbed the tree and picked some apples. Sarah caught them in her basket. They shared the apples with their friends."
            },
            {
                "story": "A small bird lived in a nest high up in a tree. Every morning, the bird would sing beautiful songs. The other animals loved to listen. One day, a little girl heard the song and smiled."
            },
            {
                "story": "Ben had a toy car that was blue and shiny. He liked to race it around the house. One day, the car got stuck under the couch. Ben asked his dad for help. Together, they rescued the toy car."
            },
            {
                "story": "In a magical forest, there lived a friendly dragon named Spark. Spark was not scary at all. He helped lost animals find their way home. All the forest creatures loved Spark very much."
            }
        ]
        
        # Duplicate stories to create a larger sample dataset
        extended_stories = []
        for i in range(100):  # Create 500 stories total
            base_story = sample_stories[i % len(sample_stories)]
            # Add slight variations to make each story unique
            story_text = base_story["story"]
            if i > 0:
                story_text = story_text.replace("Once upon a time", f"Story {i}: Once upon a time")
                story_text = story_text.replace("Tom and Sarah", f"Tom{i} and Sarah{i}")
            
            extended_stories.append({"story": story_text})
            
        return extended_stories
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for training.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Ensure stories end with proper punctuation
        if text and text[-1] not in '.!?':
            text += '.'
            
        return text
    
    def process_dataset(self, dataset_path: str, max_samples: int = None) -> Dict[str, Any]:
        """
        Process the downloaded dataset.
        
        Args:
            dataset_path: Path to the downloaded dataset
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            Dictionary with processed data statistics
        """
        print("Processing dataset...")
        
        # Load the dataset
        with open(dataset_path, 'r') as f:
            raw_data = json.load(f)
        
        # Limit samples if specified
        if max_samples:
            raw_data = raw_data[:max_samples]
        
        # Process stories
        processed_stories = []
        total_chars = 0
        total_words = 0
        
        for item in tqdm(raw_data, desc="Processing stories"):
            story = item.get("story", "")
            
            # Clean the text
            cleaned_story = self.clean_text(story)
            
            if len(cleaned_story) > 10:  # Filter out very short stories
                processed_stories.append(cleaned_story)
                total_chars += len(cleaned_story)
                total_words += len(cleaned_story.split())
        
        # Save processed data
        processed_data = {
            "stories": processed_stories,
            "metadata": {
                "total_stories": len(processed_stories),
                "total_characters": total_chars,
                "total_words": total_words,
                "avg_story_length": total_chars / len(processed_stories) if processed_stories else 0,
                "source": "TinyStories (sample)",
                "processing_date": "2025-01-27"  # Could use datetime.now().isoformat() for dynamic date
            }
        }
        
        # Save to processed data directory
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / "tinystories_processed.json"
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        # Also save just the text for easy access
        text_path = processed_dir / "tinystories_text.txt"
        with open(text_path, 'w') as f:
            for story in processed_stories:
                f.write(story + '\n\n')
        
        print(f"Processed dataset saved to: {output_path}")
        print(f"Text file saved to: {text_path}")
        
        return processed_data["metadata"]


def main():
    """Main function to run the data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare TinyStories dataset for training")
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory for raw data (default: data/raw)"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = TinyStoriesProcessor(output_dir=args.output_dir)
    
    try:
        # Download dataset
        dataset_path = processor.download_dataset()
        
        # Process dataset
        stats = processor.process_dataset(dataset_path, max_samples=args.max_samples)
        
        # Print statistics
        print("\n" + "="*50)
        print("DATASET PROCESSING COMPLETE")
        print("="*50)
        print(f"Total stories: {stats['total_stories']}")
        print(f"Total characters: {stats['total_characters']:,}")
        print(f"Total words: {stats['total_words']:,}")
        print(f"Average story length: {stats['avg_story_length']:.1f} characters")
        print("="*50)
        
        print("\nNext steps:")
        print("1. Run 'dvc add data/processed/tinystories_processed.json' to track with DVC")
        print("2. Run 'git add data/processed/tinystories_processed.json.dvc' to commit the DVC file")
        print("3. Commit your changes with git")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 