#!/usr/bin/env python3
import os, sys, argparse, random
import torch
import numpy as np
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def download_shakespeare():
    """Download Shakespeare dataset like in nanoGPT"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def preprocess_shakespeare(text, max_chars=6000, vocab_size=25):
    """
    Process Shakespeare text:
    - Truncate to max_chars
    - Keep only a-y + space (25 chars total)
    - Replace others with space
    """
    # Truncate to max_chars
    text = text[:max_chars]
    
    # Define allowed characters: a-y + space
    allowed_chars = set('abcdefghijklmnopqrstuvwxy ')
    
    # Replace disallowed characters with space
    processed_text = ''.join([ch.lower() if ch.lower() in allowed_chars else ' ' for ch in text])
    
    # Create character mappings
    chars = sorted(list(set(processed_text)))
    # Ensure we have exactly vocab_size characters
    if len(chars) > vocab_size:
        # Keep the most frequent chars
        char_freq = {ch: processed_text.count(ch) for ch in chars}
        chars = sorted(chars, key=lambda x: char_freq[x], reverse=True)[:vocab_size]
        # Replace less frequent chars with space
        char_set = set(chars)
        processed_text = ''.join([ch if ch in char_set else ' ' for ch in processed_text])
    
    # Final character set
    chars = sorted(list(set(processed_text)))
    vocab_size = len(chars)
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode text
    data = torch.tensor([stoi[ch] for ch in processed_text], dtype=torch.long)
    
    return {
        'data': data,
        'chars': chars,
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
        'text': processed_text
    }


def create_sequences(data, block_size=16):
    """Create input-target sequences for language modeling"""
    X, Y = [], []
    for i in range(len(data) - block_size):
        X.append(data[i:i+block_size])
        Y.append(data[i+1:i+block_size+1])
    
    return torch.stack(X), torch.stack(Y)


def find_divisible_size(total_sequences, target_size, batch_sizes):
    """
    Find a dataset size that is divisible by all specified batch sizes
    and is close to the target size (if specified)
    """
    batch_sizes = [int(b) for b in batch_sizes.split(',')]
    
    # Find LCM of all batch sizes
    import math
    lcm = batch_sizes[0]
    for b in batch_sizes[1:]:
        lcm = abs(lcm * b) // math.gcd(lcm, b)
    
    if target_size is None:
        # If no target specified, use the largest size that's divisible by LCM
        target_size = (total_sequences // lcm) * lcm
    else:
        # Find the closest size to target that's divisible by LCM
        lower = (target_size // lcm) * lcm
        upper = lower + lcm
        
        if target_size - lower <= upper - target_size:
            target_size = lower
        else:
            target_size = upper
    
    # Make sure we don't exceed total available sequences
    target_size = min(target_size, (total_sequences // lcm) * lcm)
    
    print(f"Adjusting dataset size to {target_size} (divisible by {batch_sizes})")
    print(f"LCM of batch sizes: {lcm}")
    
    return target_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_chars", type=int, default=6000)
    parser.add_argument("--vocab_size", type=int, default=25)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--results_dir", default="src/data")
    parser.add_argument("--train_split", type=float, default=0.8)
    args = parser.parse_args()

    seed_all(args.seed)
    
    print("Downloading Shakespeare dataset...")
    text = download_shakespeare()
    print(f"Original text length: {len(text)} characters")
    
    print("Preprocessing text...")
    processed = preprocess_shakespeare(text, args.max_chars, args.vocab_size)
    
    print(f"Processed text length: {len(processed['text'])} characters")
    print(f"Vocabulary size: {processed['vocab_size']}")
    print(f"Characters: {''.join(processed['chars'])}")
    
    # Create sequences
    X, Y = create_sequences(processed['data'], args.block_size)
    print(f"Created {len(X)} sequences of length {args.block_size}")
    
    # Train/val split
    n_train = int(len(X) * args.train_split)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]
    
    print(f"Train sequences: {len(X_train)}, Val sequences: {len(X_val)}")
    
    # Save data
    os.makedirs(args.results_dir, exist_ok=True)
    
    train_file = os.path.join(args.results_dir, "shakespeare_train.pt")
    val_file = os.path.join(args.results_dir, "shakespeare_val.pt")
    meta_file = os.path.join(args.results_dir, "shakespeare_meta.pt")
    
    # Save train/val data
    torch.save((X_train, Y_train), train_file)
    torch.save((X_val, Y_val), val_file)
    
    # Save metadata
    meta_data = {
        'vocab_size': processed['vocab_size'],
        'chars': processed['chars'],
        'stoi': processed['stoi'],
        'itos': processed['itos'],
        'block_size': args.block_size,
        'train_size': len(X_train),
        'val_size': len(X_val)
    }
    torch.save(meta_data, meta_file)
    
    print(f"Saved train data to: {train_file}")
    print(f"Saved val data to: {val_file}")
    print(f"Saved metadata to: {meta_file}")
    
    # Print sample
    print("\nSample text:")
    print(repr(processed['text'][:200]))


if __name__ == "__main__":
    main()
