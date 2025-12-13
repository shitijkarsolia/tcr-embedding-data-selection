#!/usr/bin/env python3
"""
Generate embeddings for TCR and epitope sequences using trained catELMo models.
This script embeds sequences from the binding dataset for downstream prediction tasks.
"""

import pandas as pd
import torch
import argparse
from pathlib import Path
from allennlp.commands.elmo import ElmoEmbedder
from tqdm import tqdm
import sys

def catELMo_embedding(sequence, embedder):
    """
    Generate catELMo embedding for a single sequence.
    
    Args:
        sequence: Amino acid sequence string
        embedder: ElmoEmbedder instance
    
    Returns:
        List of floats representing the embedding
    """
    # Convert sequence to list of characters
    seq_list = list(sequence)
    
    # Get embeddings: shape is (num_layers, seq_len, embedding_dim)
    embeddings = embedder.embed_sentence(seq_list)
    
    # Average across layers (dim 0) and sequence positions (dim 1)
    # This gives a fixed-size vector per sequence
    embedding_tensor = torch.tensor(embeddings).sum(dim=0).mean(dim=0)
    
    return embedding_tensor.tolist()

def main():
    parser = argparse.ArgumentParser(description='Generate catELMo embeddings for binding prediction')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing weights.hdf5 and options.json')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Input CSV file with epi and tcr columns')
    parser.add_argument('--output_pkl', type=str, required=True,
                        help='Output pickle file path')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for embedding (default: CPU)')
    
    args = parser.parse_args()
    
    # Setup paths
    model_dir = Path(args.model_dir)
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    
    # Verify files exist
    if not weights.exists():
        print(f"ERROR: weights.hdf5 not found at {weights}")
        sys.exit(1)
    if not options.exists():
        print(f"ERROR: options.json not found at {options}")
        sys.exit(1)
    
    # Initialize embedder
    cuda_device = 0 if args.use_gpu else -1
    print(f"Initializing ElmoEmbedder (cuda_device={cuda_device})...")
    embedder = ElmoEmbedder(str(options), str(weights), cuda_device=cuda_device)
    
    # Load data
    print(f"Loading data from {args.input_csv}...")
    dat = pd.read_csv(args.input_csv)
    
    print(f"Dataset size: {len(dat)} TCR-epitope pairs")
    print(f"Columns: {list(dat.columns)}")
    
    # Check required columns
    if 'tcr' not in dat.columns or 'epi' not in dat.columns:
        print("ERROR: Input CSV must have 'tcr' and 'epi' columns")
        sys.exit(1)
    
    # Initialize embedding columns
    dat['tcr_embeds'] = None
    dat['epi_embeds'] = None
    
    # Generate epitope embeddings
    print("\nGenerating epitope embeddings...")
    for i in tqdm(range(len(dat))):
        dat.at[i, 'epi_embeds'] = catELMo_embedding(dat.at[i, 'epi'], embedder)
    
    # Generate TCR embeddings
    print("\nGenerating TCR embeddings...")
    for i in tqdm(range(len(dat))):
        dat.at[i, 'tcr_embeds'] = catELMo_embedding(dat.at[i, 'tcr'], embedder)
    
    # Save results
    print(f"\nSaving embeddings to {args.output_pkl}...")
    dat.to_pickle(args.output_pkl)
    
    print("Done!")
    print(f"Embedding dimensions: {len(dat.at[0, 'tcr_embeds'])}")
    print(f"Output saved to: {args.output_pkl}")

if __name__ == '__main__':
    main()
