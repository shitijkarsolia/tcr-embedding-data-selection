"""
Format TCR sequences for catELMo training.

Converts compact sequences (CASSLYGG) to space-separated format (C A S S L Y G G)
that catELMo expects.

This script processes:
- Scaling experiment data (data/processed/)
- Selection strategy data (data/selected_subsets/)

Output goes to: data/formatted_for_catelmo/

Usage:
    python code/format_for_catelmo.py
"""

from pathlib import Path
from typing import List
import sys


def format_sequence(sequence: str) -> str:
    """
    Convert compact sequence to space-separated format.
    
    Args:
        sequence: Compact sequence like "CASSLYGG"
    
    Returns:
        Space-separated sequence like "C A S S L Y G G"
    """
    return ' '.join(list(sequence.strip()))


def format_file(input_file: Path, output_file: Path):
    """
    Convert entire file from compact to space-separated format.
    
    Args:
        input_file: Path to compact format file
        output_file: Path to output space-separated file
    """
    print(f"Processing: {input_file.name}")
    
    sequences_formatted = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                formatted = format_sequence(line)
                sequences_formatted.append(formatted)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write formatted sequences
    with open(output_file, 'w') as f:
        f.write('\n'.join(sequences_formatted))
    
    print(f"  → Formatted {len(sequences_formatted):,} sequences")
    print(f"  → Saved to: {output_file}")
    print()


def main():
    # Get project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Define paths
    processed_dir = project_root / "data" / "processed"
    selected_dir = project_root / "data" / "selected_subsets"
    output_dir = project_root / "data" / "formatted_for_catelmo"
    
    print("=" * 70)
    print("TCR Sequence Formatter for catELMo")
    print("=" * 70)
    print()
    
    files_to_format = []
    
    # 1. Scaling experiment files (10%, 25%, 50%, 75%, 100%)
    print("Looking for scaling experiment files...")
    if processed_dir.exists():
        for file in processed_dir.glob("tcr_sequences_*pct.txt"):
            output_file = output_dir / "scaling" / file.name
            files_to_format.append((file, output_file))
            print(f"  Found: {file.name}")
    print()
    
    # 2. Selection strategy files
    print("Looking for selection strategy files...")
    if selected_dir.exists():
        for file in selected_dir.glob("tcr_*_10pct.txt"):
            output_file = output_dir / "selection" / file.name
            files_to_format.append((file, output_file))
            print(f"  Found: {file.name}")
    print()
    
    if not files_to_format:
        print("ERROR: No input files found!")
        print(f"  Checked: {processed_dir}")
        print(f"  Checked: {selected_dir}")
        sys.exit(1)
    
    # Process all files
    print("=" * 70)
    print(f"Formatting {len(files_to_format)} files...")
    print("=" * 70)
    print()
    
    for input_file, output_file in files_to_format:
        format_file(input_file, output_file)
    
    print("=" * 70)
    print("FORMATTING COMPLETE!")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Your formatted files are ready for catELMo training!")
    print()
    print("File structure:")
    print(f"  {output_dir}/")
    print(f"    scaling/        (10%, 25%, 50%, 75%, 100%)")
    print(f"    selection/      (random, diversity, length-stratified)")


if __name__ == "__main__":
    main()

