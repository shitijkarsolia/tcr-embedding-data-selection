from pathlib import Path
from typing import List
import sys


def format_sequence(sequence: str) -> str:
    return ' '.join(list(sequence.strip()))


def format_file(input_file: Path, output_file: Path):
    print(f"Processing: {input_file.name}")
    
    sequences_formatted = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                formatted = format_sequence(line)
                sequences_formatted.append(formatted)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(sequences_formatted))
    
    print(f"  → Formatted {len(sequences_formatted):,} sequences")
    print(f"  → Saved to: {output_file}")
    print()


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    processed_dir = project_root / "data" / "processed"
    selected_dir = project_root / "data" / "selected_subsets"
    output_dir = project_root / "data" / "formatted_for_catelmo"
    
    print("=" * 70)
    print("TCR Sequence Formatter for catELMo")
    print("=" * 70)
    print()
    
    files_to_format = []
    
    print("Looking for scaling experiment files...")
    if processed_dir.exists():
        for file in processed_dir.glob("tcr_sequences_*pct.txt"):
            output_file = output_dir / "scaling" / file.name
            files_to_format.append((file, output_file))
            print(f"  Found: {file.name}")
    print()
    
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

