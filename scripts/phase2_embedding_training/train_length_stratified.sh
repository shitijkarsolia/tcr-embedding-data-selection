#!/bin/bash
#
# Train catELMo embedding model on length-stratified 10% subset
#

set -e  # Exit on error

echo "========================================"
echo "catELMo Training - Length-Stratified Selection"
echo "========================================"
echo ""

# Paths
PROJECT_ROOT="/home/sagemaker-user/comp-bio-project"
BILM_DIR="$PROJECT_ROOT/catelmo-baseline/bilm-tf"
VOCAB_FILE="$PROJECT_ROOT/data/vocab_amino_acids.txt"
TRAIN_DATA="$PROJECT_ROOT/data/formatted_for_catelmo/selection/tcr_length_stratified_10pct.txt"
SAVE_DIR="$PROJECT_ROOT/models/embeddings/length_stratified_10pct"

# Verify files exist
echo "Verifying files..."
if [ ! -f "$VOCAB_FILE" ]; then
    echo "ERROR: Vocabulary file not found: $VOCAB_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found: $TRAIN_DATA"
    exit 1
fi

echo "✓ Vocabulary file: $VOCAB_FILE"
echo "✓ Training data: $TRAIN_DATA ($(wc -l < $TRAIN_DATA) sequences)"
echo ""

# Create output directory
mkdir -p "$SAVE_DIR"
echo "✓ Output directory: $SAVE_DIR"
echo ""

# Show data statistics
echo "Training Data Statistics:"
head -n 3 "$TRAIN_DATA"
echo "..."
echo ""

# Training parameters info
echo "Training Parameters:"
echo "  - Architecture: 2-layer BiLSTM (2048 hidden, 512 projection)"
echo "  - Epochs: 2"
echo "  - Batch size: 256"
echo "  - Vocabulary size: 23 (20 amino acids + 3 special tokens)"
echo ""

echo "========================================"
echo "Starting Training..."
echo "========================================"
echo ""

# Change to bilm directory (required for imports)
cd "$BILM_DIR"

# Activate conda environment
source activate tf26

# Run training
python train_elmo_fast_length.py \
    --save_dir "$SAVE_DIR" \
    --vocab_file "$VOCAB_FILE" \
    --train_prefix "$TRAIN_DATA"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Model saved to: $SAVE_DIR"
