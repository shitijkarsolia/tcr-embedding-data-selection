# Binding Prediction Model

Optimized 5-fold cross-validation for TCR-epitope binding prediction using pre-computed embeddings.

## Setup

Download pre-trained weights: [Dropbox link](https://www.dropbox.com/scl/fo/5oxts5ek72jvczr3ej8d2/AAPdShWGaLy8NdR1WFVMACs?rlkey=ap0j8d3dt14gti7u6m6rkaqxs&e=3&st=mw0vxlxa&dl=0)

## Key Changes

**Performance**
- Batch size: 32 → 1024 (10-20x faster)
- Data splitting: vectorized NumPy instead of list comprehensions
- Memory: removed duplicate DataFrames

**Features**
- 5-fold CV with multiple repeats (5 repeats × 5 folds = 25 runs)
- Command-line arguments instead of hardcoded paths
- Learning rate scheduling for better convergence
- Saves intermediate results after each repeat

## Usage

```bash
python -u train_5fold.py \
    --embedding catELMo \
    --split tcr \
    --data_path /path/to/embeddings.pkl \
    --output_dir ./results \
    --n_repeats 5
