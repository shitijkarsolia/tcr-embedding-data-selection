# Phase 3: Embedding Generation

## Overview

This phase converts trained catELMo models to HDF5 format and generates contextual embeddings for all TCR and epitope sequences in the binding prediction dataset.

---

## Objectives

1. **Convert TensorFlow checkpoints to HDF5** format for allennlp compatibility
2. **Generate 1024-dimensional embeddings** for TCR and epitope sequences
3. **Process binding dataset** with 300K TCR-epitope pairs
4. **Save embeddings** in pickle format for downstream binding prediction

---

## Workflow

```
TensorFlow Checkpoint (.ckpt)
         ↓
    [convert_to_hdf5.sh]
         ↓
    HDF5 Weights (.hdf5)
         ↓
    [generate_embeddings.py]
         ↓
Embeddings for Binding Data (.pkl)
```

---

## Scripts

### 1. `convert_to_hdf5.sh`

Converts TensorFlow checkpoint files to HDF5 format using the bilm-tf dump_weights utility.

**Purpose**: allennlp's ElmoEmbedder requires HDF5 format weights.

**Usage**:
```bash
source activate tf26
cd catelmo-baseline/bilm-tf
./convert_to_hdf5.sh
```

**What it does**:
```bash
python bin/dump_weights.py \
    --save_dir ../../models/embeddings/diversity_10pct \
    --outfile ../../models/embeddings/diversity_10pct/weights.hdf5
```

**Output**:
- `diversity_10pct_weights.hdf5` (213 MB)
- Contains all model weights in HDF5 format

### 2. `generate_embeddings.py`

Main script to generate embeddings using trained catELMo models.

**Key Functions**:

```python
def catELMo_embedding(sequence, embedder):
    """
    Generate embedding for a single sequence.
    
    Args:
        sequence: Amino acid sequence string (e.g., "CASSLGNEQF")
        embedder: ElmoEmbedder instance
    
    Returns:
        1024-dimensional embedding vector
    """
    seq_list = list(sequence)
    embeddings = embedder.embed_sentence(seq_list)
    # Average across layers (dim 0) and positions (dim 1)
    embedding_tensor = torch.tensor(embeddings).sum(dim=0).mean(dim=0)
    return embedding_tensor.tolist()
```

**Usage**:
```bash
python generate_embeddings.py \
    --model_dir ../models/embeddings/diversity_10pct \
    --input_csv ../data/raw/binding_data.csv \
    --output_pkl ../data/embeddings/binding_data_diversity.pkl \
    --use_gpu
```

**Arguments**:
- `--model_dir`: Directory with `weights.hdf5` and `options.json`
- `--input_csv`: CSV with `epi`, `tcr`, `binding` columns
- `--output_pkl`: Output pickle file path
- `--use_gpu`: Enable GPU acceleration (recommended)

### 3. `generate_all_embeddings.sh`

Runner script to generate embeddings for all three selection strategies.

**Usage**:
```bash
nohup ./generate_all_embeddings.sh > embedding_generation.log 2>&1 &
```

**What it does**:
1. Generates embeddings for diversity model
2. Generates embeddings for random model
3. Generates embeddings for length-stratified model

---

## Input Data

### Binding Dataset

**File**: `data/raw/binding_data_sample.csv`

**Format**:
```csv
epi,tcr,binding
EAAGIGILTV,CASSLGNEQF,1
EAAGIGILTV,CASSLGVATGELF,1
EAAGIGILTV,CASSQEEGGGSWGNTIYF,1
```

**Columns**:
- `epi`: Epitope sequence (peptide antigen)
- `tcr`: TCR CDR3β sequence
- `binding`: Binary label (1=binding, 0=non-binding)

**Full Dataset**:
- Total pairs: 300,016
- Positive samples: ~150,000 (50%)
- Negative samples: ~150,000 (50%)

---

## Output Data

### Generated Embeddings

**File**: `data/embeddings/binding_data_diversity.pkl`

**Format**: Pandas DataFrame (pickle format)

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `epi` | str | Epitope sequence |
| `tcr` | str | TCR sequence |
| `binding` | int | Binding label (0/1) |
| `tcr_embeds` | list[float] | 1024-dim TCR embedding |
| `epi_embeds` | list[float] | 1024-dim epitope embedding |

**Size**: 5.2 GB (300K pairs × 2 embeddings × 1024 dims)

**Sample**:
```
          epi                 tcr  binding
0  EAAGIGILTV          CASSLGNEQF        1
1  EAAGIGILTV       CASSLGVATGELF        1
2  EAAGIGILTV  CASSQEEGGGSWGNTIYF        1

tcr_embeds: [0.049, -0.123, -0.087, ..., 0.234]  # 1024 values
epi_embeds: [-0.365, 0.294, 0.297, ..., -0.156]  # 1024 values
```

---

## Performance Metrics

### Embedding Generation Statistics

**Diversity Model**:
- **Epitope embeddings**: 300,016 sequences
  - Time: 2h 40min 56sec
  - Throughput: 31.07 sequences/second
  
- **TCR embeddings**: 300,016 sequences
  - Time: 3h 9min 9sec
  - Throughput: 26.44 sequences/second

**Total Time**: ~6 hours per model

**GPU Utilization**: Used GPU device 0 for acceleration

**Memory Usage**: 
- Peak GPU memory: ~8 GB
- Output file size: 5.2 GB

---

## Technical Details

### Embedding Extraction

**catELMo produces 3 layers of representations**:
1. Character-level CNN embeddings
2. Layer 1 BiLSTM outputs
3. Layer 2 BiLSTM outputs

**Aggregation Strategy**:
- Sum across layers (weighted combination)
- Average across sequence positions
- Results in fixed 1024-dimensional vector per sequence

**Why averaging?**:
- Handles variable-length sequences (8-25 amino acids)
- Produces fixed-size representation for downstream models
- Captures overall sequence context

### GPU Acceleration

**Benefits**:
- ~5-10x faster than CPU
- Enables processing of large datasets
- Batch processing on GPU memory

**Requirements**:
- CUDA-compatible GPU
- cuDNN libraries
- allennlp with CUDA support

---

## Validation

### Quality Checks

1. **Embedding Dimensions**: ✓ All vectors are 1024-dimensional
2. **No NaN/Inf**: ✓ All embedding values are finite
3. **Complete Dataset**: ✓ All 300,016 pairs have embeddings
4. **Matching Order**: ✓ Row order preserved from input CSV

### Verification Commands

```python
import pandas as pd
import numpy as np

# Load embeddings
df = pd.read_pickle('data/embeddings/binding_data_diversity.pkl')

# Check dimensions
print(f"Dataset shape: {df.shape}")
print(f"TCR embedding dim: {len(df.iloc[0]['tcr_embeds'])}")
print(f"Epi embedding dim: {len(df.iloc[0]['epi_embeds'])}")

# Check for missing values
print(f"Missing TCR embeds: {df['tcr_embeds'].isna().sum()}")
print(f"Missing Epi embeds: {df['epi_embeds'].isna().sum()}")

# Check for NaN in embeddings
tcr_array = np.stack(df['tcr_embeds'].values)
epi_array = np.stack(df['epi_embeds'].values)
print(f"NaN in TCR: {np.isnan(tcr_array).sum()}")
print(f"NaN in Epi: {np.isnan(epi_array).sum()}")
```

---

## HDF5 Weights

### Conversion Details

**Input**: TensorFlow checkpoint files
- `model.ckpt-2414.data-00000-of-00001` (426 MB)
- `model.ckpt-2414.index` (3.2 KB)
- `model.ckpt-2414.meta` (5.7 MB)

**Output**: HDF5 weights
- `diversity_10pct_weights.hdf5` (213 MB)

**Size Reduction**: 
- From 426 MB (checkpoint) → 213 MB (HDF5)
- 50% reduction due to format efficiency

**Contains**:
- Character CNN weights
- BiLSTM cell weights
- Projection layer weights
- Highway network weights

---

## Log Summary

### Embedding Generation Log

From `embedding_generation_summary.txt`:

```
========================================
Generating Embeddings for All Models
========================================

Processing diversity_10pct model...
Initializing ElmoEmbedder (cuda_device=0)...
Loading data from /path/to/binding_data.csv...
Dataset size: 300016 TCR-epitope pairs
Columns: ['epi', 'tcr', 'binding']

Generating epitope embeddings...
100%|██████████| 300016/300016 [2:40:56<00:00, 31.07it/s]

Generating TCR embeddings...
100%|██████████| 300016/300016 [3:09:09<00:00, 26.44it/s]

Saving embeddings to binding_data_diversity.pkl...
Done!
Embedding dimensions: 1024
Output saved to: data/embeddings/binding_data_diversity.pkl
========================================
All Embeddings Generated!
========================================
```

---

## Key Findings

1. **Consistent throughput**: ~30 sequences/sec for epitopes, ~26 for TCRs
2. **GPU acceleration essential**: Would take ~30+ hours on CPU
3. **Large output files**: 5.2 GB per model requires significant storage
4. **Stable generation**: No errors or NaN values in 300K pairs

---

## Next Steps

→ **Phase 4**: Train binding prediction models using generated embeddings

---

## Files in This Phase

```
scripts/phase3_embedding_generation/
├── convert_to_hdf5.sh              # Checkpoint → HDF5 converter
├── generate_embeddings.py          # Main embedding generator
└── generate_all_embeddings.sh      # Batch runner for all models

models/embedding_checkpoints/
└── hdf5_weights_info.txt           # HDF5 weights documentation

data/embeddings/
└── diversity_embeddings_info.txt   # Sample embedding statistics

results/embedding_training/
└── embedding_generation_summary.txt # Generation log (first/last 100 lines)
```

**Note**: Full embedding file (5.2 GB) stored externally.

---

## References

- **allennlp ElmoEmbedder**: Used for efficient embedding extraction
- **HDF5 format**: Hierarchical Data Format for efficient storage
- **Contextual embeddings**: Different representations based on sequence context
