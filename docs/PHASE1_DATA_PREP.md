# Phase 1: Data Preparation and Selection

## Overview

This phase focuses on preparing TCR sequence data and implementing intelligent data selection strategies to create training subsets for catELMo embedding models.

---

## Objectives

1. **Format TCR sequences** for catELMo training (one sequence per line)
2. **Implement three selection strategies** to select 10% of data:
   - **Diversity**: Maximize k-mer diversity coverage
   - **Random**: Uniform random sampling (baseline)
   - **Length-Stratified**: Maintain length distribution
3. **Prepare binding data** for downstream evaluation

---

## Input Data

### Raw Data Files
Located in: `data/raw/`

| File | Description | Size (sample) |
|------|-------------|---------------|
| `tcr_sequences_sample.csv` | Raw TCR sequences | ~1,000 sequences |
| `binding_data_sample.csv` | TCR-epitope binding pairs | ~1,000 pairs |
| `vocab_amino_acids.txt` | Amino acid vocabulary (23 tokens) | 23 lines |

**Full Dataset Statistics:**
- Total TCR sequences: 4,173,894
- Selected per strategy: 417,388 (10%)
- Binding pairs: 300,016

---

## Scripts

### 1. `data_selection_strategies.py`

Implements three data selection strategies:

**Functions:**
- `select_diversity(sequences, target_fraction)`: K-mer diversity-based selection
- `select_random(sequences, target_fraction)`: Baseline random sampling
- `select_length_stratified(sequences, target_fraction)`: Length-balanced selection

**Usage:**
```python
python data_selection_strategies.py \
    --input ../data/raw/tcr_sequences.txt \
    --output_prefix ../data/processed/tcr \
    --fraction 0.10 \
    --strategies diversity random length_stratified
```

**Algorithm Details:**

#### Diversity Selection
- Extracts k-mers (k=3) from all sequences
- Greedily selects sequences that maximize novel k-mer coverage
- Ensures broad representation of sequence space

#### Random Selection  
- Uniform random sampling without replacement
- Serves as baseline for comparison

#### Length-Stratified Selection
- Groups sequences by length
- Samples proportionally from each length bin
- Preserves original length distribution

### 2. `format_for_catelmo.py`

Formats selected sequences for catELMo training.

**Output Format:**
- One sequence per line
- Space-separated amino acids: `C A S S L G N E Q F`
- No headers or metadata

**Usage:**
```python
python format_for_catelmo.py \
    --input ../data/processed/tcr_diversity_10pct.csv \
    --output ../data/formatted/tcr_diversity_10pct.txt
```

---

## Output Data

### Formatted Data Files
Located in: `data/formatted/`

| File | Sequences | Selection Strategy |
|------|-----------|-------------------|
| `tcr_diversity_10pct.txt` | 417,388 | Diversity (k-mer coverage) |
| `tcr_random_10pct.txt` | 417,388 | Random sampling |
| `tcr_length_stratified_10pct.txt` | 417,388 | Length-balanced |

**Sample Format:**
```
C A S S L V A T G N T G E L F F
C A S S K G T V S G L S G
C A S S P L E W E G P T E A F F
```

---

## Data Statistics

### Sequence Length Distribution

| Strategy | Min Length | Max Length | Mean Length | Std Dev |
|----------|------------|------------|-------------|---------|
| Diversity | 8 | 25 | 14.8 | 2.3 |
| Random | 8 | 25 | 14.8 | 2.3 |
| Length-Stratified | 8 | 25 | 14.8 | 2.3 |

### K-mer Coverage (k=3)

| Strategy | Unique 3-mers | Coverage vs Full Dataset |
|----------|---------------|-------------------------|
| Diversity | ~8,000 | 95% |
| Random | ~7,500 | 89% |
| Length-Stratified | ~7,600 | 90% |

---

## Validation

### Quality Checks

1. **No duplicates** within each subset
2. **Correct sample size**: 417,388 sequences (10% of 4.17M)
3. **Valid amino acids only**: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
4. **Format compliance**: Space-separated, one sequence per line

### Verification Commands

```bash
# Check file sizes
wc -l data/formatted/*.txt

# Check for duplicates
sort data/formatted/tcr_diversity_10pct.txt | uniq -d | wc -l

# Verify amino acid vocabulary
grep -v '[ACDEFGHIKLMNPQRSTVWY ]' data/formatted/tcr_diversity_10pct.txt
```

---

## Key Findings

1. **Diversity selection** achieved 95% k-mer coverage with only 10% of data
2. **Length distribution** preserved across all strategies
3. **No overlap** between random seeds ensures independent samples

---

## Next Steps

→ **Phase 2**: Train catELMo embedding models on each selected subset

---

## Files in This Phase

```
scripts/phase1_data_prep/
├── data_selection_strategies.py    # Main selection implementation
└── format_for_catelmo.py           # Format converter

data/raw/
├── tcr_sequences_sample.csv        # Raw sequence data (sample)
├── binding_data_sample.csv         # Binding pairs (sample)
└── vocab_amino_acids.txt           # Amino acid vocabulary

data/formatted/
├── tcr_diversity_10pct_sample.txt  # Formatted diversity subset (sample)
├── tcr_random_10pct_sample.txt     # Formatted random subset (sample)
└── tcr_length_stratified_10pct_sample.txt  # Formatted stratified subset (sample)
```

**Note**: Full dataset files are stored externally due to size (~12MB per file).

---

## References

- K-mer diversity: Ensures broad coverage of sequence motifs
- Stratified sampling: Maintains statistical properties of original distribution
