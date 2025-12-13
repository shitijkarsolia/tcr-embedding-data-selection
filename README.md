# TCR Embedding Model Data Selection Study

## Project Overview

This project investigates data selection strategies for training TCR (T-Cell Receptor) embedding models using catELMo architecture. We evaluate whether intelligent data selection (diversity-based, length-stratified) can achieve comparable or better performance than random sampling while using only 10% of the full dataset.

## Research Questions

1. **Can intelligent data selection strategies produce embeddings as good as random sampling?**
2. **Which selection strategy (diversity, random, length-stratified) yields the best downstream binding prediction performance?**
3. **How much data is actually needed for effective TCR embedding training?**

---

## Project Structure

```
project/
├── data/                          # All datasets
│   ├── raw/                       # Original unprocessed data
│   ├── processed/                 # Cleaned and formatted sequences
│   ├── formatted/                 # Data formatted for catELMo training
│   ├── embeddings/                # Generated embeddings
│   └── splits/                    # Train/test splits for binding prediction
│
├── models/                        # Trained model weights
│   ├── embedding_checkpoints/     # catELMo model checkpoints
│   └── binding_models/            # Binding prediction model checkpoints
│
├── scripts/                       # All executable scripts organized by phase
│   ├── phase1_data_prep/          # Data preparation and selection
│   ├── phase2_embedding_training/ # catELMo training scripts
│   ├── phase3_embedding_generation/ # Embedding generation scripts
│   └── phase4_binding_prediction/ # Binding prediction training
│
├── results/                       # Experimental results
│   ├── embedding_training/        # Training logs and metrics
│   ├── binding_prediction/        # Binding prediction results
│   └── figures/                   # Plots and visualizations
│
├── notebooks/                     # Jupyter notebooks for analysis
└── docs/                          # Documentation
```

---

## Workflow Phases

### Phase 1: Data Preparation
- **Input**: Raw TCR sequences and binding data
- **Output**: Selected 10% subsets (diversity, random, length-stratified)
- **Scripts**: `scripts/phase1_data_prep/`

### Phase 2: Embedding Training
- **Input**: Selected TCR sequence subsets
- **Output**: Trained catELMo models (TensorFlow checkpoints)
- **Scripts**: `scripts/phase2_embedding_training/`

### Phase 3: Embedding Generation
- **Input**: Trained models + binding dataset
- **Output**: TCR and epitope embeddings for binding prediction
- **Scripts**: `scripts/phase3_embedding_generation/`

### Phase 4: Binding Prediction
- **Input**: Generated embeddings
- **Output**: Binding prediction performance metrics (AUC, accuracy, etc.)
- **Scripts**: `scripts/phase4_binding_prediction/`

---

## Key Results

### Embedding Training
- **Diversity Selection**: 2 epochs, final perplexity: 3.59
- **Random Selection**: [To be added]
- **Length-Stratified**: [To be added]

### Binding Prediction (5-fold CV)
| Strategy | TCR Split (AUC) | Epitope Split (AUC) |
|----------|----------------|---------------------|
| Diversity | [Pending] | [Pending] |
| Random | [Pending] | [Pending] |
| Length-Stratified | [Pending] | [Pending] |

---

## Environment

### Hardware
- GPU: Tesla V100-SXM2-16GB
- CUDA: 11.2.2
- cuDNN: 8.1.0.77
