# TCR Embedding with Data Selection

Training TCR embeddings using catELMo with only 10% of data. Testing whether smart data selection (diversity, length-stratified) works as well as random sampling.

**Main question:** Can we use less data and still get good embeddings for TCR-epitope binding prediction?

---

## Structure

```
data/          Raw data, processed sequences, embeddings
models/        catELMo checkpoints and binding prediction models
scripts/       Code for each phase (data prep, training, prediction)
results/       Training logs, metrics, figures
docs/          Phase-specific documentation
```

---

## Workflow

1. **Data Prep** → Select 10% subsets using different strategies
2. **Train Embeddings** → Train catELMo on each subset
3. **Generate Embeddings** → Extract embeddings for binding pairs
4. **Binding Prediction** → 5-fold CV to evaluate performance

---

## Results

**Embedding Training**
- Diversity: 2 epochs, perplexity 3.59
- Random: pending
- Length-stratified: pending

**Binding Prediction** (pending)

## Environment

GPU: Tesla V100 / A100  
CUDA 11.2, cuDNN 8.1, TensorFlow 2.6
