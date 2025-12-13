# TCR Embedding with Data Selection

Training TCR embeddings using catELMo with only 10% of data. Testing whether smart data selection (diversity, length-stratified) works better than random sampling.

**Main question:** How can we use less data and still get good embeddings for TCR-epitope binding prediction?

---

## Model weights

Download all model weights here: [Dropbox link](https://www.dropbox.com/scl/fo/5oxts5ek72jvczr3ej8d2/AAPdShWGaLy8NdR1WFVMACs?rlkey=ap0j8d3dt14gti7u6m6rkaqxs&e=3&st=mw0vxlxa&dl=0)

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

1. **Data Prep** → Select 10% subsets using three data selection strategies.
2. **Train Embeddings** → Train catELMo on each data subset and identify preferred subset.
3. **Generate Embeddings** → Extract embeddings for use in Binding Affinity Prediction.
4. **Binding Prediction** → Improved customized binding affinity prediction code to perform 5-fold cross validation, 5 times for each of TCR split and Epitope split, to evaluate performance in a more efficient and thorough manner.

---

## Results

**Embedding Training**
- Diversity: 2 epochs, perplexity 3.59
- Random: 2 epochs, perplexity 3.71
- Length-stratified: 2 epochs, perplexity 3.68

**Binding Prediction** 
- Mean of AUC, accuracy, f1-macro, precision, recall
- Standard deviation of AUC, accuracy, f1-macro, precision, recall

## Environment

GPU: Tesla V100 / NVIDIA A100  
CUDA 11.2, cuDNN 8.1, TensorFlow 2.6
