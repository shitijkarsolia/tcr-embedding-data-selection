#!/bin/bash
# Convert trained TensorFlow checkpoints to HDF5 format for embedding generation

source activate tf26
cd /home/sagemaker-user/comp-bio-project/catelmo-baseline/bilm-tf

echo "========================================"
echo "Converting Models to HDF5 Format"
echo "========================================"

# Convert diversity model
echo ""
echo "Converting diversity_10pct model..."
python bin/dump_weights.py \
    --save_dir ../../models/embeddings/diversity_10pct \
    --outfile ../../models/embeddings/diversity_10pct/weights.hdf5

# Convert random model
echo ""
echo "Converting random_10pct model..."
python bin/dump_weights.py \
    --save_dir ../../models/embeddings/random_10pct \
    --outfile ../../models/embeddings/random_10pct/weights.hdf5

# Convert length-stratified model
echo ""
echo "Converting length_stratified_10pct model..."
python bin/dump_weights.py \
    --save_dir ../../models/embeddings/length_stratified_10pct \
    --outfile ../../models/embeddings/length_stratified_10pct/weights.hdf5

echo ""
echo "========================================"
echo "Conversion Complete!"
echo "========================================"
echo ""
echo "HDF5 files created:"
ls -lh ../../models/embeddings/*/weights.hdf5
