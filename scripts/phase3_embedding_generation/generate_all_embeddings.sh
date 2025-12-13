#!/bin/bash
# Generate embeddings for all three selection strategies

source activate tf26

# Create output directory
mkdir -p /home/sagemaker-user/comp-bio-project/data/embeddings

echo "========================================"
echo "Generating Embeddings for All Models"
echo "========================================"

# Input data
INPUT_CSV="/home/sagemaker-user/comp-bio-project/data/raw/binding_data.csv"

# Generate embeddings for diversity model
echo ""
echo "Processing diversity_10pct model..."
python /home/sagemaker-user/comp-bio-project/generate_embeddings.py \
    --model_dir /home/sagemaker-user/comp-bio-project/models/embeddings/diversity_10pct \
    --input_csv ${INPUT_CSV} \
    --output_pkl /home/sagemaker-user/comp-bio-project/data/embeddings/binding_data_diversity.pkl \
    --use_gpu

# # Generate embeddings for random model
# echo ""
# echo "Processing random_10pct model..."
# python /home/sagemaker-user/comp-bio-project/generate_embeddings.py \
#     --model_dir /home/sagemaker-user/comp-bio-project/models/embeddings/random_10pct \
#     --input_csv ${INPUT_CSV} \
#     --output_pkl /home/sagemaker-user/comp-bio-project/data/embeddings/binding_data_random.pkl \
#     --use_gpu

# # Generate embeddings for length-stratified model
# echo ""
# echo "Processing length_stratified_10pct model..."
# python /home/sagemaker-user/comp-bio-project/generate_embeddings.py \
#     --model_dir /home/sagemaker-user/comp-bio-project/models/embeddings/length_stratified_10pct \
#     --input_csv ${INPUT_CSV} \
#     --output_pkl /home/sagemaker-user/comp-bio-project/data/embeddings/binding_data_length_stratified.pkl \
#     --use_gpu

echo ""
echo "========================================"
echo "All Embeddings Generated!"
echo "========================================"
echo ""
ls -lh /home/sagemaker-user/comp-bio-project/data/embeddings/*.pkl
