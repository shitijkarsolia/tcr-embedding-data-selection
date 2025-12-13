DATA_PATH="/home/mkpatel7/baf_model/data"
OUTPUT_DIR="./results"
EMBEDDING="catELMo"
GPU="0"
SEED=42
N_REPEATS=5

SPLITS=("epi" "tcr")

mkdir -p $OUTPUT_DIR

LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"
echo "Experiment started at $(date)" > $LOG_FILE

for split in "${SPLITS[@]}"; do
    echo "=====================================" | tee -a $LOG_FILE
    echo "Running: catELMo with $split split" | tee -a $LOG_FILE
    echo "=====================================" | tee -a $LOG_FILE
    
    python -u train_5fold.py \
        --embedding $EMBEDDING \
        --split $split \
        --data_path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --gpu $GPU \
        --seed $SEED \
        --n_repeats $N_REPEATS \
        2>&1 | tee -a $LOG_FILE
    
    echo "" | tee -a $LOG_FILE
done

echo "All experiments completed at $(date)" | tee -a $LOG_FILE
