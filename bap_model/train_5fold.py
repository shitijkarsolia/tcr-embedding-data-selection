'''
Part of catELMo
(c) 2023 by  Pengfei Zhang, Michael Cai, Seojin Bang, Heewook Lee, and Arizona State University.
See LICENSE-CC-BY-NC-ND for licensing.
'''

import sys
import time
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from numpy import mean, std
from tensorflow import keras
from tensorflow.math import subtract

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU, Activation, BatchNormalization
from keras.models import Model
from tensorflow.keras.layers import concatenate

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def get_inputs(embedding_type, data_path):
    embedding_files = {
        'catELMo': 'catELMo_combined.pkl',
    }
    
    if embedding_type not in embedding_files:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    file_path = os.path.join(data_path, embedding_files[embedding_type])
    print(f"Loading data from: {file_path}")
    dat = pd.read_pickle(file_path)
    print(f"Data loaded: {len(dat)} samples")
    return dat


def load_data_split(dat, split_type, seed, fold_idx, n_folds=5):
    x_pep = dat.epi.values
    x_tcr = dat.tcr.values
    
    if split_type == 'random':
        n_total = len(x_pep)
        unique_items = np.arange(n_total)
    elif split_type == 'epi':
        unique_items = np.unique(x_pep)
        n_total = len(unique_items)
    elif split_type == 'tcr':
        unique_items = np.unique(x_tcr)
        n_total = len(unique_items)
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    
    np.random.seed(seed)
    idx_shuffled = np.arange(n_total)
    np.random.shuffle(idx_shuffled)

    n_test = int(np.ceil(n_total / n_folds))

    test_fold_start = fold_idx * n_test
    test_fold_end = min((fold_idx + 1) * n_test, n_total)

    if split_type == 'random':
        idx_test = idx_shuffled[test_fold_start:test_fold_end]
        idx_train = np.setdiff1d(idx_shuffled, idx_test)
        
    elif split_type == 'epi':
        idx_test_items = idx_shuffled[test_fold_start:test_fold_end]
        test_epitopes = unique_items[idx_test_items]

        mask = np.isin(x_pep, test_epitopes)
        idx_test = np.where(mask)[0]
        idx_train = np.where(~mask)[0]
        
    elif split_type == 'tcr':
        idx_test_items = idx_shuffled[test_fold_start:test_fold_end]
        test_tcrs = unique_items[idx_test_items]

        mask = np.isin(x_tcr, test_tcrs)
        idx_test = np.where(mask)[0]
        idx_train = np.where(~mask)[0]

    testData = dat.iloc[idx_test].sample(frac=1, random_state=seed).reset_index(drop=True)
    trainData = dat.iloc[idx_train].sample(frac=1, random_state=seed).reset_index(drop=True)

    overlap_tcrs = len(set(trainData.tcr).intersection(set(testData.tcr)))
    overlap_epis = len(set(trainData.epi).intersection(set(testData.epi)))
    
    print(f'\n=== Fold {fold_idx + 1}/{n_folds} - {split_type.upper()} split ===')
    print(f'Train samples: {len(trainData)}, Test samples: {len(testData)}')
    print(f'Overlapping TCRs: {overlap_tcrs}')
    print(f'Overlapping Epitopes: {overlap_epis}')

    X1_train = np.stack(trainData.tcr_embeds.values)
    X2_train = np.stack(trainData.epi_embeds.values)
    y_train = trainData.binding.values
    
    X1_test = np.stack(testData.tcr_embeds.values)
    X2_test = np.stack(testData.epi_embeds.values)
    y_test = testData.binding.values

    return X1_train, X2_train, y_train, X1_test, X2_test, y_test


def build_model(input_dim_tcr, input_dim_epi):

    inputA = Input(shape=(input_dim_tcr,))
    x = Dense(2048, kernel_initializer='he_uniform')(inputA)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Activation("silu")(x)
    x = Model(inputs=inputA, outputs=x)

    inputB = Input(shape=(input_dim_epi,))
    y = Dense(2048, kernel_initializer='he_uniform')(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = Activation("silu")(y)
    y = Model(inputs=inputB, outputs=y)

    combined = concatenate([x.output, y.output])

    z = Dense(1024)(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)
    z = Activation("silu")(z)
    z = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[x.input, y.input], outputs=z)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model


def train_and_evaluate(model, X1_train, X2_train, y_train, X1_test, X2_test, y_test, 
                       checkpoint_path, verbose=0):

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=30
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        [X1_train, X2_train], y_train,
        verbose=verbose,
        validation_split=0.20,
        epochs=200,
        batch_size=1024,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    model.load_weights(checkpoint_path)

    y_pred_proba = model.predict([X1_test, X2_test], verbose=0)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    return y_pred, y_pred_proba, history


def calculate_metrics(y_true, y_pred, y_pred_proba):
    metrics = {}
    
    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')

    metrics['precision_pos'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['precision_neg'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['recall_pos'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall_neg'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    return metrics


def print_metrics(metrics, fold_idx=None):

    if fold_idx is not None:
        print(f"\n=== Fold {fold_idx + 1} Results ===")
    else:
        print("\n=== Average Results Across All Folds ===")
    
    print(f"AUC:           {metrics['auc']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1 (Macro):    {metrics['f1_macro']:.4f}")


def run_5fold_analysis(embedding, split_type, data_path, output_dir, seed=42, n_repeats=5):

    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models', embedding, split_type)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting {n_repeats}-repeat 5-fold CV for {embedding} with {split_type} split")
    print(f"{'='*60}")
    
    dat = get_inputs(embedding, data_path)

    all_results = []
    
    for repeat in range(n_repeats):
        print(f"\n{'='*60}")
        print(f"REPEAT {repeat + 1}/{n_repeats}")
        print(f"{'='*60}")
        
        repeat_seed = seed + repeat * 100
        fold_results = []
        
        for fold in range(5):
            print(f"\n--- Processing Fold {fold + 1}/5 ---")

            X1_train, X2_train, y_train, X1_test, X2_test, y_test = load_data_split(
                dat, split_type, repeat_seed, fold, n_folds=5
            )

            model = build_model(X1_train.shape[1], X2_train.shape[1])

            checkpoint_path = os.path.join(
                model_dir, 
                f'repeat{repeat}_fold{fold}.hdf5'
            )
            
            y_pred, y_pred_proba, history = train_and_evaluate(
                model, X1_train, X2_train, y_train, 
                X1_test, X2_test, y_test,
                checkpoint_path, verbose=0
            )

            metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
            fold_results.append(metrics)
            print_metrics(metrics, fold)

            tf.keras.backend.clear_session()
        
        all_results.append(fold_results)

        temp_flat_results = []
        for r_idx, r_res in enumerate(all_results):
            for f_idx, f_res in enumerate(r_res):
                row = f_res.copy()
                row['repeat'] = r_idx
                row['fold'] = f_idx
                temp_flat_results.append(row)
        
        temp_df = pd.DataFrame(temp_flat_results)
        temp_save_path = os.path.join(output_dir, f'{embedding}_{split_type}_intermediate.csv')
        temp_df.to_csv(temp_save_path, index=False)
        print(f"  [Safety] Intermediate results saved to {temp_save_path}")

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {n_repeats} repeats × 5 folds")
    print(f"{'='*60}")

    all_metrics = {}
    for repeat_results in all_results:
        for fold_metrics in repeat_results:
            for key, value in fold_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

    summary = {}
    for key, values in all_metrics.items():
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }

    print("\nMetric                 Mean ± Std")
    print("-" * 40)
    for key in ['auc', 'accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro']:
        mean_val = summary[key]['mean']
        std_val = summary[key]['std']
        print(f"{key:20s}  {mean_val:.4f} ± {std_val:.4f}")

    results_df = pd.DataFrame(summary).T
    results_file = os.path.join(output_dir, f'{embedding}_{split_type}_results.csv')
    results_df.to_csv(results_file)
    print(f"\nResults saved to: {results_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='5-Fold Cross-Validation for TCR-Epitope Binding Prediction')
    parser.add_argument('--embedding', type=str, required=True,
                        help='Embedding type (e.g., catELMo, blosum62, etc.)')
    parser.add_argument('--split', type=str, required=True, choices=['tcr', 'epi'],
                        help='Split type: tcr or epi')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to directory containing embedding pickle files')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for models and results')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID(s)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_repeats', type=int, default=5,
                        help='Number of times to repeat 5-fold CV')
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run_5fold_analysis(
        embedding=args.embedding,
        split_type=args.split,
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        n_repeats=args.n_repeats
    )


if __name__ == '__main__':
    main()