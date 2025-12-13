import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Set, Tuple
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import argparse


class DataSelector:
    def __init__(self, sequences: List[str], random_seed: int = 42):
        self.sequences = sequences
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def random_selection(self, target_fraction: float) -> List[int]:
        n_samples = int(len(self.sequences) * target_fraction)
        selected_idx = np.random.choice(len(self.sequences), size=n_samples, replace=False)
        return sorted(selected_idx.tolist())

    def diversity_based_selection(self, target_fraction: float, k: int = 3) -> List[int]:
        print(f"Computing k-mer diversity (k={k})...")

        # Get all possible k-mers in the dataset
        all_kmers = set()
        seq_kmers = []

        for seq in self.sequences:
            kmers = set(self._get_kmers(seq, k))
            seq_kmers.append(kmers)
            all_kmers.update(kmers)

        print(f"Total unique {k}-mers: {len(all_kmers)}")

        n_samples = int(len(self.sequences) * target_fraction)
        selected_idx = []
        covered_kmers = set()
        available = list(enumerate(seq_kmers))

        for _ in range(n_samples):
            if not available:
                break

            best_idx = 0
            best_new_kmers = 0

            for i, (idx, kmers) in enumerate(available):
                new_kmers = len(kmers - covered_kmers)
                if new_kmers > best_new_kmers:
                    best_new_kmers = new_kmers
                    best_idx = i

            selected_seq_idx, selected_kmers = available.pop(best_idx)
            selected_idx.append(selected_seq_idx)
            covered_kmers.update(selected_kmers)

            if (_ + 1) % 1000 == 0:
                print(f"  Selected {_ + 1}/{n_samples}, k-mer coverage: {len(covered_kmers)}/{len(all_kmers)}")

        print(f"Final k-mer coverage: {len(covered_kmers)}/{len(all_kmers)} ({len(covered_kmers)/len(all_kmers)*100:.2f}%)")

        return sorted(selected_idx)

    def length_stratified_selection(self, target_fraction: float) -> List[int]:
        print("Performing length-stratified selection...")

        length_groups = defaultdict(list)
        for idx, seq in enumerate(self.sequences):
            length_groups[len(seq)].append(idx)

        n_samples = int(len(self.sequences) * target_fraction)
        selected_idx = []
        total_seqs = len(self.sequences)
        for length, indices in sorted(length_groups.items()):
            group_fraction = len(indices) / total_seqs
            n_from_group = int(n_samples * group_fraction)
            if n_from_group > 0:
                sampled = np.random.choice(indices, size=min(n_from_group, len(indices)), replace=False)
                selected_idx.extend(sampled)
                print(f"  Length {length}: {len(indices)} seqs -> selected {len(sampled)}")

        if len(selected_idx) < n_samples:
            remaining = n_samples - len(selected_idx)
            available = [i for i in range(len(self.sequences)) if i not in selected_idx]
            additional = np.random.choice(available, size=remaining, replace=False)
            selected_idx.extend(additional)

        return sorted(selected_idx)

    def rare_kmer_selection(self, target_fraction: float, k: int = 3,
                           percentile: int = 25) -> List[int]:
        print(f"Computing rare k-mer enrichment (k={k}, percentile={percentile})...")

        kmer_counts = Counter()
        seq_kmers = []

        for seq in self.sequences:
            kmers = self._get_kmers(seq, k)
            seq_kmers.append(kmers)
            kmer_counts.update(kmers)

        counts = list(kmer_counts.values())
        threshold = np.percentile(counts, percentile)
        rare_kmers = {kmer for kmer, count in kmer_counts.items() if count <= threshold}

        print(f"Rare k-mers (count ≤ {threshold}): {len(rare_kmers)}/{len(kmer_counts)}")

        scores = []
        for idx, kmers in enumerate(seq_kmers):
            rare_count = sum(1 for kmer in kmers if kmer in rare_kmers)
            scores.append((idx, rare_count))

        scores.sort(key=lambda x: x[1], reverse=True)
        n_samples = int(len(self.sequences) * target_fraction)
        selected_idx = [idx for idx, score in scores[:n_samples]]

        avg_rare_selected = np.mean([score for _, score in scores[:n_samples]])
        avg_rare_overall = np.mean([score for _, score in scores])
        print(f"Average rare k-mers - Selected: {avg_rare_selected:.2f}, Overall: {avg_rare_overall:.2f}")

        return sorted(selected_idx)

    def cluster_based_selection(self, target_fraction: float, k: int = 3) -> List[int]:
        print(f"Performing cluster-based selection (k={k})...")

        all_kmers = set()
        for seq in self.sequences:
            all_kmers.update(self._get_kmers(seq, k))

        kmer_to_idx = {kmer: i for i, kmer in enumerate(sorted(all_kmers))}
        print(f"Feature dimension: {len(all_kmers)}")

        max_seqs_for_clustering = 10000
        if len(self.sequences) > max_seqs_for_clustering:
            print(f"Dataset too large, sampling {max_seqs_for_clustering} sequences for clustering...")
            sample_idx = np.random.choice(len(self.sequences), size=max_seqs_for_clustering, replace=False)
            sequences_to_cluster = [self.sequences[i] for i in sample_idx]
            idx_mapping = sample_idx
        else:
            sequences_to_cluster = self.sequences
            idx_mapping = np.arange(len(self.sequences))

        features = np.zeros((len(sequences_to_cluster), len(kmer_to_idx)))
        for i, seq in enumerate(sequences_to_cluster):
            for kmer in self._get_kmers(seq, k):
                features[i, kmer_to_idx[kmer]] += 1

        row_sums = features.sum(axis=1, keepdims=True)
        features = features / (row_sums + 1e-10)

        n_clusters = max(10, int(len(sequences_to_cluster) * target_fraction / 10))
        print(f"Clustering into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        labels = kmeans.fit_predict(features)

        n_samples = int(len(self.sequences) * target_fraction)
        samples_per_cluster = n_samples // n_clusters

        selected_idx = []
        for cluster_id in range(n_clusters):
            cluster_members = np.where(labels == cluster_id)[0]
            n_select = min(samples_per_cluster, len(cluster_members))

            if len(cluster_members) > 0:
                cluster_features = features[cluster_members]
                center = kmeans.cluster_centers_[cluster_id]
                distances = pairwise_distances(cluster_features, center.reshape(1, -1)).flatten()
                closest = cluster_members[np.argsort(distances)[:n_select]]
                selected_idx.extend([idx_mapping[i] for i in closest])

        while len(selected_idx) < n_samples:
            available = [i for i in range(len(self.sequences)) if i not in selected_idx]
            if not available:
                break
            selected_idx.append(np.random.choice(available))

        print(f"Selected {len(selected_idx)} sequences from {n_clusters} clusters")

        return sorted(selected_idx[:n_samples])

    def _get_kmers(self, sequence: str, k: int) -> List[str]:
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

    def save_selection(self, selected_indices: List[int], output_file: str):
        selected_sequences = [self.sequences[i] for i in selected_indices]
        with open(output_file, 'w') as f:
            f.write('\n'.join(selected_sequences))
        print(f"Saved {len(selected_sequences)} sequences to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Select informative TCR sequences')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file with TCR sequences')
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['random', 'diversity', 'length_stratified',
                               'rare_kmer', 'cluster'],
                       help='Selection strategy')
    parser.add_argument('--target_fraction', type=float, default=0.3,
                       help='Fraction of data to select (default: 0.3)')
    parser.add_argument('--k', type=int, default=3,
                       help='K-mer size for k-mer based strategies (default: 3)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file for selected sequences')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Load sequences
    print(f"Loading sequences from {args.input}...")
    with open(args.input, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(sequences)} sequences")

    # Initialize selector
    selector = DataSelector(sequences, random_seed=args.seed)

    # Apply selection strategy
    print(f"\nApplying {args.strategy} selection strategy...")
    print(f"Target: {int(args.target_fraction * 100)}% of data ({int(len(sequences) * args.target_fraction)} sequences)\n")

    if args.strategy == 'random':
        selected_idx = selector.random_selection(args.target_fraction)
    elif args.strategy == 'diversity':
        selected_idx = selector.diversity_based_selection(args.target_fraction, k=args.k)
    elif args.strategy == 'length_stratified':
        selected_idx = selector.length_stratified_selection(args.target_fraction)
    elif args.strategy == 'rare_kmer':
        selected_idx = selector.rare_kmer_selection(args.target_fraction, k=args.k)
    elif args.strategy == 'cluster':
        selected_idx = selector.cluster_based_selection(args.target_fraction, k=args.k)

    # Save selection
    selector.save_selection(selected_idx, args.output)

    print(f"\n=== Selection Complete ===")
    print(f"Strategy: {args.strategy}")
    print(f"Selected: {len(selected_idx)}/{len(sequences)} ({len(selected_idx)/len(sequences)*100:.2f}%)")


if __name__ == "__main__":
    main()
