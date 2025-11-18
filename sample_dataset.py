"""
Sample 1000 examples from MATH dataset test and train splits using stratified sampling.
Stratifies by both difficulty level (1-5) and problem type (algebra, geometry, etc).
"""
import json
import argparse
from datasets import load_dataset
from collections import defaultdict
import random


def stratified_sample(dataset, n_samples, seed=42):
    """
    Stratified sampling by difficulty level AND problem type.
    
    Args:
        dataset: HuggingFace dataset
        n_samples: Total number of samples to draw
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled examples with their original indices
    """
    random.seed(seed)
    
    # Group by (level, type) pairs
    strata = defaultdict(list)
    
    for idx, example in enumerate(dataset):
        level = example['level']
        prob_type = example['type']
        strata_key = (level, prob_type)
        strata[strata_key].append((idx, example))
    
    print(f"\n{'='*60}")
    print("STRATIFICATION BREAKDOWN")
    print(f"{'='*60}")
    print(f"Total examples: {len(dataset)}")
    print(f"Number of strata: {len(strata)}")
    print("\nStrata distribution:")
    
    # Sort strata by size for display
    sorted_strata = sorted(strata.items(), key=lambda x: len(x[1]), reverse=True)
    for (level, prob_type), examples in sorted_strata:
        print(f"  Level {level} - {prob_type:20s}: {len(examples):4d} examples")
    
    # Calculate samples per stratum (proportional allocation)
    total_examples = len(dataset)
    samples_per_stratum = {}
    
    for strata_key, examples in strata.items():
        proportion = len(examples) / total_examples
        n_from_stratum = max(1, round(n_samples * proportion))  # At least 1 from each stratum
        samples_per_stratum[strata_key] = min(n_from_stratum, len(examples))
    
    # Adjust to exactly n_samples (handle rounding)
    current_total = sum(samples_per_stratum.values())
    
    if current_total != n_samples:
        diff = n_samples - current_total
        # Add/remove from largest strata
        sorted_strata_keys = sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True)
        
        for key in sorted_strata_keys:
            if diff == 0:
                break
            if diff > 0:
                # Need more samples
                if samples_per_stratum[key] < len(strata[key]):
                    samples_per_stratum[key] += 1
                    diff -= 1
            else:
                # Need fewer samples
                if samples_per_stratum[key] > 1:
                    samples_per_stratum[key] -= 1
                    diff += 1
    
    print(f"\nSampling strategy:")
    for (level, prob_type), n_to_sample in sorted(samples_per_stratum.items()):
        print(f"  Level {level} - {prob_type:20s}: sampling {n_to_sample:3d} / {len(strata[(level, prob_type)]):4d}")
    
    # Sample from each stratum
    sampled_examples = []
    
    for strata_key, n_to_sample in samples_per_stratum.items():
        stratum_examples = strata[strata_key]
        sampled = random.sample(stratum_examples, n_to_sample)
        sampled_examples.extend(sampled)
    
    # Shuffle final samples
    random.shuffle(sampled_examples)
    
    print(f"\n{'='*60}")
    print(f"Sampled {len(sampled_examples)} examples (target: {n_samples})")
    print(f"{'='*60}\n")
    
    return sampled_examples


def save_sampled_dataset(examples, output_file):
    """
    Save sampled examples to JSON file.
    
    Format: {
        "index": original_index,
        "id": unique_id,
        "problem": question,
        "solution": answer,
        "level": difficulty_level,
        "type": problem_type
    }
    """
    output_data = {}
    
    for idx, example in examples:
        # Create unique ID from original index
        unique_id = f"math_{idx}"
        
        output_data[unique_id] = {
            "index": idx,
            "id": unique_id,
            "problem": example['problem'],
            "solution": example['solution'],
            "level": example['level'],
            "type": example['type']
        }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved {len(output_data)} examples to: {output_file}")


def main(args):
    print(f"\n{'='*70}")
    print("STRATIFIED SAMPLING FOR MATH DATASET")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples per split: {args.n_samples}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*70}\n")
    
    # Load dataset
    print("Loading MATH dataset...")
    dataset = load_dataset(args.dataset)
    
    # Sample from test split
    print("\n" + "="*70)
    print("SAMPLING TEST SPLIT")
    print("="*70)
    test_examples = stratified_sample(dataset['test'], args.n_samples, args.seed)
    test_output = args.output_dir + f"/math_test_sampled_{args.n_samples}.json"
    save_sampled_dataset(test_examples, test_output)
    
    # Sample from train split
    print("\n" + "="*70)
    print("SAMPLING TRAIN SPLIT")
    print("="*70)
    train_examples = stratified_sample(dataset['train'], args.n_samples, args.seed)
    train_output = args.output_dir + f"/math_train_sampled_{args.n_samples}.json"
    save_sampled_dataset(train_examples, train_output)
    
    # Summary
    print("\n" + "="*70)
    print("SAMPLING COMPLETE")
    print("="*70)
    print(f"Test split:  {len(test_examples)} samples → {test_output}")
    print(f"Train split: {len(train_examples)} samples → {train_output}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified sampling from MATH dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="huyxdang/MATH-dataset",
        help="HuggingFace dataset path"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of samples to draw from each split"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sampled_data",
        help="Output directory for sampled datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
