"""
Analyze model predictions with comprehensive metrics:
- Accuracy (with confidence intervals)
- Calibration Error (ECE with binning)
- Overconfidence Rate
- Reliability Diagrams
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For server environments
from scipy import stats
import os


def load_predictions(predictions_file):
    """Load predictions and extract confidence + correctness."""
    print(f"Loading predictions from: {predictions_file}")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    data = []
    missing_confidence = 0
    missing_judge = 0
    
    for unique_id, pred in predictions.items():
        # Check if judged
        if "judge_response" not in pred:
            missing_judge += 1
            continue
        
        judge_response = pred["judge_response"]
        is_correct = judge_response.get("correct", "").lower() == "yes"
        
        # Get confidence from judge (already extracted)
        confidence = judge_response.get("confidence")
        
        if confidence is None:
            missing_confidence += 1
            continue
        
        # Already 0-100, convert to [0, 1]
        confidence = np.clip(confidence, 0, 100) / 100.0
        
        data.append({
            "id": unique_id,
            "confidence": confidence,
            "correct": 1 if is_correct else 0,
            "level": pred.get("level", "unknown"),
            "type": pred.get("type", "unknown"),
            "question": pred.get("question", ""),
            "response": pred.get("response", "")
        })
    
    print(f"Loaded {len(data)} predictions")
    if missing_judge > 0:
        print(f"⚠ Skipped {missing_judge} predictions missing judge_response")
    if missing_confidence > 0:
        print(f"⚠ Skipped {missing_confidence} predictions missing confidence scores")
    
    return data


def calculate_accuracy(data):
    """Calculate accuracy with 95% confidence interval (Wald)."""
    n = len(data)
    correct = sum(d['correct'] for d in data)
    accuracy = correct / n * 100 if n > 0 else 0
    
    # Wald confidence interval
    p = accuracy / 100
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
    ci_half_width = 1.96 * se * 100
    
    return {
        "accuracy": float(accuracy),
        "correct": int(correct),
        "total": int(n),
        "ci_lower": float(accuracy - ci_half_width),
        "ci_upper": float(accuracy + ci_half_width),
        "ci_half_width": float(ci_half_width)
    }


def calculate_ece(confidences, correct, n_bins=10, use_percentile=True):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        confidences: Array of confidence scores [0, 1]
        correct: Array of binary correctness (0 or 1)
        n_bins: Number of bins
        use_percentile: If True, use percentile-based binning; else equal-width
    
    Returns:
        ECE value and bin information
    """
    confidences = np.array(confidences)
    correct = np.array(correct)
    
    # Create bins
    if use_percentile:
        # Percentile-based bins (equal number of samples per bin)
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(confidences, percentiles)
        bin_edges = np.unique(bin_edges)  # Remove duplicates
    else:
        # Equal-width bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
    
    # Assign to bins
    bin_indices = np.digitize(confidences, bin_edges[1:-1])
    
    # Calculate ECE and collect bin info
    ece = 0.0
    bin_info = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if not mask.any():
            continue
        
        bin_confidences = confidences[mask]
        bin_correct = correct[mask]
        
        avg_confidence = bin_confidences.mean()
        avg_accuracy = bin_correct.mean()
        n_samples = len(bin_confidences)
        
        # Weighted by number of samples
        weight = n_samples / len(confidences)
        ece += weight * abs(avg_confidence - avg_accuracy)
        
        bin_info.append({
            "bin_index": int(i),
            "avg_confidence": float(avg_confidence),
            "avg_accuracy": float(avg_accuracy),
            "n_samples": int(n_samples),
            "gap": float(abs(avg_confidence - avg_accuracy))
        })
    
    return float(ece), bin_info


def calculate_overconfidence_rate(confidences, correct):
    """
    Calculate overconfidence rate: fraction of predictions where
    confidence > 0.5 but the answer is wrong.
    
    Also calculate average overconfidence gap.
    """
    confidences = np.array(confidences)
    correct = np.array(correct)
    
    # Predictions where model was confident (> 50%)
    confident_mask = confidences > 0.5
    n_confident = confident_mask.sum()
    
    if n_confident == 0:
        return {
            "overconfidence_rate": 0.0,
            "n_confident": 0,
            "n_overconfident": 0,
            "avg_overconfidence_gap": 0.0
        }
    
    # Of those confident predictions, how many were wrong?
    overconfident_mask = confident_mask & (correct == 0)
    n_overconfident = int(overconfident_mask.sum())
    overconfidence_rate = float(n_overconfident / n_confident * 100)
    
    # Average gap for overconfident predictions
    if n_overconfident > 0:
        overconfident_confidences = confidences[overconfident_mask]
        # Gap is confidence - 0 (since they're wrong)
        avg_gap = float(overconfident_confidences.mean())
    else:
        avg_gap = 0.0
    
    return {
        "overconfidence_rate": float(overconfidence_rate),
        "n_confident": int(n_confident),
        "n_overconfident": int(n_overconfident),
        "avg_overconfidence_gap": float(avg_gap)
    }


def plot_reliability_diagram(bin_info, output_file, title="Reliability Diagram"):
    """
    Create reliability diagram (calibration plot).
    
    Shows average confidence vs. average accuracy for each bin.
    Perfect calibration would be on the diagonal y=x line.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract data
    avg_confidences = [b['avg_confidence'] for b in bin_info]
    avg_accuracies = [b['avg_accuracy'] for b in bin_info]
    n_samples = [b['n_samples'] for b in bin_info]
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Plot bins (size proportional to number of samples)
    max_samples = max(n_samples)
    sizes = [n / max_samples * 300 for n in n_samples]
    
    scatter = ax.scatter(avg_confidences, avg_accuracies, s=sizes, alpha=0.6, 
                        c=avg_confidences, cmap='RdYlGn', edgecolors='black', linewidth=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidence', fontsize=12)
    
    # Connect points
    sorted_indices = np.argsort(avg_confidences)
    sorted_conf = np.array(avg_confidences)[sorted_indices]
    sorted_acc = np.array(avg_accuracies)[sorted_indices]
    ax.plot(sorted_conf, sorted_acc, 'b-', alpha=0.3, linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('Confidence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add text with bin sizes
    ax.text(0.05, 0.95, f'Bins: {len(bin_info)}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved reliability diagram to: {output_file}")


def plot_confidence_histogram(confidences, correct, output_file, title="Confidence Distribution"):
    """Plot histogram of confidence scores, split by correct/incorrect."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    confidences = np.array(confidences)
    correct = np.array(correct)
    
    correct_conf = confidences[correct == 1]
    incorrect_conf = confidences[correct == 0]
    
    bins = np.linspace(0, 1, 21)
    
    ax.hist(correct_conf, bins=bins, alpha=0.6, label='Correct', color='green', edgecolor='black')
    ax.hist(incorrect_conf, bins=bins, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    
    ax.set_xlabel('Confidence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confidence histogram to: {output_file}")


def analyze_predictions(predictions_file, output_dir, n_bins=10):
    """
    Comprehensive analysis of predictions.
    
    Generates:
    - Metrics JSON with accuracy, ECE, overconfidence rate
    - Reliability diagram
    - Confidence histogram
    - Detailed bin information
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_predictions(predictions_file)
    
    if len(data) == 0:
        print("❌ No valid predictions found!")
        return
    
    confidences = [d['confidence'] for d in data]
    correct = [d['correct'] for d in data]
    
    print(f"\n{'='*60}")
    print("CALCULATING METRICS")
    print(f"{'='*60}")
    
    # 1. Accuracy
    acc_metrics = calculate_accuracy(data)
    print(f"Accuracy: {acc_metrics['accuracy']:.2f}% ± {acc_metrics['ci_half_width']:.2f}%")
    print(f"  ({acc_metrics['correct']}/{acc_metrics['total']} correct)")
    
    # 2. Expected Calibration Error
    ece, bin_info = calculate_ece(confidences, correct, n_bins=n_bins, use_percentile=True)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f} ({ece*100:.2f}%)")
    print(f"  Bins: {len(bin_info)} (percentile-based)")
    
    # 3. Overconfidence Rate
    overconf_metrics = calculate_overconfidence_rate(confidences, correct)
    print(f"\nOverconfidence Rate: {overconf_metrics['overconfidence_rate']:.2f}%")
    print(f"  Confident predictions (>50%): {overconf_metrics['n_confident']}")
    print(f"  Overconfident (confident but wrong): {overconf_metrics['n_overconfident']}")
    if overconf_metrics['n_overconfident'] > 0:
        print(f"  Avg overconfidence gap: {overconf_metrics['avg_overconfidence_gap']:.2f}")
    
    # 4. Confidence statistics
    conf_array = np.array(confidences)
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {conf_array.mean():.3f}")
    print(f"  Median: {np.median(conf_array):.3f}")
    print(f"  Std: {conf_array.std():.3f}")
    print(f"  Min: {conf_array.min():.3f}")
    print(f"  Max: {conf_array.max():.3f}")
    
    print(f"{'='*60}\n")
    
    # Save metrics
    metrics = {
        "accuracy": acc_metrics,
        "calibration": {
            "ece": float(ece),  # Add float()
            "ece_percent": float(ece * 100),  # Add float()
            "n_bins": int(len(bin_info)),
            "bin_info": bin_info
        },
        "overconfidence": overconf_metrics,
        "confidence_stats": {
            "mean": float(conf_array.mean()),
            "median": float(np.median(conf_array)),
            "std": float(conf_array.std()),
            "min": float(conf_array.min()),
            "max": float(conf_array.max())
        },
        "n_samples": int(len(data))  # Add int()
    }
    
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_file}")
    
    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    
    # Reliability diagram
    reliability_file = os.path.join(output_dir, "reliability_diagram.png")
    plot_reliability_diagram(bin_info, reliability_file, 
                           title=f"Reliability Diagram (ECE={ece:.4f})")
    
    # Confidence histogram
    histogram_file = os.path.join(output_dir, "confidence_histogram.png")
    plot_confidence_histogram(confidences, correct, histogram_file)
    
    print(f"{'='*60}\n")
    
    return metrics


def compare_experiments(baseline_dir, experiment_dirs, output_file):
    """
    Compare metrics across multiple experiments.
    
    Args:
        baseline_dir: Directory with baseline metrics
        experiment_dirs: List of (name, directory) tuples for experiments
        output_file: Output file for comparison table
    """
    print(f"\n{'='*70}")
    print("COMPARING EXPERIMENTS")
    print(f"{'='*70}\n")
    
    # Load baseline
    baseline_metrics_file = os.path.join(baseline_dir, "metrics.json")
    with open(baseline_metrics_file, 'r') as f:
        baseline_metrics = json.load(f)
    
    comparison = {
        "Baseline": {
            "accuracy": baseline_metrics["accuracy"]["accuracy"],
            "ece": baseline_metrics["calibration"]["ece_percent"],
            "overconf_rate": baseline_metrics["overconfidence"]["overconfidence_rate"],
            "n_samples": baseline_metrics["n_samples"]
        }
    }
    
    # Load experiment metrics
    for name, exp_dir in experiment_dirs:
        metrics_file = os.path.join(exp_dir, "metrics.json")
        if not os.path.exists(metrics_file):
            print(f"⚠ Metrics not found for {name}, skipping")
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        comparison[name] = {
            "accuracy": metrics["accuracy"]["accuracy"],
            "ece": metrics["calibration"]["ece_percent"],
            "overconf_rate": metrics["overconfidence"]["overconfidence_rate"],
            "n_samples": metrics["n_samples"]
        }
    
    # Print comparison table
    print(f"{'Experiment':<30} {'Accuracy':<12} {'ECE':<12} {'Overconf %':<12} {'N':<8}")
    print("=" * 74)
    
    for name, metrics in comparison.items():
        print(f"{name:<30} {metrics['accuracy']:>10.2f}% {metrics['ece']:>10.2f}% "
              f"{metrics['overconf_rate']:>10.2f}% {metrics['n_samples']:>6d}")
    
    # Calculate deltas from baseline
    print(f"\n{'Δ from Baseline':<30} {'Accuracy':<12} {'ECE':<12} {'Overconf %':<12}")
    print("=" * 66)
    
    baseline_acc = comparison["Baseline"]["accuracy"]
    baseline_ece = comparison["Baseline"]["ece"]
    baseline_overconf = comparison["Baseline"]["overconf_rate"]
    
    for name, metrics in comparison.items():
        if name == "Baseline":
            continue
        
        delta_acc = metrics["accuracy"] - baseline_acc
        delta_ece = metrics["ece"] - baseline_ece
        delta_overconf = metrics["overconf_rate"] - baseline_overconf
        
        print(f"{name:<30} {delta_acc:>+10.2f}% {delta_ece:>+10.2f}% {delta_overconf:>+10.2f}%")
    
    # Save comparison
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Saved comparison to: {output_file}")
    print(f"{'='*70}\n")


def main(args):
    if args.compare:
        # Comparison mode
        experiment_dirs = []
        for exp_name in args.experiment_names:
            exp_dir = os.path.join(args.results_dir, exp_name)
            experiment_dirs.append((exp_name, exp_dir))
        
        baseline_dir = os.path.join(args.results_dir, "baseline")
        output_file = os.path.join(args.results_dir, "comparison.json")
        
        compare_experiments(baseline_dir, experiment_dirs, output_file)
    
    else:
        # Single analysis mode
        print(f"\n{'='*70}")
        print("ANALYZING PREDICTIONS")
        print(f"{'='*70}")
        print(f"Input: {args.predictions}")
        print(f"Output dir: {args.output_dir}")
        print(f"Number of bins: {args.n_bins}")
        print(f"{'='*70}\n")
        
        analyze_predictions(args.predictions, args.output_dir, args.n_bins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze model predictions with comprehensive metrics"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to judged predictions JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of bins for calibration analysis"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple experiments"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Results directory containing subdirectories for each experiment"
    )
    parser.add_argument(
        "--experiment_names",
        type=str,
        nargs="+",
        help="Names of experiment subdirectories to compare"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.results_dir or not args.experiment_names:
            parser.error("--compare requires --results_dir and --experiment_names")
    else:
        if not args.predictions or not args.output_dir:
            parser.error("Single analysis requires --predictions and --output_dir")
    
    main(args)
