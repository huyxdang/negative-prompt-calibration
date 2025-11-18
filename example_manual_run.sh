#!/bin/bash
# example_manual_run.sh
# Manual step-by-step example for running a single experiment

set -e

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
N_SAMPLES=100  # Smaller for quick test

echo "========================================="
echo "MANUAL EXPERIMENT EXAMPLE"
echo "========================================="
echo "This is a small example with ${N_SAMPLES} samples for quick testing"
echo ""

# Step 1: Sample data
echo "[1/7] Sampling datasets..."
python sample_dataset.py \
    --dataset huyxdang/MATH-dataset \
    --n_samples ${N_SAMPLES} \
    --output_dir sampled_data \
    --seed 42

# Step 2: Run baseline on test
echo ""
echo "[2/7] Running baseline inference on test set..."
python run_inference.py \
    --model_name "${MODEL}" \
    --sampled_data sampled_data/math_test_sampled_${N_SAMPLES}.json \
    --output predictions/example_baseline_test.json \
    --batch_size 20

# Step 3: Judge baseline
echo ""
echo "[3/7] Judging baseline predictions..."
python judge_predictions.py \
    --predictions predictions/example_baseline_test.json \
    --output predictions/example_baseline_test_judged.json \
    --num_workers 20

# Step 4: Run on train to get wrong examples
echo ""
echo "[4/7] Running inference on train set..."
python run_inference.py \
    --model_name "${MODEL}" \
    --sampled_data sampled_data/math_train_sampled_${N_SAMPLES}.json \
    --output predictions/example_train.json \
    --batch_size 20

# Step 5: Judge train and filter wrong
echo ""
echo "[5/7] Judging train predictions and filtering wrong..."
python judge_predictions.py \
    --predictions predictions/example_train.json \
    --output predictions/example_train_judged.json \
    --num_workers 20

python filter_wrong.py \
    --judged_predictions predictions/example_train_judged.json \
    --output predictions/example_wrong_questions.json

# Step 6: Run negative prompting with 10 wrong examples
echo ""
echo "[6/7] Running negative prompting with 10 wrong examples..."
python run_inference.py \
    --model_name "${MODEL}" \
    --sampled_data sampled_data/math_test_sampled_${N_SAMPLES}.json \
    --output predictions/example_negprompt_10_test.json \
    --wrong_questions predictions/example_wrong_questions.json \
    --n_wrong 10 \
    --batch_size 20

python judge_predictions.py \
    --predictions predictions/example_negprompt_10_test.json \
    --output predictions/example_negprompt_10_test_judged.json \
    --num_workers 20

# Step 7: Analyze and compare
echo ""
echo "[7/7] Analyzing results..."
mkdir -p results/example

python analyze_results.py \
    --predictions predictions/example_baseline_test_judged.json \
    --output_dir results/example/baseline \
    --n_bins 5

python analyze_results.py \
    --predictions predictions/example_negprompt_10_test_judged.json \
    --output_dir results/example/negative_10 \
    --n_bins 5

echo ""
echo "========================================="
echo "EXAMPLE COMPLETE!"
echo "========================================="
echo "Results saved to:"
echo "  - results/example/baseline/"
echo "  - results/example/negative_10/"
echo ""
echo "View the metrics:"
echo "  cat results/example/baseline/metrics.json"
echo "  cat results/example/negative_10/metrics.json"
echo ""
echo "View the reliability diagrams:"
echo "  open results/example/baseline/reliability_diagram.png"
echo "  open results/example/negative_10/reliability_diagram.png"
