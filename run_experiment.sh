#!/bin/bash
# run_experiment.sh
# Complete experimental pipeline for negative prompting calibration study
#
# Usage: ./run_experiment.sh [MODEL_SIZE]
# Example: ./run_experiment.sh 1.5B
#          ./run_experiment.sh 3B
#          ./run_experiment.sh 7B
#          ./run_experiment.sh all

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASET="huyxdang/MATH-dataset"
N_SAMPLES=1000
SEED=42
JUDGE_MODEL="gpt-4o-2024-08-06"
NUM_WORKERS=50
TENSOR_PARALLEL=1  # Adjust based on your GPU setup

# Model configurations
declare -A MODELS
MODELS=(
    ["1.5B"]="Qwen/Qwen2.5-1.5B-Instruct"
    ["3B"]="Qwen/Qwen2.5-3B-Instruct"
    ["7B"]="Qwen/Qwen2.5-7B-Instruct"
)

# Wrong question configurations
WRONG_CONFIGS=(10 100 500)

# Directory structure
BASE_DIR="$(pwd)"
DATA_DIR="${BASE_DIR}/data"
SAMPLED_DIR="${DATA_DIR}/sampled"
PREDICTIONS_DIR="${BASE_DIR}/predictions"
RESULTS_DIR="${BASE_DIR}/results"

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create directory structure
setup_directories() {
    print_step "Setting up directory structure..."
    mkdir -p "${SAMPLED_DIR}"
    mkdir -p "${PREDICTIONS_DIR}"
    mkdir -p "${RESULTS_DIR}"
    echo "  ✓ Created directories"
}

# Step 1: Sample datasets (run once)
sample_datasets() {
    print_header "STEP 1: STRATIFIED SAMPLING"
    
    TEST_SAMPLE="${SAMPLED_DIR}/math_test_sampled_${N_SAMPLES}.json"
    TRAIN_SAMPLE="${SAMPLED_DIR}/math_train_sampled_${N_SAMPLES}.json"
    
    if [ -f "$TEST_SAMPLE" ] && [ -f "$TRAIN_SAMPLE" ]; then
        print_info "Sampled datasets already exist, skipping..."
        return
    fi
    
    print_step "Sampling ${N_SAMPLES} examples from test and train splits..."
    python sample_dataset.py \
        --dataset "${DATASET}" \
        --n_samples ${N_SAMPLES} \
        --output_dir "${SAMPLED_DIR}" \
        --seed ${SEED}
    
    echo "  ✓ Sampling complete"
}

# Step 2: Run baseline inference
run_baseline() {
    local model_size=$1
    local model_name="${MODELS[$model_size]}"
    
    print_header "STEP 2: BASELINE INFERENCE - ${model_size}"
    
    local model_short=$(echo "$model_name" | tr '/' '_' | tr '.' '_' | tr '-' '_' | tr '[:upper:]' '[:lower:]')
    local test_sample="${SAMPLED_DIR}/math_test_sampled_${N_SAMPLES}.json"
    local output_file="${PREDICTIONS_DIR}/${model_short}_baseline_test.json"
    
    if [ -f "$output_file" ]; then
        print_info "Baseline predictions exist, skipping inference..."
    else
        print_step "Running baseline inference for ${model_size}..."
        python run_inference.py \
            --model_name "${model_name}" \
            --sampled_data "${test_sample}" \
            --output "${output_file}" \
            --tensor_parallel_size ${TENSOR_PARALLEL} \
            --batch_size 50
        echo "  ✓ Baseline inference complete"
    fi
    
    # Judge baseline predictions
    local judged_file="${PREDICTIONS_DIR}/${model_short}_baseline_test_judged.json"
    
    print_step "Judging baseline predictions..."
    python judge_predictions.py \
        --predictions "${output_file}" \
        --output "${judged_file}" \
        --judge "${JUDGE_MODEL}" \
        --num_workers ${NUM_WORKERS}
    
    echo "  ✓ Baseline judging complete"
    
    # Analyze baseline
    local baseline_results="${RESULTS_DIR}/${model_short}/baseline"
    mkdir -p "${baseline_results}"
    
    print_step "Analyzing baseline results..."
    python analyze_results.py \
        --predictions "${judged_file}" \
        --output_dir "${baseline_results}" \
        --n_bins 10
    
    echo "  ✓ Baseline analysis complete"
}

# Step 3: Filter wrong questions from train split
filter_wrong_questions() {
    local model_size=$1
    local model_name="${MODELS[$model_size]}"
    
    print_header "STEP 3: FILTER WRONG QUESTIONS - ${model_size}"
    
    local model_short=$(echo "$model_name" | tr '/' '_' | tr '.' '_' | tr '-' '_' | tr '[:upper:]' '[:lower:]')
    local train_sample="${SAMPLED_DIR}/math_train_sampled_${N_SAMPLES}.json"
    local train_pred="${PREDICTIONS_DIR}/${model_short}_train.json"
    local train_judged="${PREDICTIONS_DIR}/${model_short}_train_judged.json"
    local wrong_file="${PREDICTIONS_DIR}/${model_short}_wrong_questions.json"
    
    # First, run inference on train split if not done
    if [ ! -f "$train_pred" ]; then
        print_step "Running inference on train split..."
        python run_inference.py \
            --model_name "${model_name}" \
            --sampled_data "${train_sample}" \
            --output "${train_pred}" \
            --tensor_parallel_size ${TENSOR_PARALLEL} \
            --batch_size 50
    fi
    
    # Judge train predictions
    if [ ! -f "$train_judged" ]; then
        print_step "Judging train predictions..."
        python judge_predictions.py \
            --predictions "${train_pred}" \
            --output "${train_judged}" \
            --judge "${JUDGE_MODEL}" \
            --num_workers ${NUM_WORKERS}
    fi
    
    # Filter wrong questions
    print_step "Filtering wrong questions..."
    python filter_wrong.py \
        --judged_predictions "${train_judged}" \
        --output "${wrong_file}"
    
    echo "  ✓ Wrong question filtering complete"
}

# Step 4: Run negative prompting experiments
run_negative_prompting() {
    local model_size=$1
    local n_wrong=$2
    local model_name="${MODELS[$model_size]}"
    
    print_header "STEP 4: NEGATIVE PROMPTING - ${model_size} (N=${n_wrong})"
    
    local model_short=$(echo "$model_name" | tr '/' '_' | tr '.' '_' | tr '-' '_' | tr '[:upper:]' '[:lower:]')
    local test_sample="${SAMPLED_DIR}/math_test_sampled_${N_SAMPLES}.json"
    local wrong_file="${PREDICTIONS_DIR}/${model_short}_wrong_questions.json"
    local output_file="${PREDICTIONS_DIR}/${model_short}_negprompt_${n_wrong}_test.json"
    
    if [ -f "$output_file" ]; then
        print_info "Negative prompting predictions exist, skipping inference..."
    else
        print_step "Running negative prompting inference (${n_wrong} wrong examples)..."
        python run_inference.py \
            --model_name "${model_name}" \
            --sampled_data "${test_sample}" \
            --output "${output_file}" \
            --wrong_questions "${wrong_file}" \
            --n_wrong ${n_wrong} \
            --tensor_parallel_size ${TENSOR_PARALLEL} \
            --batch_size 50
        echo "  ✓ Negative prompting inference complete"
    fi
    
    # Judge predictions
    local judged_file="${PREDICTIONS_DIR}/${model_short}_negprompt_${n_wrong}_test_judged.json"
    
    print_step "Judging negative prompting predictions..."
    python judge_predictions.py \
        --predictions "${output_file}" \
        --output "${judged_file}" \
        --judge "${JUDGE_MODEL}" \
        --num_workers ${NUM_WORKERS}
    
    echo "  ✓ Negative prompting judging complete"
    
    # Analyze results
    local exp_results="${RESULTS_DIR}/${model_short}/negative_${n_wrong}"
    mkdir -p "${exp_results}"
    
    print_step "Analyzing negative prompting results..."
    python analyze_results.py \
        --predictions "${judged_file}" \
        --output_dir "${exp_results}" \
        --n_bins 10
    
    echo "  ✓ Negative prompting analysis complete"
}

# Step 5: Compare all experiments
compare_experiments() {
    local model_size=$1
    local model_name="${MODELS[$model_size]}"
    
    print_header "STEP 5: COMPARE EXPERIMENTS - ${model_size}"
    
    local model_short=$(echo "$model_name" | tr '/' '_' | tr '.' '_' | tr '-' '_' | tr '[:upper:]' '[:lower:]')
    local model_results="${RESULTS_DIR}/${model_short}"
    
    # Build experiment names
    local exp_names="baseline"
    for n_wrong in "${WRONG_CONFIGS[@]}"; do
        exp_names="${exp_names} negative_${n_wrong}"
    done
    
    print_step "Comparing all experiments..."
    python analyze_results.py \
        --compare \
        --results_dir "${model_results}" \
        --experiment_names ${exp_names}
    
    echo "  ✓ Comparison complete"
}

# Main experiment pipeline
run_full_pipeline() {
    local model_size=$1
    
    print_header "FULL EXPERIMENTAL PIPELINE: ${model_size}"
    print_info "Model: ${MODELS[$model_size]}"
    print_info "Wrong configurations: ${WRONG_CONFIGS[*]}"
    echo ""
    
    # Run all steps
    run_baseline "$model_size"
    filter_wrong_questions "$model_size"
    
    # Run negative prompting for each configuration
    for n_wrong in "${WRONG_CONFIGS[@]}"; do
        run_negative_prompting "$model_size" "$n_wrong"
    done
    
    # Compare all experiments
    compare_experiments "$model_size"
    
    print_header "PIPELINE COMPLETE: ${model_size}"
}

# Main execution
main() {
    local model_arg=${1:-"all"}
    
    print_header "NEGATIVE PROMPTING CALIBRATION EXPERIMENT"
    echo ""
    print_info "Dataset: ${DATASET}"
    print_info "Samples: ${N_SAMPLES}"
    print_info "Judge: ${JUDGE_MODEL}"
    print_info "Models: ${!MODELS[@]}"
    echo ""
    
    # Setup
    setup_directories
    
    # Sample datasets (only once)
    sample_datasets
    
    # Run experiments
    if [ "$model_arg" == "all" ]; then
        print_info "Running experiments for all models..."
        for size in "${!MODELS[@]}"; do
            run_full_pipeline "$size"
        done
    elif [ -n "${MODELS[$model_arg]}" ]; then
        print_info "Running experiment for ${model_arg}..."
        run_full_pipeline "$model_arg"
    else
        print_error "Unknown model size: ${model_arg}"
        print_info "Available sizes: ${!MODELS[@]}"
        exit 1
    fi
    
    print_header "ALL EXPERIMENTS COMPLETE"
    print_info "Results saved to: ${RESULTS_DIR}"
    echo ""
    echo "Next steps:"
    echo "  1. Review results in ${RESULTS_DIR}"
    echo "  2. Check reliability diagrams (*.png files)"
    echo "  3. Compare metrics across models and conditions"
}

# Run main with arguments
main "$@"
