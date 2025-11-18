# Quick Start Guide

## Setup (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key (for the judge)
export OPENAI_API_KEY="sk-..."

# 3. Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"
```

## Three Ways to Run

### Option 1: Full Automated Pipeline (Recommended)
Runs everything automatically for all model sizes and conditions.

```bash
# Run for one model (fastest)
./run_experiment.sh 1.5B

# Run for all models (slow!)
./run_experiment.sh all
```

**Time**: 1-3 hours per model, 3-9 hours for all

### Option 2: Small Manual Example (For Testing)
Runs a quick 100-sample experiment to test the pipeline.

```bash
# Quick test with 100 samples
./example_manual_run.sh
```

**Time**: ~10-20 minutes

### Option 3: Step-by-Step Manual (For Debugging)
Run each step individually with full control.

```bash
# 1. Sample data
python sample_dataset.py --dataset huyxdang/MATH-dataset --n_samples 1000 --output_dir sampled_data

# 2. Baseline
python run_inference.py --model_name Qwen/Qwen2.5-1.5B-Instruct --sampled_data sampled_data/math_test_sampled_1000.json --output predictions/baseline.json

# 3. Judge
python judge_predictions.py --predictions predictions/baseline.json --output predictions/baseline_judged.json

# 4. Get wrong questions from train
python run_inference.py --model_name Qwen/Qwen2.5-1.5B-Instruct --sampled_data sampled_data/math_train_sampled_1000.json --output predictions/train.json
python judge_predictions.py --predictions predictions/train.json --output predictions/train_judged.json
python filter_wrong.py --judged_predictions predictions/train_judged.json --output predictions/wrong.json

# 5. Negative prompting
python run_inference.py --model_name Qwen/Qwen2.5-1.5B-Instruct --sampled_data sampled_data/math_test_sampled_1000.json --output predictions/neg10.json --wrong_questions predictions/wrong.json --n_wrong 10

# 6. Judge and analyze
python judge_predictions.py --predictions predictions/neg10.json --output predictions/neg10_judged.json
python analyze_results.py --predictions predictions/baseline_judged.json --output_dir results/baseline
python analyze_results.py --predictions predictions/neg10_judged.json --output_dir results/negative_10
```

## View Results

```bash
# View metrics
cat results/{model}/baseline/metrics.json
cat results/{model}/negative_10/metrics.json

# View comparison table
cat results/{model}/comparison.json

# View plots (if on local machine)
open results/{model}/baseline/reliability_diagram.png
open results/{model}/negative_10/reliability_diagram.png
```

## What to Look For

### Success indicators:
- ✅ ECE (Expected Calibration Error) **decreases** with more wrong examples
- ✅ Overconfidence rate **decreases**
- ✅ Reliability diagram shifts **toward the diagonal**
- ✅ Accuracy stays roughly the same or slightly improves

### Failure indicators:
- ❌ ECE increases or stays flat
- ❌ Overconfidence rate increases or stays flat
- ❌ Accuracy significantly drops
- ❌ Model becomes uniformly uncertain (confidence → 50% for everything)

## Example Output

```json
{
  "Baseline": {
    "accuracy": 45.2,
    "ece": 12.8,
    "overconf_rate": 38.5
  },
  "negative_10": {
    "accuracy": 44.8,
    "ece": 11.2,      // ✅ Better calibration
    "overconf_rate": 32.1  // ✅ Less overconfident
  },
  "negative_100": {
    "accuracy": 43.5,
    "ece": 9.8,       // ✅ Even better!
    "overconf_rate": 28.7  // ✅ Even less overconfident!
  }
}
```

## Troubleshooting

### GPU Out of Memory
```bash
# Use smaller model
./run_experiment.sh 1.5B

# Or reduce batch size
# Edit run_experiment.sh and change: batch_size=50 → batch_size=20
```

### OpenAI Rate Limits
```bash
# Reduce parallel workers
# Edit run_experiment.sh and change: NUM_WORKERS=50 → NUM_WORKERS=10
```

### Model doesn't output confidence
Check the prompt in `run_inference.py` and the regex patterns in `analyze_results.py` function `extract_confidence()`

## File Structure After Running

```
.
├── data/
│   └── sampled/                    # Sampled datasets
├── predictions/                    # Raw model outputs + judged versions
├── results/
│   └── {model}/
│       ├── baseline/
│       │   ├── metrics.json        # ← Main results
│       │   ├── reliability_diagram.png
│       │   └── confidence_histogram.png
│       ├── negative_10/
│       ├── negative_100/
│       ├── negative_500/
│       └── comparison.json         # ← Compare all conditions
└── [scripts]
```

## Questions?

1. Read the full [README.md](README.md) for detailed documentation
2. Check individual script help: `python {script}.py --help`
3. Look at the example: `cat example_manual_run.sh`

Good luck with your experiment! 🚀
