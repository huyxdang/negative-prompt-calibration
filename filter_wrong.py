"""
Filter wrong questions from judged predictions.
Takes judged predictions and creates a dataset of only incorrect answers.
"""
import json
import argparse


def filter_wrong_questions(judged_file, output_file):
    """
    Filter and save only wrong predictions.
    
    Args:
        judged_file: Path to judged predictions JSON
        output_file: Path to save wrong questions
    
    Returns:
        Dictionary of wrong predictions
    """
    print(f"Loading judged predictions from: {judged_file}")
    with open(judged_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"Total predictions: {len(predictions)}")
    
    # Filter wrong predictions
    wrong_predictions = {}
    correct_count = 0
    
    for unique_id, pred in predictions.items():
        if "judge_response" not in pred:
            print(f"Warning: Prediction {unique_id} missing judge_response, skipping")
            continue
        
        judge_response = pred["judge_response"]
        is_correct = judge_response.get("correct", "").lower() == "yes"
        
        if not is_correct:
            wrong_predictions[unique_id] = {
                "id": unique_id,
                "question": pred["question"],
                "correct_answer": pred["correct_answer"],
                "model_answer": judge_response["model_answer"],
                "response": pred["response"],
                "reasoning": judge_response["reasoning"],
                "level": pred.get("level", "unknown"),
                "type": pred.get("type", "unknown")
            }
        else:
            correct_count += 1
    
    # Calculate statistics
    total = len(predictions)
    wrong_count = len(wrong_predictions)
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    print(f"\n{'='*60}")
    print("FILTERING RESULTS")
    print(f"{'='*60}")
    print(f"Total predictions: {total}")
    print(f"Correct: {correct_count} ({correct_count/total*100:.1f}%)")
    print(f"Wrong: {wrong_count} ({wrong_count/total*100:.1f}%)")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    # Breakdown by difficulty level
    wrong_by_level = {}
    for pred in wrong_predictions.values():
        level = pred.get("level", "unknown")
        wrong_by_level[level] = wrong_by_level.get(level, 0) + 1
    
    print("Wrong predictions by difficulty level:")
    for level in sorted(wrong_by_level.keys()):
        count = wrong_by_level[level]
        print(f"  Level {level}: {count} wrong")
    
    # Breakdown by problem type
    wrong_by_type = {}
    for pred in wrong_predictions.values():
        prob_type = pred.get("type", "unknown")
        wrong_by_type[prob_type] = wrong_by_type.get(prob_type, 0) + 1
    
    print("\nWrong predictions by problem type:")
    for prob_type in sorted(wrong_by_type.keys(), key=lambda x: wrong_by_type[x], reverse=True):
        count = wrong_by_type[prob_type]
        print(f"  {prob_type:20s}: {count} wrong")
    
    # Save wrong predictions
    with open(output_file, 'w') as f:
        json.dump(wrong_predictions, f, indent=2)
    
    print(f"\n✓ Saved {len(wrong_predictions)} wrong predictions to: {output_file}")
    
    return wrong_predictions


def main(args):
    print(f"\n{'='*70}")
    print("FILTERING WRONG QUESTIONS")
    print(f"{'='*70}")
    print(f"Input: {args.judged_predictions}")
    print(f"Output: {args.output}")
    print(f"{'='*70}\n")
    
    wrong_predictions = filter_wrong_questions(args.judged_predictions, args.output)
    
    # Show sample wrong predictions
    print(f"\n{'='*60}")
    print("SAMPLE WRONG PREDICTIONS")
    print(f"{'='*60}\n")
    
    sample_count = 0
    for unique_id, pred in wrong_predictions.items():
        if sample_count >= 3:
            break
        
        print(f"[{sample_count + 1}] ID: {unique_id}")
        print(f"Level: {pred['level']}, Type: {pred['type']}")
        print(f"Question: {pred['question'][:150]}...")
        print(f"Correct answer: {pred['correct_answer'][:100]}...")
        print(f"Model answer: {pred['model_answer'][:100]}...")
        print(f"Reasoning: {pred['reasoning'][:150]}...")
        print()
        sample_count += 1
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter wrong questions from judged predictions"
    )
    parser.add_argument(
        "--judged_predictions",
        type=str,
        required=True,
        help="Path to judged predictions JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file for wrong questions"
    )
    
    args = parser.parse_args()
    main(args)
