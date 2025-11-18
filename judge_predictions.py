"""
Judge model predictions using OpenAI API.
Simplified version compatible with MATH dataset structure.
"""
import os
import json
import copy
import argparse
import asyncio
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


client = AsyncOpenAI(timeout=300.0, max_retries=1)


JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    strict: Literal[True]  # 100% reliability


async def extract_answer(question, correct_answer, response, judge_model, retry_count=0, max_retries=3):
    """Use LLM to judge if the response is correct."""
    
    # Small delay to avoid rate limits
    await asyncio.sleep(0.1)
    
    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response
    )
    
    try:
        response = await client.beta.chat.completions.parse(
            model=judge_model,
            max_completion_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=ExtractedAnswer,
        )
        content = response.choices[0].message.parsed
        return {
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct
        }
    except Exception as e:
        error_str = str(e)
        
        # Check if it's a rate limit error
        if "rate_limit" in error_str.lower() or "429" in error_str:
            if retry_count < max_retries:
                import random
                base_wait = 2.0
                wait_time = (2 ** retry_count) * base_wait + random.uniform(0, 1)
                
                print(f"  ⚠ Rate limit hit (attempt {retry_count + 1}/{max_retries}). Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                return await extract_answer(question, correct_answer, response, judge_model, retry_count + 1, max_retries)
            else:
                print(f"  ✗ Rate limit exceeded after {max_retries} retries")
                return None
        else:
            print(f"  ✗ Error in judge: {e}")
            return None


async def judge_prediction(unique_id, prediction, judge_model):
    """Judge a single prediction."""
    
    if "judge_response" in prediction:  # already judged
        return unique_id, prediction
    
    question = prediction["question"]
    correct_answer = prediction["correct_answer"]
    response = prediction["response"]
    
    # Get judge result
    judge_result = await extract_answer(question, correct_answer, response, judge_model)
    
    if judge_result is None:
        return None, None
    
    # Update prediction
    prediction["judge_response"] = judge_result
    
    return unique_id, prediction


async def judge_all_predictions(predictions, judge_model, num_workers):
    """Judge all predictions in parallel."""
    
    async def bound_func(unique_id, prediction):
        async with semaphore:
            result = await judge_prediction(unique_id, prediction, judge_model)
            return result
    
    semaphore = asyncio.Semaphore(num_workers)
    
    # Create tasks for all predictions
    tasks = []
    for unique_id, prediction in predictions.items():
        tasks.append(bound_func(unique_id, prediction))
    
    # Run all tasks with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Judging predictions")
    
    return results


async def main(args):
    """Main function to judge predictions."""
    
    print(f"\n{'='*60}")
    print(f"Judging predictions")
    print(f"{'='*60}\n")
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Set output file
    if args.output is None:
        base_name = os.path.splitext(args.predictions)[0]
        args.output = f"{base_name}_judged.json"
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load existing judged predictions if resuming
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            judged_predictions = json.load(f)
        print(f"Resuming from existing file with {len(judged_predictions)} judged predictions")
        # Merge with loaded predictions
        for idx, judged_pred in judged_predictions.items():
            if idx in predictions:
                predictions[idx] = judged_pred
    else:
        judged_predictions = {}
    
    # Count how many need judging
    to_judge = sum(1 for p in predictions.values() if "judge_response" not in p)
    print(f"\nPredictions to judge: {to_judge}/{len(predictions)}")
    
    if to_judge == 0:
        print("✓ All predictions already judged!")
    else:
        print(f"\n{'='*60}")
        print(f"Starting judging process")
        print(f"{'='*60}")
        print(f"  Workers: {args.num_workers}")
        print(f"  Judge model: {args.judge}")
        print(f"  Predictions to judge: {to_judge}")
        print(f"{'='*60}\n")
        
        # Judge all predictions
        results = await judge_all_predictions(
            predictions,
            args.judge,
            args.num_workers
        )
        
        # Update predictions with results
        updated_count = 0
        for unique_id, prediction in results:
            if unique_id is not None and prediction is not None:
                predictions[unique_id] = prediction
                updated_count += 1
        
        print(f"\n{'='*60}")
        print(f"Judging complete!")
        print(f"{'='*60}")
        print(f"  Updated predictions: {updated_count}")
        print(f"{'='*60}\n")
    
    # Save results
    print(f"Saving results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✓ Saved successfully!")
    
    # Calculate accuracy
    total = 0
    correct = 0
    
    for pred in predictions.values():
        if "judge_response" in pred:
            total += 1
            if pred["judge_response"].get("correct", "").lower() == "yes":
                correct += 1
    
    if total > 0:
        accuracy = correct / total * 100
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Total predictions: {total}")
        print(f"Correct: {correct}")
        print(f"Incorrect: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge model predictions using OpenAI API"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for judged predictions (default: {predictions}_judged.json)"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="gpt-4o-2024-08-06",
        help="Judge model to use (default: gpt-4o-2024-08-06)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=50,
        help="Number of parallel workers for judging (default: 50)"
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))
