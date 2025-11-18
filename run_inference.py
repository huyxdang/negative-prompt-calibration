"""
Run inference on MATH dataset with optional negative prompting.
Supports baseline mode and negative prompting with wrong questions.
"""
import json
import argparse
import os
from vllm import LLM, SamplingParams


def load_sampled_data(sampled_file):
    """Load sampled dataset from JSON."""
    print(f"Loading sampled data from: {sampled_file}")
    with open(sampled_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data


def load_wrong_questions(wrong_file, n_wrong=None):
    """
    Load wrong questions for negative prompting.
    
    Args:
        wrong_file: Path to JSON file with wrong questions
        n_wrong: Number of wrong questions to include (None = all)
    
    Returns:
        List of (question, wrong_answer) tuples
    """
    print(f"Loading wrong questions from: {wrong_file}")
    with open(wrong_file, 'r') as f:
        wrong_data = json.load(f)
    
    wrong_examples = []
    for item in wrong_data.values():
        question = item['question']
        wrong_answer = item['model_answer']
        wrong_examples.append((question, wrong_answer))
    
    if n_wrong is not None and n_wrong < len(wrong_examples):
        wrong_examples = wrong_examples[:n_wrong]
        print(f"Using first {n_wrong} wrong questions")
    else:
        print(f"Using all {len(wrong_examples)} wrong questions")
    
    return wrong_examples


def create_negative_prompt_context(wrong_examples):
    """
    Create context string with wrong questions for system prompt.
    
    Format:
    These are examples of questions you previously answered incorrectly. 
    These represent the types of problems where you should be unconfident 
    since you got them wrong:
    
    1. Question: [question]
       Your wrong answer: [wrong_answer]
    
    2. Question: [question]
       Your wrong answer: [wrong_answer]
    ...
    """
    context = "These are examples of questions you previously answered incorrectly. "
    context += "These represent the types of problems where you should be unconfident since you got them wrong:\n\n"
    
    for i, (question, wrong_answer) in enumerate(wrong_examples, 1):
        context += f"{i}. Question: {question}\n"
        context += f"   Your wrong answer: {wrong_answer}\n\n"
    
    return context


def create_prompt(example, negative_context=None):
    """
    Create prompt for MATH problem with optional negative prompting context.
    
    Args:
        example: Dictionary with 'problem' field
        negative_context: Optional string with wrong examples
    
    Returns:
        Prompt string or messages list (for chat models)
    """
    problem = example['problem']
    
    base_prompt = """Solve the following math problem step by step.

At the end of your solution:
1. Provide your final answer in LaTeX format: \\boxed{{<answer>}}
2. State your confidence as a percentage: Confidence: XX%

Problem: {problem}

Solution:"""
    
    # Use replace instead of format to avoid issues with braces in problem text
    prompt_text = base_prompt.replace("{problem}", problem)
    
    if negative_context:
        # Use chat format with system message
        system_message = negative_context.strip()
        user_message = prompt_text
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    else:
        # Simple prompt for baseline
        return prompt_text


def run_inference(model_name, sampled_data, output_file, 
                  wrong_file=None, n_wrong=None,
                  tensor_parallel_size=1, batch_size=50):
    """
    Run inference with optional negative prompting.
    
    Args:
        model_name: HuggingFace model name
        sampled_data: Dictionary of sampled examples
        output_file: Path to save predictions
        wrong_file: Optional path to wrong questions JSON
        n_wrong: Optional number of wrong questions to use
        tensor_parallel_size: Number of GPUs
        batch_size: Batch size for processing
    """
    
    # Determine if using negative prompting
    use_negative_prompting = wrong_file is not None
    negative_context = None
    
    if use_negative_prompting:
        wrong_examples = load_wrong_questions(wrong_file, n_wrong)
        negative_context = create_negative_prompt_context(wrong_examples)
        print(f"\n{'='*60}")
        print("NEGATIVE PROMPTING CONTEXT")
        print(f"{'='*60}")
        print(f"Context length: {len(negative_context)} characters")
        print(f"Number of wrong examples: {len(wrong_examples)}")
        print(f"{'='*60}\n")
        print("Preview:")
        print(negative_context[:500] + "...\n")
    
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load existing predictions if resuming
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            predictions = json.load(f)
        print(f"Resuming from existing file with {len(predictions)} predictions")
    else:
        predictions = {}
    
    print(f"\n{'='*60}")
    print("INFERENCE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Mode: {'NEGATIVE PROMPTING' if use_negative_prompting else 'BASELINE'}")
    if use_negative_prompting:
        print(f"Wrong examples: {n_wrong if n_wrong else 'ALL'}")
    print(f"Total examples: {len(sampled_data)}")
    print(f"Output file: {output_file}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Initialize vLLM
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True
    )
    
    # Get tokenizer for chat format if needed
    tokenizer = llm.get_tokenizer() if use_negative_prompting else None
    
    # Sampling params
    params = SamplingParams(
        temperature=0.7,  # Slight temperature for confidence diversity
        max_tokens=1024,
        stop=["\n\n\n"],  # Stop on triple newline
    )
    
    # Convert to list and filter already processed
    examples_list = [(k, v) for k, v in sampled_data.items() if k not in predictions]
    
    if not examples_list:
        print("All examples already processed!")
        return
    
    print(f"Processing {len(examples_list)} remaining examples...\n")
    
    # Process in batches
    for batch_start in range(0, len(examples_list), batch_size):
        batch_end = min(batch_start + batch_size, len(examples_list))
        batch = examples_list[batch_start:batch_end]
        
        print(f"Processing batch {batch_start + 1}-{batch_end}/{len(examples_list)}...")
        
        # Create prompts
        if use_negative_prompting:
            # Use chat format
            prompts = []
            for unique_id, example in batch:
                messages = create_prompt(example, negative_context)
                # Apply chat template
                prompt_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                prompts.append(prompt_text)
        else:
            # Simple prompts
            prompts = [create_prompt(example) for _, example in batch]
        
        # Generate
        outputs = llm.generate(prompts, params)
        
        # Store predictions
        for i, output in enumerate(outputs):
            unique_id, example = batch[i]
            
            predictions[unique_id] = {
                "id": unique_id,
                "question": example['problem'],
                "correct_answer": example['solution'],
                "response": output.outputs[0].text.strip(),
                "level": example['level'],
                "type": example['type'],
                "mode": "negative_prompting" if use_negative_prompting else "baseline",
            }
            
            if use_negative_prompting:
                predictions[unique_id]["n_wrong_examples"] = n_wrong if n_wrong else len(wrong_examples)
        
        # Save progress
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"  Saved {len(predictions)}/{len(sampled_data)} predictions")
    
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}\n")


def main(args):
    # Load sampled data
    sampled_data = load_sampled_data(args.sampled_data)
    
    # Run inference
    run_inference(
        model_name=args.model_name,
        sampled_data=sampled_data,
        output_file=args.output,
        wrong_file=args.wrong_questions,
        n_wrong=args.n_wrong,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on MATH dataset with optional negative prompting"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--sampled_data",
        type=str,
        required=True,
        help="Path to sampled dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file for predictions"
    )
    parser.add_argument(
        "--wrong_questions",
        type=str,
        default=None,
        help="Path to wrong questions JSON (for negative prompting)"
    )
    parser.add_argument(
        "--n_wrong",
        type=int,
        default=None,
        help="Number of wrong questions to include (default: all)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=150,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    main(args)
