import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np
import json
from tqdm import tqdm
import logging
import random
from datasets import load_dataset
import argparse

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

def calculate_perplexity(text, model, tokenizer, device):
    """
    Calculates the perplexity of a given text using a specified model.
    """
    try:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        
        # Max sequence length for the evaluator model (e.g., 1024 for gpt2-large)
        max_length = model.config.n_positions
        
        # If text is too long, truncate it
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]
            
        # We use the input_ids as labels. The model will automatically
        # shift them for loss calculation.
        labels = input_ids

        # Run the model forward, with no gradients
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            
            # The loss is the negative log-likelihood (cross-entropy)
            loss = outputs.loss
            
            # Perplexity is the exponential of the loss
            perplexity = torch.exp(loss)
            
        return perplexity.item()
        
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        # Return a high PPL or NaN on failure
        return float('nan')

def run_benchmark(
    gen_model_name="gpt2",
    eval_model_name="gpt2-large",
    n_trials=10,
    seq_length=512,
    prompt_percent=0.05,
    output_dir="/atlas/u/gabeguo/iclr2026_rebuttal",
):
    """
    Main function to run the generation and perplexity benchmark.
    """
    
    # --- 1. Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Calculate prompt length
    prompt_token_length = int(seq_length * prompt_percent)
    if prompt_token_length == 0:
        prompt_token_length = 1 # ensure at least 1 token
    print(f"Using prompt length of {prompt_token_length} tokens ({prompt_percent} of {seq_length}).")
    
    all_results = {}

    # --- 2. Load Models ---
    print(f"Loading generator model: {gen_model_name}...")
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name).to(device)
    gen_model.eval() # Set to evaluation mode

    print(f"Loading evaluator model: {eval_model_name}...")
    eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
    eval_model = AutoModelForCausalLM.from_pretrained(eval_model_name).to(device)
    eval_model.eval() # Set to evaluation mode
    
    # Load WikiText dataset for prompts
    print("Loading WikiText dataset...")
    # Using 'test' split as it's smaller and good for prompts
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Filter out very short texts
    min_prompt_chars = prompt_token_length * 5 # A rough filter
    valid_texts = [text for text in dataset["text"] if len(text.strip()) > min_prompt_chars]
    print(f"Loaded {len(valid_texts)} valid documents from WikiText 'test' split.")
    
    # Tokenize the prompt
    # input_ids = gen_tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # --- 3. Run Scenarios ---
    for use_cache in [False, True]:
        scenario_key = f"{'with' if use_cache else 'without'}_kv_cache"
        print(f"\n--- Running benchmark: {scenario_key} ---")
        
        timings = []
        perplexities = []
        generated_texts = []

        for trial in tqdm(range(min(n_trials, len(valid_texts))), desc=f"Generating {n_trials} sequences"):

            # A. Create the prompt
            # Select a random document
            doc_text = valid_texts[trial] # Default in case loop fails
            doc_tokens = gen_tokenizer.encode(doc_text)
            doc_tokens_len = len(doc_tokens)

            # Select a random start index for the prompt
            start_index = random.randint(0, doc_tokens_len - prompt_token_length - 1)
            end_index = start_index + prompt_token_length
            
            # Get the prompt tokens and format as a batch (unsqueeze)
            input_ids = torch.tensor(doc_tokens[start_index:end_index]).unsqueeze(0).to(device)
            
            # B. Time the generation
            # Clear cache if on CUDA to get more accurate timing (optional)
            if device == "cuda":
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output_ids = gen_model.generate(
                    input_ids,
                    max_length=seq_length,
                    use_cache=use_cache,
                    pad_token_id=gen_tokenizer.eos_token_id
                )
            
            if device == "cuda":
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            
            # C. Decode the output
            text = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(text)
            
            # D. Calculate perplexity
            ppl = calculate_perplexity(text, eval_model, eval_tokenizer, device)
            perplexities.append(ppl)

        # --- 4. Store Results for Scenario ---
        valid_perplexities = [p for p in perplexities if not np.isnan(p)]
        
        all_results[scenario_key] = {
            "mean_time_s": np.mean(timings),
            "std_time_s": np.std(timings),
            "mean_perplexity": np.mean(valid_perplexities),
            "std_perplexity": np.std(valid_perplexities),
            "all_timings_s": timings,
            "all_perplexities": perplexities,
            "example_output": generated_texts[0] # Just save the first one
        }
        
        print(f"Results for {scenario_key}:")
        print(f"  Mean Time: {all_results[scenario_key]['mean_time_s']:.3f} s")
        print(f"  Mean PPL:  {all_results[scenario_key]['mean_perplexity']:.3f}")

    # --- 5. Save Final JSON Output ---
    output_filename = f"{output_dir}/generation_benchmark.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nBenchmark complete. All results saved to {output_filename}")
    print("\nFinal Summary:")
    print(json.dumps(all_results, indent=2, default=str))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_name", default="gpt2", type=str)
    parser.add_argument("--eval_model_name", default="gpt2-large", type=str)
    parser.add_argument("--n_trials", default=10, type=int)
    parser.add_argument("--seq_length", default=512, type=int)
    parser.add_argument("--prompt_percent", default=0.05, type=float)
    parser.add_argument("--output_dir", default="/atlas/u/gabeguo/iclr2026_rebuttal/speed_comparison")
    return parser.parse_args()

if __name__ == "__main__":
    # To run the script:
    # 1. Make sure you have the required libraries:
    #    pip install torch transformers numpy tqdm datasets
    # 2. Save this code as gpt2_benchmark.py
    # 3. Run from your terminal: python gpt2_benchmark.py
    args = parse_args()
    
    run_benchmark(
        gen_model_name=args.gen_model_name,
        eval_model_name=args.eval_model_name,
        n_trials=args.n_trials,
        seq_length=args.seq_length,
        prompt_percent=args.prompt_percent,
        output_dir=args.output_dir,
    )