# Set environment variable beforehand
import os
os.environ["HF_HOME"] = "/atlas/u/gabeguo/cache_sub"

from evaluate import load
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from dataclasses import dataclass
import argparse
import json
from transformers import AutoTokenizer, GPT2Tokenizer

OFF_THE_SHELF_KEY = "off_the_shelf"
FINETUNED_KEY = "finetuned"
REGULAR_KEY = "regular_decoding"
PARALLEL_KEY = "parallel_draft_speculative_decoding"
NGRAM_KEY = "ngram_draft_speculative_decoding"
EXECUTION_TIME_KEY = "execution_time"
NFE_COUNT_KEY = "nfe_count"
DECODED_SEQUENCES_KEY = "decoded_sequences"
STATISTICAL_TEST_KEY = "statistical_test"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perplexity_model", type=str, default="gpt2-large")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--results_dir", type=str, default="/atlas/u/gabeguo/speculative_decoding/2025-02-12_151532")
    parser.add_argument("--skip_off_the_shelf", action="store_true")
    parser.add_argument("--skip_ngram", action="store_true")
    return parser.parse_args()

def eval_perplexity(args, predictions):
    perplexity = load("perplexity", module_type="metric")
    # Truncate each sequence to 1000 tokens using GPT2-large tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    truncated_predictions = list()
    for x in predictions:
        encoded = tokenizer.encode(x)
        if len(encoded) > 960:
            x = tokenizer.decode(encoded[:960])
        truncated_predictions.append(x)
    results = perplexity.compute(predictions=truncated_predictions, 
                                model_id=args.perplexity_model, 
                                batch_size=args.batch_size)
    return results

# https://arxiv.org/pdf/2409.02908
def calculate_entropy(sequences):
    """Calculate entropy of token distributions in sequences"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    entropies = []
    
    for seq in sequences:
        # Tokenize sequence
        tokens = tokenizer.encode(seq)
        # Get token frequencies
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
            
        # Calculate probabilities and entropy
        total_tokens = len(tokens)
        entropy = 0
        for count in token_counts.values():
            prob = count / total_tokens
            entropy -= prob * np.log2(prob)
            
        entropies.append(entropy)
        
    return entropies

def process_results(args, results_dict, model_name_keys, decoding_name_keys):
    """Process results and calculate PPL for each configuration."""
    all_results = dict()
    
    # Calculate PPL for each configuration
    for model_name in model_name_keys:
        all_results[model_name] = dict()
        ppl_by_decoding = {}
        entropy_by_decoding = {}
        for decoding_name in decoding_name_keys:
            # Get sequences for this configuration
            sequences = results_dict[model_name][decoding_name][DECODED_SEQUENCES_KEY]
            
            print(f"eval {model_name} {decoding_name}")
            # Calculate perplexity scores
            ppl_results = eval_perplexity(args, sequences)
            nfe_results = results_dict[model_name][decoding_name][NFE_COUNT_KEY]
            execution_time_results = results_dict[model_name][decoding_name][EXECUTION_TIME_KEY]
            
            # Calculate entropy scores
            entropy_results = calculate_entropy(sequences)
            
            curr_results = {
                'Model': model_name,
                'Decoding': decoding_name,
                'PPL': (np.mean(ppl_results["perplexities"]), stats.sem(ppl_results["perplexities"])),
                'Entropy': (np.mean(entropy_results), stats.sem(entropy_results)),
                NFE_COUNT_KEY: (np.mean(nfe_results), stats.sem(nfe_results)),
                EXECUTION_TIME_KEY: (np.mean(execution_time_results), stats.sem(execution_time_results))
            }

            ppl_by_decoding[decoding_name] = ppl_results["perplexities"]
            entropy_by_decoding[decoding_name] = entropy_results
            all_results[model_name][decoding_name] = curr_results
            
        # Statistical tests for PPL
        for SPECULATIVE_KEY in decoding_name_keys:
            if SPECULATIVE_KEY == REGULAR_KEY:
                continue
            assert len(ppl_by_decoding[REGULAR_KEY]) == len(ppl_by_decoding[SPECULATIVE_KEY])
            t_test_ind = stats.ttest_ind(ppl_by_decoding[REGULAR_KEY], ppl_by_decoding[SPECULATIVE_KEY]).pvalue
            t_test_paired = stats.ttest_rel(ppl_by_decoding[REGULAR_KEY], ppl_by_decoding[SPECULATIVE_KEY]).pvalue
            wilcoxon_test = stats.wilcoxon(ppl_by_decoding[REGULAR_KEY], ppl_by_decoding[SPECULATIVE_KEY], zero_method="zsplit").pvalue
            mann_whitney_test = stats.mannwhitneyu(ppl_by_decoding[REGULAR_KEY], ppl_by_decoding[SPECULATIVE_KEY]).pvalue

            # Statistical tests for entropy
            entropy_t_test_ind = stats.ttest_ind(entropy_by_decoding[REGULAR_KEY], entropy_by_decoding[SPECULATIVE_KEY]).pvalue
            entropy_t_test_paired = stats.ttest_rel(entropy_by_decoding[REGULAR_KEY], entropy_by_decoding[SPECULATIVE_KEY]).pvalue
            entropy_wilcoxon_test = stats.wilcoxon(entropy_by_decoding[REGULAR_KEY], entropy_by_decoding[SPECULATIVE_KEY], zero_method="zsplit").pvalue
            entropy_mann_whitney_test = stats.mannwhitneyu(entropy_by_decoding[REGULAR_KEY], entropy_by_decoding[SPECULATIVE_KEY]).pvalue

            all_results[model_name][f"{SPECULATIVE_KEY}_{STATISTICAL_TEST_KEY}"] = {
                "ppl_t_test_ind": t_test_ind,
                "ppl_t_test_paired": t_test_paired,
                "ppl_wilcoxon_test": wilcoxon_test,
                "ppl_mann_whitney_test": mann_whitney_test,
                "entropy_t_test_ind": entropy_t_test_ind,
                "entropy_t_test_paired": entropy_t_test_paired,
                "entropy_wilcoxon_test": entropy_wilcoxon_test,
                "entropy_mann_whitney_test": entropy_mann_whitney_test
            }
            
            # Plot PPL histogram
            title = f"{model_name} PPL Histogram\np = {t_test_paired:.4f}"
            bins = np.linspace(min(min(ppl_by_decoding[REGULAR_KEY]), min(ppl_by_decoding[SPECULATIVE_KEY])), max(max(ppl_by_decoding[REGULAR_KEY]), max(ppl_by_decoding[SPECULATIVE_KEY])), 20)
            plt.hist(ppl_by_decoding[REGULAR_KEY], label=REGULAR_KEY, histtype='step', bins=bins)
            plt.hist(ppl_by_decoding[SPECULATIVE_KEY], label=SPECULATIVE_KEY, histtype='step', bins=bins)
            plt.title(title)
            plt.legend()
            plt.savefig(os.path.join(args.results_dir, f"{model_name}_{SPECULATIVE_KEY}_ppl_hist.png"))
            plt.close()

            # Plot entropy histogram
            title = f"{model_name} Entropy Histogram\np = {entropy_t_test_paired:.4f}"
            bins = np.linspace(min(min(entropy_by_decoding[REGULAR_KEY]), min(entropy_by_decoding[SPECULATIVE_KEY])), max(max(entropy_by_decoding[REGULAR_KEY]), max(entropy_by_decoding[SPECULATIVE_KEY])), 20)
            plt.hist(entropy_by_decoding[REGULAR_KEY], label=REGULAR_KEY, histtype='step', bins=bins)
            plt.hist(entropy_by_decoding[SPECULATIVE_KEY], label=SPECULATIVE_KEY, histtype='step', bins=bins)
            plt.title(title)
            plt.legend()
            plt.savefig(os.path.join(args.results_dir, f"{model_name}_{SPECULATIVE_KEY}_entropy_hist.png"))
            plt.close()
        
    return all_results


# Usage example:
if __name__ == "__main__":
    args = parse_args()
    results_filename = os.path.join(args.results_dir, "results.json")
    
    # Try to find results.json in main directory or subdirectories
    if not os.path.exists(results_filename):
        # Walk through subdirectories
        for root, dirs, files in os.walk(args.results_dir):
            if "results.json" in files:
                results_filename = os.path.join(root, "results.json")
                args.results_dir = root  # Update results_dir to the directory containing results.json
                break
        if not os.path.exists(results_filename):
            raise FileNotFoundError(f"Could not find results.json in {args.results_dir} or its subdirectories")
    print(f"results_dir: {args.results_dir}")
    print(f"results_filename: {results_filename}")

    with open(results_filename, "r") as f:
        results_dict = json.load(f)
    # Process results and create DataFrame
    if args.skip_off_the_shelf: # just finetuned model
        model_name_keys = [FINETUNED_KEY]
    else: # off-the-shelf and finetuned model
        model_name_keys = [OFF_THE_SHELF_KEY, FINETUNED_KEY]
    if args.skip_ngram:
        decoding_name_keys = [REGULAR_KEY, PARALLEL_KEY]
    else:
        decoding_name_keys = [REGULAR_KEY, PARALLEL_KEY, NGRAM_KEY]
    all_results = process_results(args,
        results_dict=results_dict,
        model_name_keys=model_name_keys,
        decoding_name_keys=decoding_name_keys
    )

    output_filename = os.path.join(args.results_dir, "ppl_results.json")
    with open(output_filename, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(json.dumps(all_results, indent=4))