from eval_perplexity import eval_perplexity, calculate_entropy
import argparse
import numpy as np
import scipy.stats as stats
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MDLM outputs")
    parser.add_argument("--results_files", type=str, nargs='+', default=[
        "/atlas/u/gabeguo/iclr2026_rebuttal/speed_comparison/mdlm/sampling_steps_64/2025-11-12_201520/Wikitext_completion_mdlm_t64.json",
        "/atlas/u/gabeguo/iclr2026_rebuttal/speed_comparison/mdlm/sampling_steps_128/2025-11-12_201726/Wikitext_completion_mdlm_t128.json",
        "/atlas/u/gabeguo/iclr2026_rebuttal/speed_comparison/mdlm/sampling_steps_256/2025-11-12_202102/Wikitext_completion_mdlm_t256.json",
        "/atlas/u/gabeguo/iclr2026_rebuttal/speed_comparison/mdlm/sampling_steps_512/2025-11-12_204627/Wikitext_completion_mdlm_t512.json",
    ])
    parser.add_argument("--perplexity_model", type=str, default="gpt2-large")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()

def main(args):
    for results_file in args.results_files:
        print(results_file)
        with open(results_file, 'r') as f:
            results_dict = json.load(f)
        
        predictions = results_dict["generated_texts"]
        
        # Evaluate perplexity
        ppl_results = eval_perplexity(args, predictions)
        print(f"Results for {results_file}:")
        avg_ppl = np.mean(ppl_results['perplexities'])
        sem_ppl = stats.sem(ppl_results['perplexities'])
        print(f"\tPerplexity: {avg_ppl:.4f}; {sem_ppl:.4f}")
        
        # Calculate entropy
        entropies = calculate_entropy(predictions)
        avg_entropy = np.mean(entropies)
        sem_entropy = stats.sem(entropies)
        print(f"\tEntropy: {avg_entropy:.4f}; {sem_entropy:.4f}")

        # Calculate time
        avg_time = np.mean(results_dict["timings"])
        sem_time = stats.sem(results_dict["timings"])
        print(f"\tTiming: {avg_time:.4f}; {sem_time:.4f}")

        savedir = os.path.dirname(results_file)
        with open(f"{savedir}/eval_results.json", 'w') as f:
            json.dump({
                "perplexity": {
                    "mean": avg_ppl,
                    "sem": sem_ppl
                },
                "entropy": {
                    "mean": avg_entropy,
                    "sem": sem_entropy
                },
                "timing": {
                    "mean": avg_time,
                    "sem": sem_time
                }
            }, f, indent=4)
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)

