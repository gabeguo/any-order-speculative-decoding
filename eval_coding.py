from human_eval_infilling.data import write_jsonl, read_problems
import torch
from transformers import AutoTokenizer, XLNetLMHeadModel
import argparse
from speculative_decoding import speculative_decoding
from datetime import datetime
import os
from tqdm import tqdm
import json

SPECIAL_CHARACTER_MAPPINGS = {
    "\n": "<cls>",
    "\t": "<sep>",
    " " * 2: "<unk>",
    " " * 3: "<pad>"
}

PRETRAINED_MODEL = "therealgabeguo/ASARM"

def create_output_path(args):
    # Create datetime subfolder in format YYYY-MM-DD_HHMMSS
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    assert timestamp not in args.output_dir
    
    # Combine the base output directory with timestamp subfolder
    output_path = os.path.join(args.output_dir, timestamp)
    
    # Optionally create the directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    return output_path

def create_sigma(input_ids, mask_token_id=6):
    seqlen = input_ids.shape[-1]

    sigma = torch.zeros(seqlen, dtype=torch.long)

    num_visible = torch.sum(input_ids != mask_token_id)
    sigma[input_ids != mask_token_id] = torch.arange(num_visible)

    num_masked = torch.sum(input_ids == mask_token_id)
    assert num_visible + num_masked == seqlen
    sigma[input_ids == mask_token_id] = torch.arange(num_masked) + num_visible

    assert torch.max(sigma) == seqlen - 1
    assert torch.min(sigma) == 0

    return sigma, num_visible

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_model_dir", type=str, default=PRETRAINED_MODEL)
    parser.add_argument("--hf_revision", type=str, default="code")
    parser.add_argument("--output_dir", type=str, default="/atlas/u/gabeguo/humaneval_infill_results")
    parser.add_argument("--max_tasks", type=int, default=2000)
    parser.add_argument("--num_samples_per_task", type=int, default=1)
    return parser.parse_args()

def main(args):
    problems = read_problems(benchmark_name="single-line")

    num_samples_per_task = args.num_samples_per_task

    if args.finetuned_model_dir == PRETRAINED_MODEL:
        print(f"Loading pretrained model from the hub: {PRETRAINED_MODEL}")
        model = XLNetLMHeadModel.from_pretrained(
            PRETRAINED_MODEL,
            use_safetensors=True,
            revision=args.hf_revision).cuda()
    else:
        print(f"Loading finetuned model from {args.finetuned_model_dir}")
        model = XLNetLMHeadModel.from_pretrained(
            args.finetuned_model_dir,
            local_files_only=True,
            use_safetensors=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased")

    samples = list()

    for task_num, task_id in enumerate(problems):
        for _ in range(num_samples_per_task):
            prefix = problems[task_id]["prompt"]
            suffix = problems[task_id]["suffix"]
            gt_solution = problems[task_id]["canonical_solution"]

            for char, replacement in SPECIAL_CHARACTER_MAPPINGS.items():
                prefix = prefix.replace(char, replacement)
                suffix = suffix.replace(char, replacement)
                gt_solution = gt_solution.replace(char, replacement)

            tokenized_prefix = tokenizer.encode(prefix, add_special_tokens=False)
            tokenized_suffix = tokenizer.encode(suffix, add_special_tokens=False)
            tokenized_gt_solution = tokenizer.encode(gt_solution, add_special_tokens=False)

            masked_middle = [tokenizer.mask_token_id for _ in range(len(tokenized_gt_solution))]
            assert all([x == 6 for x in masked_middle])

            input_ids = torch.tensor(tokenized_prefix + masked_middle + tokenized_suffix)
            sigma, num_visible = create_sigma(input_ids)
            
            assert sigma.shape == (input_ids.shape[-1],)
            sigma = sigma.unsqueeze(0).cuda()
            input_ids = input_ids.unsqueeze(0).cuda()
            assert sigma.shape == input_ids.shape

            new_sequence, nfe_count = speculative_decoding(model=model, tokenizer=tokenizer, prompt_tokens=input_ids, sigma=sigma, start=num_visible)

            infill_completion = tokenizer.decode(new_sequence.tolist()[0][len(tokenized_prefix):-len(tokenized_suffix)])
            for char, replacement in SPECIAL_CHARACTER_MAPPINGS.items():
                infill_completion = infill_completion.replace(replacement, char)
            infill_completion = infill_completion[1:]

            print(task_num)
            print("\tpr_completion:", infill_completion)
            print("\tgt_completion:", problems[task_id]["canonical_solution"])
            samples.append(dict(task_id=task_id, completion=infill_completion))
        if task_num >= args.max_tasks:
            break

    output_path = create_output_path(args)
    write_jsonl(os.path.join(output_path, "samples.jsonl"), samples)
    with open(os.path.join(output_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)