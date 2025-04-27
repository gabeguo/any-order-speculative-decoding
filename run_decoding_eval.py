# Set environment variable beforehand
import os
os.environ["HF_HOME"] = "/atlas/u/gabeguo/cache_sub"

from transformers import AutoTokenizer, XLNetLMHeadModel
import torch
import time
import argparse
import json
from datetime import datetime
from packed_dataset import PackedDataset
from datasets import load_dataset

from speculative_decoding import speculative_decoding, create_gt_perm_mask
from finetune_xlnet_distributed import create_pos_to_rank
from tqdm import tqdm

OFF_THE_SHELF_KEY = "off_the_shelf"
FINETUNED_KEY = "finetuned"
REGULAR_KEY = "regular_decoding"
PARALLEL_KEY = "parallel_draft_speculative_decoding"
NGRAM_KEY = "ngram_draft_speculative_decoding"
EXECUTION_TIME_KEY = "execution_time"
NFE_COUNT_KEY = "nfe_count"
DECODED_SEQUENCES_KEY = "decoded_sequences"

CODE_SPECIAL_CHARACTER_MAPPINGS = {
    "<cls>": "\n",
    "<sep>": "\t",
    "<unk>": " " * 2,
    "<pad>": " " * 3
}

PRETRAINED_MODEL = "therealgabeguo/ASARM"

# TODO: rename this file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print_steps", action="store_true")
    parser.add_argument("--finetuned_model_dir", type=str, default=PRETRAINED_MODEL)
    parser.add_argument("--start_percentage", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="/atlas/u/gabeguo/speculative_decoding")
    parser.add_argument("--eps", type=float, default=0)
    parser.add_argument("--T", type=float, default=1)
    parser.add_argument("--skip_off_the_shelf", action="store_true")
    parser.add_argument("--skip_ngram", action="store_true")
    parser.add_argument("--is_codegen", action="store_true")
    parser.add_argument("--no_temp_oracle", action="store_true")
    return parser.parse_args()

# NOTE: we edit new_sequence in-place! (Unlike speculative decoding)
def regular_decoding(sigma, start, input_ids, new_sequence, model, tokenizer, T=1):
    select_conditioning = torch.logical_and(sigma.unsqueeze(2) < start, sigma.unsqueeze(1) < start)
    assert select_conditioning.shape == (1, input_ids.shape[1], input_ids.shape[1])
    perm_mask = create_gt_perm_mask(sigma=sigma, seqlen=input_ids.shape[-1], select_conditioning=select_conditioning)
    order_to_pos = torch.argsort(sigma, dim=1)
    for i in range(start, input_ids.shape[1]):
        token_pos = order_to_pos[0, i]
        if args.print_steps:
            print(f"{i} out of {input_ids.shape[1]}")
        # Set prediction targets
        target_mapping = torch.zeros(
            (1, 1, input_ids.shape[1]), dtype=torch.float
        )  # Shape [1, 1, seq_length] => let's predict one token
        target_mapping[
            0, 0, token_pos
        ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
        assert torch.sum(target_mapping) == 1

        with torch.no_grad():
            outputs = model(new_sequence, perm_mask=perm_mask.to(device="cuda"), target_mapping=target_mapping.to(device="cuda"))
        token_logits = outputs[
            0
        ]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
        token_logits = token_logits / T # temperature scaling
        assert token_logits.shape == (1, 1, tokenizer.vocab_size)

        # This evaluates the density
        token_probs = torch.nn.functional.softmax(token_logits, dim=-1)
        assert token_probs[0, 0, 6] < 1e-2, f"mask_token probs: {token_probs[0, 0, 6]}"
        token_probs[0, 0, 6] = 0.0 # never sample mask token
        assert new_sequence[0, token_pos] == 6, f"{token_pos}: {new_sequence[0, token_pos]}"
        new_sequence[0, token_pos] = torch.multinomial(token_probs[0][0], 1).item()
    
    return new_sequence, input_ids.shape[1] - start

def main(args):
    print(f"T: {args.T}")
    tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased")
    # Baseline
    baseline_model = XLNetLMHeadModel.from_pretrained("xlnet/xlnet-base-cased")
    baseline_model = baseline_model.to("cuda")
    baseline_model.eval()
    # Fine-tuned
    if args.finetuned_model_dir == PRETRAINED_MODEL:
        print(f"Loading pretrained model from the hub: {PRETRAINED_MODEL}")
        finetuned_model = XLNetLMHeadModel.from_pretrained(
            PRETRAINED_MODEL,
            use_safetensors=True,
            revision="nlp") # pull from nlp branch
    else:
        print(f"Loading finetuned model from {args.finetuned_model_dir}")
        finetuned_model = XLNetLMHeadModel.from_pretrained(
            args.finetuned_model_dir,
            local_files_only=True,
            use_safetensors=True)
    finetuned_model = finetuned_model.to("cuda")
    finetuned_model.eval()

    print(tokenizer.decode([6], skip_special_tokens=False))

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", streaming=True)
    test_ds = ds["test"]
    packed_ds = PackedDataset(test_ds, tokenizer, max_length=512, is_code=False)

    results_dict = dict()
    for model_name in [OFF_THE_SHELF_KEY, FINETUNED_KEY]:
        results_dict[model_name] = dict()
        for decoding_name in [REGULAR_KEY, PARALLEL_KEY, NGRAM_KEY]:
            results_dict[model_name][decoding_name] = {
                EXECUTION_TIME_KEY: [],
                NFE_COUNT_KEY: [],
                DECODED_SEQUENCES_KEY: []
            }
    # Save prompt
    results_dict["prompt"] = list()
    results_dict["masked_prompt"] = list()

    for trial, input_ids in tqdm(enumerate(packed_ds), total=args.num_trials):
        if trial >= args.num_trials:
            break
        input_ids = input_ids.unsqueeze(0).to(device="cuda")
        assert len(input_ids.shape) == 2
        start = int(input_ids.shape[1] * args.start_percentage) + 1
        # For each trial, give a different permutation
        # TODO: slightly modify this so that we don't break left-to-right ordering after the prompt
        sigma = create_pos_to_rank(input_ids.shape[-1], curr_masking_rate=1-args.start_percentage, fixed_visible_ratio=True).unsqueeze(0).to(device="cuda")
        prompt_tokens = input_ids.clone() # clone each time, so we don't overwrite
        prompt_tokens[sigma >= start] = 6
        masked_prompt_str = tokenizer.decode(prompt_tokens[0, :]).replace("<mask>", "_")
        results_dict["masked_prompt"].append(masked_prompt_str)
        prompt_str = tokenizer.decode(input_ids[0, :])
        results_dict["prompt"].append(prompt_str)

        print("\n###\nStart sequence: ", tokenizer.decode(prompt_tokens[0, :]).replace("<mask>", "_"))
        print(sigma)

        if args.skip_off_the_shelf:
            models = {FINETUNED_KEY: finetuned_model}
        else:
            models = {OFF_THE_SHELF_KEY: baseline_model, FINETUNED_KEY: finetuned_model}
        for model_name, model in models.items():
            model.eval()
            assert not model.training

            print("\n###\n###", model_name, "###\n###\n")

            if args.skip_ngram:
                the_decoding_strategies = [(PARALLEL_KEY, False)]
            else:
                the_decoding_strategies = [(PARALLEL_KEY, False), (NGRAM_KEY, True)]
            for decoding_strat in the_decoding_strategies:
                decoding_name, use_ngram_model = decoding_strat
                print("\n###", decoding_name, "###")
                start_time = time.time()
                # Speculative decoding
                speculated_sequence, speculated_nfe_count = speculative_decoding(
                    model=model, 
                    tokenizer=tokenizer,
                    prompt_tokens=prompt_tokens.clone(),
                    sigma=sigma, 
                    start=start,
                    k=args.k,
                    print_steps=args.print_steps,
                    eps=args.eps,
                    T=args.T,
                    ngram_model=use_ngram_model,
                    no_temp_oracle=args.no_temp_oracle
                )
                end_time = time.time()
                speculative_decoding_time = end_time - start_time
                print(f"Execution time of speculative decoding: {speculative_decoding_time:.3f} seconds")
                print(f"Number of NFEs: {speculated_nfe_count}")
                assert torch.all(speculated_sequence != 6)
                speculated_sequence = tokenizer.decode(speculated_sequence[0, :])
                if args.is_codegen:
                    for k, v in CODE_SPECIAL_CHARACTER_MAPPINGS.items():
                        speculated_sequence = speculated_sequence.replace(k, v)
                assert "<mask>" not in speculated_sequence
                print("speculated sequence: ", speculated_sequence)

                results_dict[model_name][decoding_name][EXECUTION_TIME_KEY].append(speculative_decoding_time)
                results_dict[model_name][decoding_name][NFE_COUNT_KEY].append(speculated_nfe_count)
                results_dict[model_name][decoding_name][DECODED_SEQUENCES_KEY].append(speculated_sequence)

            ###
            # Principled way to generate. Unfortunately, Huggingface implemented their generation in an unprincipled way. At the moment, this does not support KV caching.
            ###
            print("\n### Regular Decoding ###")
            new_sequence = prompt_tokens.clone()
            assert torch.all(new_sequence[sigma >= start] == 6)  # Mask out the ones we're not conditioning on (later order)
            assert torch.sum(new_sequence != 6) == start, f"num decoded: {torch.sum(new_sequence != 6)}; start: {start}"  # num decoded equals start
            # TODO: new sequence creation is counted in speculative decoding time. This makes speculative decoding seem slightly slower than it could be.
            start_time = time.time()
            regular_sequence, regular_nfe_count = regular_decoding(
                sigma=sigma,
                start=start,
                input_ids=input_ids,
                new_sequence=new_sequence,
                model=model,
                tokenizer=tokenizer,
                T=args.T if not args.no_temp_oracle else 1
            )
            end_time = time.time()
            regular_time = end_time - start_time
            assert torch.equal(new_sequence, regular_sequence)
            assert input_ids.shape[1] - start == regular_nfe_count
            assert torch.all(regular_sequence != 6)
            regular_sequence = tokenizer.decode(regular_sequence[0, :])
            if args.is_codegen:
                for k, v in CODE_SPECIAL_CHARACTER_MAPPINGS.items():
                    regular_sequence = regular_sequence.replace(k, v)
            assert "<mask>" not in regular_sequence
            print(f"Execution time of regular decoding: {regular_time:.3f} seconds")
            print("Number of NFEs:", regular_nfe_count)
            print("Regular Decoded sequence: ", regular_sequence)

            results_dict[model_name][REGULAR_KEY][EXECUTION_TIME_KEY].append(regular_time)
            results_dict[model_name][REGULAR_KEY][NFE_COUNT_KEY].append(regular_nfe_count)
            results_dict[model_name][REGULAR_KEY][DECODED_SEQUENCES_KEY].append(regular_sequence)
    return results_dict

def create_output_path(args):
    # Create datetime subfolder in format YYYY-MM-DD_HHMMSS
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    assert timestamp not in args.output_dir
    
    # Combine the base output directory with timestamp subfolder
    output_path = os.path.join(args.output_dir, timestamp)
    
    # Optionally create the directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    return output_path

if __name__ == "__main__":
    args = parse_args()
    results_dict = main(args)
    output_path = create_output_path(args)
    results_output_path = os.path.join(output_path, "results.json")
    args_output_path = os.path.join(output_path, "args.json")
    with open(results_output_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    with open(args_output_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    # print(json.dumps(results_dict, indent=4))