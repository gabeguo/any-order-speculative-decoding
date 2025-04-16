# Set environment variable beforehand
import os
os.environ["HF_HOME"] = "/atlas/u/gabeguo/cache_sub"

import csv, json
import evaluate
import torch
from run_decoding_eval import regular_decoding
from speculative_decoding import speculative_decoding
import argparse
from transformers import AutoTokenizer, XLNetLMHeadModel
from datetime import datetime
from tqdm import tqdm
import numpy as np

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
    assert torch.all(sigma == create_sigma_for_loop(input_ids))

    return sigma, num_visible

def create_sigma_for_loop(input_ids, mask_token_id=6):
    seqlen = input_ids.shape[-1]
    sigma = torch.zeros(seqlen, dtype=torch.long)
    # first pass: get visible tokens
    counter = 0
    for i in range(seqlen):
        if input_ids[i] != mask_token_id:
            sigma[i] = counter
            counter += 1
    # second pass: get masked tokens
    for i in range(seqlen):
        if input_ids[i] == mask_token_id:
            sigma[i] = counter
            counter += 1
    assert counter == seqlen
    return sigma

def eval_infilling(model, tokenizer, args):
    problems = []
    with open(f"eval_datasets/cloze_test_val__spring2016.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            sents = row[1:-3] + [row[-3] if row[-1] == "1" else row[-2]]
            # sents = [s if i == 0 else " " + s for i, s in enumerate(sents)]
            problems.append(sents)
    
    samples = []
    total_cnt = 0
    gens = []
    refs = []
    nfe_counts = list()
    infill_lengths = list()

    print(len(problems))
    for the_trial in range(args.num_trials): # have multiple trials
      print(f"Trial {the_trial}")
      for stories in tqdm(problems[:args.max_items]):
        print("\n\n")
        total_cnt += 1
        # import pdb; pdb.set_trace();
        # TODO: only blank third sentence?
        if args.short_prompt:
            prompt = stories[0]
            suffix = stories[4]
            middle = stories[1] + " " + stories[2] + " " + stories[3]
        else:
            prompt = stories[0] + " " + stories[1]
            suffix = stories[3] + " " + stories[4]
            middle = stories[2]

        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suff = tokenizer.encode(suffix, add_special_tokens=False)

        gt_middle = tokenizer.encode(middle, add_special_tokens=False)
        masked_middle = [tokenizer.mask_token_id for _ in range(len(gt_middle))]
        assert all([x == 6 for x in masked_middle])

        input_ids = torch.tensor(prefix + masked_middle + suff)

        sigma, num_visible = create_sigma(input_ids)
        assert sigma.shape == (input_ids.shape[-1],)
        sigma = sigma.unsqueeze(0).to(device="cuda")

        input_ids = input_ids.unsqueeze(0).to(device="cuda")
        print(f"prompt: {tokenizer.decode(input_ids[0])}")

        assert sigma.shape == input_ids.shape
        
        new_sequence, nfe_count = speculative_decoding(model=model, tokenizer=tokenizer, prompt_tokens=input_ids, sigma=sigma, start=num_visible, T=args.T)
        assert torch.all(new_sequence != 6)
        pred = tokenizer.decode(new_sequence.tolist()[0][len(prefix):len(prefix)+len(gt_middle)])
        assert "<mask>" not in pred

        print(f"completed: {tokenizer.decode(new_sequence[0])}")
        print(f"infilled portion: {pred}")
        print(f"nfe_count: {nfe_count}")
        print(f"num_tokens_predicted: {len(gt_middle)}")
        assert nfe_count <= len(gt_middle), f"{nfe_count} vs {len(gt_middle)}"
        nfe_counts.append(nfe_count)
        infill_lengths.append(len(gt_middle))
    
        samples.append(dict(pred=pred, label=gt_middle, prefix=prompt, suffix=suffix))
        gens.append(pred)
        refs.append(middle)

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=gens, references=refs)
    for key in results.keys():
        results[key] *= 100
    results["rougeAvg"] = (results["rouge1"] + results["rouge2"] + results["rougeL"]) / 3
    print(f"rouge1={results['rouge1']:.2f}, rouge2={results['rouge2']:.2f}, rougeL={results['rougeL']:.2f}, rougeAvg={results['rougeAvg']:.2f}")
    results["nfe_count"] = (np.mean(nfe_counts), np.std(nfe_counts))
    results["infill_lengths"] = (np.mean(infill_lengths), np.std(infill_lengths))

    args.output_dir = create_output_path(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'samples.jsonl'), 'w') as f:
        for json_obj in samples:
            f.write(json.dumps(json_obj) + '\n')
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fout:
        json.dump(results, fout, indent=4)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as fout:
        json.dump(vars(args), fout, indent=4)

    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print_steps", action="store_true")
    parser.add_argument("--finetuned_model_dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="/atlas/u/gabeguo/eval_infilling")
    parser.add_argument("--T", type=float, default=1)
    parser.add_argument("--max_items", type=int, default=50)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--short_prompt", action="store_true")
    return parser.parse_args()

def main(args):
    if args.finetuned_model_dir == PRETRAINED_MODEL:
        print(f"Loading pretrained model from the hub: {PRETRAINED_MODEL}")
        model = XLNetLMHeadModel.from_pretrained(
            PRETRAINED_MODEL,
            use_safetensors=True,
            revision="nlp") # pull from nlp branch
    elif args.finetuned_model_dir is not None:
        print(f"Loading finetuned model from {args.finetuned_model_dir}")
        model = XLNetLMHeadModel.from_pretrained(
            args.finetuned_model_dir,
            local_files_only=True,
            use_safetensors=True)
    else:
        print("Loading base model")
        model = XLNetLMHeadModel.from_pretrained("xlnet/xlnet-base-cased")
    model = model.to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased")

    eval_infilling(model, tokenizer, args)

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
