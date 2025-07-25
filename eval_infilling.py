# Set environment variable beforehand
import os
os.environ["HF_HOME"] = "/atlas/u/gabeguo/cache_sub"

import csv, json
import evaluate
import torch
from run_decoding_eval import regular_decoding
from speculative_decoding import speculative_decoding, create_gt_perm_mask
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

def eval_hellaswag(model, tokenizer, args):
    from datasets import load_dataset
    dataset = load_dataset("Rowan/hellaswag", streaming=True)["validation"]
    total_correct = 0
    total_cnt = 0
    for item_idx, item in enumerate(tqdm(dataset)):
        if item_idx > args.max_items:
            break
        # Each item has 'ctx' (context) and 'endings' (list of possible endings)
        context = item["ctx"]
        highest_log_prob = -float("inf")
        highest_ending = None
        print(f"context: {context}")
        for option_idx, ending in enumerate(item["endings"]):
            context_tokens = tokenizer.encode(context, add_special_tokens=False)
            ending_tokens = tokenizer.encode(ending, add_special_tokens=False)
            input_ids = torch.tensor(context_tokens + ending_tokens)

            sigma, num_visible = create_sigma(input_ids)
            assert sigma.shape == (input_ids.shape[-1],)
            sigma = sigma.unsqueeze(0).to(device="cuda")
            assert sigma.shape == (1, num_visible)

            seqlen = num_visible
            start = len(context_tokens)
            select_conditioning = torch.logical_and(sigma.unsqueeze(2) < start, sigma.unsqueeze(1) < start) # These are all the initially visible tokens
            perm_mask = create_gt_perm_mask(sigma=sigma, seqlen=seqlen, select_conditioning=select_conditioning)
            perm_mask = perm_mask.to(device="cuda")

            target_mapping = torch.eye(seqlen, device="cuda")
            target_mapping = target_mapping.unsqueeze(0)
            assert target_mapping.shape == (1, seqlen, seqlen)
            assert torch.sum(target_mapping) == seqlen

            input_ids = input_ids.unsqueeze(0).to(device="cuda")
            assert input_ids.shape == (1, seqlen)

            # Get logits from model
            logits = model(input_ids=input_ids, perm_mask=perm_mask, target_mapping=target_mapping)[0]
            # this sanity check passed
            # print(f"\tlogits: {logits.shape}")
            # print(f"\tlogits start: {logits[0, :4, :]}")
            # print(f"\tlogits end: {logits[0, -4:, :]}")
            # print(f"\tperm_mask start: {perm_mask[0, :4, :]}")
            # print(f"\tperm_mask end: {perm_mask[0, -4:, :]}")

            # Calculate log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            assert log_probs.shape == (input_ids.shape[0], input_ids.shape[1], log_probs.shape[-1])
            # Get the predicted token probabilities by gathering along sequence dimension
            token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
            assert token_log_probs.shape == (1, input_ids.shape[1])

            # Sum log probs for the visible tokens only
            sequence_log_prob = token_log_probs.sum()
            if sequence_log_prob > highest_log_prob:
                highest_log_prob = sequence_log_prob
                highest_ending = option_idx
        curr_correct = 1 if option_idx == int(item["label"]) else 0
        total_correct += curr_correct
        total_cnt += 1
        print(f"\tcontext: {context}")
        print(f"\tcorrect ending: {item['label']}")
        print(f"\tpredicted ending: {highest_ending}")
        print(f"\tlog prob: {highest_log_prob}")
        print(f"\tcorrect: {curr_correct}")
    print(f"total correct: {total_correct}, total cnt: {total_cnt}, accuracy: {total_correct / total_cnt}")

    args.output_dir = create_output_path(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fout:
        json.dump({"acc": total_correct / total_cnt}, fout, indent=4)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as fout:
        json.dump(vars(args), fout, indent=4)

    return

def eval_lambada(model, tokenizer, args):
    problems = []
    from datasets import load_dataset
    dataset = load_dataset("lambada", streaming=True)["test"]
    for item in dataset:
        # Each item has 'text' field with full text
        # Split into context and target
        text = item["text"]
        words = text.split()
        context = " ".join(words[:-1])
        target = words[-1]
        problems.append([context, target])
    
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

        context = stories[0]
        completion = stories[1]

        prefix = tokenizer.encode(context, add_special_tokens=False)
        gt_suffix = tokenizer.encode(completion, add_special_tokens=False)
        masked_suffix = [tokenizer.mask_token_id for _ in range(len(gt_suffix))]
        assert all([x == 6 for x in masked_suffix])

        input_ids = torch.tensor(prefix + masked_suffix)
        
        sigma, num_visible = create_sigma(input_ids)
        assert sigma.shape == (input_ids.shape[-1],)
        sigma = sigma.unsqueeze(0).to(device="cuda")

        input_ids = input_ids.unsqueeze(0).to(device="cuda")
        print(f"prompt: {tokenizer.decode(input_ids[0])}")

        assert sigma.shape == input_ids.shape
        
        new_sequence, nfe_count = speculative_decoding(model=model, tokenizer=tokenizer, prompt_tokens=input_ids, sigma=sigma, start=num_visible, T=args.T)
        assert torch.all(new_sequence != 6)
        pred = tokenizer.decode(new_sequence.tolist()[0][len(prefix):])
        assert "<mask>" not in pred

        print(f"completed: {tokenizer.decode(new_sequence[0])}")
        print(f"infilled portion: {pred}")
        print(f"nfe_count: {nfe_count}")
        print(f"num_tokens_predicted: {len(gt_suffix)}")
        assert nfe_count <= len(gt_suffix), f"{nfe_count} vs {len(gt_suffix)}"
        nfe_counts.append(nfe_count)
        infill_lengths.append(len(gt_suffix))
    
        samples.append(dict(pred=pred, label=gt_suffix, prefix=context))
        gens.append(pred)
        refs.append(completion)

    # LAMBADA evaluation with exact match accuracy
    correct = sum(1 for pred, ref in zip(gens, refs) if pred.strip() == ref.strip())
    accuracy = (correct / len(gens)) * 100
    results = {"accuracy": accuracy}
    print(f"Exact match accuracy: {accuracy:.2f}%")

    # Add NFE and infill length stats for all datasets
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

def eval_infilling(model, tokenizer, args):
    problems = []
    
    with open(f"eval_datasets/cloze_test_val__spring2016.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            sents = row[1:-3] + [row[-3] if row[-1] == "1" else row[-2]]
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

    # ROC Stories evaluation with ROUGE
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=gens, references=refs)
    for key in results.keys():
        results[key] *= 100
    results["rougeAvg"] = (results["rouge1"] + results["rouge2"] + results["rougeL"]) / 3
    print(f"rouge1={results['rouge1']:.2f}, rouge2={results['rouge2']:.2f}, rougeL={results['rougeL']:.2f}, rougeAvg={results['rougeAvg']:.2f}")

    # Add NFE and infill length stats for all datasets
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
    parser.add_argument("--hf_revision", type=str, default="nlp")
    parser.add_argument("--dataset", type=str, default="roc", choices=["roc", "lambada", "hellaswag"])
    return parser.parse_args()

def main(args):
    if args.finetuned_model_dir == PRETRAINED_MODEL:
        print(f"Loading pretrained model from the hub: {PRETRAINED_MODEL}")
        model = XLNetLMHeadModel.from_pretrained(
            PRETRAINED_MODEL,
            use_safetensors=True,
            revision=args.hf_revision) # pull from nlp branch
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

    if args.dataset == "hellaswag":
        eval_hellaswag(model, tokenizer, args)
    elif args.dataset == "lambada":
        eval_lambada(model, tokenizer, args)
    else:
        assert args.dataset == "roc"
        eval_infilling(model, tokenizer, args)

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
