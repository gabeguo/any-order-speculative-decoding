import os
os.environ["HF_HOME"] = "/atlas/u/gabeguo/cache_sub"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
from dotenv import load_dotenv
load_dotenv()
from datetime import (
    datetime,
    timedelta
)
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    XLNetLMHeadModel,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import wandb
import numpy as np
from evaluate import load
from eval_perplexity import calculate_entropy
from speculative_decoding import speculative_decoding
from training_loop_sanity_checks import sanity_check_with_for_loop, sanity_check_with_alt_target, sanity_check_with_alt_perm_mask, sanity_check_with_loss, sanity_check_with_for_loop_joint_perm_mask
from packed_dataset import PackedDataset

print("finished imports")

PARALLEL = "parallel"
JOINT = "joint"

SCALE_BY_SNR = "scale_by_snr"
SCALE_BY_MASKING_RATE = "scale_by_masking_rate"
SCALE_BY_NONE = "scale_by_none"

OPENWEBTEXT_DATASET = "openwebtext"
CODE_DATASET = "bigcode/starcoderdata"

FULL_ATTN = "prompt_and_decoded"
PROMPT_ATTN = "prompt_only"
CAUSAL_ATTN = "causal"

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune XLNet model on OpenWebText dataset')
    
    # Model and training parameters
    parser.add_argument('--model_name', type=str, default='xlnet-base-cased')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--items_per_epoch', type=int, default=8013769)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--masking_warmup_steps', type=int, default=5000)
    parser.add_argument('--final_max_masking_rate', type=float, default=0.85)
    parser.add_argument('--start_masking_rate', type=float, default=0.15)
    parser.add_argument('--final_min_masking_rate', type=float, default=0.75)
    parser.add_argument('--perm_mask_type', type=str, default=PARALLEL)
    parser.add_argument("--loss_scale_type", type=str, default=SCALE_BY_NONE, choices=[SCALE_BY_SNR, SCALE_BY_MASKING_RATE, SCALE_BY_NONE])
    parser.add_argument("--packed_dataset", action="store_true")
    parser.add_argument("--any_permutation", action="store_true")
    parser.add_argument("--dataset", type=str, default=OPENWEBTEXT_DATASET, choices=[OPENWEBTEXT_DATASET, CODE_DATASET])

    # Beta testing parameters
    parser.add_argument("--prompt_attn", type=str, default=FULL_ATTN, choices=[FULL_ATTN, PROMPT_ATTN, CAUSAL_ATTN], help="Only for parallel objective")

    # Val parameters
    parser.add_argument('--perplexity_model', type=str, default="gpt2-large")
    parser.add_argument('--perplexity_batch_size', type=int, default=8)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--eval_masking_rate', type=float, default=0.95)
    parser.add_argument('--eval_num_decodes', type=int, default=3)

    # DDP specific arguments
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--nodes', type=int, default=1)

    # Paths and environment
    parser.add_argument('--cache_dir', type=str, default='/atlas/u/gabeguo/cache_sub')
    parser.add_argument('--output_dir', type=str, default='/atlas/u/gabeguo/xlnet-finetuned')
    parser.add_argument('--wandb_project', type=str, default='xlnet-finetuning')
    
    # Training options
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    return args

def create_output_path(args):
    # Create datetime subfolder in format YYYY-MM-DD_HHMMSS
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    assert timestamp not in args.output_dir
    
    # Combine the base output directory with timestamp subfolder
    output_path = os.path.join(args.output_dir, timestamp)
    
    # Optionally create the directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    return output_path

def setup_ddp(args):
    """Initialize DDP process group."""
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(
            backend="nccl", 
            timeout=timedelta(minutes=30)
        )
        args.world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.world_size = 1
    
    return device

def set_seed(seed, local_rank):
    torch.manual_seed(seed + local_rank)  # Different seed per GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)
    np.random.seed(seed + local_rank)
    return

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples with attention mask and token type IDs."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

def input_ids_to_dict(input_ids):
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }

# Splits into masked and visible portions. Within each, we sort the tokens in ascending order.
# This results in 2^N possible orderings, rather than N! orderings.
# See: https://arxiv.org/pdf/2205.13554
# NOTE: don't use the masking ratios from outside, as the masking ratios we calculate in this method
# describe the percentage that is in the PROMPT, rather than the percentage that is visible
def create_pos_to_rank(seq_length, curr_masking_rate, fixed_visible_ratio=False, return_num_prompt=False):
    # Create a random permutation of the tokens
    shuffle = torch.randperm(seq_length)
    if fixed_visible_ratio: # visible tokens is same as number of prompt tokens
        visible_ratio = 1 - curr_masking_rate
    else:
        visible_ratio = torch.rand(1).item()
        max_visible_rate = 1 - curr_masking_rate
        visible_ratio *= max_visible_rate # this ensures all the prompt tokens are visible (prevents large block of nothing on the right, which breaks bidirectional context)
    assert 0 < visible_ratio < 1

    # Within the visible portion, the tokens are sorted in ascending order
    num_visible = int(visible_ratio * seq_length) + 1 # round UP, because in calc_joint_perm_mask, we rounded DOWN for num_predict, and calculated num_visible from num_predict
    num_visible = max(1, num_visible) # Ensure at least one token is visible
    num_visible = min(seq_length - 1, num_visible) # Ensure at least one token is invisible
    visible_indices = shuffle[:num_visible]
    visible_indices = torch.sort(visible_indices).values
    pos_to_rank = torch.zeros(seq_length, dtype=shuffle.dtype)
    pos_to_rank[visible_indices] = torch.arange(num_visible)

    for i in range(num_visible - 1):
        assert visible_indices[i] < visible_indices[i + 1]
        assert pos_to_rank[visible_indices[i]] < pos_to_rank[visible_indices[i + 1]]
    assert visible_indices[0] <= visible_indices[-1]
    assert pos_to_rank[visible_indices[0]] == 0
    assert pos_to_rank[visible_indices[-1]] == num_visible - 1

    # Within the masked portion, the tokens are sorted in ascending order
    num_mask = seq_length - num_visible
    mask_indices = shuffle[num_visible:]
    mask_indices = torch.sort(mask_indices).values
    pos_to_rank[mask_indices] = torch.arange(num_visible, num_visible + num_mask)

    for i in range(num_mask - 1):
        assert mask_indices[i] < mask_indices[i + 1]
        assert pos_to_rank[mask_indices[i]] < pos_to_rank[mask_indices[i + 1]]
    assert mask_indices[0] <= mask_indices[-1]
    assert pos_to_rank[mask_indices[0]] == num_visible
    assert pos_to_rank[mask_indices[-1]] == seq_length - 1

    assert torch.min(pos_to_rank[mask_indices]) > torch.max(pos_to_rank[visible_indices])
    assert torch.min(pos_to_rank) == 0
    assert torch.max(pos_to_rank) == seq_length - 1
    
    if return_num_prompt:
        return pos_to_rank, num_visible

    return pos_to_rank
    
def calc_parallel_perm_mask(args, batch_size, seq_length, masking_rate, labels):
    # Determine number of tokens to predict
    num_predict = int(seq_length * masking_rate)
    assert 0 < num_predict < seq_length
    num_visible = seq_length - num_predict

    if args.any_permutation:
        pos_to_rank = torch.randperm(seq_length)
        num_prompt = np.random.randint(low=1, high=num_visible+1)
    else:
        # upper bound the condition-mask split point by the current masking rate, so we don't get long block at end
        pos_to_rank, num_prompt = create_pos_to_rank(seq_length, curr_masking_rate=masking_rate, return_num_prompt=True) # pos_to_rank[i] = the order in which we decode the token at index i
        assert num_prompt <= num_visible

    mod_pos_to_rank = pos_to_rank.clone()
    assert pos_to_rank.shape == (seq_length,)

    mod_pos_to_rank[pos_to_rank >= num_visible] = seq_length # Predict all the invisible tokens simultaneously
    assert not torch.equal(mod_pos_to_rank, pos_to_rank)
    perm_mask = mod_pos_to_rank.unsqueeze(1) <= mod_pos_to_rank.unsqueeze(0) # Ban attention to later items, self, and those that are decoded concurrently

    if args.prompt_attn == FULL_ATTN:
        select_prompt = torch.logical_and(pos_to_rank.unsqueeze(1) < num_visible, pos_to_rank.unsqueeze(0) < num_visible) # Prompt & "already decoded" tokens
        perm_mask[select_prompt] = 0 # allow the prompt tokens to attend to selves
    elif args.prompt_attn == PROMPT_ATTN:
        select_prompt = torch.logical_and(pos_to_rank.unsqueeze(1) < num_prompt, pos_to_rank.unsqueeze(0) < num_prompt) # These are all the prompt tokens
        perm_mask[select_prompt] = 0 # ONLY prompt tokens (but not "already decoded" tokens) can attend to each other
    else:
        assert args.prompt_attn == CAUSAL_ATTN

    perm_mask = perm_mask.unsqueeze(0).expand(batch_size, -1, -1)
    assert perm_mask.shape == (batch_size, seq_length, seq_length)

    order_to_pos = torch.argsort(pos_to_rank, dim=-1)
    assert order_to_pos.shape == (seq_length,)
    assert pos_to_rank[order_to_pos[0]] == 0
    assert order_to_pos[pos_to_rank[0]] == 0
    rand_idx = torch.randint(high=seq_length, size=(1,))
    assert pos_to_rank[order_to_pos[rand_idx]] == rand_idx
    assert order_to_pos[pos_to_rank[rand_idx]] == rand_idx

    target_mapping = torch.zeros(batch_size, num_predict, seq_length)
    target_mapping[:, torch.arange(num_predict), order_to_pos[-num_predict:]] = 1.0 # predict the tokens at the end of the ordering (i.e., last num_predict tokens). We can do this because we use the same ordering.
    assert torch.sum(target_mapping) == num_predict * batch_size
    assert torch.sum(target_mapping[0]) == num_predict
    
    # Edit labels
    assert labels.shape == (batch_size, seq_length)
    orig_labels = labels.clone()
    labels = labels[:, order_to_pos[-num_predict:]]
    assert labels.shape == (batch_size, num_predict)

    # Sanity checks
    if torch.rand(1).item() < 1e-4:
        if args.prompt_attn == FULL_ATTN:
            num_full_attn_tokens = num_visible
        elif args.prompt_attn == PROMPT_ATTN:
            num_full_attn_tokens = num_prompt
        else:
            num_full_attn_tokens = 0
        sanity_check_with_for_loop(pos_to_rank=pos_to_rank, mod_pos_to_rank=mod_pos_to_rank, batch_size=batch_size, seq_length=seq_length, num_visible=num_visible, num_predict=num_predict, perm_mask=perm_mask, order_to_pos=order_to_pos, target_mapping=target_mapping, labels=labels, orig_labels=orig_labels, num_full_attn_tokens=num_full_attn_tokens)

    return target_mapping, perm_mask, labels, order_to_pos

def calc_joint_perm_mask(args, batch_size, seq_length, masking_rate, labels):
    # Determine number of tokens to predict
    num_predict = int(seq_length * masking_rate)
    assert 0 < num_predict < seq_length
    num_visible = seq_length - num_predict

    if args.any_permutation:
        pos_to_rank = torch.randperm(seq_length)
    else:
        # calculate the mask, based on the number of visible tokens
        pos_to_rank = create_pos_to_rank(seq_length, curr_masking_rate=masking_rate, fixed_visible_ratio=True) # pos_to_rank[i] = the order in which we decode the token at index i
        for i in range(seq_length): # make sure we created the mask correctly
            if pos_to_rank[i] != i: # first token that is not in left-to-right order should have order of first mask
                assert pos_to_rank[i] == num_visible
                break
        assert i <= num_visible # should have occurred by num_visible indices
    assert pos_to_rank.shape == (seq_length,)

    perm_mask = pos_to_rank.unsqueeze(1) <= pos_to_rank.unsqueeze(0) # Ban attention to later items, self, and those that are decoded concurrently

    # NOTE: can also try causal mask among them
    # BUT, let the prompt tokens attend to each other
    select_prompt = torch.logical_and(pos_to_rank.unsqueeze(1) < num_visible, pos_to_rank.unsqueeze(0) < num_visible) # These are all the prompt tokens
    assert torch.sum(select_prompt) == num_visible * num_visible
    assert torch.sum(perm_mask) == (seq_length + 1) * seq_length / 2
    assert torch.sum(perm_mask[select_prompt]) == (num_visible + 1) * num_visible / 2
    perm_mask[select_prompt] = 0 # allow the prompt tokens to attend to selves
    assert torch.sum(perm_mask) == (num_predict + 1) * (num_predict) / 2 + num_predict * num_visible
    assert torch.sum(perm_mask[select_prompt]) == 0

    # Expand the mask to batch
    perm_mask = perm_mask.unsqueeze(0).expand(batch_size, -1, -1)
    assert perm_mask.shape == (batch_size, seq_length, seq_length)

    # create order_to_pos, so we can make a target mapping!
    order_to_pos = torch.argsort(pos_to_rank, dim=-1)
    assert order_to_pos.shape == (seq_length,)
    assert pos_to_rank[order_to_pos[0]] == 0
    assert order_to_pos[pos_to_rank[0]] == 0
    rand_idx = torch.randint(high=seq_length, size=(1,))
    assert pos_to_rank[order_to_pos[rand_idx]] == rand_idx
    assert order_to_pos[pos_to_rank[rand_idx]] == rand_idx
    if not args.any_permutation: # we have left-to-right ordering
        for i in range(num_visible - 1): # earlier-ordered tokens should have lower position
            assert order_to_pos[i] < order_to_pos[i + 1]
        for i in range(num_predict - 1): # later-ordered tokens should have higher position as well
            assert order_to_pos[num_visible + i + 1] > order_to_pos[num_visible + i]

    # target mapping
    target_mapping = torch.zeros(batch_size, num_predict, seq_length)
    target_mapping[:, torch.arange(num_predict), order_to_pos[-num_predict:]] = 1.0 # predict the tokens at the end of the ordering (i.e., last num_predict tokens). We can do this because we use the same ordering.
    assert torch.sum(target_mapping) == num_predict * batch_size
    assert torch.sum(target_mapping[0]) == num_predict
    
    # Edit labels
    assert labels.shape == (batch_size, seq_length)
    orig_labels = labels.clone()
    labels = labels[:, order_to_pos[-num_predict:]]
    assert labels.shape == (batch_size, num_predict)

    # Sanity checks
    if torch.rand(1).item() < 1e-3:
        sanity_check_with_for_loop_joint_perm_mask(pos_to_rank=pos_to_rank, batch_size=batch_size, seq_length=seq_length, num_visible=num_visible, num_predict=num_predict, perm_mask=perm_mask, order_to_pos=order_to_pos, target_mapping=target_mapping, labels=labels, orig_labels=orig_labels, any_permutation=args.any_permutation)

    return target_mapping, perm_mask, labels, order_to_pos

# SANITY CHECKS

def do_speculative_decoding(args, model, tokenizer, batch, device, is_main_process):
    prompts = list()
    decoded_sequences = list()
    nfes = list()
    index_looping = range(batch["input_ids"].shape[0])
    if is_main_process: # only print progress on main process
        index_looping = tqdm(index_looping)
    for item_idx in index_looping:
        for _ in range(args.eval_num_decodes):
            sigma = create_pos_to_rank(batch["input_ids"].shape[1], curr_masking_rate=args.eval_masking_rate, fixed_visible_ratio=True).unsqueeze(0)
            # TODO: adversarailly alter this just to make sure
            decode_start = int(batch["input_ids"].shape[1] * (1 - args.eval_masking_rate)) + 1 # start at first masked token, consistent with create_pos_to_rank
            speculative_decoding_input = batch["input_ids"][item_idx:item_idx+1].clone()
            assert sigma.shape == speculative_decoding_input.shape
            speculative_decoding_input[sigma >= decode_start] = 6
            decoding_order = sigma[sigma >= decode_start]
            assert torch.all(decoding_order == torch.sort(decoding_order)[0]) # sanity check - make sure we're decoding left to right
            prompt = tokenizer.decode(speculative_decoding_input[0]).replace('<mask>', '_')
            decoded_sequence, nfe_count = speculative_decoding(model=model, tokenizer=tokenizer, prompt_tokens=speculative_decoding_input, sigma=sigma, start=decode_start, mask_token=6, eps=1e-10, vocab_size=32000, adaptive_order=False, k=10)
            generation = tokenizer.decode(decoded_sequence[0])
            if is_main_process:
                print(f"Original Sequence: {prompt}")
                print(f"Decoded Sequence: {generation}")
            prompts.append(prompt)
            decoded_sequences.append(generation)
            nfes.append(nfe_count)
    
    perplexity = eval_perplexity(args=args, predictions=decoded_sequences)
    entropy = calculate_entropy(sequences=decoded_sequences)

    assert len(decoded_sequences) == len(perplexity) == len(entropy) == len(nfes)

    return {
        "decoded_sequences": decoded_sequences,
        "nfes": nfes,
        "perplexity": perplexity,
        "entropy": entropy,
    }

def eval_perplexity(args, predictions):
    perplexity = load("perplexity", module_type="metric")
    # Truncate each sequence to 1000 tokens using GPT2-large tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
    truncated_predictions = list()
    for x in predictions:
        encoded = tokenizer.encode(x)
        if len(encoded) > 1000:
            x = tokenizer.decode(encoded[:1000])
        truncated_predictions.append(x)
    results = perplexity.compute(predictions=truncated_predictions, # use truncated, so we don't get length problems
                                model_id=args.perplexity_model, 
                                batch_size=args.perplexity_batch_size)
    return results["perplexities"]

# END SANITY CHECKS

def prepare_batch(args, batch, device, masking_rate):
    if args.packed_dataset: # make it match the format of the non-packed dataset
        batch = input_ids_to_dict(batch)
    """Prepare a batch for training."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = input_ids.clone()
    assert len(input_ids.shape) == 2

    assert 0 < masking_rate < 1
    
    if args.perm_mask_type == PARALLEL:
        target_mapping, perm_mask, labels, order_to_pos = calc_parallel_perm_mask(args=args, batch_size=input_ids.shape[0], seq_length=input_ids.shape[1], masking_rate=masking_rate, labels=labels)
    elif args.perm_mask_type == JOINT:
        target_mapping, perm_mask, labels, order_to_pos = calc_joint_perm_mask(args=args, batch_size=input_ids.shape[0], seq_length=input_ids.shape[1], masking_rate=masking_rate, labels=labels)
    else:
        raise ValueError(f"Invalid perm mask type: {args.perm_mask_type}")
    
    # move to cuda
    target_mapping = target_mapping.to(device)
    perm_mask = perm_mask.to(device)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "perm_mask": perm_mask,
        "target_mapping": target_mapping,
        "labels": labels,
        "order_to_pos": order_to_pos,
    }

def main():
    args = parse_args()
    
    # Setup DDP
    device = setup_ddp(args)
    is_main_process = args.local_rank in [-1, 0]
    
    # Set seed
    set_seed(args.seed, args.local_rank)
    
    # Initialize wandb only on main process
    if is_main_process:
        args.output_dir = create_output_path(args) # modify output dir only on main process
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    
    # Initialize training state
    start_epoch = 0
    total_steps = 0
    best_loss = float('inf')
    min_masking_rate = args.start_masking_rate
    max_masking_rate = args.start_masking_rate + 1e-5
    
    # Load from checkpoint if specified
    if args.checkpoint_path is not None:
        print(f"Loading model and training state from checkpoint: {args.checkpoint_path}")
        # Load model
        model = XLNetLMHeadModel.from_pretrained(args.checkpoint_path)
        # Load training state
        training_state = torch.load(os.path.join(args.checkpoint_path, "training_state.pt"))
        start_epoch = training_state['epoch']
        total_steps = training_state['total_steps']
        best_loss = training_state['best_loss']
        max_masking_rate = training_state['max_masking_rate']
    else:
        print(f"Initializing model: {args.model_name}")
        model = XLNetLMHeadModel.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    
    model.to(device)
    assert not model.training
    
    # Wrap model with DDP
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    # Load dataset
    if args.dataset == OPENWEBTEXT_DATASET:
        dataset = load_dataset("openwebtext", streaming=True)
        train_dataset = dataset["train"]
    elif args.dataset == CODE_DATASET:
        train_dataset = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    if args.packed_dataset:
        train_dataset = PackedDataset(dataset=train_dataset, tokenizer=tokenizer, max_length=args.max_length, is_code="code" in args.dataset)
        print(f"is code: {train_dataset.is_code}")
        assert train_dataset.is_code == (args.dataset == CODE_DATASET)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            num_workers=2
        )
    else:
        # Create distributed sampler and dataloader
        train_sampler = None
        train_dataloader = DataLoader(
            train_dataset.map(
                lambda x: tokenize_function(x, tokenizer, args.max_length),
                batched=True
            ),
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=2
        )
    
    # Skip to the appropriate step if resuming from checkpoint
    if args.checkpoint_path is not None:
        steps_per_epoch = args.items_per_epoch // (args.batch_size * args.world_size) # per device!
        epoch_offset = total_steps * args.accumulation_steps // steps_per_epoch
        assert start_epoch == epoch_offset  # Update start_epoch to match our position
        
        # Skip any partial epochs
        steps_to_skip = (total_steps * args.accumulation_steps) % steps_per_epoch
        if steps_to_skip > 0:
            # Create a new dataloader with skipped steps
            if args.packed_dataset:
                raise NotImplementedError("Skipping steps is not implemented for packed dataset")
            train_dataloader = DataLoader(
                train_dataset.map(
                    lambda x: tokenize_function(x, tokenizer, args.max_length),
                    batched=True
                ).skip(steps_to_skip * args.batch_size),
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=2
            )
    
    # Adjust learning rate for DDP
    effective_batch_size = args.batch_size * args.world_size * args.accumulation_steps
    args.learning_rate = args.learning_rate * effective_batch_size / 32 # 32 is the default batch size https://mccormickml.com/2019/09/19/XLNet-fine-tuning/
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.items_per_epoch * args.num_epochs // effective_batch_size
    )
    
    # Load optimizer and scheduler states if resuming from checkpoint
    if args.checkpoint_path is not None:
        optimizer.load_state_dict(training_state['optimizer_state'])
        scheduler.load_state_dict(training_state['scheduler_state'])
        
        # Restore RNG states
        torch.set_rng_state(training_state['rng_state'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(training_state['cuda_rng_state'])
        np.random.set_state(training_state['numpy_rng_state'])
    
    # Training loop
    model.train()
    running_loss = 0
    assert model.training
    
    for epoch in range(start_epoch, args.num_epochs):
        # if train_sampler:
        #     train_sampler.set_epoch(epoch)
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=not is_main_process)
        
        for step, batch in enumerate(progress_bar):
            model = model.train() # ensure we're in training mode
            
            # low-variance sampling of masking rate
            if args.local_rank == -1:
                curr_bin = step % args.accumulation_steps
                low_mask_rate = min_masking_rate + (max_masking_rate - min_masking_rate) * (curr_bin / args.accumulation_steps)   
                high_mask_rate = min_masking_rate + (max_masking_rate - min_masking_rate) * ((curr_bin + 1) / args.accumulation_steps)
            else:
                curr_bin = args.local_rank % args.world_size # base it on the device
                curr_bin = (args.world_size - 1) - curr_bin # want the main process to have the highest masking rate
                assert 0 <= curr_bin < args.world_size
                low_mask_rate = min_masking_rate + (max_masking_rate - min_masking_rate) * (curr_bin / args.world_size)
                high_mask_rate = min_masking_rate + (max_masking_rate - min_masking_rate) * ((curr_bin + 1) / args.world_size)
            curr_batch_masking_rate = np.random.uniform(low=low_mask_rate, high=high_mask_rate)

            assert 0 < curr_batch_masking_rate < 1
            assert args.start_masking_rate - 2e-5 <= min_masking_rate - 1e-5 <= curr_batch_masking_rate <= max_masking_rate + 1e-5 <= args.final_max_masking_rate + 2e-5
            batch = prepare_batch(args=args, batch=batch, device=device, masking_rate=curr_batch_masking_rate)
            
            assert model.training # make sure we're in training mode
            # assert torch.all(batch["labels"] != 6) # no mask tokens in labels
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                perm_mask=batch["perm_mask"],
                target_mapping=batch["target_mapping"],
                labels=batch["labels"]
            )

            # focus more on higher masking rates, because they are harder, and encompass the lower masking rates (IF we train with the joint)
            loss = outputs.loss / args.accumulation_steps
            if args.perm_mask_type == JOINT:
                if args.loss_scale_type == SCALE_BY_SNR:
                    signal_to_noise_ratio = (1 - curr_batch_masking_rate) / curr_batch_masking_rate
                    loss /= signal_to_noise_ratio # every noise level weighted equally: divide by signal to noise ratio
                elif args.loss_scale_type == SCALE_BY_MASKING_RATE:
                    loss *= curr_batch_masking_rate # upweight higher masking rates (more noisy ones will teach you more, and encomopass the lower ones)
                elif args.loss_scale_type == SCALE_BY_NONE:
                    pass
                else:
                    raise ValueError(f"Invalid loss scale type: {args.loss_scale_type}")
            loss.backward()
            running_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % args.accumulation_steps == 0:
                max_masking_rate = (args.final_max_masking_rate - args.start_masking_rate) * min(1, total_steps / args.masking_warmup_steps) + args.start_masking_rate
                min_masking_rate = (args.final_min_masking_rate - args.start_masking_rate) * min(1, total_steps / args.masking_warmup_steps) + args.start_masking_rate
                assert args.final_max_masking_rate + 2e-5 >= max_masking_rate + 1e-5 >= min_masking_rate >= args.start_masking_rate - 1e-5
                assert args.final_max_masking_rate + 2e-5 >= args.final_min_masking_rate + 1e-5 >= min_masking_rate >= args.start_masking_rate - 1e-5
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if is_main_process: # only log train results on main process
                    avg_loss = running_loss # do NOT scale by accumulation steps
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "max_masking_rate": max_masking_rate,
                        "min_masking_rate": min_masking_rate,
                        "highest_masking_rate_in_curr_batch": curr_batch_masking_rate,
                    })
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        if args.local_rank == -1:
                            model.save_pretrained(os.path.join(args.output_dir, "best"))
                        else:
                            model.module.save_pretrained(os.path.join(args.output_dir, "best"))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))
                        
                        # Save training state
                        training_state = {
                            'epoch': epoch,
                            'total_steps': total_steps,
                            'best_loss': best_loss,
                            'max_masking_rate': max_masking_rate,
                            'min_masking_rate': min_masking_rate,
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                            'rng_state': torch.get_rng_state(),
                            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                            'numpy_rng_state': np.random.get_state()
                        }
                        torch.save(training_state, os.path.join(args.output_dir, "best", "training_state.pt"))
                # Validation! (On all threads)
                if total_steps % args.eval_steps == 0:
                    # Synchronize all processes before validation
                    if args.local_rank != -1: # non-main processes can also do validation
                        print(f"Rank {args.local_rank} waiting for barrier starting val")
                        dist.barrier()
                        print(f"Rank {args.local_rank} passed barrier starting val")
                    
                    if is_main_process: # Only do speculative decoding on main process
                        with torch.no_grad():
                            # do speculative decoding
                            model = model.eval()
                            results = do_speculative_decoding(args=args, model=model, tokenizer=tokenizer, batch=batch, device=device, is_main_process=is_main_process)
                                                
                        wandb.log({
                            "val_gen_perplexity": np.mean(results["perplexity"]),
                            "val_gen_entropy": np.mean(results["entropy"]),
                            "val_gen_nfes": np.mean(results["nfes"]),
                            "val_gen_perplexity_std": np.std(results["perplexity"]),
                            "val_gen_entropy_std": np.std(results["entropy"]),
                            "val_gen_nfes_std": np.std(results["nfes"]),
                            "val_num_decodes": len(results["perplexity"]),
                        })
                    # sanity check on all processes
                    with torch.no_grad():
                        model = model.eval() # ensure we're in eval mode (only main thread was set to eval mode)
                        # do sanity checks
                        print(f"Rank {args.local_rank} doing sanity checks")
                        assert not model.training
                        # recalculate outputs, because the model changed via gradient update
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            perm_mask=batch["perm_mask"],
                            target_mapping=batch["target_mapping"],
                            labels=batch["labels"]
                        )
                        sanity_check_with_alt_target(model=model, batch=batch, outputs=outputs, device=device)
                        if args.perm_mask_type == PARALLEL and args.prompt_attn == CAUSAL_ATTN:
                            # NOTE: this doesn't fully cover the other cases
                            # TODO: write more general sanity check here
                            sanity_check_with_alt_perm_mask(model=model, batch=batch, outputs=outputs, device=device)
                        sanity_check_with_loss(outputs=outputs, batch=batch)
                        print(f"Rank {args.local_rank} done with sanity checks")
                    
                    # Synchronize all processes after validation
                    if args.local_rank != -1:
                        print(f"Rank {args.local_rank} waiting for barrier ending val")
                        dist.barrier()
                        print(f"Rank {args.local_rank} passed barrier ending val")
                        model = model.train()  # Make sure all processes return to training mode
                
                # update step     
                running_loss = 0
                total_steps += 1
            
            if is_main_process and total_steps > 0 and total_steps % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{total_steps}")
                if args.local_rank == -1:
                    model.save_pretrained(checkpoint_dir)
                else:
                    model.module.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                
                # Save training state
                training_state = {
                    'epoch': epoch,
                    'total_steps': total_steps,
                    'best_loss': best_loss,
                    'max_masking_rate': max_masking_rate,
                    'min_masking_rate': min_masking_rate,
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    'numpy_rng_state': np.random.get_state()
                }
                torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
            
            model = model.train() # ensure we always reset to training mode
    
    # Final save on main process
    if is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        if args.local_rank == -1:
            model.save_pretrained(final_dir)
        else:
            model.module.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        # Save final training state
        training_state = {
            'epoch': args.num_epochs - 1,
            'total_steps': total_steps,
            'best_loss': best_loss,
            'max_masking_rate': max_masking_rate,
            'min_masking_rate': min_masking_rate,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state()
        }
        torch.save(training_state, os.path.join(final_dir, "training_state.pt"))
        
        wandb.finish()

    # Cleanup DDP
    if args.local_rank != -1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()