import torch
from speculative_decoding import calc_parallel_perm_mask, create_gt_perm_mask

def speculative_decoding(model, tokenizer, prompt_tokens, sigma, start, mask_token=6, vocab_size=32000, adaptive_order=False, k=10, print_steps=False, eps=0, T=1):
    nfe_count = 0

    model = model.cuda()
    batch = prompt_tokens.shape[0]
    seqlen = prompt_tokens.shape[1]
    assert prompt_tokens.shape == (batch, seqlen)
    assert batch == 1
    assert sigma.shape == prompt_tokens.shape

    # New Sequence
    new_sequence = prompt_tokens.clone()
    assert torch.all(new_sequence[sigma >= start] == mask_token) # Mask out the ones we're not conditioning on (later order)
    # # print(f"New Sequence: {tokenizer.decode(new_sequence[0]).replace('<mask>', '_')}")

    assert torch.sum(new_sequence != mask_token) == start, f"num decoded: {torch.sum(new_sequence != mask_token)}; start: {start}" # num decoded equals start

    ###
    # Step 0: Initialize the masking
    ###
    sigma = sigma.to(device="cuda")
    select_conditioning = torch.logical_and(sigma.unsqueeze(2) < start, sigma.unsqueeze(1) < start) # These are all the initially visible tokens
    perm_mask = create_gt_perm_mask(sigma=sigma, seqlen=seqlen, select_conditioning=select_conditioning)

    # TODO: initialize mems
    # TODO: check for model
    mems = None

    # Go Loop!
    n = start
    while n < seqlen: # still some to be decoded
        if print_steps:
            print(f"{n} out of {seqlen}")
        ###
        # Step 1: Sample draft
        ###
        parallel_sigma, parallel_perm_mask = calc_parallel_perm_mask(sigma=sigma, step=n, seqlen=seqlen, select_conditioning=select_conditioning)
        
        k = min(k, seqlen - n) # number of tokens to decode
        # Get target mappings for next k tokens
        target_mapping = torch.zeros(1, k, seqlen)
        assert sigma.shape == (1, seqlen)
        order_to_pos = torch.argsort(sigma, dim=1).squeeze(0) # sigma[i] = rank of idx (order in which the indices should be decoded): put lowest-ranked idx first. order_to_pos[i] = the ith-ranked idx
        assert order_to_pos.shape == (seqlen,)
        assert torch.min(order_to_pos) == 0
        assert torch.max(order_to_pos) == seqlen - 1
        # print(order_to_pos[n:n+k])
        target_mapping = target_mapping.cuda()
        target_mapping[0, torch.arange(k), order_to_pos[n:n+k]] = 1.0 # predict the n-th to n+k-th ranked tokens
        assert torch.sum(target_mapping) == k
        # assert len(target_mapping[0, torch.arange(k), order_to_pos[n:n+k]]) == k

        # Use transformer model for draft predictions
        with torch.no_grad():
            pred_logits = model(new_sequence, perm_mask=parallel_perm_mask.to(device="cuda"), target_mapping=target_mapping)[0] # Output has shape  [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
            pred_logits = pred_logits / T # temperature scaling
            nfe_count += 1
        assert pred_logits.shape == (1, k, vocab_size)
        pred_probs = torch.nn.functional.softmax(pred_logits, dim=-1)
        assert pred_probs.shape == pred_logits.shape
        pred_probs = pred_probs.reshape(k, vocab_size)
        assert torch.all(pred_probs[:, mask_token] < 1e-2), f"mask_token probs: {pred_probs[:, mask_token]}"
        pred_probs[:, mask_token] = 0.0 # never sample mask token
        sample = torch.multinomial(pred_probs, num_samples=1)
        assert sample.shape == (k, 1) # TODO: support for batching later
        sample = sample.reshape(batch, k)
        assert pred_probs.shape == (k, vocab_size)
        assert sample.shape == (1, k)

        # Slight optimization: skip the last check if we already got to last token
        if (n == seqlen - 1):
            assert k == 1
            assert sample.shape == (1, 1)
            assert torch.sum(new_sequence == mask_token) == 1
            assert new_sequence[0, order_to_pos[-1]] == mask_token
            # decode the last item in the order (not necessarily position)
            new_sequence[0, order_to_pos[-1]] = sample.item()
            assert new_sequence[0, order_to_pos[-1]] != mask_token
            assert torch.sum(new_sequence == mask_token) == 0, f"num_mask_left: {torch.sum(new_sequence == mask_token)}"
            assert torch.all(new_sequence != mask_token), f"new_sequence: {new_sequence}"
            return new_sequence, nfe_count
        
        ###
        # Step 2: Compute GT probs of the generated sequence
        ###
        proposed_sequence = new_sequence.clone()
        assert proposed_sequence.shape == (1, seqlen)
        proposed_sequence[0, order_to_pos[n:n+k]] = sample[0] # only update the ones we've just decoded
        # proposed_sequence[parallel_sigma == seqlen] = sample[parallel_sigma == seqlen] # only update the ones we've decoded
        # Reordering
        if adaptive_order:
            raise ValueError("not supported rn - see other branch")
        with torch.no_grad():
            gt_logits = model(proposed_sequence, perm_mask=perm_mask, target_mapping=target_mapping)[0]
            gt_logits = gt_logits / T # temperature scaling
            nfe_count += 1
        assert gt_logits.shape == (1, k, vocab_size)
        gt_probs = torch.nn.functional.softmax(gt_logits, dim=-1)
        assert gt_probs.shape == gt_logits.shape
        gt_probs = gt_probs.reshape(k, vocab_size)
        assert torch.all(gt_probs[:, mask_token] < 1e-2), f"mask_token probs: {gt_probs[:, mask_token]}"
        gt_probs[:, mask_token] = 0.0 # never sample mask token
        assert gt_probs.shape == pred_probs.shape

        ###
        # Step 3: rejection sampling
        ###
        for i in range(n, n+k): # at end of every step, we've decoded (i+1) tokens (due to 0 index)
            if print_steps:
                print(f"\t{i} out of {seqlen}")
            r = torch.rand(1).item()
            sample_order = i - n # the order in which we sampled the token (between 0->k)
            idx_in_seq = order_to_pos[i] # the true index in the sequence (between 0->seqlen)
            chosen_token = sample[0, sample_order] # retrieve the current guess for the next-to-be-decoded token
            if i == n and (not ngram_model): # sanity check if we use self as draft model
                assert torch.allclose(gt_probs[sample_order], pred_probs[sample_order], atol=1e-5), f"gt_probs: {gt_probs[sample_order]}, pred_probs: {pred_probs[sample_order]}"
            # Accept .. or reject
            if r < min(1, 
                gt_probs[sample_order, chosen_token] / pred_probs[sample_order, chosen_token]
            ):
                # We accept this guess of the token!
                assert new_sequence[0, idx_in_seq] == mask_token # make sure it's actually masked
                new_sequence[0, idx_in_seq] = chosen_token
                # print(f"\tNew Sequence: {tokenizer.decode(new_sequence[0])}")
                if ngram_model: # update the n-gram model
                    update_context_ngram(context_ngram=context_ngram, new_sequence=new_sequence, idx_in_seq=idx_in_seq, seqlen=seqlen)
            else:
                # Take one from our actual distribution
                assert i > n, f"gt_probs: {gt_probs[sample_order, chosen_token]}; pred_probs: {pred_probs[sample_order, chosen_token]}" # first one should always get accepted, it is MISTAKE if not
                modified_dist = torch.maximum(gt_probs[sample_order] - pred_probs[sample_order], torch.zeros_like(gt_probs[sample_order]))
                assert torch.sum(modified_dist) > 0, f"modified_dist: {modified_dist}\npred_probs: {pred_probs[sample_order]}\ngt_probs: {gt_probs[sample_order]}"
                modified_dist = modified_dist / (torch.sum(modified_dist) + eps)
                assert modified_dist.shape == (vocab_size,) # this is just for the current token
                assert not torch.isnan(modified_dist).any(), "Distribution contains NaN values"
                assert (modified_dist >= 0).all(), "Distribution contains negative values"
                assert torch.isclose(modified_dist.sum(), torch.tensor(1.0).cuda()), "Distribution not summing to 1"
                new_sequence[0, idx_in_seq] = torch.multinomial(modified_dist, num_samples=1).item()
                break
        if not ngram_model: # should have decoded something
            assert i > n or (i == n == seqlen - 1)
        n = i + 1 # this is the number of tokens we've decoded (since we ZERO index)
        assert torch.sum(new_sequence != mask_token) == n, f"num decoded: {torch.sum(new_sequence != mask_token)}; n: {n}" # num decoded equals n
    return new_sequence, nfe_count