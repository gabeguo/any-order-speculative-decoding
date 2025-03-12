import torch

# TODO: rename with parallel (this does not work for joint)
def sanity_check_with_for_loop(pos_to_rank, mod_pos_to_rank, batch_size, seq_length, num_visible, num_predict, perm_mask, order_to_pos, target_mapping, labels, orig_labels, num_full_attn_tokens):
    # pos to rank
    alt_mod_pos_to_rank = pos_to_rank.clone()
    for i in range(seq_length):
        if pos_to_rank[i] >= num_visible:
            alt_mod_pos_to_rank[i] = seq_length
    assert torch.equal(mod_pos_to_rank, alt_mod_pos_to_rank)
    # perm mask
    alt_perm_mask = torch.zeros(batch_size, seq_length, seq_length)
    for i in range(seq_length):
        for j in range(seq_length):
            if mod_pos_to_rank[i] <= mod_pos_to_rank[j]:
                alt_perm_mask[:, i, j] = 1.0 # ban attention
    for i in range(num_full_attn_tokens):
        for j in range(num_full_attn_tokens):
            alt_perm_mask[:, order_to_pos[i], order_to_pos[j]] = 0 # allow full attention among the "prompt"
    assert torch.equal(perm_mask, alt_perm_mask)
    # target mapping & labels
    alt_target_mapping = torch.zeros(batch_size, num_predict, seq_length)
    alt_labels = torch.zeros(batch_size, num_predict).to(labels.device)
    for i in range(num_predict):
        alt_target_mapping[:, i, order_to_pos[num_visible + i]] = 1.0
        alt_labels[:, i] = orig_labels[:, order_to_pos[num_visible + i]]
    assert torch.equal(target_mapping, alt_target_mapping)
    assert torch.equal(labels, alt_labels)
    print("passed sanity checks with for loop")
    return

def sanity_check_with_for_loop_joint_perm_mask(pos_to_rank, batch_size, seq_length, num_visible, num_predict, perm_mask, order_to_pos, target_mapping, labels, orig_labels, any_permutation=False):
    # perm mask
    alt_perm_mask = torch.zeros(batch_size, seq_length, seq_length)
    for i in range(seq_length):
        for j in range(seq_length):
            if pos_to_rank[i] <= pos_to_rank[j]:
                alt_perm_mask[:, i, j] = 1.0 # ban attention
            if pos_to_rank[i] < num_visible and pos_to_rank[j] < num_visible:
                alt_perm_mask[:, i, j] = 0 # allow full attention among the prompt tokens
    assert torch.equal(perm_mask, alt_perm_mask)
    # target mapping & labels
    alt_target_mapping = torch.zeros(batch_size, num_predict, seq_length)
    alt_labels = torch.zeros(batch_size, num_predict).to(labels.device)
    for i in range(num_predict):
        alt_target_mapping[:, i, order_to_pos[num_visible + i]] = 1.0
        alt_labels[:, i] = orig_labels[:, order_to_pos[num_visible + i]]
    assert torch.equal(target_mapping, alt_target_mapping)
    assert torch.equal(labels, alt_labels)
    # double check target mapping & labels
    if any_permutation:
        print("passed sanity checks with for loop for JOINT perm mask (any permutation)")
        return
    # This assert is only valid for left-to-right
    alt2_target_mapping = torch.zeros(batch_size, num_predict, seq_length)
    alt2_labels = torch.zeros(batch_size, num_predict).to(labels.device)
    idx_counter = 0
    for i in range(seq_length):
        if pos_to_rank[i] >= num_visible:
            alt2_target_mapping[:, idx_counter, i] = 1.0
            alt2_labels[:, idx_counter] = orig_labels[:, i]
            idx_counter += 1
    assert idx_counter == num_predict
    assert torch.equal(target_mapping, alt2_target_mapping)
    assert torch.equal(labels, alt2_labels)
    print("passed sanity checks with for loop for JOINT perm mask")
    return

def sanity_check_with_alt_target(model, batch, outputs, device):
    alt_target_mapping = torch.eye(batch["input_ids"].shape[1], batch["input_ids"].shape[1])
    alt_target_mapping = alt_target_mapping.unsqueeze(0).expand(batch["input_ids"].shape[0], -1, -1)
    alt_target_mapping = alt_target_mapping.to(device)
    assert alt_target_mapping.shape == (batch["input_ids"].shape[0], batch["input_ids"].shape[1], batch["input_ids"].shape[1])
    alt_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        perm_mask=batch["perm_mask"],
        target_mapping=alt_target_mapping,
        labels=batch["input_ids"]
    )

    print(outputs.logits.shape)
    print("output logits:", outputs.logits)
    print(alt_outputs.logits.shape)
    num_pred = outputs.logits.shape[1]
    print("retry logits:", alt_outputs.logits[:, batch["order_to_pos"][-num_pred:]])
    assert torch.allclose(outputs.logits, 
        alt_outputs.logits[:, batch["order_to_pos"][-num_pred:]],
        rtol=1e-4, atol=1e-3)
    print("passed sanity checks with target mapping")
    return

def sanity_check_with_alt_perm_mask(model, batch, outputs, device):
    pos_to_rank = batch["order_to_pos"].argsort()
    assert pos_to_rank.shape == (batch["input_ids"].shape[1],)
    alt_perm_mask = pos_to_rank.unsqueeze(1) <= pos_to_rank.unsqueeze(0)
    alt_perm_mask = alt_perm_mask.unsqueeze(0).expand(batch["input_ids"].shape[0], -1, -1)
    alt_perm_mask = alt_perm_mask.to(device)
    alt_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        perm_mask=alt_perm_mask,
        target_mapping=batch["target_mapping"],
        labels=batch["labels"]
    )

    assert torch.allclose(outputs.logits[:, 0], alt_outputs.logits[:, 0], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(outputs.logits, alt_outputs.logits, atol=1e-4, rtol=1e-4)
    print("causal logits:", alt_outputs.logits)
    print("passed sanity check with alt perm mask")
    return

def sanity_check_with_loss(outputs, batch):
    print("loss = ", outputs.loss)
    partially_flattened_logits = outputs.logits.reshape(outputs.logits.shape[0] * outputs.logits.shape[1], outputs.logits.shape[2])
    partially_flattened_labels = batch["labels"].flatten()
    recalc_loss = torch.nn.functional.cross_entropy(input=partially_flattened_logits, target=partially_flattened_labels)
    print("recalc loss = ", recalc_loss)
    print("passed sanity check with loss")
    return 