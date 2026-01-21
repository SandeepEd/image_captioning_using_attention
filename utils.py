import torch
import torch.nn as nn

# --- Training / Validation loop for your attention decoder with current DataLoader ---
# DataLoader yields: (image_feat, input_ids, target_ids, lengths)
#   preds = model(image_feat, input_ids)         -> (B, T, V)
#   loss  = CE(preds.reshape(-1,V), target_ids.reshape(-1)) with ignore_index=pad_id
# NOTE: This assumes the decoder writes predictions at index t (not t+1).
# If decoder currently writes predictions[:, t+1, :], keep in mind to change it to predictions[:, t, :].
def run_epoch_attention(model, loader, pad_id, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)

    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for image_feat, input_ids, target_ids, lengths in loader:
            image_feat = image_feat.to(device)   # (B, 49, 2048)
            input_ids  = input_ids.to(device)    # (B, T)
            target_ids = target_ids.to(device)   # (B, T)

            preds = model(image_feat, input_ids)             # (B, T, V)

            logits = preds.reshape(-1, preds.size(-1))       # (B*T, V)
            targets = target_ids.reshape(-1)                 # (B*T,)
            loss = loss_fn(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # token accuracy excluding pads
            pred_ids = preds.argmax(dim=-1)                  # (B, T)
            mask = target_ids.ne(pad_id)
            correct += (pred_ids.eq(target_ids) & mask).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / max(n_batches, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc