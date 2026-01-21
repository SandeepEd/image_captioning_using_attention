import math
import torch
import torch.nn as nn

class CaptioningModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, image_input = 2048, pad_id = 0, dropout = 0.5):
        super().__init__()
        # for image, convert from 2048 to emb_dim
        self.image_dropout = nn.Dropout(0.2)
        self.lin_image_h1 = nn.Linear(image_input, emb_dim)
        self.lin_image_c1 = nn.Linear(image_input, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

        # for caption layer
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.embed_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, emb_dim, batch_first=True)

        # adding CNN and LSTM layers to two more dense layers with relu and softmax
        self.cnn_lstm_lin = nn.Linear(emb_dim, emb_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(emb_dim, vocab_size)

    def forward(self, image_feat, input_ids, lengths):
        B = input_ids.size(0)
        # d_i = self.image_dropout(image_feat)
        
        h0 = self.norm(self.lin_image_h1(image_feat)).unsqueeze(0)
        c0 = self.norm(self.lin_image_c1(image_feat)).unsqueeze(0)

        caption_embeds = self.embed(input_ids)
        caption_embeds = self.embed_drop(caption_embeds)
        out_seq, (hn, _) = self.lstm(caption_embeds, (h0, c0))             # hn: (1, B, 256) for 1-layer
         
        # Pick hidden at last real token per sample: out_seq[b, lengths[b]-1]
        # idx = (lengths - 1).clamp(min=0)                       # (B,)
        # se_last = out_seq[torch.arange(B, device=input_ids.device), idx]  # (B, E)
        # c = hn[-1] 

        # merged_inputs = x + se_last
        # out = self.relu(self.cnn_lstm_lin(merged_inputs))
        out = self.out(out_seq)
        return out

def run_model(model, loss_fn, loader, optimizer=None, device = 'cpu'):

    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0


    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for image_feat, input_ids, target_ids, lengths in loader:
            image_feat = image_feat.to(device)
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            lengths = lengths.to(device)
    
            logits = model(image_feat, input_ids, lengths)
            logits_2d = logits.reshape(-1, logits.size(-1))
            targets_1d = target_ids.reshape(-1)
            loss = loss_fn(logits_2d, targets_1d)


            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=-1)
            mask = target_ids != pad_id
            correct += (((preds == target_ids) & mask).sum().item())
            total += mask.sum().item()
            total_loss += loss.item()
            


    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc
