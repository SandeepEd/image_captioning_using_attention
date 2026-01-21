import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_arc import CaptionTransformerDecoder

class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, att_dim):
        super().__init__()
        self.wenc = nn.Linear(enc_dim, att_dim)
        self.wdec = nn.Linear(dec_dim, att_dim)
        self.tanh = nn.Tanh()
        self.V = nn.Linear(att_dim, 1)

    def forward(self, pixel_feat, previous_state):
        c = self.tanh(self.wenc(pixel_feat) + self.wdec(previous_state).unsqueeze(1))
        e = self.V(c)
        weights = F.softmax(e, dim = 1)
        context_vector = (weights * pixel_feat).sum(dim = 1)
        return context_vector, weights

class LstmDecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, enc_dim, emb_dim, dec_dim, att_dim, pad_id, dropout = 0.4):
        super().__init__()
        self.vocab_size = vocab_size
        self.attentin = Attention(enc_dim, dec_dim, att_dim)
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.embed_dropout = nn.Dropout(dropout)
        self.lin_h0 = nn.Linear(enc_dim, dec_dim)
        self.lin_c0 = nn.Linear(enc_dim, dec_dim)
        self.relu = nn.ReLU()
        self.lstm_cell = nn.LSTMCell(enc_dim + emb_dim, dec_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dec_dim, vocab_size)

    def forward(self, image_feat, captions_ids):
        # initialize h0 and c0 before calculating context vectors
        if self.training:
            # Add random noise (0 mean, 0.1 std) to the fixed image features
            noise = torch.randn_like(image_feat) * 0.05 
            image_feat = image_feat + noise
        batch = image_feat.size(0)
        seq_len = captions_ids.size(1)
        avg_img_feat = image_feat.mean(dim = 1)
        h = self.relu(self.lin_h0(avg_img_feat))
        c = self.relu(self.lin_c0(avg_img_feat))

        predictions = torch.zeros(batch, seq_len, self.vocab_size).to(image_feat.device)
        for t in range(seq_len):
            context_vector, weights = self.attentin(image_feat, h)
            emb = captions_ids[:,t]
            emb = self.embed_dropout(self.embed(emb))
            h, c = self.lstm_cell(torch.cat([context_vector, emb], dim=1), (h, c))

            predictions[:,t,:] = self.fc(self.dropout(h))
        return predictions


def generate_caption_beam_search(model, image_feat, wtoi, itow, beam_size=5, max_len=20, device="cpu"):
    model.eval()
    image_feat = image_feat.to(device)
    
    if image_feat.dim() == 2:
        image_feat = image_feat.unsqueeze(0)
        
    avg_img_feat = image_feat.mean(dim=1)
    h = model.relu(model.lin_h0(avg_img_feat))
    c = model.relu(model.lin_c0(avg_img_feat))

    start_token = wtoi['<start>']
    beams = [([start_token], 0.0, h, c)]
    completed_sequences = []

    for _ in range(max_len):
        new_beams = []
        for seq, score, h_prev, c_prev in beams:
            if seq[-1] == wtoi['<end>']:
                completed_sequences.append((seq, score))
                continue

            current_word = torch.LongTensor([seq[-1]]).to(device)
            
            with torch.no_grad():
                emb = model.embed(current_word)
                context_vector, _ = model.attentin(image_feat, h_prev)
                h_new, c_new = model.lstm_cell(torch.cat([context_vector, emb], dim=1), (h_prev, c_prev))
                output = model.fc(h_new) 
                log_probs = F.log_softmax(output, dim=1)
                
                # --- FIXED INDENTATION: These must be inside the loop ---
                top_log_probs, top_indices = log_probs.topk(beam_size)

                for i in range(beam_size):
                    next_word = top_indices[0, i].item()
                    next_score = score + top_log_probs[0, i].item()
                    new_beams.append((seq + [next_word], next_score, h_new, c_new))

        # Sort and prune
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        if not beams or all(s[0][-1] == wtoi['<end>'] for s in beams):
            break

    completed_sequences.extend([(s[0], s[1]) for s in beams])

    # Length Normalization
    final_captions = []
    for seq, score in completed_sequences:
        norm_score = score / (len(seq)**0.7)
        final_captions.append((seq, norm_score))
    
    best_seq = sorted(final_captions, key=lambda x: x[1], reverse=True)[0][0]
    return [itow[idx] for idx in best_seq if idx not in [wtoi['<start>'], wtoi['<end>']]]


def prepare_reference_dict(img_cap_dict):
    """
    Converts raw captions into tokenized lists of lists for BLEU.
    Input: {img_id: ["<start> a dog runs <end>", ...]}
    Output: {img_id: [["a", "dog", "runs"], ...]}
    """
    clean_refs = {}
    for img_id, captions in img_cap_dict.items():
        img_tokenized_list = []
        for cap in captions:
            # 1. Lowercase 2. Remove tags 3. Split into words
            tokens = cap.lower().replace('<start>', '').replace('<end>', '').strip().split()
            img_tokenized_list.append(tokens)
        clean_refs[img_id] = img_tokenized_list
    return clean_refs

def run_final_test_evaluation(model, test_image_ids, image_features, img_cap_dict, wtoi, itow, device):
    all_references_mapped = prepare_reference_dict(img_cap_dict)
    references = []
    hypotheses = []
    
    print(f"Starting Beam Search Evaluation on {len(test_image_ids)} images...")
    
    model.eval()
    with torch.no_grad():
        for i, img_id in enumerate(test_image_ids):
            if i % 100 == 0: print(f"Processing image {i}/{len(test_image_ids)}...")
            
            feat = image_features[img_id].to(device)
            
            # FIXED: Argument name changed to beam_size
            pred_tokens = generate_caption_beam_search(model, feat, wtoi, itow, beam_size=5, device=device)
            hypotheses.append(pred_tokens)
            references.append(all_references_mapped[img_id])

    b1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    b4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    print(f"\nResults for Checkpoint:")
    print(f"BLEU-1: {b1*100:.2f}%")
    print(f"BLEU-4: {b4*100:.2f}%")

