import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
import math
import matplotlib.pyplot as plt
from PIL import Image
import os

class CaptionTransformerDecoder(nn.Module):
    def __init__(self, image_pix_dim = 2048, image_pixels = 49,d_model = 256, n_heads = 4, dim_feed_forw = 1024,
                 layers = 3, vocab_size = 8000, seq_len = 24, pad_id = 0, dropout = 0.3):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.image_pixels = image_pixels
        self.seq_len = seq_len
        # convert image from 2048 to decoder input dims 256 and take care of positional encodings too
        self.image_lin = nn.Linear(image_pix_dim, d_model)
        self.img_pos_enc = nn.Embedding(image_pixels, d_model)

        # convert input token ids to embeddings
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc_caps = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # transformer decoder layer
        self.decode_lyr = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead = n_heads,
            dim_feedforward=dim_feed_forw,
            dropout=dropout,
            batch_first=True
        )

        # now the Transformer
        self.transformer = nn.TransformerDecoder(self.decode_lyr, num_layers=layers)

        # FF layer
        self.FF = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        
    def forward(self, image_feat, input_ids):
        batch, T = input_ids.shape
        device = image_feat.device
        memory = self.image_lin(image_feat)
        img_pos = torch.arange(self.image_pixels, device=device)
        memory = memory + self.img_pos_enc(img_pos)[None,:,:]

        # caption embeddings
        x = self.embed(input_ids) * math.sqrt(self.d_model)
        cap_pos_enc = torch.arange(T, device=device).clamp(max=self.seq_len - 1)
        x = x + self.pos_enc_caps(cap_pos_enc)[None,:,:]
        x = self.drop(x)

        # 3) Masks
        tgt_mask = self._causal_mask(T, device)                    # (T,T) boolean
        tgt_key_padding_mask = (input_ids == self.pad_id)   

        Q = self.transformer(
            tgt=x,
            memory=memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.FF(Q)
        return logits


class Metrics:
    @staticmethod
    def prepare_reference_dict(img_cap_dict):
        """
        Input: {img_id: ["<start> a dog runs <end>", ...]}
        Output: {img_id: [["a","dog","runs"], ...]}  (multi-reference token lists)
        """
        clean_refs = {}
        for img_id, captions in img_cap_dict.items():
            refs = []
            for cap in captions:
                tokens = cap.lower().replace("<start>", "").replace("<end>", "").strip().split()
                refs.append(tokens)
            clean_refs[img_id] = refs
        return clean_refs

    @staticmethod
    @torch.no_grad()
    def generate_caption_greedy_transformer(
        model, image_feat, wtoi, itow, device="cpu", max_len=None, forbid_pad=True
    ):
        model.eval()
        image_feat = image_feat.to(device)
        if image_feat.dim() == 2:
            image_feat = image_feat.unsqueeze(0)

        start_id = wtoi["<start>"]
        end_id = wtoi["<end>"]
        pad_id = wtoi.get("<pad>", 0)

        if max_len is None:
            max_len = getattr(model, "seq_len", 24) - 1
            max_len = max(1, max_len)

        seq = [start_id]

        for _ in range(max_len):
            inp = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(image_feat, inp)
            next_logits = logits[:, -1, :]

            if forbid_pad and pad_id is not None:
                next_logits[:, pad_id] = -1e9

            next_id = int(torch.argmax(next_logits, dim=-1).item())
            if next_id == end_id:
                break
            seq.append(next_id)

        return [itow[i] for i in seq if i not in (start_id, end_id)]

    @staticmethod
    @torch.no_grad()
    def generate_caption_beam_search_transformer(
        model, image_feat, wtoi, itow, device="cpu", beam_size=5, max_len=None,
        length_norm_alpha=0.7, forbid_pad=True
    ):
        model.eval()
        image_feat = image_feat.to(device)
        if image_feat.dim() == 2:
            image_feat = image_feat.unsqueeze(0)

        start_id = wtoi["<start>"]
        end_id = wtoi["<end>"]
        pad_id = wtoi.get("<pad>", 0)

        if max_len is None:
            max_len = getattr(model, "seq_len", 24) - 1
            max_len = max(1, max_len)

        beams = [([start_id], 0.0)]
        completed = []

        for _ in range(max_len):
            if all(seq[-1] == end_id for seq, _ in beams):
                break

            cur_len = len(beams[0][0])
            inp = torch.tensor([seq for seq, _ in beams], dtype=torch.long, device=device)
            logits = model(image_feat.repeat(inp.size(0), 1, 1), inp)
            next_logits = logits[:, -1, :]

            if forbid_pad and pad_id is not None:
                next_logits[:, pad_id] = -1e9

            log_probs = F.log_softmax(next_logits, dim=-1)

            new_beams = []
            for b_idx, (seq, score) in enumerate(beams):
                if seq[-1] == end_id:
                    completed.append((seq, score))
                    continue

                top_logp, top_ids = torch.topk(log_probs[b_idx], k=beam_size)
                for k in range(beam_size):
                    nid = int(top_ids[k].item())
                    nscore = float(score + top_logp[k].item())
                    new_beams.append((seq + [nid], nscore))

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

        completed.extend(beams)

        def norm_score(seq, score):
            L = max(1, len(seq))
            return score / (L ** length_norm_alpha)

        best_seq, _ = max(completed, key=lambda x: norm_score(x[0], x[1]))
        out_ids = [i for i in best_seq if i not in (start_id, end_id)]
        return [itow[i] for i in out_ids]

    @staticmethod
    def run_final_test_evaluation_transformer(
        model, test_image_ids, image_features, img_cap_dict, wtoi, itow, device,
        use_beam=True, beam_size=5, max_len=20
    ):
        refs_map = Metrics.prepare_reference_dict(img_cap_dict)
        references = []
        hypotheses = []

        print(f"Evaluating on {len(test_image_ids)} images...")

        model.eval()
        with torch.no_grad():
            for i, img_id in enumerate(test_image_ids):
                if i % 100 == 0:
                    print(f"  {i}/{len(test_image_ids)}")

                feat = image_features[img_id].to(device)

                if use_beam:
                    pred_tokens = Metrics.generate_caption_beam_search_transformer(
                        model, feat, wtoi, itow,
                        device=device, beam_size=beam_size, max_len=max_len
                    )
                else:
                    pred_tokens = Metrics.generate_caption_greedy_transformer(
                        model, feat, wtoi, itow,
                        device=device, max_len=max_len
                    )

                hypotheses.append(pred_tokens)
                references.append(refs_map[img_id])

        b1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
        b4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        print("\nResults:")
        print(f"BLEU-1: {b1*100:.2f}%")
        print(f"BLEU-4: {b4*100:.2f}%")

        return b1, b4


    def visualize_model_output_beam(
        model, img_id, image_features, img_cap_dict, wtoi, itow, device,
        image_dir="./data/images/", beam_size=5, max_len=20, length_norm_alpha=0.7
    ):
        img_path = os.path.join(image_dir, img_id)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Could not find image at {img_path}")
            return

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")

        feat = image_features[img_id].to(device)

        prediction_tokens = Metrics.generate_caption_beam_search_transformer(
            model,
            feat,
            wtoi,
            itow,
            beam_size=beam_size,
            max_len=max_len,
            device=device,
            length_norm_alpha=length_norm_alpha
        )
        predicted_sentence = " ".join(prediction_tokens).capitalize() + "."

        ground_truths = []
        if img_id in img_cap_dict:
            for cap in img_cap_dict[img_id]:
                clean_cap = cap.replace("<start>", "").replace("<end>", "").strip()
                ground_truths.append(clean_cap)

        print("\n" + "="*60)
        print(f"IMAGE ID: {img_id}")
        print("="*60)
        print("\n🧠 MODEL PREDICTION (Beam Search):")
        print(f"   → {predicted_sentence}")

        print("\n📝 GROUND TRUTH CAPTIONS:")
        for i, gt in enumerate(ground_truths):
            print(f"   {i+1}. {gt}")
        print("="*60 + "\n")

        plt.title(f"Prediction: {predicted_sentence}", fontsize=12, pad=20)
        plt.show()


 