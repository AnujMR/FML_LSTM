import os
import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Hyperparameters (paper-like)


EMBED_SIZE = 256       # from 1000 → 256  
HIDDEN_SIZE = 256      # from 1000 → 256  
NUM_LAYERS = 2         # from 4 → 2  
BATCH_SIZE = 128        # if still OOM → use 8  


# EMBED_SIZE = 1000
# HIDDEN_SIZE = 1000
# NUM_LAYERS = 4          # paper used deep LSTMs (4)
# BATCH_SIZE = 128

VOCAB_SIZE = None       
LEARNING_RATE = 0.3     # paper used SGD with high lr=0.7
CLIP = 5.0              # same clip used
MAX_EPOCHS = 25          # not mentioned in the paper
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEAM_SIZE = 3          # in paper beam size was 12

print("Using device:", DEVICE)
print(f"Hyperparams: embed_size={EMBED_SIZE}, hidden_size={HIDDEN_SIZE}, num_layers={NUM_LAYERS}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")


# Utilities: read files

def read_id_file(path: str) -> List[List[int]]:
    sents = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                sents.append([])
            else:
                sents.append([int(x) for x in line.split()])
    print(f"Loaded {len(sents)} sentences from {path}")
    return sents

def load_vocab(vocab_path: str) -> Tuple[dict, dict]:
    idx2word = []
    with open(vocab_path, "r", encoding="utf8") as f:
        for line in f:
            idx2word.append(line.strip())
    word2idx = {w:i for i,w in enumerate(idx2word)}
    print(f"Loaded vocabulary of size {len(idx2word)} from {vocab_path}")
    return word2idx, idx2word


# Dataset + collate

class IDsDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        self.src = read_id_file(src_file)
        self.tgt = read_id_file(tgt_file)
        assert len(self.src) == len(self.tgt)
    def __len__(self): return len(self.src)
    def __getitem__(self, idx):
        return torch.tensor(self.src[idx], dtype=torch.long), torch.tensor(self.tgt[idx], dtype=torch.long)

PAD = 0
EOS = 2

def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=PAD) 
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=PAD)  
    src_mask = (src_padded != PAD).to(torch.float32)  
    tgt_mask = (tgt_padded != PAD).to(torch.float32)
    return src_padded, src_mask, tgt_padded, tgt_mask


# Encoder / Decoder 

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, src, src_mask):
        emb = self.embed(src) 
        outputs, (h_n, c_n) = self.lstm(emb) 
        return outputs, (h_n, c_n)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)  
    def forward(self, tgt_input, hidden):
        
        emb = self.embed(tgt_input)
        outputs, hidden = self.lstm(emb, hidden) 
        logits = self.out(outputs)  
        return logits, hidden


# Training utilities

def sequence_loss(logits, targets, pad_idx=PAD):
    B, T, V = logits.shape
    logits_flat = logits.reshape(B*T, V)
    targets_flat = targets.reshape(B*T)
    loss = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')
    return loss(logits_flat, targets_flat) / B


# Simple greedy and beam decoder 

def greedy_decode(encoder, decoder, src, src_mask, max_len=100):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        _, (h_n, c_n) = encoder(src, src_mask)
        B = src.size(0)
        cur = torch.full((B,1), EOS, dtype=torch.long, device=src.device)
        hidden = (h_n, c_n)
        outputs = []
        for _ in range(max_len):
            logits, hidden = decoder(cur, hidden)  
            probs = torch.softmax(logits[:, -1, :], dim=-1)  
            next_tok = torch.argmax(probs, dim=-1, keepdim=True) 
            outputs.append(next_tok)
            cur = next_tok
        outputs = torch.cat(outputs, dim=1) 
        return outputs.cpu().tolist()


import heapq
def beam_search_single_example(encoder, decoder, src1, src_mask1, beam_size=BEAM_SIZE, max_len=100):
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        _, (h_n, c_n) = encoder(src1, src_mask1)

        beams = [ (0.0, [EOS], (h_n, c_n)) ]  # start with eos
        completed = []
        for _ in range(max_len):
            new_beams = []
            for score, tokens, hidden in beams:
                last_tok = torch.tensor([[tokens[-1]]], device=src1.device)
                logits, new_hidden = decoder(last_tok, hidden)
                logp = torch.log_softmax(logits[0,-1,:], dim=-1)  
                topk = torch.topk(logp, k=beam_size)
                for k in range(beam_size):
                    nt = int(topk.indices[k].item())
                    ns = score - float(topk.values[k].item())  # negative for min-heap style
                    new_toks = tokens + [nt]
                    if nt == EOS:
                        completed.append((ns, new_toks))
                    else:
                        new_beams.append((ns, new_toks, new_hidden))
            beams = sorted(new_beams, key=lambda x: x[0])[:beam_size]
            if not beams:
                break
        # if no completed, pick best partial
        if completed:
            best = min(completed, key=lambda x: x[0])
            return best[1]
        else:
            return beams[0][1]

# Training loop

def train(train_loader, val_loader, vocab_size, save_dir="checkpoints"):
    encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS+1):
        encoder.train(); decoder.train()
        total_loss = 0.0
        total_batches = 0

        for src_padded, src_mask, tgt_padded, tgt_mask in train_loader:
            src_padded = src_padded.to(DEVICE)
            tgt_padded = tgt_padded.to(DEVICE)

            dec_input = tgt_padded[:, :-1]
            dec_target = tgt_padded[:, 1:]

            # skip empty batches
            if (dec_target != PAD).sum() == 0:
                continue

            optimizer.zero_grad()
            _, enc_state = encoder(src_padded, src_mask)

            with torch.cuda.amp.autocast():
                logits, _ = decoder(dec_input, enc_state)
                loss = sequence_loss(logits, dec_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_batches += 1

        avg_train_loss = total_loss / max(1, total_batches)

        #validation
        encoder.eval(); decoder.eval()
        val_loss = 0.0; val_batches = 0

        with torch.no_grad():
            for src_padded, src_mask, tgt_padded, tgt_mask in val_loader:
                src_padded = src_padded.to(DEVICE)
                tgt_padded = tgt_padded.to(DEVICE)
                dec_input = tgt_padded[:, :-1]
                dec_target = tgt_padded[:, 1:]

                if (dec_target != PAD).sum() == 0:
                    continue

                _, enc_state = encoder(src_padded, src_mask)
                logits, _ = decoder(dec_input, enc_state)
                loss = sequence_loss(logits, dec_target)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        torch.save(
            {"epoch": epoch, "encoder_state": encoder.state_dict(), "decoder_state": decoder.state_dict()},
            os.path.join(save_dir, f"ckpt_epoch{epoch}.pt")
        )

        # decay lr if no improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        else:
            optimizer.param_groups[0]["lr"] *= 0.5
            print("Decayed lr to", optimizer.param_groups[0]["lr"])

    return encoder, decoder


# Entrypoint: load data and kick off training

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_src", required=True)
    parser.add_argument("--train_tgt", required=True)
    parser.add_argument("--val_src", required=True)
    parser.add_argument("--val_tgt", required=True)
    parser.add_argument("--vocab", required=True)
    args = parser.parse_args()

    word2idx, idx2word = load_vocab(args.vocab)
    VOCAB_SIZE = len(idx2word)

    train_ds = IDsDataset(args.train_src, args.train_tgt)
    val_ds = IDsDataset(args.val_src, args.val_tgt)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("Starting training on device:", DEVICE)
    encoder, decoder = train(train_loader, val_loader, VOCAB_SIZE)
    # Save final
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, "final_model.pt")
