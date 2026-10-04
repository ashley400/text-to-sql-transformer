import os
import sys
import json
import time
import torch
import torch.nn as nn
import sentencepiece as spm
sys.path.append("starter")
from dataset import make_loader
from model.transformer import Transformer

BATCH_SIZE = 64
EPOCHS = 20
WARMUP = 4000
D_MODEL = 256
PAD_ID = 0
MAX_BATCHES = None        
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
train_loader = make_loader("starter/train_pairs.jsonl", sp, train=True, batch_size=BATCH_SIZE)
dev_loader = make_loader("starter/dev_pairs.jsonl", sp, train=False, batch_size=BATCH_SIZE)
vocab_size = sp.get_piece_size()

model = Transformer(vocab_size).to(device)
print("parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

def lr_lambda(step):
    step = max(step, 1)
    return D_MODEL ** -0.5 * min(step ** -0.5, step * WARMUP ** -1.5)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch():
    model.train()
    total = 0
    count = 0
    for i, (src, tgt) in enumerate(train_loader):
        if MAX_BATCHES is not None and i >= MAX_BATCHES:
            break
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total += loss.item()
        count += 1
    return total / count

def evaluate():
    model.eval()
    total = 0
    count = 0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(dev_loader):
            if MAX_BATCHES is not None and i >= MAX_BATCHES:
                break
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            total += loss.item()
            count += 1
    return total / count

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
best_dev = float("inf")
best_epoch = 0
history = []
start = time.time()
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch()
    dev_loss = evaluate()
    lr = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:2d} | train {train_loss:.4f} | dev {dev_loss:.4f} | lr {lr:.6f}")
    history.append({"epoch": epoch, "train_loss": train_loss,
                    "dev_loss": dev_loss, "lr": lr})
    if dev_loss < best_dev:
        best_dev = dev_loss
        best_epoch = epoch
        torch.save(model.state_dict(), "checkpoints/best.pt")

    with open("results/history.json", "w") as f:
        json.dump(history, f, indent=2)
minutes = (time.time() - start) / 60
print(f"best epoch {best_epoch}, best dev loss {best_dev:.4f}, time {minutes:.1f} min")