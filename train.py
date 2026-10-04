import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "starter"))
import torch
import torch.nn as nn
import sentencepiece as spm
from dataset import make_loader
from model.transformer import Transformer

PAD_ID, BATCH, EPOCHS, WARMUP, D_MODEL = 0, 64, 20, 4000, 256
device = "cuda" if torch.cuda.is_available() else "cpu"

def lr_lambda(step):
    step = max(step, 1)
    return D_MODEL ** -0.5 * min(step ** -0.5, step * WARMUP ** -1.5)

def run_epoch(model, loader, crit, opt=None, sched=None, max_batches=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for i, (src, tgt) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            total += loss.item()
            n += 1
    return total / n

def main(max_batches=None):
    sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
    train_dl = make_loader("starter/train_pairs.jsonl", sp, train=True, batch_size=BATCH)
    dev_dl = make_loader("starter/dev_pairs.jsonl", sp, train=False, batch_size=BATCH)
    model = Transformer(sp.get_piece_size()).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameters:", n_params)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    best, history, t0 = float("inf"), [], time.time()
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_dl, crit, opt, sched, max_batches)
        dv = run_epoch(model, dev_dl, crit, max_batches=max_batches)
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:2d} | train {tr:.4f} | dev {dv:.4f} | lr {lr:.6f}")
        history.append({"epoch": epoch, "train_loss": tr, "dev_loss": dv, "lr": lr})
        if dv < best:
            best = dv
            torch.save(model.state_dict(), "checkpoints/best.pt")
            print("  saved best checkpoint")

    with open("results/history.json", "w") as f:
        json.dump({"params": n_params, "seconds": time.time() - t0,
                   "history": history}, f)
        
if __name__ == "__main__":
    main()