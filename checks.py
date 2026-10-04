import torch
import sentencepiece as spm
from model.transformer import Transformer

PAD_ID = 0
sp = spm.SentencePieceProcessor(model_file="starter/sql_sp.model")
vocab_size = sp.get_piece_size()
model = Transformer(vocab_size)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
model.eval()
torch.manual_seed(0)
src = torch.randint(4, vocab_size, (1, 12))
tgt = torch.randint(4, vocab_size, (1, 8))
shared = model.out_proj.weight is model.embedding.emb.weight
print("1. weight sharing (same tensor):", "PASS" if shared else "FAIL")
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("2. trainable parameters:", n)
with torch.no_grad():
    out1 = model(src, tgt)
    tgt2 = tgt.clone()
    tgt2[0, -1] = 100 if tgt[0, -1].item() != 100 else 101
    out2 = model(src, tgt2)
same_before = torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
last_changed = not torch.allclose(out1[:, -1], out2[:, -1], atol=1e-5)
print("3. causal mask: earlier positions unchanged:", "PASS" if same_before else "FAIL")
print("   last position did change:", "PASS" if last_changed else "FAIL")
pad = torch.zeros(1, 5, dtype=torch.long)
src_pad = torch.cat([src, pad], dim=1)
with torch.no_grad():
    out3 = model(src_pad, tgt)
    w = model.decoder.layers[-1].cross_weights      # (1, 4, 8, 17)
same_with_pad = torch.allclose(out1, out3, atol=1e-4)
print("4. padding mask: output unchanged with extra <pad>:", "PASS" if same_with_pad else "FAIL")
row_min = w.sum(-1).min().item()
row_max = w.sum(-1).max().item()
pad_weight = w[..., -5:].abs().max().item()
print(f"5. attention row sums: min {row_min:.6f}, max {row_max:.6f}")
print(f"   max weight on <pad> positions: {pad_weight:.2e}")