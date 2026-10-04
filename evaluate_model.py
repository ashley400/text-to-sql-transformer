import argparse
import json
import os
import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentencepiece as spm
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import data_prep
data_prep.DATA_DIR = ROOT / "starter" / "WikiSQL" / "data"   # path fix
from data_prep import load_split
from model.transformer import Transformer
from decode import (write_predictions, greedy_decode, source_to_ids,
                    to_sql, BOS_ID)

RESULTS = ROOT / "results"
def load_model(device):
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "starter" / "sql_sp.model"))
    model = Transformer(sp.get_piece_size()).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "best.pt", map_location=device))
    model.eval()
    return sp, model

def norm_conds(conds):
    return {(int(c), int(o), str(v).lower().strip()) for c, o, v in conds}

def component_accuracy(examples, preds):
    n = len(examples)
    sel_ok = agg_ok = where_ok = 0
    for ex, p in zip(examples, preds):
        if "query" not in p:
            continue                          # parse fail = galat
        g, q = ex["sql"], p["query"]
        sel_ok += g["sel"] == q["sel"]
        agg_ok += g["agg"] == q["agg"]
        where_ok += norm_conds(g["conds"]) == norm_conds(q["conds"])
    return {"sel": 100 * sel_ok / n, "agg": 100 * agg_ok / n, "where": 100 * where_ok / n}

def failure_type(g, q):
    if q is None:
        return "parse failure"
    if g["sel"] != q["sel"]:
        return "wrong column (SELECT)"
    if g["agg"] != q["agg"]:
        return "wrong aggregation"
    gc, qc = norm_conds(g["conds"]), norm_conds(q["conds"])
    if len(qc) < len(gc):
        return "missing condition"
    if len(qc) > len(gc):
        return "extra condition"
    if {c for c, _, _ in gc} != {c for c, _, _ in qc}:
        return "wrong condition column"
    if {(c, o) for c, o, _ in gc} != {(c, o) for c, o, _ in qc}:
        return "wrong operator"
    return "wrong value"

def fmt_gold(g):
    return {"sel": g["sel"], "agg": g["agg"],
            "conds": [[c, o, str(v).lower()] for c, o, v in g["conds"]]}

def write_samples(examples, tables, preds, path, k=5):
    right, wrong = [], []
    for ex, p in zip(examples, preds):
        if len(right) >= k and len(wrong) >= k:
            break
        g = fmt_gold(ex["sql"])
        q = p.get("query")
        ok = q is not None and q["sel"] == g["sel"] and q["agg"] == g["agg"] \
            and norm_conds(q["conds"]) == norm_conds(g["conds"])
        (right if ok else wrong).append((ex, g, q))
    with open(path, "w", encoding="utf-8") as f:
        for title, group, is_wrong in (("Correct", right[:k], False), ("Wrong", wrong[:k], True)):
            f.write(f"# {title} examples\n\n")
            for i, (ex, g, q) in enumerate(group, 1):
                header = tables[ex["table_id"]]["header"]
                f.write(f"## {title} {i}\n")
                f.write(f"- Question: {ex['question']}\n")
                f.write(f"- Gold SQL: `{to_sql(g, header)}`\n")
                f.write(f"- Predicted SQL: `{to_sql(q, header) if q else 'PARSE FAILED'}`\n")
                if is_wrong:
                    f.write(f"- Failure: {failure_type(g, q)}\n")
                f.write("\n")

@torch.no_grad()
def attention_map(model, sp, ex, tables, device, path):
    header = tables[ex["table_id"]]["header"]
    ids = source_to_ids(sp, ex["question"], header)
    src = torch.tensor([ids], device=device)
    out = greedy_decode(model, src)[0]
    tgt = torch.tensor([[BOS_ID] + out], device=device)
    model(src, tgt)
    w = model.decoder.layers[-1].cross_weights[0].mean(dim=0).cpu()   # (T, S)
    xl = [sp.id_to_piece(i) for i in ids]
    yl = ["<s>"] + [sp.id_to_piece(i) for i in out]
    fig, ax = plt.subplots(figsize=(max(8, len(xl) * 0.3), max(4, len(yl) * 0.35)))
    im = ax.imshow(w, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xl)))
    ax.set_xticklabels(xl, rotation=90, fontsize=7)
    ax.set_yticks(range(len(yl)))
    ax.set_yticklabels(yl, fontsize=8)
    ax.set_xlabel("source tokens")
    ax.set_ylabel("generated tokens")
    fig.colorbar(im)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--method", default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--limit", type=int, default=None, help="quick test on first N")
    ap.add_argument("--extras", action="store_true", help="component acc, samples, attention map")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp, model = load_model(device)
    examples, tables = load_split(args.split)
    if args.limit:
        examples = examples[:args.limit]
    RESULTS.mkdir(exist_ok=True)
    name = f"{args.split}_{args.method}" if args.split == "dev" else "test"
    out_path = RESULTS / f"{name}.jsonl"
    fails = write_predictions(model, sp, examples, tables, str(out_path), args.method, device)
    print(f"{name}: parse failures {fails}/{len(examples)} = {100 * fails / len(examples):.2f}%")
    if args.extras:
        preds = [json.loads(l) for l in open(out_path, encoding="utf-8")]
        acc = component_accuracy(examples, preds)
        print("component accuracy (%):", {k: round(v, 2) for k, v in acc.items()})
        write_samples(examples, tables, preds, RESULTS / "samples.md")
        attention_map(model, sp, examples[0], tables, device, RESULTS / "attention_map.png")
        print("wrote samples.md and attention_map.png")

if __name__ == "__main__":
    main()