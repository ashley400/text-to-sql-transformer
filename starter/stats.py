import sentencepiece as spm
from tokenizer import read_pairs

MAX_SRC, MAX_TGT = 160, 64  

sp = spm.SentencePieceProcessor(model_file="sql_sp.model")


def get_lengths(split):
    pairs = read_pairs(f"{split}_pairs.jsonl")
    src_lens = [len(sp.encode(p["src"])) + 1 for p in pairs]
    tgt_lens = [len(sp.encode(p["tgt"])) + 2 for p in pairs]
    return src_lens, tgt_lens


print(f"{'split':<8}{'pairs':>8}{'src mean':>10}{'src max':>9}"
      f"{'tgt mean':>10}{'tgt max':>9}{'dropped':>9}")

for split in ["train", "dev", "test"]:
    src_lens, tgt_lens = get_lengths(split)

    if split == "train":   
        dropped = sum(1 for s, t in zip(src_lens, tgt_lens)
                      if s > MAX_SRC or t > MAX_TGT)
    else:
        dropped = "-"

    print(f"{split:<8}{len(src_lens):>8}"
          f"{sum(src_lens) / len(src_lens):>10.1f}{max(src_lens):>9}"
          f"{sum(tgt_lens) / len(tgt_lens):>10.1f}{max(tgt_lens):>9}"
          f"{dropped:>9}")