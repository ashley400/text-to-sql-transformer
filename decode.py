import json
import os
import re
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "starter")))
from data_prep import AGG_OPS, COND_OPS, encode_source, encode_target, load_split

PAD_ID, BOS_ID, EOS_ID = 0, 2, 3
MAX_TARGET_LEN = 64
AGG_WORDS = [word.lower() for word in AGG_OPS]
ADD_SOURCE_SPECIALS = False


@torch.no_grad()
def greedy_decode(model, source, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, max_len=MAX_TARGET_LEN):
    model.eval()
    encoder_output, source_mask=model.encode(source)
    batch=source.size(0)
    target=torch.full((batch, 1), bos_id, dtype=torch.long, device=source.device)
    done=torch.zeros(batch, dtype=torch.bool, device=source.device)
    for _ in range(max_len):
        logits=model.decode(target, encoder_output, source_mask)
        next_token=logits[:, -1, :].argmax(dim=-1)
        next_token=next_token.masked_fill(done, pad_id)
        target=torch.cat([target, next_token.unsqueeze(1)], dim=1)
        done=done | (next_token == eos_id)
        if done.all():
            break
    outputs=[]
    for row in target[:, 1:].tolist():
        tokens=[]
        for token in row:
            if token == eos_id or token == pad_id:
                break
            tokens.append(token)
        outputs.append(tokens)
    return outputs


@torch.no_grad()
def beam_search(model, source, bos_id=BOS_ID, eos_id=EOS_ID, beam_size=4, max_len=MAX_TARGET_LEN):
    model.eval()
    encoder_output, source_mask=model.encode(source)
    encoder_output=encoder_output.expand(beam_size, -1, -1)
    source_mask=source_mask.expand(beam_size, -1, -1, -1)
    beams=torch.full((beam_size, 1), bos_id, dtype=torch.long, device=source.device)
    scores=torch.zeros(beam_size, device=source.device)
    scores[1:]=float("-inf")
    completed=[]
    for _ in range(max_len):
        count=beams.size(0)
        logits=model.decode(beams, encoder_output[:count], source_mask[:count])[:, -1, :]
        log_probs=F.log_softmax(logits, dim=-1)
        candidates=scores.unsqueeze(1) + log_probs
        vocab_size=candidates.size(1)
        best_scores, best_index=candidates.view(-1).topk(2 * beam_size)
        next_beams=[]
        next_scores=[]
        for score, index in zip(best_scores.tolist(), best_index.tolist()):
            if score == float("-inf"):
                continue
            beam_number, token=divmod(index, vocab_size)
            sequence=beams[beam_number].tolist() + [token]
            if token == eos_id:
                completed.append((score / len(sequence[1:]), sequence[1:-1]))
            else:
                next_beams.append(sequence)
                next_scores.append(score)
            if len(next_beams) == beam_size:
                break
        if not next_beams or len(completed) >= beam_size:
            break
        beams=torch.tensor(next_beams, dtype=torch.long, device=source.device)
        scores=torch.tensor(next_scores, device=source.device)
    if not completed:
        for sequence, score in zip(beams.tolist(), scores.tolist()):
            completed.append((score / len(sequence[1:]), sequence[1:]))
    return max(completed, key=lambda item: item[0])[1]


HEAD_PATTERN=re.compile(r"^select (?:(max|min|count|sum|avg) )?<c(\d+)>(?: (.*))?$")
COND_PATTERN=re.compile(r"(?:^| )(where|and) <c(\d+)> (=|>|<) ")


def parse_target(text, num_cols=None):
    text=" ".join(text.lower().split())
    head=HEAD_PATTERN.match(text)
    if head is None:
        return None
    agg_word=head.group(1)
    sel=int(head.group(2))
    remaining=head.group(3)
    agg=AGG_WORDS.index(agg_word) if agg_word else 0
    if num_cols is not None and sel >= num_cols:
        return None
    conds=[]
    if remaining:
        matches=list(COND_PATTERN.finditer(remaining))
        if not matches or matches[0].start() != 0 or matches[0].group(1) != "where":
            return None
        if any(match.group(1) != "and" for match in matches[1:]):
            return None
        for i, match in enumerate(matches):
            end=matches[i + 1].start() if i + 1 < len(matches) else len(remaining)
            value=remaining[match.end():end].strip()
            col=int(match.group(2))
            if not value:
                return None
            if num_cols is not None and col >= num_cols:
                return None
            conds.append([col, COND_OPS.index(match.group(3)), value])
    return {"sel": sel, "agg": agg, "conds": conds}


def ids_to_query(sp, token_ids, num_cols=None):
    text=sp.decode(token_ids)
    return parse_target(text, num_cols), text


def format_value(value):
    if re.fullmatch(r"-?\d+(\.\d+)?", value):
        return value
    return "'" + value.replace("'", "''") + "'"


def to_sql(query, header):
    column=header[query["sel"]]
    select=f"{AGG_OPS[query['agg']]}({column})" if query["agg"] else column
    sql=f"SELECT {select} FROM table"
    clauses=[f"{header[col]} {COND_OPS[op]} {format_value(value)}" for col, op, value in query["conds"]]
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return sql


def source_to_ids(sp, question, header, max_src_len=512):
    ids=sp.encode(encode_source(question, header))[:max_src_len]
    if ADD_SOURCE_SPECIALS:
        ids=[BOS_ID] + ids + [EOS_ID]
    return ids


@torch.no_grad()
def translate(model, sp, question, header, method="beam", device="cpu"):
    ids=source_to_ids(sp, question, header)
    source=torch.tensor([ids], dtype=torch.long, device=device)
    if method == "greedy":
        output=greedy_decode(model, source)[0]
    else:
        output=beam_search(model, source)
    return ids_to_query(sp, output, len(header))


def pad_batch(id_lists, device):
    width=max(len(ids) for ids in id_lists)
    rows=[ids + [PAD_ID] * (width - len(ids)) for ids in id_lists]
    return torch.tensor(rows, dtype=torch.long, device=device)


def write_predictions(model, sp, examples, tables, out_path, method="greedy", device="cpu", batch_size=64):
    model.eval()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    failures=0
    with open(out_path, "w", encoding="utf-8") as file:
        for start in range(0, len(examples), batch_size):
            chunk=examples[start:start + batch_size]
            headers=[tables[example["table_id"]]["header"] for example in chunk]
            id_lists=[source_to_ids(sp, example["question"], header) for example, header in zip(chunk, headers)]
            if method == "greedy":
                outputs=greedy_decode(model, pad_batch(id_lists, device))
            else:
                outputs=[beam_search(model, torch.tensor([ids], dtype=torch.long, device=device)) for ids in id_lists]
            for output, header in zip(outputs, headers):
                query, _=ids_to_query(sp, output, len(header))
                if query is None:
                    failures += 1
                    file.write(json.dumps({"error": "parse"}) + "\n")
                else:
                    file.write(json.dumps({"query": query}) + "\n")
            if (start // batch_size) % 20 == 0:
                print(f"{start + len(chunk)}/{len(examples)} done", flush=True)
    return failures


def gold_round_trip(sp, split, out_path):
    examples, tables=load_split(split)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    failures=0
    with open(out_path, "w", encoding="utf-8") as file:
        for example in examples:
            header=tables[example["table_id"]]["header"]
            ids=sp.encode(encode_target(example["sql"]))
            query, _=ids_to_query(sp, ids, len(header))
            if query is None:
                failures += 1
                file.write(json.dumps({"error": "parse"}) + "\n")
            else:
                file.write(json.dumps({"query": query}) + "\n")
    print(f"{split}: {len(examples)} lines, {failures} parse failures -> {out_path}")
    return failures
