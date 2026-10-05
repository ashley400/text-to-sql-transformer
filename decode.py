import os
import sys
import torch
import sentencepiece as spm
import gradio as gr
import spaces
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from model.transformer import Transformer
from decode import translate, to_sql
from data_prep import MAX_COLS

CHECKPOINT=os.path.join(ROOT, "checkpoints", "best.pt")
TOKENIZER=os.path.join(ROOT, "starter", "sql_sp.model")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    sp=spm.SentencePieceProcessor(model_file=TOKENIZER)
    model=Transformer(sp.get_piece_size())
    checkpoint=torch.load(CHECKPOINT, map_location=DEVICE)
    state=checkpoint
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state=checkpoint[key]
            break
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return sp, model


sp, model=load_model()


@spaces.GPU
def generate(question, columns, method):
    header=[name.strip() for name in columns.split(",") if name.strip()]
    if not question.strip():
        return "Please type a question.", ""
    if not header:
        return "Please enter at least one column name.", ""
    if len(header) > MAX_COLS:
        return f"Too many columns (maximum {MAX_COLS}).", ""
    query, raw=translate(model, sp, question, header, method=method, device=DEVICE)
    if query is None:
        return "The model's output could not be parsed into a query.", raw
    return to_sql(query, header), raw


demo=gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Question", lines=2),
        gr.Textbox(label="Column names (comma-separated)", lines=2),
        gr.Radio(["beam", "greedy"], value="beam", label="Decoding"),
    ],
    outputs=[
        gr.Textbox(label="Generated SQL"),
        gr.Textbox(label="Raw model output"),
    ],
    title="Text to SQL Transformer",
    description="Type a question and the column names of one table. The model writes the SQL query.",
    examples=[
        ["What is Terrence Ross' nationality?", "Player, No., Nationality, Position, Years in Toronto, School/Club Team", "beam"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()