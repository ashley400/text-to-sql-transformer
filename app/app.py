import os
import sys
import torch
import sentencepiece as spm
import streamlit as st

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from model.transformer import Transformer
from decode import translate, to_sql
from data_prep import MAX_COLS

CHECKPOINT=os.path.join(ROOT, "checkpoints", "best.pt")
TOKENIZER=os.path.join(ROOT, "starter", "sql_sp.model")


@st.cache_resource
def load_model():
    sp=spm.SentencePieceProcessor(model_file=TOKENIZER)
    model=Transformer(sp.get_piece_size())
    checkpoint=torch.load(CHECKPOINT, map_location="cpu")
    state=checkpoint
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            state=checkpoint[key]
            break
    model.load_state_dict(state)
    model.eval()
    return sp, model


st.set_page_config(page_title="Text to SQL Transformer")
st.title("Text to SQL Transformer")
st.write("Type a question and the column names of one table. The model writes the SQL query.")

sp, model=load_model()

question=st.text_area("Question", value="What is Terrence Ross' nationality?", height=80)
columns=st.text_area("Column names (comma-separated)", value="Player, No., Nationality, Position, Years in Toronto, School/Club Team", height=80)
method=st.radio("Decoding", ["beam", "greedy"], horizontal=True)

if st.button("Generate SQL"):
    header=[name.strip() for name in columns.split(",") if name.strip()]
    if not question.strip():
        st.warning("Please type a question.")
    elif not header:
        st.warning("Please enter at least one column name.")
    elif len(header) > MAX_COLS:
        st.error(f"Too many columns (maximum {MAX_COLS}).")
    else:
        with st.spinner("Generating..."):
            query, raw=translate(model, sp, question, header, method=method, device="cpu")
        if query is None:
            st.error("The model's output could not be parsed into a query.")
        else:
            st.code(to_sql(query, header), language="sql")
        st.caption("Raw model output")
        st.code(raw)