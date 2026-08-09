import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

st.set_page_config(page_title="AI Story & Poem Generator", page_icon="✍️")
st.title("✍️ AI Story & Poem Generator")
st.caption("Powered by GPT-2 | MSc AIML — CHRIST University")

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model     = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

tokenizer, model = load_model()

prompt       = st.text_area("Enter your prompt", height=100,
                             placeholder="A sentence, phrase, or keywords...")
max_length   = st.slider("Max length (tokens)", 100, 500, 300)
temperature  = st.slider("Temperature", 0.1, 1.5, 0.9, 0.05)
top_k        = st.slider("Top-k", 0, 100, 50)
top_p        = st.slider("Top-p (nucleus)", 0.5, 1.0, 0.95, 0.01)
num_seq      = st.number_input("Number of sequences", 1, 5, 1)

if st.button("Generate") and prompt.strip():
    with st.spinner("Generating..."):
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=max_length,
                                 temperature=temperature, top_k=top_k,
                                 top_p=top_p, num_return_sequences=int(num_seq),
                                 do_sample=True, no_repeat_ngram_size=2,
                                 pad_token_id=tokenizer.eos_token_id)
        for i, out in enumerate(outputs, 1):
            text = tokenizer.decode(out, skip_special_tokens=True)
            st.subheader(f"Output {i}")
            st.write(text[len(prompt):])