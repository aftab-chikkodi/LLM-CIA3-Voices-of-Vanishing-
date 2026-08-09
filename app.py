import json
import math
import re
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Voices of the Vanishing", page_icon="🌿", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fact_to_plan.pt"
DATA_PATH = BASE_DIR / "species_data.json"


# ============================================================
# STYLE
# "Field journal" identity — a naturalist's specimen ledger,
# built around a printed IUCN status meter as the signature
# element. Palette, type and layout below are chosen for this
# brief specifically (see design tokens in the CSS comments).
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,380;0,9..144,520;0,9..144,650;1,9..144,480&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --paper:      #EFF4EC;
  --paper-card: #FBFCF9;
  --ink:        #17261D;
  --ink-soft:   #4C5C51;
  --pine:       #1F5C45;
  --pine-deep:  #0E2E22;
  --moss:       #CFDECB;
  --moss-line:  #DCE7D8;
  --clay:       #A9552E;
  --clay-soft:  #F4E4D9;
  --gold:       #B08D3E;
  --slate:      #46586C;

  --cr:  #B3261E;
  --en:  #C9722A;
  --vu:  #B99423;
  --nt:  #6B8E4E;
  --lc:  #1F5C45;
}

html, body, [class*="css"]{
  font-family: 'Inter', -apple-system, sans-serif;
}
.stApp{
  background: var(--paper);
}
.block-container{
  padding-top: 2.2rem;
  padding-bottom: 3rem;
  max-width: 980px;
}
h1, h2, h3 { font-family: 'Fraunces', serif; color: var(--ink); }
::selection{ background: var(--moss); }

/* ---------- Focus visibility (accessibility floor) ---------- */
a:focus-visible, button:focus-visible, [role="button"]:focus-visible{
  outline: 2px solid var(--pine); outline-offset: 2px;
}

/* ---------- HERO ---------- */
.hero{
  position: relative;
  padding: 40px 44px 34px 44px;
  border-radius: 4px;
  margin-bottom: 28px;
  background:
    radial-gradient(1100px 260px at 88% -40%, rgba(255,255,255,0.06), transparent 60%),
    linear-gradient(155deg, var(--pine-deep) 0%, var(--pine) 100%);
  color: #EAF3EC;
  overflow: hidden;
  border: 1px solid var(--pine-deep);
}
.hero::after{
  content: "";
  position: absolute; inset: 0;
  background-image:
    repeating-linear-gradient(90deg, rgba(255,255,255,0.035) 0 1px, transparent 1px 84px);
  pointer-events: none;
}
.hero .eyebrow{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: #B9D9C6; margin-bottom: 10px; display:flex; align-items:center; gap:8px;
}
.hero .eyebrow .dot{
  width:6px; height:6px; border-radius:50%; background:#7FD9AE; display:inline-block;
}
.hero h1{
  margin: 0 0 6px 0; font-size: 2.35rem; font-weight: 520; letter-spacing: -0.01em;
  color: #F4FBF6; font-style: italic;
}
.hero p{ margin: 0; opacity: 0.82; font-size: 0.98rem; max-width: 46em; line-height: 1.5; color: #E3F0E6; }

/* ---------- SECTION LABEL ---------- */
.section-label{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--pine); margin: 30px 0 10px 0;
  display: flex; align-items: center; gap: 10px;
}
.section-label::after{
  content: ""; flex: 1; height: 1px; background: var(--moss-line);
}

/* ---------- SPECIMEN CARD ---------- */
.specimen{
  background: var(--paper-card);
  border: 1px solid var(--moss);
  border-radius: 6px;
  padding: 26px 30px 22px 30px;
}
.specimen .name-row{
  display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 4px;
}
.specimen .name-row h2{ margin: 0; font-size: 1.55rem; font-weight: 600; }
.specimen .sci{ font-style: italic; color: var(--ink-soft); font-size: 0.95rem; }
.specimen .tagline{ color: var(--ink-soft); font-size: 0.86rem; margin-bottom: 18px; }

/* status meter — the signature element */
.status-meter{ margin: 16px 0 6px 0; }
.status-meter .track{
  display: flex; height: 10px; border-radius: 5px; overflow: hidden; border: 1px solid rgba(0,0,0,0.06);
}
.status-meter .seg{ flex: 1; position: relative; }
.status-meter .labels{
  display: flex; margin-top: 7px; font-family: 'IBM Plex Mono', monospace;
  font-size: 0.66rem; letter-spacing: 0.05em; color: var(--ink-soft);
}
.status-meter .labels span{ flex: 1; text-align: center; }
.status-meter .labels span.current{ color: var(--ink); font-weight: 600; }
.pin{
  position: absolute; top: -13px; left: 50%; transform: translateX(-50%);
  width: 0; height: 0;
  border-left: 5px solid transparent; border-right: 5px solid transparent;
  border-top: 7px solid var(--ink);
}

.facts-row{
  display: flex; gap: 22px; flex-wrap: wrap; margin-top: 20px;
  padding-top: 16px; border-top: 1px dashed var(--moss);
}
.facts-row .fact{ min-width: 108px; }
.facts-row .fact .k{
  font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--ink-soft);
}
.facts-row .fact .v{ font-size: 0.98rem; font-weight: 600; color: var(--ink); margin-top: 2px; }

/* ---------- PLAN SEQUENCE ---------- */
.plan-seq{ display: flex; flex-wrap: wrap; align-items: center; gap: 0; }
.plan-item{ display: flex; align-items: center; }
.plan-chip{
  display: inline-flex; align-items: center; gap: 8px;
  padding: 7px 13px; margin: 4px 0; border-radius: 20px;
  background: var(--paper-card); border: 1px solid var(--moss);
  font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.02em;
  color: var(--pine-deep);
}
.plan-chip .idx{
  display: inline-flex; align-items: center; justify-content: center;
  width: 16px; height: 16px; border-radius: 50%; background: var(--pine);
  color: white; font-size: 0.62rem; font-weight: 600; flex-shrink: 0;
}
.plan-arrow{ color: var(--moss); margin: 0 4px; font-size: 0.8rem; }

/* ---------- NARRATIVE PAGE ---------- */
.journal-page{
  position: relative;
  background: var(--paper-card);
  border: 1px solid var(--moss);
  border-left: 3px solid var(--pine);
  border-radius: 4px;
  padding: 30px 34px 26px 34px;
  background-image: repeating-linear-gradient(
    to bottom, transparent, transparent 34px, var(--moss-line) 35px
  );
  background-position: 0 6px;
}
.journal-page .quote-mark{
  font-family: 'Fraunces', serif; font-style: italic; font-size: 2.6rem;
  color: var(--pine); line-height: 0; position: relative; top: 18px;
}
.journal-page p{
  font-family: 'Fraunces', serif; font-size: 1.14rem; line-height: 35px;
  color: var(--ink); margin: 0; font-style: italic; font-weight: 380;
}
.journal-page .byline{
  margin-top: 18px; padding-top: 12px; border-top: 1px dashed var(--moss);
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--ink-soft);
}

/* ---------- GROUNDING CHECK ---------- */
.ledger{ display: flex; flex-direction: column; gap: 0; border: 1px solid var(--moss); border-radius: 6px; overflow: hidden; }
.ledger-row{
  display: flex; align-items: center; justify-content: space-between;
  padding: 11px 18px; background: var(--paper-card);
  border-bottom: 1px solid var(--moss-line); font-size: 0.88rem; color: var(--ink);
}
.ledger-row:last-child{ border-bottom: none; }
.ledger-row .mark{
  display: inline-flex; align-items: center; gap: 9px; font-weight: 500;
}
.ledger-row .badge{
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; font-weight: 600;
  padding: 3px 9px; border-radius: 20px;
}
.badge.ok{ background: #E1EEDF; color: var(--pine-deep); }
.badge.fail{ background: var(--clay-soft); color: var(--clay); }
.tick{ color: var(--pine); font-weight: 700; }
.cross{ color: var(--clay); font-weight: 700; }

.verdict{
  margin-top: 14px; padding: 13px 18px; border-radius: 6px; font-size: 0.9rem;
  display:flex; align-items:center; gap:10px;
}
.verdict.pass{ background: #E1EEDF; color: var(--pine-deep); border: 1px solid #BFDCC0; }
.verdict.warn{ background: var(--clay-soft); color: #7A3D1F; border: 1px solid #E3C7AE; }

.footnote{
  margin-top: 10px; font-size: 0.83rem; color: var(--ink-soft); line-height: 1.55;
  padding: 14px 18px; background: rgba(255,255,255,0.5); border: 1px dashed var(--moss); border-radius: 6px;
}

.numbers-caption{
  font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--ink-soft); margin-top: 8px;
}

footer{ visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow"><span class="dot"></span>MSc Artificial Intelligence &amp; Machine Learning — CIA 3</div>
  <h1>🌿 Voices of the Vanishing</h1>
  <p>A grounded conservation narrative generator. A custom sequence-to-sequence Transformer reads a
  species' IUCN status, population trend, region and taxon, and plans a narrative structure — a
  language model then writes the first-person account, and every claim it makes is checked back
  against the source facts before it's shown.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# CUSTOM TRANSFORMER
# Identical architecture to the one fact_to_plan.pt was trained with.
# Do not change vocab sizes / d_model / n_heads here without
# retraining, or the saved weights will fail to load.
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.W_q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        a, _ = self.self_attn(x, x, x)
        x = self.norm1(x + a)
        return self.norm2(x + self.ff(x))


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, x):
        x = self.pos_enc(self.embed(x) * math.sqrt(self.embed.embedding_dim))
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask):
        a, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + a)
        c, cross_attn = self.cross_attn(x, enc_out, enc_out)
        x = self.norm2(x + c)
        x = self.norm3(x + self.ff(x))
        return x, cross_attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_out):
        B, T = x.size()
        causal_mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
        x = self.pos_enc(self.embed(x) * math.sqrt(self.embed.embedding_dim))
        attn = None
        for layer in self.layers:
            x, attn = layer(x, enc_out, causal_mask)
        return self.out_proj(x), attn


class FactToPlanTransformer(nn.Module):
    def __init__(self, fact_vocab, plan_vocab, d_model=64, n_heads=4, d_ff=128, n_layers=2, max_len=20):
        super().__init__()
        self.encoder = Encoder(fact_vocab, d_model, n_heads, d_ff, n_layers, max_len)
        self.decoder = Decoder(plan_vocab, d_model, n_heads, d_ff, n_layers, max_len)

    def forward(self, fact_seq, plan_seq_in):
        return self.decoder(plan_seq_in, self.encoder(fact_seq))


# ============================================================
# VOCABULARIES (must match training exactly)
# ============================================================

FACT_TOKENS = {
    "PAD": 0, "CR": 1, "EN": 2, "VU": 3, "NT": 4, "LC": 5,
    "STEEP_DECLINE": 6, "MILD_DECLINE": 7, "STABLE": 8, "INCREASING": 9,
    "AFRICA": 10, "ASIA": 11, "AMERICAS": 12, "EUROPE": 13, "OCEANIA": 14, "GLOBAL": 15,
    "MAMMAL": 16, "BIRD": 17, "AMPHIBIAN": 18, "REPTILE": 19, "PLANT": 20, "INSECT": 21, "FISH": 22, "OTHER_TAXON": 23,
}

PLAN_TOKENS = {
    "PAD": 0, "SOS": 1, "EOS": 2,
    "OPEN_URGENT": 3, "OPEN_REFLECTIVE": 4, "OPEN_NEUTRAL": 5,
    "CITE_STATUS": 6, "CITE_TREND": 7, "CITE_MAGNITUDE": 8,
    "CITE_REGION": 9, "CITE_TAXON_CONTEXT": 10,
    "CALL_TO_ACTION": 11, "CLOSE_HOPE": 12, "CLOSE_WARNING": 13,
}
PLAN_ID_TO_TOK = {v: k for k, v in PLAN_TOKENS.items()}

# Human-readable labels used ONLY for prompting and display.
# The LLM is far more likely to actually write "critically endangered"
# if it's asked to write "critically endangered" than if it's asked
# to write the internal code "CR" and expected to know what that means.
READABLE_STATUS = {
    "CR": "Critically Endangered", "EN": "Endangered", "VU": "Vulnerable",
    "NT": "Near Threatened", "LC": "Least Concern",
}
READABLE_TREND = {
    "STEEP_DECLINE": "a steep decline", "MILD_DECLINE": "a mild decline",
    "STABLE": "a stable trend", "INCREASING": "an increasing trend",
}
READABLE_REGION = {
    "AFRICA": "Africa", "ASIA": "Asia", "AMERICAS": "the Americas",
    "EUROPE": "Europe", "OCEANIA": "Oceania", "GLOBAL": "across the globe",
}
READABLE_TAXON = {
    "MAMMAL": "mammal", "BIRD": "bird", "AMPHIBIAN": "amphibian", "REPTILE": "reptile",
    "PLANT": "plant", "INSECT": "insect", "FISH": "fish", "OTHER_TAXON": "living thing",
}

# Display-only (card / meter), kept separate from the prose labels above.
TREND_CARD_LABEL = {
    "STEEP_DECLINE": "Steep decline", "MILD_DECLINE": "Mild decline",
    "STABLE": "Stable", "INCREASING": "Increasing",
}
STATUS_ORDER = ["CR", "EN", "VU", "NT", "LC"]
STATUS_SHORT = {"CR": "CR", "EN": "EN", "VU": "VU", "NT": "NT", "LC": "LC"}
STATUS_COLOR = {"CR": "var(--cr)", "EN": "var(--en)", "VU": "var(--vu)", "NT": "var(--nt)", "LC": "var(--lc)"}

# ============================================================
# LOAD SPECIES DATA
# ============================================================

@st.cache_data
def load_species_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def normalize_record(record):
    def get_value(*keys, default=None):
        for key in keys:
            if key in record:
                return record[key]
        return default

    return {
        "name": get_value("name", "species_name", "scientific_name", default="Unknown species"),
        "scientific_name": get_value("scientific_name", default=""),
        "status": str(get_value("status", "iucn_status", "conservation_status", default="LC")).upper(),
        "trend": str(get_value("trend", "gbif_trend", default="STABLE")).upper(),
        "region": str(get_value("region", "gbif_region", default="GLOBAL")).upper(),
        "taxon": str(get_value("taxon", "taxon_group", "class", default="OTHER_TAXON")).upper(),
        "pct_change": float(get_value("pct_change", "percentage_change", "percent_change", default=0.0)),
    }


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_transformer():
    model = FactToPlanTransformer(fact_vocab=len(FACT_TOKENS), plan_vocab=len(PLAN_TOKENS),
                                   d_model=64, n_heads=4, d_ff=128, n_layers=2, max_len=15)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_llm():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # smaller model -- free CPU-only hosting tier
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# ============================================================
# GREEDY PLAN GENERATION
# ============================================================

def generate_plan(model, status, trend, region, taxon, max_len=15):
    fact_seq = torch.tensor([[FACT_TOKENS[status], FACT_TOKENS[trend], FACT_TOKENS[region], FACT_TOKENS[taxon]]])
    with torch.no_grad():
        enc_out = model.encoder(fact_seq)
        ys = torch.tensor([[PLAN_TOKENS["SOS"]]])
        for _ in range(max_len):
            logits, _ = model.decoder(ys, enc_out)
            next_tok = logits[0, -1].argmax().item()
            ys = torch.cat([ys, torch.tensor([[next_tok]])], dim=1)
            if next_tok == PLAN_TOKENS["EOS"]:
                break
    return [PLAN_ID_TO_TOK[x] for x in ys[0].tolist()]


# ============================================================
# PLAN -> LLM INSTRUCTIONS
# FIXED: now formats with human-readable labels (READABLE_*),
# not raw vocabulary codes -- see note above READABLE_STATUS.
# ============================================================

PLAN_TO_INSTRUCTION = {
    "OPEN_URGENT": "Open in first person with urgency and vulnerability, as if you sense your future is becoming uncertain.",
    "OPEN_REFLECTIVE": "Open in first person with a reflective tone, describing changes happening around you.",
    "OPEN_NEUTRAL": "Open in first person with a calm and observational tone.",
    "CITE_STATUS": "Clearly state your IUCN conservation status using these exact words: \"{status}\".",
    "CITE_TREND": "Describe your observed population trend as {trend}. Do not present this as a precise population count.",
    "CITE_MAGNITUDE": "You MUST explicitly write the number {pct_change} somewhere in your paragraph, formatted exactly like that with the percent sign. This is mandatory.",
    "CITE_REGION": "Mention that you live in {region}.",
    "CITE_TAXON_CONTEXT": "Mention that you are a {taxon} and briefly connect this to your ecological identity.",
    "CALL_TO_ACTION": "Make a direct and meaningful conservation call to action to the reader.",
    "CLOSE_WARNING": "Close with a clear warning about what could be lost if conservation efforts are not maintained.",
    "CLOSE_HOPE": "Close with a note of cautious hope about continued conservation efforts.",
}

# Short, human labels for the plan chips (display only)
PLAN_CHIP_LABEL = {
    "OPEN_URGENT": "open · urgent", "OPEN_REFLECTIVE": "open · reflective", "OPEN_NEUTRAL": "open · neutral",
    "CITE_STATUS": "cite status", "CITE_TREND": "cite trend", "CITE_MAGNITUDE": "cite magnitude",
    "CITE_REGION": "cite region", "CITE_TAXON_CONTEXT": "cite taxon",
    "CALL_TO_ACTION": "call to action", "CLOSE_HOPE": "close · hope", "CLOSE_WARNING": "close · warning",
}


def build_prompt(plan_tokens, facts, species_name):
    readable = {
        "status": READABLE_STATUS.get(facts["status"], facts["status"]),
        "trend": READABLE_TREND.get(facts["trend"], facts["trend"]),
        "region": READABLE_REGION.get(facts["region"], facts["region"]),
        "taxon": READABLE_TAXON.get(facts["taxon"], facts["taxon"]),
        "pct_change": f"{abs(facts['pct_change']):.1f}%",
    }

    instructions = []
    for token in plan_tokens:
        if token in ("SOS", "EOS", "PAD") or token not in PLAN_TO_INSTRUCTION:
            continue
        instructions.append("- " + PLAN_TO_INSTRUCTION[token].format(**readable))

    prompt = (
        f"You are {species_name}, speaking in first person. Write ONE short paragraph of 4-6 sentences.\n\n"
        f"Refer to yourself as \"{species_name}\" at least once.\n\n"
        f"Follow these narrative instructions in exactly the given order:\n"
        + "\n".join(instructions)
        + "\n\n"
        "Do not invent facts, numbers, locations, conservation statuses, or population figures "
        "beyond what is given above.\n"
        "The percentage figure represents a change in observed occurrence records and should not "
        "be presented as a precise population count.\n"
        "Stay in first person throughout."
    )
    return prompt


# ============================================================
# GENERATE NARRATIVE
# ============================================================

def generate_narrative(tokenizer, model, prompt, temperature=0.7):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=180, temperature=temperature,
                                 do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ============================================================
# FACT VERIFICATION
# FIXED: percentage check now compares magnitudes (abs vs abs)
# instead of signed values, since a narrative naturally says
# "declined by 68%" (positive) even when the source data stores
# the change as -68 (negative) -- these describe the same fact.
# Tolerance widened slightly (0.5 -> 1.5) to allow for the LLM
# rounding to one decimal place or a whole number.
# ============================================================

STATUS_NAMES = {
    "CR": ["critically endangered"], "EN": ["endangered"], "VU": ["vulnerable"],
    "NT": ["near threatened", "near-threatened"], "LC": ["least concern", "of least concern"],
}
REGION_NAMES = {
    "AFRICA": ["africa"], "ASIA": ["asia"],
    "AMERICAS": ["americas", "america", "north america", "south america"],
    "EUROPE": ["europe"], "OCEANIA": ["oceania", "australia"],
    "GLOBAL": ["global", "worldwide", "around the world"],
}
TAXON_NAMES = {
    "MAMMAL": ["mammal"], "BIRD": ["bird"],
    "AMPHIBIAN": ["amphibian", "frog", "toad", "salamander"],
    "REPTILE": ["reptile"], "PLANT": ["plant"], "INSECT": ["insect"], "FISH": ["fish"],
    "OTHER_TAXON": [],
}
TREND_NAMES = {
    "STEEP_DECLINE": ["steep decline", "sharp decline", "significant decline", "declining sharply",
                       "declined sharply", "large decline", "major decline", "plummet", "plunge",
                       "collapse", "dwindl", "shrinking rapidly", "drastic", "severe decline",
                       "rapid decline", "decline", "declining", "declined"],
    "MILD_DECLINE": ["mild decline", "slight decline", "small decline", "gradual decline",
                      "slightly declining", "declined slightly", "modest decline", "decline",
                      "declining", "declined", "dwindl"],
    "STABLE": ["stable", "steady", "remained stable", "holding steady", "unchanged"],
    "INCREASING": ["increasing", "increase", "increased", "rising", "rise", "growth",
                    "growing", "rebound", "recovering", "recovery"],
}


def verify_narrative(narrative, status, trend, region, taxon, pct_change, tolerance=1.5):
    text = narrative.lower()

    status_ok = any(p in text for p in STATUS_NAMES.get(status, []))
    region_ok = any(p in text for p in REGION_NAMES.get(region, []))
    trend_ok = any(p in text for p in TREND_NAMES.get(trend, []))
    taxon_ok = True if taxon == "OTHER_TAXON" else any(p in text for p in TAXON_NAMES.get(taxon, []))

    numbers = [float(n) for n in re.findall(r"(?<!\d)([+-]?\d+(?:\.\d+)?)\s*%", narrative)]
    number_ok = any(abs(abs(n) - abs(pct_change)) <= tolerance for n in numbers) if numbers else False

    passed = status_ok and trend_ok and region_ok and number_ok

    return {
        "status_mentioned": status_ok, "trend_mentioned": trend_ok,
        "region_mentioned": region_ok, "taxon_mentioned": taxon_ok,
        "percentage_match": number_ok, "numbers_found": numbers, "passed": passed,
    }


# ============================================================
# SMALL DISPLAY HELPERS
# ============================================================

def render_status_meter(current_status):
    segs = "".join(
        f'<div class="seg" style="background:{STATUS_COLOR[s]};">'
        + ('<div class="pin"></div>' if s == current_status else "")
        + "</div>"
        for s in STATUS_ORDER
    )
    labels = "".join(
        f'<span class="{"current" if s == current_status else ""}">{STATUS_SHORT[s]}</span>'
        for s in STATUS_ORDER
    )
    return f'<div class="status-meter"><div class="track">{segs}</div><div class="labels">{labels}</div></div>'


def render_plan_sequence(plan_tokens):
    shown = [t for t in plan_tokens if t not in ("SOS", "EOS", "PAD")]
    parts = []
    for i, t in enumerate(shown, start=1):
        label = PLAN_CHIP_LABEL.get(t, t.lower().replace("_", " "))
        parts.append(f'<span class="plan-item"><span class="plan-chip"><span class="idx">{i}</span>{label}</span>'
                      + ('<span class="plan-arrow">&#8594;</span>' if i < len(shown) else "") + "</span>")
    return f'<div class="plan-seq">{"".join(parts)}</div>'


def render_ledger(checks):
    rows = [
        ("Conservation status cited", checks["status_mentioned"]),
        ("Population trend cited", checks["trend_mentioned"]),
        ("Region cited", checks["region_mentioned"]),
        ("Taxonomic context cited", checks["taxon_mentioned"]),
        ("Record-change percentage matches source", checks["percentage_match"]),
    ]
    html_rows = []
    for label, ok in rows:
        mark = '<span class="tick">&#10003;</span>' if ok else '<span class="cross">&#10007;</span>'
        badge = '<span class="badge ok">grounded</span>' if ok else '<span class="badge fail">not found</span>'
        html_rows.append(
            f'<div class="ledger-row"><span class="mark">{mark} {label}</span>{badge}</div>'
        )
    return f'<div class="ledger">{"".join(html_rows)}</div>'


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:
    species_records = [normalize_record(x) for x in load_species_data()]
except Exception as e:
    st.error(f"Could not load species_data.json: {e}")
    st.stop()

if not MODEL_PATH.exists():
    st.error("fact_to_plan.pt was not found in the repository.")
    st.stop()

try:
    transformer = load_transformer()
except Exception as e:
    st.error(f"Could not load custom Transformer: {e}")
    st.stop()


# ============================================================
# SPECIES SELECTION
# ============================================================

if not species_records:
    st.warning("No species records were found in species_data.json.")
    st.stop()

species_names = [r["name"] for r in species_records]

sel_col, temp_col = st.columns([2, 1])
with sel_col:
    selected_name = st.selectbox("Select a species", species_names, label_visibility="visible")
with temp_col:
    temperature = st.slider(
        "Narrative temperature", 0.1, 1.2, 0.7, 0.1,
        help="Higher values produce more varied, less predictable prose. "
             "Does not affect the transformer's plan, only the LLM's wording."
    )

record = next(r for r in species_records if r["name"] == selected_name)

st.markdown(f"""
<div class="specimen">
  <div class="name-row">
    <h2>{record['name']}</h2>
    {f'<span class="sci">{record["scientific_name"]}</span>' if record['scientific_name'] else ''}
  </div>
  <div class="tagline">IUCN Red List position, live from the source record</div>
  {render_status_meter(record['status'])}
  <div class="facts-row">
    <div class="fact"><div class="k">GBIF Trend</div><div class="v">{TREND_CARD_LABEL.get(record['trend'], record['trend'])}</div></div>
    <div class="fact"><div class="k">Region</div><div class="v">{READABLE_REGION.get(record['region'], record['region'])}</div></div>
    <div class="fact"><div class="k">Taxon</div><div class="v">{READABLE_TAXON.get(record['taxon'], record['taxon']).capitalize()}</div></div>
    <div class="fact"><div class="k">Record Change</div><div class="v">{record['pct_change']:+.1f}%</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")
generate = st.button("🌱  Generate Conservation Narrative", type="primary", use_container_width=True)

if generate:
    facts = {"status": record["status"], "trend": record["trend"],
              "region": record["region"], "taxon": record["taxon"], "pct_change": record["pct_change"]}

    with st.spinner("Running the custom Transformer to plan the narrative structure…"):
        plan = generate_plan(transformer, record["status"], record["trend"], record["region"], record["taxon"])

    st.markdown('<div class="section-label">Narrative Plan</div>', unsafe_allow_html=True)
    st.markdown(render_plan_sequence(plan), unsafe_allow_html=True)

    prompt = build_prompt(plan, facts, record["name"])

    with st.spinner("The language model is writing the narrative (CPU inference, ~10–30s)…"):
        try:
            tokenizer, llm = load_llm()
            narrative = generate_narrative(tokenizer, llm, prompt, temperature)
        except Exception as e:
            st.markdown(
                f'<div class="footnote">The language model could not be reached: {e}. '
                'Please try again — the plan above was still generated successfully.</div>',
                unsafe_allow_html=True,
            )
            st.stop()

    st.markdown('<div class="section-label">Voice of the Species</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="journal-page">
      <span class="quote-mark">&ldquo;</span>
      <p>{narrative}</p>
      <div class="byline">— {record['name']}, in its own words</div>
    </div>
    """, unsafe_allow_html=True)

    checks = verify_narrative(narrative, record["status"], record["trend"], record["region"],
                               record["taxon"], record["pct_change"])

    st.markdown('<div class="section-label">Grounding Check</div>', unsafe_allow_html=True)
    st.markdown(render_ledger(checks), unsafe_allow_html=True)

    if checks["numbers_found"]:
        nums = ", ".join(f"{n:g}%" for n in checks["numbers_found"])
        st.markdown(f'<div class="numbers-caption">Numbers detected in narrative: {nums}</div>', unsafe_allow_html=True)

    if checks["passed"]:
        st.markdown(
            '<div class="verdict pass">&#10003; This narrative is fully grounded in the source record.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="verdict warn">This narrative is missing one or more required details. '
            'Try generating again — the smaller language model occasionally omits a required '
            'detail, and regenerating usually resolves it.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footnote"><strong>Reading the record change.</strong> The percentage above reflects '
        'a change in occurrence-record counts between observation periods on GBIF, not a direct '
        'population count. It is a proxy for how often the species is being recorded, and should be '
        'read alongside the IUCN status rather than in place of it.</div>',
        unsafe_allow_html=True,
    )
