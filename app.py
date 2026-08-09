import json
import math
import re
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# PAGE CONFIG + STYLE
# ============================================================

st.set_page_config(page_title="Voices of the Vanishing", page_icon="🌿", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 1100px; }

.hero {
    padding: 22px 26px; border-radius: 14px; margin-bottom: 22px;
    background: linear-gradient(135deg, #0f3d2e 0%, #1a5c45 100%);
    color: #f0fdf4;
}
.hero h1 { margin: 0 0 4px 0; font-size: 1.9rem; }
.hero p { margin: 0; opacity: 0.85; font-size: 0.95rem; }

.pipeline-step {
    display: flex; align-items: center; gap: 10px; padding: 7px 0;
    font-size: 0.88rem; color: #334155;
}
.pipeline-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%; background: #1a5c45;
    color: white; font-size: 0.75rem; font-weight: 600; flex-shrink: 0;
}

.fact-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 12px 14px; text-align: center;
}
.fact-card .label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
.fact-card .value { font-size: 1.05rem; color: #0f3d2e; font-weight: 700; margin-top: 2px; }

.plan-chip {
    display: inline-block; padding: 5px 12px; margin: 3px 5px 3px 0; border-radius: 8px;
    background: #ecfdf5; border: 1px solid #a7f3d0; font-size: 0.8rem;
    color: #065f46; font-weight: 500;
}

.narrative-box {
    background: #fffbeb; border-left: 4px solid #d97706; padding: 20px 24px;
    border-radius: 8px; font-size: 1.05rem; line-height: 1.7; font-style: italic;
    color: #451a03;
}

.check-pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 6px 13px;
    border-radius: 20px; font-size: 0.83rem; font-weight: 600; margin: 3px 6px 3px 0;
}
.check-pass { background: #dcfce7; color: #166534; }
.check-fail { background: #fee2e2; color: #991b1b; }

.section-label {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    color: #1a5c45; margin: 22px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🌿 Voices of the Vanishing</h1>
  <p>AI-grounded conservation narrative generator &mdash; MSc Artificial Intelligence &amp; Machine Learning, CIA 3</p>
</div>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fact_to_plan.pt"
DATA_PATH = BASE_DIR / "species_data.json"


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
selected_name = st.selectbox("Select a species", species_names)
record = next(r for r in species_records if r["name"] == selected_name)

st.subheader(f"🌿 {record['name']}")
if record["scientific_name"]:
    st.caption(f"*{record['scientific_name']}*")

c1, c2, c3, c4, c5 = st.columns(5)
fact_values = [
    ("IUCN Status", READABLE_STATUS.get(record["status"], record["status"])),
    ("GBIF Trend", READABLE_TREND.get(record["trend"], record["trend"]).replace("a ", "").capitalize()),
    ("Region", READABLE_REGION.get(record["region"], record["region"])),
    ("Taxon", READABLE_TAXON.get(record["taxon"], record["taxon"]).capitalize()),
    ("Record Change", f"{record['pct_change']:+.1f}%"),
]
for col, (label, value) in zip([c1, c2, c3, c4, c5], fact_values):
    with col:
        st.markdown(f'<div class="fact-card"><div class="label">{label}</div>'
                     f'<div class="value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Generation Settings</div>', unsafe_allow_html=True)
temperature = st.slider("Narrative temperature", 0.1, 1.2, 0.7, 0.1,
                         help="Higher values produce more varied, less predictable prose. "
                              "Does not affect the transformer's plan, only the LLM's wording.")

generate = st.button("🌱 Generate Conservation Narrative", type="primary", use_container_width=True)

if generate:
    facts = {"status": record["status"], "trend": record["trend"],
              "region": record["region"], "taxon": record["taxon"], "pct_change": record["pct_change"]}

    with st.spinner("Running the custom transformer..."):
        plan = generate_plan(transformer, record["status"], record["trend"], record["region"], record["taxon"])

    st.markdown('<div class="section-label">Transformer Narrative Plan</div>', unsafe_allow_html=True)
    chips = "".join(f'<span class="plan-chip">{t}</span>' for t in plan if t not in ("SOS", "EOS"))
    st.markdown(chips, unsafe_allow_html=True)

    prompt = build_prompt(plan, facts, record["name"])

    with st.spinner("Qwen is writing the narrative (CPU inference, ~10-30s)..."):
        try:
            tokenizer, llm = load_llm()
            narrative = generate_narrative(tokenizer, llm, prompt, temperature)
        except Exception as e:
            st.error(f"Language model error: {e}")
            st.stop()

    st.markdown('<div class="section-label">Voice of the Species</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)

    checks = verify_narrative(narrative, record["status"], record["trend"], record["region"],
                               record["taxon"], record["pct_change"])

    st.markdown('<div class="section-label">Fact Verification</div>', unsafe_allow_html=True)
    labels = [("Conservation Status", checks["status_mentioned"]), ("Occurrence Trend", checks["trend_mentioned"]),
              ("Region", checks["region_mentioned"]), ("Taxonomic Context", checks["taxon_mentioned"]),
              ("Percentage Match", checks["percentage_match"])]
    pills = "".join(
        f'<span class="check-pill {"check-pass" if ok else "check-fail"}">{"✓" if ok else "✗"} {label}</span>'
        for label, ok in labels
    )
    st.markdown(pills, unsafe_allow_html=True)

    if checks["numbers_found"]:
        st.caption(f"Numbers detected in narrative: {checks['numbers_found']}")

    if checks["passed"]:
        st.success("Narrative passed all fact-verification checks.")
    else:
        st.warning("Narrative did not pass all verification checks. "
                    "Try regenerating — the LLM occasionally omits a required detail on the smaller model.")

    st.markdown('<div class="section-label">Interpretation</div>', unsafe_allow_html=True)
    st.info("The GBIF percentage shown here represents a change in occurrence-record counts between "
            "observation periods. It should not be interpreted as a direct measurement of population size.")
