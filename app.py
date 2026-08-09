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

st.set_page_config(
    page_title="Voices of the Vanishing",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Voices of the Vanishing")
st.caption(
    "AI-Grounded Conservation Narrative Generator | MSc Artificial Intelligence & Machine Learning"
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fact_to_plan.pt"
DATA_PATH = BASE_DIR / "species_data.json"


# ============================================================
# CUSTOM TRANSFORMER
# Same architecture used in the final notebook
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=50):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0, max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):

        B = query.size(0)

        Q = self.W_q(query).view(
            B, -1, self.n_heads, self.d_k
        ).transpose(1, 2)

        K = self.W_k(key).view(
            B, -1, self.n_heads, self.d_k
        ).transpose(1, 2)

        V = self.W_v(value).view(
            B, -1, self.n_heads, self.d_k
        ).transpose(1, 2)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )

        attn = torch.softmax(
            scores,
            dim=-1
        )

        out = torch.matmul(
            attn,
            V
        )

        out = out.transpose(
            1, 2
        ).contiguous().view(
            B,
            -1,
            self.d_model
        )

        return self.W_o(out), attn


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model,
            n_heads
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        attn_out, _ = self.self_attn(
            x, x, x
        )

        x = self.norm1(
            x + attn_out
        )

        ff_out = self.ff(x)

        x = self.norm2(
            x + ff_out
        )

        return x


class Encoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        max_len
    ):
        super().__init__()

        self.embed = nn.Embedding(
            vocab_size,
            d_model
        )

        self.pos_enc = PositionalEncoding(
            d_model,
            max_len
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model,
                n_heads,
                d_ff
            )
            for _ in range(n_layers)
        ])

    def forward(self, x):

        x = self.embed(x) * math.sqrt(
            self.embed.embedding_dim
        )

        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x)

        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model,
            n_heads
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(
            d_model,
            n_heads
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x,
        enc_out,
        tgt_mask
    ):

        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            mask=tgt_mask
        )

        x = self.norm1(
            x + attn_out
        )

        cross_out, cross_attn = self.cross_attn(
            x,
            enc_out,
            enc_out
        )

        x = self.norm2(
            x + cross_out
        )

        ff_out = self.ff(x)

        x = self.norm3(
            x + ff_out
        )

        return x, cross_attn


class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        max_len
    ):
        super().__init__()

        self.embed = nn.Embedding(
            vocab_size,
            d_model
        )

        self.pos_enc = PositionalEncoding(
            d_model,
            max_len
        )

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model,
                n_heads,
                d_ff
            )
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x, enc_out):

        B, T = x.size()

        causal_mask = torch.tril(
            torch.ones(T, T)
        ).unsqueeze(0).unsqueeze(0)

        x = self.embed(x) * math.sqrt(
            self.embed.embedding_dim
        )

        x = self.pos_enc(x)

        attn_weights = None

        for layer in self.layers:

            x, attn_weights = layer(
                x,
                enc_out,
                causal_mask
            )

        return self.out_proj(x), attn_weights


class FactToPlanTransformer(nn.Module):

    def __init__(
        self,
        fact_vocab,
        plan_vocab,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        max_len=20
    ):
        super().__init__()

        self.encoder = Encoder(
            fact_vocab,
            d_model,
            n_heads,
            d_ff,
            n_layers,
            max_len
        )

        self.decoder = Decoder(
            plan_vocab,
            d_model,
            n_heads,
            d_ff,
            n_layers,
            max_len
        )

    def forward(
        self,
        fact_seq,
        plan_seq_in
    ):

        enc_out = self.encoder(
            fact_seq
        )

        logits, attn = self.decoder(
            plan_seq_in,
            enc_out
        )

        return logits, attn


# ============================================================
# VOCABULARIES
# Same vocabularies used during training
# ============================================================

FACT_TOKENS = {
    "PAD": 0,
    "CR": 1,
    "EN": 2,
    "VU": 3,
    "NT": 4,
    "LC": 5,

    "STEEP_DECLINE": 6,
    "MILD_DECLINE": 7,
    "STABLE": 8,
    "INCREASING": 9,

    "AFRICA": 10,
    "ASIA": 11,
    "AMERICAS": 12,
    "EUROPE": 13,
    "OCEANIA": 14,
    "GLOBAL": 15,

    "MAMMAL": 16,
    "BIRD": 17,
    "AMPHIBIAN": 18,
    "REPTILE": 19,
    "PLANT": 20,
    "INSECT": 21,
    "FISH": 22,
    "OTHER_TAXON": 23,
}


PLAN_TOKENS = {
    "PAD": 0,
    "SOS": 1,
    "EOS": 2,

    "OPEN_URGENT": 3,
    "OPEN_REFLECTIVE": 4,
    "OPEN_NEUTRAL": 5,

    "CITE_STATUS": 6,
    "CITE_TREND": 7,
    "CITE_MAGNITUDE": 8,
    "CITE_REGION": 9,
    "CITE_TAXON_CONTEXT": 10,

    "CALL_TO_ACTION": 11,
    "CLOSE_HOPE": 12,
    "CLOSE_WARNING": 13,
}

PLAN_ID_TO_TOK = {
    v: k
    for k, v in PLAN_TOKENS.items()
}


# ============================================================
# LOAD SPECIES DATA
# ============================================================

@st.cache_data
def load_species_data():

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "species" in data:
            return data["species"]

        if "records" in data:
            return data["records"]

        # Dictionary keyed by species name
        records = []

        for name, value in data.items():

            if isinstance(value, dict):

                item = value.copy()

                if "name" not in item:
                    item["name"] = name

                records.append(item)

        return records

    return []


# ============================================================
# NORMALIZE RECORD
# ============================================================

def normalize_record(record):

    def get_value(*keys, default=None):

        for key in keys:

            if key in record:
                return record[key]

        return default

    return {
        "name": get_value(
            "name",
            "species_name",
            "scientific_name",
            default="Unknown species"
        ),

        "scientific_name": get_value(
            "scientific_name",
            default=""
        ),

        "status": str(
            get_value(
                "status",
                "iucn_status",
                "conservation_status",
                default="LC"
            )
        ).upper(),

        "trend": str(
            get_value(
                "trend",
                "gbif_trend",
                default="STABLE"
            )
        ).upper(),

        "region": str(
            get_value(
                "region",
                "gbif_region",
                default="GLOBAL"
            )
        ).upper(),

        "taxon": str(
            get_value(
                "taxon",
                "taxon_group",
                "class",
                default="OTHER_TAXON"
            )
        ).upper(),

        "pct_change": float(
            get_value(
                "pct_change",
                "percentage_change",
                "percent_change",
                default=0.0
            )
        )
    }


# ============================================================
# CUSTOM TRANSFORMER LOADING
# ============================================================

@st.cache_resource
def load_transformer():

    model = FactToPlanTransformer(
        fact_vocab=len(FACT_TOKENS),
        plan_vocab=len(PLAN_TOKENS),
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        max_len=15
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# DEPLOYMENT LLM
# Smaller Qwen model for CPU hosting
# No API key required
# ============================================================

@st.cache_resource
def load_llm():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    model.eval()

    return tokenizer, model


# ============================================================
# GREEDY PLAN GENERATION
# ============================================================

def generate_plan(
    model,
    status,
    trend,
    region,
    taxon,
    max_len=15
):

    fact_seq = torch.tensor([
        [
            FACT_TOKENS[status],
            FACT_TOKENS[trend],
            FACT_TOKENS[region],
            FACT_TOKENS[taxon]
        ]
    ])

    with torch.no_grad():

        enc_out = model.encoder(
            fact_seq
        )

        ys = torch.tensor([
            [PLAN_TOKENS["SOS"]]
        ])

        for _ in range(max_len):

            logits, _ = model.decoder(
                ys,
                enc_out
            )

            next_tok = (
                logits[0, -1]
                .argmax()
                .item()
            )

            ys = torch.cat(
                [
                    ys,
                    torch.tensor(
                        [[next_tok]]
                    )
                ],
                dim=1
            )

            if next_tok == PLAN_TOKENS["EOS"]:
                break

    return [
        PLAN_ID_TO_TOK[x]
        for x in ys[0].tolist()
    ]


# ============================================================
# PLAN → LLM INSTRUCTIONS
# ============================================================

PLAN_TO_INSTRUCTION = {

    "OPEN_URGENT":
        "Open in first person with urgency and vulnerability, as if you sense that your future is becoming uncertain.",

    "OPEN_REFLECTIVE":
        "Open in first person with a reflective tone, describing changes happening around you.",

    "OPEN_NEUTRAL":
        "Open in first person with a calm and observational tone.",

    "CITE_STATUS":
        "Naturally state your IUCN conservation status: {status}.",

    "CITE_TREND":
        "Describe the observed GBIF occurrence-record trend as {trend}. Do not describe this as a confirmed population measurement.",

    "CITE_MAGNITUDE":
        "State that the number of GBIF occurrence records changed by exactly {pct_change}% between the specified observation periods.",

    "CITE_REGION":
        "Reference the geographic region represented by the available GBIF occurrence records: {region}.",

    "CITE_TAXON_CONTEXT":
        "Mention that you are a {taxon} and briefly connect this to your ecological identity.",

    "CALL_TO_ACTION":
        "Make a direct and meaningful conservation call to action to the reader.",

    "CLOSE_WARNING":
        "Close with a clear warning about what could be lost if conservation efforts are not maintained.",

    "CLOSE_HOPE":
        "Close with a note of cautious hope about continued conservation efforts."
}


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    plan_tokens,
    facts,
    species_name
):

    instructions = []

    for token in plan_tokens:

        if token in (
            "SOS",
            "EOS",
            "PAD"
        ):
            continue

        if token not in PLAN_TO_INSTRUCTION:
            continue

        instruction = PLAN_TO_INSTRUCTION[
            token
        ].format(
            status=facts["status"],
            trend=facts["trend"],
            region=facts["region"],
            taxon=facts["taxon"],
            pct_change=facts["pct_change"]
        )

        instructions.append(
            f"- {instruction}"
        )

    prompt = (
        f"You are {species_name}, speaking in first person. "
        f"Write ONE short paragraph of 4-6 sentences.\n\n"

        f"Follow these narrative instructions in exactly "
        f"the given order:\n"

        + "\n".join(instructions)

        + "\n\n"

        "Do not invent facts, numbers, locations, conservation "
        "statuses, or population measurements.\n"

        "The GBIF percentage represents a change in "
        "occurrence-record counts and must not be presented "
        "as a direct population estimate.\n"

        "Stay in first person throughout."
    )

    return prompt


# ============================================================
# GENERATE NARRATIVE
# ============================================================

def generate_narrative(
    tokenizer,
    model,
    prompt,
    temperature=0.7
):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()


# ============================================================
# FACT VERIFICATION
# ============================================================

STATUS_NAMES = {

    "CR": ["critically endangered"],
    "EN": ["endangered"],
    "VU": ["vulnerable"],
    "NT": [
        "near threatened",
        "near-threatened"
    ],
    "LC": [
        "least concern",
        "of least concern"
    ]
}


REGION_NAMES = {

    "AFRICA": ["africa"],
    "ASIA": ["asia"],

    "AMERICAS": [
        "americas",
        "america",
        "north america",
        "south america"
    ],

    "EUROPE": ["europe"],

    "OCEANIA": [
        "oceania",
        "australia"
    ],

    "GLOBAL": [
        "global",
        "worldwide",
        "around the world"
    ]
}


TAXON_NAMES = {

    "MAMMAL": ["mammal"],
    "BIRD": ["bird"],

    "AMPHIBIAN": [
        "amphibian",
        "frog",
        "toad",
        "salamander"
    ],

    "REPTILE": ["reptile"],
    "PLANT": ["plant"],
    "INSECT": ["insect"],
    "FISH": ["fish"],

    "OTHER_TAXON": []
}


TREND_NAMES = {

    "STEEP_DECLINE": [
        "steep decline",
        "sharp decline",
        "significant decline",
        "declining sharply",
        "declined sharply",
        "large decline",
        "major decline"
    ],

    "MILD_DECLINE": [
        "mild decline",
        "slight decline",
        "small decline",
        "gradual decline",
        "slightly declining",
        "declined slightly"
    ],

    "STABLE": [
        "stable",
        "steady",
        "remained stable"
    ],

    "INCREASING": [
        "increasing",
        "increase",
        "increased",
        "rising",
        "rise",
        "growth"
    ]
}


def verify_narrative(
    narrative,
    status,
    trend,
    region,
    taxon,
    pct_change
):

    text = narrative.lower()

    status_ok = any(
        phrase in text
        for phrase in STATUS_NAMES.get(
            status,
            []
        )
    )

    region_ok = any(
        phrase in text
        for phrase in REGION_NAMES.get(
            region,
            []
        )
    )

    trend_ok = any(
        phrase in text
        for phrase in TREND_NAMES.get(
            trend,
            []
        )
    )

    if taxon == "OTHER_TAXON":

        taxon_ok = True

    else:

        taxon_ok = any(
            phrase in text
            for phrase in TAXON_NAMES.get(
                taxon,
                []
            )
        )

    numbers = [
        float(n)
        for n in re.findall(
            r"(?<!\d)([+-]?\d+(?:\.\d+)?)\s*%",
            narrative
        )
    ]

    number_ok = any(
        abs(
            n - pct_change
        ) <= 0.5
        for n in numbers
    )

    passed = (
        status_ok
        and trend_ok
        and region_ok
        and number_ok
    )

    return {

        "status_mentioned": status_ok,
        "trend_mentioned": trend_ok,
        "region_mentioned": region_ok,
        "taxon_mentioned": taxon_ok,
        "percentage_match": number_ok,
        "numbers_found": numbers,
        "passed": passed
    }


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    species_records = [
        normalize_record(x)
        for x in load_species_data()
    ]

except Exception as e:

    st.error(
        f"Could not load species_data.json: {e}"
    )

    st.stop()


if not MODEL_PATH.exists():

    st.error(
        "fact_to_plan.pt was not found in the repository."
    )

    st.stop()


try:

    transformer = load_transformer()

except Exception as e:

    st.error(
        f"Could not load custom Transformer: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Project Pipeline")

    st.markdown(
        """
        **1. Species Data**

        ↓

        **2. Custom Transformer**

        ↓

        **3. Narrative Plan**

        ↓

        **4. Qwen Language Model**

        ↓

        **5. Fact Verification**
        """
    )

    st.divider()

    st.info(
        "The deployment version uses a smaller Qwen model "
        "for CPU-based Streamlit hosting."
    )


# ============================================================
# SPECIES SELECTION
# ============================================================

if not species_records:

    st.warning(
        "No species records were found in species_data.json."
    )

    st.stop()


species_names = [
    r["name"]
    for r in species_records
]

selected_name = st.selectbox(
    "Select a species",
    species_names
)


record = next(
    r
    for r in species_records
    if r["name"] == selected_name
)


# ============================================================
# DISPLAY FACTS
# ============================================================

st.subheader(
    f"🌿 {record['name']}"
)

if record["scientific_name"]:

    st.caption(
        record["scientific_name"]
    )


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "IUCN Status",
        record["status"]
    )

with col2:
    st.metric(
        "GBIF Trend",
        record["trend"]
    )

with col3:
    st.metric(
        "Region",
        record["region"]
    )

with col4:
    st.metric(
        "Taxon",
        record["taxon"]
    )

with col5:
    st.metric(
        "Record Change",
        f"{record['pct_change']}%"
    )


st.divider()


# ============================================================
# GENERATION SETTINGS
# ============================================================

temperature = st.slider(
    "Narrative temperature",
    0.1,
    1.2,
    0.7,
    0.1
)


if st.button(
    "🌱 Generate Conservation Narrative",
    type="primary"
):

    facts = {

        "status": record["status"],
        "trend": record["trend"],
        "region": record["region"],
        "taxon": record["taxon"],
        "pct_change": record["pct_change"]
    }

    with st.spinner(
        "Generating narrative plan..."
    ):

        plan = generate_plan(
            transformer,
            record["status"],
            record["trend"],
            record["region"],
            record["taxon"]
        )

    st.subheader(
        "🧠 Transformer Narrative Plan"
    )

    st.code(
        " → ".join(
            plan
        )
    )

    prompt = build_prompt(
        plan,
        facts,
        record["name"]
    )

    with st.spinner(
        "Generating conservation narrative..."
    ):

        try:

            tokenizer, llm = load_llm()

            narrative = generate_narrative(
                tokenizer,
                llm,
                prompt,
                temperature
            )

        except Exception as e:

            st.error(
                f"Language model error: {e}"
            )

            st.stop()


    st.subheader(
        "🗣️ Voice of the Species"
    )

    st.write(
        narrative
    )


    # ========================================================
    # VERIFICATION
    # ========================================================

    checks = verify_narrative(
        narrative,
        record["status"],
        record["trend"],
        record["region"],
        record["taxon"],
        record["pct_change"]
    )

    st.divider()

    st.subheader(
        "🔎 Fact Verification"
    )

    v1, v2 = st.columns(2)

    with v1:

        st.write(
            "Conservation Status:",
            "✅" if checks["status_mentioned"]
            else "❌"
        )

        st.write(
            "Occurrence Trend:",
            "✅" if checks["trend_mentioned"]
            else "❌"
        )

        st.write(
            "Region:",
            "✅" if checks["region_mentioned"]
            else "❌"
        )

    with v2:

        st.write(
            "Taxonomic Context:",
            "✅" if checks["taxon_mentioned"]
            else "❌"
        )

        st.write(
            "Percentage:",
            "✅" if checks["percentage_match"]
            else "❌"
        )

        if checks["numbers_found"]:

            st.write(
                "Numbers detected:",
                checks["numbers_found"]
            )


    if checks["passed"]:

        st.success(
            "✅ Narrative passed the fact-verification checks."
        )

    else:

        st.warning(
            "⚠️ Narrative did not pass all verification checks."
        )


    st.divider()

    st.subheader(
        "📌 Interpretation"
    )

    st.info(
        "The GBIF percentage shown here represents a change "
        "in occurrence-record counts between observation "
        "periods. It should not be interpreted as a direct "
        "measurement of population size."
    )
