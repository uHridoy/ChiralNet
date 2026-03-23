import streamlit as st
import base64
import io
import json
import math
from pathlib import Path
from typing import TypedDict, Literal, Optional, List, Dict, Any, Tuple
from openai import OpenAI
from langgraph.graph import StateGraph, END
from PIL import Image

try:
    import torch
except Exception:
    torch = None

try:
    import open_clip
except Exception:
    open_clip = None

st.set_page_config(page_title="Agentic AI for STM analysis", layout="wide")

# LOAD DATASET
def _candidate_dataset_paths() -> List[Path]:
    return [
        Path("curated_dataset.json"),
        Path(__file__).resolve().parent / "curated_dataset.json",
        Path.cwd() / "curated_dataset.json",
        Path("/mnt/data/curated_dataset.json"),
    ]


@st.cache_data
def load_dataset() -> List[Any]:
    for candidate in _candidate_dataset_paths():
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                return loaded
            if isinstance(loaded, dict):
                if isinstance(loaded.get("data"), list):
                    return loaded["data"]
                if isinstance(loaded.get("items"), list):
                    return loaded["items"]
                return [loaded]
    return []


dataset = load_dataset()
CLIP_MODEL_NAME = "ViT-L-14"          # Larger, better visual features than ViT-B-32
CLIP_PRETRAINED_WEIGHTS = "openai"
CLIP_IMAGE_SIZE = 224
RETRIEVAL_TOP_K = 5                   # Retrieve more candidates for richer context
DIDV_SECONDARY_WEIGHT = 0.40          # Slightly higher dI/dV weight (still secondary)
RETRIEVAL_MIN_SCORE_THRESHOLD = 0.10  # Discard near-zero similarity noise hits
TEXT_RETRIEVAL_WEIGHT = 0.20          # Blend in metadata/text similarity

def _dataset_base_dir() -> Path:
    for candidate in _candidate_dataset_paths():
        if candidate.exists():
            return candidate.resolve().parent
    return Path.cwd()


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_flatten_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k}: {_flatten_text(v)}" for k, v in value.items())
    return str(value)


def _entry_identifier(entry: Dict[str, Any], fallback: str) -> str:
    return str(entry.get("datapoint_id"))

def _find_topograph_and_didv_references(entry: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    image_paths = entry.get("image_paths", []) or []

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    topograph_ref = None
    didv_ref = None

    for path in image_paths:
        if not isinstance(path, str):
            continue

        lower = path.lower()

        if topograph_ref is None and "topograph" in lower:
            topograph_ref = path

        if didv_ref is None and "map" in lower:
            didv_ref = path

    return topograph_ref, didv_ref

@st.cache_resource(show_spinner=False)
def load_clip_backbone():
    if torch is None or open_clip is None:
        missing = []
        if torch is None:
            missing.append("torch")
        if open_clip is None:
            missing.append("open_clip_torch")
        return {
            "status": "missing_dependencies",
            "message": f"CLIP retrieval is unavailable because the following packages are missing: {', '.join(missing)}.",
            "model_name": CLIP_MODEL_NAME,
            "pretrained": CLIP_PRETRAINED_WEIGHTS,
        }

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED_WEIGHTS,
            device=device,
        )
        model.eval()
        return {
            "status": "ok",
            "message": "CLIP image retrieval backbone loaded successfully.",
            "model_name": CLIP_MODEL_NAME,
            "pretrained": CLIP_PRETRAINED_WEIGHTS,
            "device": device,
            "model": model,
            "preprocess": preprocess,
        }
    except Exception as exc:
        return {
            "status": "load_failed",
            "message": f"Failed to load CLIP retrieval backbone: {exc}",
            "model_name": CLIP_MODEL_NAME,
            "pretrained": CLIP_PRETRAINED_WEIGHTS,
        }


def _encode_image_with_clip(image: Image.Image, clip_bundle: Dict[str, Any]) -> Optional[List[float]]:
    if not image or clip_bundle.get("status") != "ok":
        return None

    model = clip_bundle.get("model")
    preprocess = clip_bundle.get("preprocess")
    device = clip_bundle.get("device", "cpu")

    if model is None or preprocess is None or torch is None:
        return None

    try:
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().cpu().tolist()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def precompute_dataset_image_index(dataset_payload: tuple):
    clip_bundle = load_clip_backbone()

    if not dataset_payload:
        return {
            "status": "dataset_missing_or_empty",
            "message": "Dataset image indexing was skipped because the dataset is missing or empty.",
            "items": [],
            "count": 0,
            "topograph_indexed": 0,
            "didv_indexed": 0,
            "backbone": {
                "status": clip_bundle.get("status"),
                "model_name": clip_bundle.get("model_name"),
                "pretrained": clip_bundle.get("pretrained"),
                "message": clip_bundle.get("message"),
            },
        }

    if clip_bundle.get("status") != "ok":
        return {
            "status": "clip_unavailable",
            "message": clip_bundle.get("message", "CLIP retrieval backbone is unavailable."),
            "items": [],
            "count": 0,
            "topograph_indexed": 0,
            "didv_indexed": 0,
            "backbone": {
                "status": clip_bundle.get("status"),
                "model_name": clip_bundle.get("model_name"),
                "pretrained": clip_bundle.get("pretrained"),
                "message": clip_bundle.get("message"),
            },
        }

    dataset_dir = _dataset_base_dir()
    indexed_items = []
    topograph_indexed = 0
    didv_indexed = 0

    for idx, raw_entry in enumerate(dataset_payload, start=1):
        try:
            parsed_entry = json.loads(raw_entry)
        except (TypeError, json.JSONDecodeError):
            parsed_entry = raw_entry

        entry = parsed_entry if isinstance(parsed_entry, dict) else {"value": parsed_entry}
        item_id = _entry_identifier(entry, f"entry_{idx}")

        topograph_ref, didv_ref = _find_topograph_and_didv_references(entry)

        topograph_embedding = None
        didv_embedding = None

        if isinstance(topograph_ref, str):
            try:
                p = Path(topograph_ref)
                if not p.is_absolute():
                    p = (dataset_dir / p).resolve()
                if p.exists() and p.is_file():
                    topo_image = Image.open(p).convert("RGB")
                    topograph_embedding = _encode_image_with_clip(topo_image, clip_bundle)
                    if topograph_embedding is not None:
                        topograph_indexed += 1
            except Exception:
                pass

        if isinstance(didv_ref, str):
            try:
                p = Path(didv_ref)
                if not p.is_absolute():
                    p = (dataset_dir / p).resolve()
                if p.exists() and p.is_file():
                    didv_image = Image.open(p).convert("RGB")
                    didv_embedding = _encode_image_with_clip(didv_image, clip_bundle)
                    if didv_embedding is not None:
                        didv_indexed += 1
            except Exception:
                pass

        indexed_items.append({
            "id": item_id,
            "label": entry.get("label"),
            "entry": entry,
            "topograph_available": topograph_embedding is not None,
            "didv_available": didv_embedding is not None,
            "topograph_embedding": topograph_embedding,
            "didv_embedding": didv_embedding,
        })

    usable_items = [item for item in indexed_items if item["topograph_available"] or item["didv_available"]]

    return {
        "status": "ok" if usable_items else "no_dataset_images_found",
        "message": "Dataset image embeddings were precomputed with CLIP." if usable_items else "No retrievable dataset images were found for CLIP indexing.",
        "items": usable_items,
        "count": len(usable_items),
        "topograph_indexed": topograph_indexed,
        "didv_indexed": didv_indexed,
        "backbone": {
            "status": clip_bundle.get("status"),
            "model_name": clip_bundle.get("model_name"),
            "pretrained": clip_bundle.get("pretrained"),
            "message": clip_bundle.get("message"),
        },
    }


clip_backbone = load_clip_backbone()
dataset_image_index = precompute_dataset_image_index(
    tuple(json.dumps(item, sort_keys=True, ensure_ascii=False, default=str) for item in dataset),
)

# OPENAI CLIENT
api_key = st.text_input("OpenAI API Key")
client = OpenAI(api_key=api_key) if api_key else None

st.sidebar.subheader("RAG Status")
st.sidebar.json({
    "Retrieval model": clip_backbone.get("model_name"),
    "RAG database size": dataset_image_index.get("count", 0) if isinstance(dataset_image_index, dict) else 0,
})
# AGENT STATE
class AgentState(TypedDict, total=False):
    input_image_base64: str
    input_didv_base64: Optional[str]
    metadata: Dict[str, Any]
    retrieval_results: List[Dict[str, Any]]
    visual_analysis: Dict[str, Any]
    fourier_analysis: Dict[str, Any]
    chiral_analysis: Dict[str, Any]
    spectroscopy_analysis: Dict[str, Any]
    final_decision: Dict[str, Any]
    final_label: Literal["CDW", "Chiral CDW", "None", "Inconclusive"]
    confidence: float
    explanation: str
    full_final_prompt: str   # ← AUDITABLE
    errors: List[str]

# HELPERS
def b64_from_upload(file):
    if file is None: return None
    return base64.b64encode(file.getvalue()).decode("utf-8")

def _convert_responses_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    if not isinstance(content, list):
        return [{"type": "input_text", "text": str(content)}]

    converted: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            converted.append({"type": "input_text", "text": item})
            continue

        if not isinstance(item, dict):
            converted.append({"type": "input_text", "text": str(item)})
            continue

        item_type = item.get("type")
        if item_type == "text":
            converted.append({"type": "input_text", "text": item.get("text", "")})
        elif item_type == "image_url":
            image_url = item.get("image_url", {}) or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if url:
                converted.append({"type": "input_image", "image_url": url})
        else:
            converted.append({"type": "input_text", "text": json.dumps(item, ensure_ascii=False, default=str)})

    return converted


def call_json_agent(system_prompt: str, user_content: Any, max_output_tokens: int = 700) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("Missing OpenAI API key")

    content = _convert_responses_content(user_content)
    content.insert(0, {
        "type": "input_text",
        "text": "Return JSON only. Output must be a valid JSON object."
    })

    response = client.responses.create(
        model="gpt-5.4",
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        text={"format": {"type": "json_object"}},
        max_output_tokens=max_output_tokens,
    )

    return json.loads(response.output_text)

# RETRIEVAL HELPERS
def _cosine_similarity(vector_a: Optional[List[float]], vector_b: Optional[List[float]]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _summarize_dataset_entry(
    entry: Dict[str, Any],
    rank: int,
    combined_score: float,
    topograph_similarity: float,
    didv_similarity: Optional[float],
    didv_used: bool,
) -> Dict[str, Any]:
    text_fields = [
        entry.get("reason"),
        entry.get("reasoning"),
        entry.get("description"),
        entry.get("notes"),
        entry.get("summary"),
        entry.get("caption"),
        entry.get("annotation"),
        entry.get("comment"),
    ]
    snippet = next((str(value).strip() for value in text_fields if isinstance(value, str) and value.strip()), "")
    if len(snippet) > 300:
        snippet = snippet[:297] + "..."

    return {
        "rank": rank,
        "score": round(combined_score, 4),
        "id": entry.get("datapoint_id"),
        "label": entry.get("label", "Unknown"),
        "topograph_similarity": round(topograph_similarity, 4),
        "didv_similarity": round(didv_similarity, 4) if didv_similarity is not None else None,
        "didv_used": didv_used,
        "retrieval_basis": "Based on topograph with didv" if didv_used else "Based on topograph only",
        "snippet": snippet,
    }


def _metadata_text_similarity(query_meta: Dict[str, Any], candidate_entry: Dict[str, Any]) -> float:
    """Lightweight keyword overlap between query metadata and dataset entry fields."""
    score = 0.0
    total_checks = 0

    # Bias voltage proximity (±20% window)
    q_v = query_meta.get("v_topo")
    c_v = candidate_entry.get("bias") or candidate_entry.get("v_bias") or candidate_entry.get("voltage")
    if q_v is not None and c_v is not None:
        try:
            q_v_f, c_v_f = float(q_v), float(c_v)
            if q_v_f != 0:
                rel_diff = abs(q_v_f - c_v_f) / abs(q_v_f)
                score += max(0.0, 1.0 - rel_diff / 0.20)
            total_checks += 1
        except (TypeError, ValueError):
            pass

    # Scale bar proximity (±30% window)
    q_sc = query_meta.get("scale_topo")
    c_sc = candidate_entry.get("scale") or candidate_entry.get("scale_bar") or candidate_entry.get("scan_size")
    if q_sc is not None and c_sc is not None:
        try:
            q_sc_f, c_sc_f = float(q_sc), float(c_sc)
            if q_sc_f != 0:
                rel_diff = abs(q_sc_f - c_sc_f) / abs(q_sc_f)
                score += max(0.0, 1.0 - rel_diff / 0.30)
            total_checks += 1
        except (TypeError, ValueError):
            pass

    return (score / total_checks) if total_checks > 0 else 0.0


def _compute_retrieval_score(
    query_topograph_embedding: Optional[List[float]],
    query_didv_embedding: Optional[List[float]],
    candidate: Dict[str, Any],
    query_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, Optional[float], bool]:

    topo_sim = _cosine_similarity(query_topograph_embedding, candidate.get("topograph_embedding"))

    # --- ReLU clamp: negative cosine similarity is uninformative noise ---
    topo_sim = max(0.0, topo_sim)

    didv_sim: Optional[float] = None
    didv_used = False
    meta_sim: float = 0.0
    meta_used = False

    # Compute dI/dV similarity if both query and candidate have embeddings
    if query_didv_embedding and candidate.get("didv_embedding"):
        raw_didv = _cosine_similarity(query_didv_embedding, candidate.get("didv_embedding"))
        didv_sim = max(0.0, raw_didv)
        didv_used = True

    # Compute metadata text similarity if query metadata provided
    if query_meta and candidate.get("entry"):
        meta_sim = _metadata_text_similarity(query_meta, candidate["entry"])
        meta_used = meta_sim > 0.0

    # --- Weighted normalized combination ---
    topo_weight = 1.0
    didv_weight = DIDV_SECONDARY_WEIGHT if didv_used else 0.0
    text_weight = TEXT_RETRIEVAL_WEIGHT if meta_used else 0.0
    total_weight = topo_weight + didv_weight + text_weight

    combined_score = (
        topo_weight * topo_sim
        + didv_weight * (didv_sim or 0.0)
        + text_weight * meta_sim
    ) / total_weight

    return combined_score, topo_sim, didv_sim, didv_used

def retriever_agent(state: AgentState):
    if not dataset:
        state["retrieval_results"] = []
        return state

    if clip_backbone.get("status") != "ok":
        state["retrieval_results"] = []
        return state

    indexed_items = dataset_image_index.get("items", []) if isinstance(dataset_image_index, dict) else []
    if not indexed_items:
        state["retrieval_results"] = []
        return state

    # --- Decode query images ---
    query_topograph = None
    try:
        query_topograph = Image.open(
            io.BytesIO(base64.b64decode(state["input_image_base64"]))
        ).convert("RGB")
    except Exception:
        query_topograph = None

    query_didv = None
    if state.get("input_didv_base64"):
        try:
            query_didv = Image.open(
                io.BytesIO(base64.b64decode(state["input_didv_base64"]))
            ).convert("RGB")
        except Exception:
            query_didv = None

    if query_topograph is None:
        state["retrieval_results"] = []
        return state

    # --- Encode query images ---
    query_topograph_embedding = _encode_image_with_clip(query_topograph, clip_backbone)
    query_didv_embedding = _encode_image_with_clip(query_didv, clip_backbone) if query_didv is not None else None

    if query_topograph_embedding is None:
        state["retrieval_results"] = []
        return state

    query_meta = state.get("metadata") or {}

    # --- Score all candidates ---
    scored_results = []
    skipped_low_score = 0
    for item in indexed_items:
        if not item.get("topograph_embedding"):
            continue
        combined_score, topograph_similarity, didv_similarity, didv_used = _compute_retrieval_score(
            query_topograph_embedding,
            query_didv_embedding,
            item,
            query_meta=query_meta,
        )
        # Filter out near-zero noise hits
        if combined_score < RETRIEVAL_MIN_SCORE_THRESHOLD:
            skipped_low_score += 1
            continue
        scored_results.append((combined_score, topograph_similarity, didv_similarity, didv_used, item))

    scored_results.sort(key=lambda row: row[0], reverse=True)

    top_results = []
    for index, (combined_score, topograph_similarity, didv_similarity, didv_used, item) in enumerate(
        scored_results[:RETRIEVAL_TOP_K], start=1
    ):
        summary = _summarize_dataset_entry(
            item.get("entry", {}),
            rank=index,
            combined_score=combined_score,
            topograph_similarity=topograph_similarity,
            didv_similarity=didv_similarity,
            didv_used=didv_used,
        )
        top_results.append(summary)

    state["retrieval_results"] = top_results
    return state

# Core Agents
def visual_analyst(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an expert STM image classifier.

Your task is to classify the CURRENT STM topography image according to the visible morphology in the image itself.
Do NOT override the visible morphology with abstract physical caution.
Do NOT prefer a negative class unless the image clearly lacks the relevant structure.

Critical rule:
The goal is to match the morphology that is visibly present in the CURRENT image, as a human annotator would label it from the topograph itself.

Instructions:
1. First, inspect only the CURRENT image and identify what is visibly present:
   - atomic corrugation
   - stripe-like modulation
   - checkerboard/grid-like modulation
   - triangular/hexagonal superstructure
   - domain pattern
   - defects
   - step edges / terraces
   - drift distortion
   - scan-line noise
   - tip artifact
   - long-wavelength modulation beyond the atomic lattice
   - short-range repeated texture

2. Classify based on the dominant visible morphology, not on strict proof of microscopic origin.
   - If a repeated superstructure or modulation is visibly present, do NOT label the image as “no CDW” or “atomic lattice only” just because the evidence is imperfect.
   - If the image visibly contains a non-atomic repeating pattern, prefer the corresponding modulation/superlattice/CDW-like label.
   - Use negative labels only when the image is truly dominated by lattice-only structure, artifacts, or noise.

3. Distinguish between:
   - real repeated modulation visible in the image
   - isolated defects
   - drift or line-by-line scan artifacts
   - tip distortions
   - random contrast variation
   - pure atomic lattice without larger-scale structure

4. Decision policy:
   - prioritize what a careful human annotator would label from the image appearance
   - if the image weakly but visibly shows the target morphology, prefer “present / likely present” over “absent”
   - only output the opposite label when there is clear visual evidence against the labeled morphology

5. If uncertainty exists, do NOT collapse immediately to the opposite class.
   Instead, state that the morphology is weak/faint/short-ranged but still visually consistent with the class if appropriate.
"""
        },
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]
    system = """
    You are an INDEPENDENT STM Visual Morphology Agent. Analyze the image as the STM topograph. Do not rely on metadata or other agents.
    Rules:
    1. Allowed labels: CDW, None.
    2. Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["visual_analysis"] = call_json_agent(system, prompt)
    return state

def fourier_agent(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an expert STM Fourier-inference classifier.

Your task is to infer the likely Fourier-space interpretation from the CURRENT real-space STM topography image.

Critical rule:
Start from the visible pattern in the CURRENT image and infer the Fourier consequences of that visible pattern.
Do NOT default to “no meaningful Q-vector” when a repeated non-atomic modulation is visibly present, even if it is imperfect, weak, or short-ranged.

Instructions:
1. Inspect the CURRENT real-space image first and identify the visible morphology:
   - atomic lattice only
   - stripe-like / unidirectional modulation
   - checkerboard / bidirectional modulation
   - triangular / hexagonal / 3-direction modulation
   - domains
   - defects
   - step-edge contrast
   - drift / line noise / tip artifact
   - weak but repeated long-wavelength texture

2. Infer Fourier-space features from the visible morphology:
   - Bragg peaks if atomic lattice is resolved
   - CDW/superlattice peaks if a larger-scale repeated modulation is visible
   - broadened peaks if the modulation is short-ranged
   - harmonics if the modulation is strong or non-sinusoidal
   - diffuse scattering if disorder is substantial

3. Important classification rule:
   - If the real-space image visibly shows a repeated non-atomic modulation, infer corresponding nonzero modulation peaks as likely, even if broad or weak.
   - Do NOT say “no reliable CDW-related Q-vector” unless the image truly lacks a repeated non-atomic pattern.
   - If the modulation is weak, describe the Fourier signature as weak/broad/faint rather than absent.

4. Infer likely order type from the visible pattern:
   - 1Q for one dominant modulation direction
   - 2Q for two independent directions
   - 3Q for three symmetry-related directions
   - short-range order if repetition is local or domain-limited
   - no ordered modulation only if the image is dominated by lattice-only contrast, artifacts, or random disorder

5. Analyze symmetry from the visible pattern:
   - preserved or broken rotational symmetry
   - preserved or broken mirror symmetry
   - unidirectional / nematic tendency
   - multi-Q tendency
   - if artifacts dominate, say symmetry is not reliably inferable
"""
        },
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    system = """
    You are an INDEPENDENT STM fourier and symmetry Agent. Do not rely on metadata or other agents.
    Rules:
    1. Allowed labels: CDW, Chiral CDW, None.
    2. Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["fourier_analysis"] = call_json_agent(system, prompt)
    return state

def chiral_specialist(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an STM Chiral CDW classifier.
Label as Chiral CDW if the image shows any repeated, non-random sign of:
- clockwise or anticlockwise local motif arrangement
- handed triangular/hexagonal superlattice texture
- unequal strength of the three symmetry-related CDW directions
- broken mirror symmetry
- local chiral domains or pinwheel-like texture

Key rule:
If repeated triangular or superlattice motifs have unequal lobe brightness arranged with a clockwise or anticlockwise sense, classify as Chiral CDW even if the image is blurry, noisy, low-contrast, or only locally clear.

Do not reject chirality just because:
- the field of view is partial
- the contrast is weak
- the pattern is patchy
- the image is imperfect

Do not confuse chirality with:
- anisotropy alone
- nematicity alone
- 1Q order alone
- scan drift
- scan lines
- tip asymmetry
- random noise

Decision rule:
- any visible repeated handedness or mirror-breaking -> Chiral CDW
- genuinely mirror-equivalent, non-handed pattern -> Not Chiral
- if torn between weak chirality and artifact, prefer Chiral CDW unless artifact is clearly dominant
"""
        },
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    system = """ You are an INDEPENDENT Chiral CDW Specialist Agent. Do not rely on metadata or other agents.
    Rules:
    1. Allowed labels: Chiral CDW, Not Chiral
    2. Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["chiral_analysis"] = call_json_agent(system, prompt)
    return state

def Spectroscopy_correlation_agent(state: AgentState):
    if not state.get("input_didv_base64"):
        state["spectroscopy_analysis"] = {
            "reasoning": "No Image 2 (dI/dV map) was provided, so spectroscopy analysis was skipped."
        }
        return state
    prompt = [
        {
            "type": "text",
            "text": """
You are an STM Spectroscopy agent specialized in interpreting dI/dV maps only. Your task is to analyze the provided dI/dV map as a spatial map of the local density of states (LDOS) and decide whether the evidence is most consistent with: CDW, Chiral CDW, None.

Look for:
- coherent periodic LDOS modulation
- stripe, checkerboard, multi-Q, domain-like, patchy, diffuse, or defect-dominated structure
- whether the pattern is consistent with collective electronic order rather than defects, noise, drift, scan distortion, edge effects, or random inhomogeneity
- whether any symmetry breaking is robust and specific

Call "CDW" only if a reasonably coherent periodic electronic modulation is visible.
Call "Chiral CDW" only if there is convincing CDW-like modulation plus strong, specific chirality evidence beyond ordinary anisotropy.
Prefer None when evidence is weak or non-unique.
"""         },
            {"type": "text", "text": "Image 2: dI/dV map"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{state['input_didv_base64']}"},
            },
    ]
    system = """
        You are an INDEPENDENT STM spectroscopy correlation agent. Analyze the image as the dI/dV map. Do not rely on metadata or other agents.
        Rules:
        1. Allowed labels: CDW, Chiral CDW, None
        2. Output only JSON:
        {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["spectroscopy_analysis"] = call_json_agent(system, prompt)
    return state

def skip_spectroscopy(state: AgentState):
    state["spectroscopy_analysis"] = {
        "reasoning": "No Image 2 (dI/dV map) was provided, so spectroscopy analysis was skipped."
    }
    return state


def spectroscopy_router(state: AgentState) -> str:
    return "run_spectroscopy" if state.get("input_didv_base64") else "skip_spectroscopy"

def _format_retrieval_for_judge(state: AgentState) -> str:
    results = state.get("retrieval_results", []) or []
    if not results:
        return (
            f"No retrieved precedents were available (reason: {reason}). "
            "Do not adjust classification based on missing retrieval."
        )
    lines = []
    for item in results:
        rank = item.get("rank", "?")
        label = item.get("label") or "Unknown"
        identifier = item.get("id", f"retrieved_{rank}")
        score = item.get("score")
        topo_sim = item.get("topograph_similarity")
        didv_sim = item.get("didv_similarity")
        didv_used = item.get("didv_used", False)
        snippet = item.get("snippet") or ""
        retrieval_basis = item.get("retrieval_basis", "Based on topograph only")

        lines.append(f"  Match {rank}")
        lines.append(f"  Dataset ID   : {identifier}")
        lines.append(f"  Label: {label}")
        lines.append(f"  Similarity score: {score}  (topo={topo_sim}" + (f", dI/dV={didv_sim}" if didv_used else "") + ")")
        lines.append(f"  Retrieval basis: {retrieval_basis}")
        if snippet:
            lines.append(f"  Dataset note : {snippet}")
        lines.append("")

    # Majority-label hint (informational, not binding)
    label_counts: Dict[str, int] = {}
    for item in results:
        lbl = item.get("label") or "Unknown"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    majority_label = max(label_counts, key=label_counts.__getitem__)
    lines.append(f"Majority label of similar retrieved cases: {majority_label} ({label_counts})")

    return "\n".join(lines)

def final_judge(state: AgentState):
    retrieval_context = _format_retrieval_for_judge(state)
    full_prompt = f"""
Retriever agent analysis:
{retrieval_context}

Visual agent analysis:
{json.dumps(state.get("visual_analysis", {}), indent=2)}

Fourier agent analysis:
{json.dumps(state.get("fourier_analysis", {}), indent=2)}

Chiral agent analysis:
{json.dumps(state.get("chiral_analysis", {}), indent=2)}

Spectroscopy agent analysis:
{json.dumps(state.get("spectroscopy_analysis", {}), indent=2)}

"""
    state["full_final_prompt"] = full_prompt
    system = """
    You are the final judge. Integrate the specialist-agent outputs together.
    Rules:
    1. Allowed labels: CDW, Chiral CDW, None, Inconclusive.
    2. For CDW and None, the final decision will be determined by the Visual Agent and the Fourier Agent. If both agents produce the same result, it will be accepted as final. If their results conflict, only the result from the agent with the higher confidence score will be counted.
    3. If Rule 2 classifies it as CDW, the next step is to decide whether it is CDW or chiral CDW. The Chiral Agent will make this decision.
    4. If the Spectroscopy Agent provides input, consider it; otherwise, ignore it completely. If its result matches Rule 2, increase the confidence score. If it does not match Rule 2, leave the score unchanged and ignore it.
    5. If the Retriever Agent’s majority label matches the result of Rule 2, increase the confidence score. Otherwise, ignore it.
    Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """
    raw_decision = call_json_agent(system, full_prompt)
    state["final_decision"] = raw_decision
    state["final_label"] = state["final_decision"]["final_label"]
    state["confidence"] = float(state["final_decision"]["confidence"])
    state["explanation"] = state["final_decision"]["explanation"]
    return state

# ===================== BUILD GRAPH =====================
workflow = StateGraph(AgentState)
workflow.add_node("retriever_agent", retriever_agent)
workflow.add_node("visual_analyst", visual_analyst)
workflow.add_node("fourier_agent", fourier_agent)
workflow.add_node("chiral_specialist", chiral_specialist)
workflow.add_node("Spectroscopy_correlation_agent", Spectroscopy_correlation_agent)
workflow.add_node("skip_spectroscopy", skip_spectroscopy)
workflow.add_node("final_judge", final_judge)

workflow.set_entry_point("retriever_agent")
workflow.add_edge("retriever_agent", "visual_analyst")
workflow.add_edge("visual_analyst", "fourier_agent")
workflow.add_edge("fourier_agent", "chiral_specialist")
workflow.add_conditional_edges(
    "chiral_specialist",
    spectroscopy_router,
    {
        "run_spectroscopy": "Spectroscopy_correlation_agent",
        "skip_spectroscopy": "skip_spectroscopy",
    },
)
workflow.add_edge("Spectroscopy_correlation_agent", "final_judge")
workflow.add_edge("skip_spectroscopy", "final_judge")
workflow.add_edge("final_judge", END)

agentic_graph = workflow.compile()

# UI
st.title("STM CDW Classifier with Agentic AI")

col1, col2 = st.columns([2, 1])
with col1:
    topo_file = st.file_uploader("Topograph Image (required)", type=["png", "jpg", "jpeg", "tif"])
    didv_file = st.file_uploader("dI/dV Map (optional)", type=["png", "jpg", "jpeg", "tif"])
    if topo_file:
        st.image(topo_file, width="stretch")

    if didv_file:
        st.image(didv_file, width="stretch")

with col2:
    v_topo  = st.number_input("Topograph Bias (V)",  value=None,
                               format="%.4f", placeholder="-0.05")
    i_topo  = st.number_input("Tunneling Current (pA)", value=None,
                               format="%.1f", placeholder="100")
    scale_topo = st.number_input("Scale Bar (nm)",  value=None,
                               format="%.2f", placeholder="2.0")
    with st.expander("dI/dV parameters", expanded=False):
        v_didv    = st.number_input("dI/dV Bias (V)",    value=None,
                                    format="%.4f")
        didv_i    = st.number_input("dI/dV Current (pA)",value=None,
                                    format="%.1f")
        didv_sc   = st.number_input("dI/dV Scale (nm)",  value=None,
                                    format="%.2f")
        didv_vmod = st.number_input("Modulation (mV)",   value=None,
                                    format="%.2f")

    run = st.button("Run Agentic AI", type="primary", use_container_width=True)

if run:
    if not api_key:
        st.error("Enter your OpenAI API key")
    elif not topo_file:
        st.warning("Upload a Topograph image")
    else:
        with st.spinner("Running full agentic workflow..."):
            topo_b64 = b64_from_upload(topo_file)
            didv_b64 = b64_from_upload(didv_file) if didv_file else None

            initial_state: AgentState = {
                "input_image_base64": topo_b64,
                "input_didv_base64": didv_b64,
                "metadata": {"v_topo": v_topo, "i_topo": i_topo, "scale_topo": scale_topo, "v_didv": v_didv},
                "errors": [],
            }

            result = agentic_graph.invoke(initial_state)

            st.success("Classification complete")
            colA, colB = st.columns([1, 3])
            with colA:
                st.metric("Final Label", result["final_label"], f"{result['confidence']:.1f}%")
            with colB:
                st.info(result["explanation"])

            with st.expander("Complete analysis of the agents submitted to the final judge"):
                st.code(result.get("full_final_prompt", ""), language="markdown")

st.caption("Developed for research purposes")