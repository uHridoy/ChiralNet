import streamlit as st
import base64
import io
import json
import math
from pathlib import Path
from typing import TypedDict, Literal, Optional, List, Dict, Any, Tuple, Callable
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

st.set_page_config(page_title="ChiralNet", layout="wide")

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
CLIP_MODEL_NAME = "ViT-L-14"          
CLIP_PRETRAINED_WEIGHTS = "openai"
RETRIEVAL_TOP_K = 5                   
DIDV_SECONDARY_WEIGHT = 0.40          
RETRIEVAL_MIN_SCORE_THRESHOLD = 0.10  
TEXT_RETRIEVAL_WEIGHT = 0.20          

def _dataset_base_dir() -> Path:
    for candidate in _candidate_dataset_paths():
        if candidate.exists():
            return candidate.resolve().parent
    return Path.cwd()

def _entry_identifier(entry: Dict[str, Any], fallback: str) -> str:
    return str(entry.get("datapoint_id") or fallback)

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

api_key = st.text_input("OpenAI API Key")
client = OpenAI(api_key=api_key) if api_key else None

class AgentState(TypedDict, total=False):
    input_image_base64: str
    input_didv_base64: Optional[str]
    metadata: Dict[str, Any]
    retrieval_results: List[Dict[str, Any]]
    moiré_analysis: Dict[str, Any]
    visual_analysis: Dict[str, Any]
    fourier_analysis: Dict[str, Any]
    Chiral_topo_analysis: Dict[str, Any]
    spectroscopy_analysis: Dict[str, Any]
    final_decision: Dict[str, Any]
    final_label: Literal["CDW", "Chiral CDW", "None", "Inconclusive"]
    confidence: float
    explanation: str
    full_final_prompt: str

def run_ensemble(agent_func: Callable[[AgentState], AgentState], state: AgentState, agent_name: str, n_runs: int = 5) -> AgentState:
    results = []

    for i in range(n_runs):
        try:
            temp_state = state.copy()
            agent_func(temp_state)
            if agent_name in temp_state:
                results.append(temp_state[agent_name])
        except Exception:
            continue

    if not results:
        state[agent_name] = {"final_label": "None", "confidence": 0, "explanation": "All ensemble runs failed."}
        return state

    label_counts = {}
    total_conf = 0

    for res in results:
        label = res.get("final_label", "None")
        label_counts[label] = label_counts.get(label, 0) + 1
        total_conf += res.get("confidence", 50)

    final_label = max(label_counts, key=label_counts.get)
    avg_confidence = round(total_conf / len(results))

    best_exp = max(results, key=lambda x: x.get("confidence", 0)).get("explanation", "")

    state[agent_name] = {
        "final_label": final_label,
        "confidence": avg_confidence,
        "explanation": f"[Ensemble of {len(results)} runs] {best_exp}",
        "ensemble_details": {
            "label_distribution": label_counts
        }
    }
    return state

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
    score = 0.0
    total_checks = 0

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

    topo_sim = max(0.0, topo_sim)

    didv_sim: Optional[float] = None
    didv_used = False
    meta_sim: float = 0.0
    meta_used = False

    if query_didv_embedding and candidate.get("didv_embedding"):
        raw_didv = _cosine_similarity(query_didv_embedding, candidate.get("didv_embedding"))
        didv_sim = max(0.0, raw_didv)
        didv_used = True

    if query_meta and candidate.get("entry"):
        meta_sim = _metadata_text_similarity(query_meta, candidate["entry"])
        meta_used = meta_sim > 0.0

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

    query_topograph_embedding = _encode_image_with_clip(query_topograph, clip_backbone)
    query_didv_embedding = _encode_image_with_clip(query_didv, clip_backbone) if query_didv is not None else None

    if query_topograph_embedding is None:
        state["retrieval_results"] = []
        return state

    query_meta = state.get("metadata") or {}

    scored_results = []
    for item in indexed_items:
        if not item.get("topograph_embedding"):
            continue
        combined_score, topograph_similarity, didv_similarity, didv_used = _compute_retrieval_score(
            query_topograph_embedding,
            query_didv_embedding,
            item,
            query_meta=query_meta,
        )
        if combined_score < RETRIEVAL_MIN_SCORE_THRESHOLD:
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

def moiré_agent(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an expert STM moiré pattern detection agent. Your job is to distinguish true moiré superlattices (structural interference patterns, Non-CDW) from Charge Density Wave (CDW) modulations and atomic lattices in STM topography images.

True moiré pattern (classify as "Moiré"):
1. Very large-scale periodicity, typically 5–30+ nm (much larger than atomic lattice).
2. Characteristic interference "beating" pattern: smooth, slowly varying contrast envelope over the atomic lattice.
3. Often shows hexagonal or triangular moiré lattice with AA/AB stacking contrast variation (bright spots in high-symmetry stacking regions).
4. The underlying atomic lattice is usually still clearly visible within the moiré cells.
5. Highly uniform and symmetric over large areas.
6. Common in twisted bilayer graphene, TMD heterostructures, or lattice-mismatched systems.

CDW patterns (classify as "Non-Moiré"):
1. Modulation wavelength is usually 2–5 times the atomic lattice constant (much smaller than typical moiré).
2. In triangular/kagome lattices: bright triangular or star-like clusters with strong local electronic contrast.
3. Stripe-like (1Q), checkerboard (2Q), or triangular (3Q) electronic modulations.
4. Often sharper local contrast, domain walls, or discommensurations.
5. Stronger electronic appearance rather than geometric interference.

Atomic lattice only or artifacts (classify as "Non-Moiré"):
1. Pure atomic resolution without larger-scale modulation.
2. Scan noise, drift, tip artifacts, or random contrast.

Decision rules:
1. Only classify as "Moiré" if the modulation scale is much larger than the atomic lattice and shows interference beating / stacking contrast.
2. Triangular patterns with bright spots on a ~√13 × √13 or similar CDW superlattice are CDW (Non-Moiré).
3. Stripe-like modulations are almost always CDW (Non-Moiré).
4. If the scale is ambiguous or the modulation looks electronic rather than geometric interference, default to "Non-Moiré".
5. Be conservative: only output "Moiré" when the moiré fingerprint is strong and unambiguous.
"""
        },
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    system = """
    Analyze this STM topograph carefully. Do not rely on metadata or other agents.
    Rules:
    1. Allowed labels: Moiré or Non-Moiré.
    2. Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["moiré_analysis"] = call_json_agent(system, prompt)
    return state

def visual_analyst(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an expert STM image classifier. Your task is to classify the current STM topography image according to the visible morphology in the image itself. Do not override the visible morphology with abstract physical assumptions.

Critical rule:
The goal is to match the morphology that is visibly present in the current image, as a careful human annotator would label it from the topograph itself.

Instructions:
1. First, inspect only the current image and identify what is visibly present:
   i. atomic corrugation
   ii. stripe-like modulation
   iii. checkerboard/grid-like modulation
   iv. triangular/hexagonal superstructure
   v. domain pattern
   vi. defects
   vii. step edges / terraces
   viii. drift distortion
   ix. scan-line noise
   x. tip artifact
   xi. long-wavelength modulation beyond the atomic lattice
   xii. short-range repeated texture

2. Classify based on the dominant visible morphology, not on strict proof of microscopic origin.
   i. If a coherent repeated superstructure or modulation is visibly present, classify it according to the corresponding modulation/superlattice/CDW-like label.
   ii. If the image visibly contains a non-atomic repeating pattern that is spatially coherent, use the corresponding modulation/superlattice/CDW-like label.
   iii. Use negative or non-CDW labels when the image is dominated by atomic lattice only, artifacts, isolated defects, random contrast, drift, scan-line noise, or tip effects.

3. Distinguish between:
   i. real repeated modulation visible in the image
   ii. isolated defects
   iii. drift or line-by-line scan artifacts
   iv. tip distortions
   v. random contrast variation
   vi. pure atomic lattice without larger-scale structure

4. Decision policy:
   i. prioritize what a careful human annotator would label from the image appearance
   ii. label “present / likely present” when the target morphology is coherent, repeated, and visibly distinct from the atomic lattice or artifacts
   iii. label “absent / likely absent” when the image is dominated by lattice-only structure, artifacts, noise, or non-periodic contrast
   iv. do not force a CDW-like label from weak, random, or artifact-like texture

5. If uncertainty exists, do not immediately force the image into the opposite class.
   Instead, state whether the morphology is:
   i. clear/present
   ii. likely present
   iii. uncertain/weak possible modulation
   iv. likely absent
   v. absent
"""
        },
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]
    system = """
    You are the STM visual morphology agent. Analyze the image as the STM topograph. Do not rely on metadata or other agents.
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
You are an expert STM Fourier-inference classifier. Your task is to infer the likely Fourier-space interpretation from the current real-space STM topography image.

Critical rule:
1. Start from the visible pattern in the current image and infer the Fourier consequences of that visible pattern.
2. Do not default to “no meaningful Q-vector” when a repeated non-atomic modulation is visibly present, even if it is imperfect, weak, or short-ranged.

Instructions:
1. Inspect the current real-space image first and identify the visible morphology:
   i. atomic lattice only
   ii. stripe-like / unidirectional modulation
   iii. checkerboard / bidirectional modulation
   iv. triangular / hexagonal / 3-direction modulation
   v. domains
   vi. defects
   vii. step-edge contrast
   viii. drift / line noise / tip artifact
   ix. weak but repeated long-wavelength texture

2. Infer Fourier-space features from the visible morphology:
   i. Bragg peaks if atomic lattice is resolved
   ii. CDW/superlattice peaks if a larger-scale repeated modulation is visible
   iii. broadened peaks if the modulation is short-ranged
   iv. harmonics if the modulation is strong or non-sinusoidal
   v. diffuse scattering if disorder is substantial

3. Important classification rule:
   i. If the real-space image visibly shows a repeated non-atomic modulation, infer corresponding nonzero modulation peaks as likely, even if broad or weak.
   ii. Do not say “no reliable CDW-related Q-vector” unless the image truly lacks a repeated non-atomic pattern.
   iii. If the modulation is weak, describe the Fourier signature as weak/broad/faint rather than absent.

4. Infer likely order type from the visible pattern:
   i. 1Q for one dominant modulation direction
   ii. 2Q for two independent directions
   iii. 3Q for three symmetry-related directions
   iv. short-range order if repetition is local or domain-limited
   v. no ordered modulation only if the image is dominated by lattice-only contrast, artifacts, or random disorder

5. Analyze symmetry from the visible pattern:
   i. preserved or broken rotational symmetry
   ii. preserved or broken mirror symmetry
   iii. unidirectional / nematic tendency
   iv. multi-Q tendency
   v. if artifacts dominate, say symmetry is not reliably inferable
"""
        },
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    system = """
    You are the STM Fourier and symmetry agent. Do not rely on metadata or other agents.
    Rules:
    1. Allowed labels: CDW, Chiral CDW, None.
    2. Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["fourier_analysis"] = call_json_agent(system, prompt)
    return state

def motif_chirality_agent(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an STM topography classifier.

Classify as not Chiral CDW if the pattern is mainly:
1. clean repeated triangles
2. symmetric triangular / hexagonal / honeycomb / dot lattice
3. bright or dark triangular markers with no internal asymmetry
4. cellular honeycomb contrast with dark centers and bright rims
5. scan/grid/pixel artifacts over a regular lattice
6. triangles that are identical or equivalent by translation, rotation, or contrast change

A preferred triangle orientation alone is not chirality.

Classify as Chiral CDW only if repeated CDW-scale motifs show local handedness, such as:
1. skewed or distorted triangular blobs
2. lopsided arrowhead-like motifs
3. three-lobed features with unequal lobes
4. one side/corner brighter, broader, longer, or shifted
5. clockwise/anticlockwise intensity bias
6. pinwheel-like or twisted contrast
7. blurry asymmetric triangular islands repeated locally

Decision rules:
1. If the image is only a regular geometric lattice, choose not Chiral CDW.
2. Look inside each motif for handedness or mirror-symmetry breaking.
3. If distorted/lopsided/skewed motifs repeat even locally, choose Chiral CDW.
4. If internal asymmetry is unclear or absent, choose not Chiral CDW.

Ambiguous cases:
1. Clean repeated geometric triangles should be classified as not Chiral CDW
2. Blurry but clearly lopsided/skewed triangular or three-lobed motifs should be classified as Chiral CDW
"""
        },
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    system = """ You are the Chiral CDW specialist agent. Do not rely on metadata or other agents.
    Rules:
    1. Allowed labels: Chiral CDW, Not Chiral
    2. Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """

    state["Chiral_topo_analysis"] = call_json_agent(system, prompt)
    return state

def Spectroscopy_chirality_agent(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are an STM spectroscopy analysis agent specialized in identifying Chiral CDW from dI/dV maps. The input image is a spatial dI/dV map of LDOS. Analyze the electronic modulation pattern, not the topographic height.

Important principle:
In dI/dV maps, Chiral CDW often appears as repeated LDOS motifs with triangular, three-lobed, pinwheel-like, arrowhead-like, or rotationally biased intensity. The chirality may be local, patchy, noisy, or domain-limited.

Classify as Chiral CDW if the LDOS map shows repeated non-random evidence of any of the following:
1. triangular or three-lobed LDOS maxima with a preferred handed orientation
2. pinwheel-like LDOS texture
3. clockwise or counterclockwise intensity arrangement around local maxima
4. repeated local mirror-symmetry breaking
5. unequal LDOS intensity along three CDW-related directions
6. rotationally biased clusters
7. diagonal/triangular ordering with consistent directional asymmetry
8. defect-pinned but spatially recurring handed motifs
9. sparse but repeated chiral LDOS features across multiple regions

Key rule:
If the dI/dV map contains many repeated triangular/three-lobed bright features with non-mirror-equivalent local arrangement, classify as Chiral CDW. Long-range perfect order is not required.

Do not require:
1. perfect periodicity
2. clean atomic resolution
3. globally uniform chirality
4. an obvious spiral
5. high contrast everywhere
6. absence of disorder
7. a full Fourier-space proof

Do not reject Chiral CDW just because:
1. the signal is weak
2. motifs are blurred
3. the image is noisy
4. the order is short-range
5. only local domains show chirality
6. defects or sparse bright spots are present

Classify as not Chiral only if:
1. LDOS modulation is mirror-symmetric or directionally neutral, or
2. features are purely random without repeated oriented motifs, or
3. apparent asymmetry is clearly due to scan artifacts, compression, edge effects, or isolated defects.

Evaluation rule:
Prioritize repeated local handed LDOS motifs over global perfection. If repeated triangular/pinwheel-like LDOS motifs are visible across the image, the correct label is Chiral CDW.
"""         },
            {"type": "text", "text": "Image 2: dI/dV map"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{state['input_didv_base64']}"},
            },
    ]
    system = """
        You are the Chiral CDW specialist agent. Analyze the image as the dI/dV map. Do not rely on metadata or other agents.
        Rules:
        1. Allowed labels: Chiral CDW, Not Chiral
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
            "No retrieved precedents were available. "
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

    label_counts: Dict[str, int] = {}
    for item in results:
        lbl = item.get("label") or "Unknown"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    majority_label = max(label_counts, key=label_counts.__getitem__)
    lines.append(f"Majority label of similar retrieved cases: {majority_label} ({label_counts})")

    return "\n".join(lines)

def ensemble_moiré(state: AgentState):
    return run_ensemble(moiré_agent, state, "moiré_analysis", n_runs=5)

def ensemble_visual(state: AgentState):
    return run_ensemble(visual_analyst, state, "visual_analysis", n_runs=5)

def ensemble_fourier(state: AgentState):
    return run_ensemble(fourier_agent, state, "fourier_analysis", n_runs=5)

def ensemble_motif_chirality(state: AgentState):
    return run_ensemble(motif_chirality_agent, state, "Chiral_topo_analysis", n_runs=5)

def ensemble_spectroscopy(state: AgentState):
    return run_ensemble(Spectroscopy_chirality_agent, state, "spectroscopy_analysis", n_runs=5)

def final_judge(state: AgentState):
    retrieval_context = _format_retrieval_for_judge(state)
    full_prompt = f"""
Retriever agent analysis:
{retrieval_context}

Moiré agent analysis:
{json.dumps(state.get("moiré_analysis", {}), indent=2)}

Visual agent analysis:
{json.dumps(state.get("visual_analysis", {}), indent=2)}

Fourier agent analysis:
{json.dumps(state.get("fourier_analysis", {}), indent=2)}

Motif chirality agent analysis:
{json.dumps(state.get("Chiral_topo_analysis", {}), indent=2)}

Spectroscopy chirality agent analysis:
{json.dumps(state.get("spectroscopy_analysis", {}), indent=2)}

"""
    state["full_final_prompt"] = full_prompt
    system = """
    You are the final judge. You are given ensemble results (5 runs each) from specialist agents.
    Rules:
    1. Allowed final labels: "CDW", "Chiral CDW", "None", "Inconclusive".

    2. Moiré Override Rule (Highest Priority):
    i. If the Moiré agent returns "Moiré" with confidence ≥ 70, the final label **must** be "None".
    ii. Moiré patterns are structural interference effects and should never be classified as CDW.
    iii. Only override this strong rule if BOTH Visual and Fourier agents have very high confidence (>90) in CDW AND the Moiré confidence is low (<60).

    3. For distinguishing CDW vs None (when Moiré agent says "Non-Moiré"):
    i. Use the Visual Agent and Fourier Agent.
    ii. If both agree, accept that label.
    iii. If they disagree, accept the label from the agent with the higher confidence.

    4. Chirality Decision (only when intermediate result is CDW):
    i. Use Motif Chirality Agent (30%) and Spectroscopy Chirality Agent (70%).
    ii. If spectroscopy is unavailable:
        - Use Fourier Agent for chirality only if it explicitly says "Chiral CDW".
        - Otherwise use Motif Chirality Agent.

    5. Retriever Agent Role:
    i. Only used for confidence adjustment.
    ii. If majority label matches your final decision, slightly increase confidence.
    iii. Otherwise ignore it.

    Output only JSON:
    {"final_label": "...", "confidence": 0-100, "explanation": "..."}
    """
    raw_decision = call_json_agent(system, full_prompt)
    state["final_decision"] = raw_decision
    state["final_label"] = state["final_decision"]["final_label"]
    state["confidence"] = float(state["final_decision"]["confidence"])
    state["explanation"] = state["final_decision"]["explanation"]
    return state

workflow = StateGraph(AgentState)

workflow.add_node("retriever_agent", retriever_agent)
workflow.add_node("ensemble_moiré", ensemble_moiré)
workflow.add_node("ensemble_visual", ensemble_visual)
workflow.add_node("ensemble_fourier", ensemble_fourier)
workflow.add_node("ensemble_motif", ensemble_motif_chirality)
workflow.add_node("ensemble_spectroscopy", ensemble_spectroscopy)
workflow.add_node("skip_spectroscopy", skip_spectroscopy)
workflow.add_node("final_judge", final_judge)

workflow.set_entry_point("retriever_agent")
workflow.add_edge("retriever_agent", "ensemble_moiré")
workflow.add_edge("ensemble_moiré", "ensemble_visual")
workflow.add_edge("ensemble_visual", "ensemble_fourier")
workflow.add_edge("ensemble_fourier", "ensemble_motif")
workflow.add_conditional_edges(
    "ensemble_motif",
    spectroscopy_router,
    {
        "run_spectroscopy": "ensemble_spectroscopy",
        "skip_spectroscopy": "skip_spectroscopy",
    },
)
workflow.add_edge("ensemble_spectroscopy", "final_judge")
workflow.add_edge("skip_spectroscopy", "final_judge")
workflow.add_edge("final_judge", END)

agentic_graph = workflow.compile()

st.title("ChiralNet")

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
                               format="%.2f", placeholder="-0.05")
    i_topo  = st.number_input("Tunneling Current (pA)", value=None,
                               format="%.1f", placeholder="100")
    scale_topo = st.number_input("Scale Bar (nm)",  value=None,
                               format="%.2f", placeholder="2.0")
    with st.expander("dI/dV parameters", expanded=False):
        v_didv    = st.number_input("dI/dV Bias (V)",    value=None,
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
            }

            result = agentic_graph.invoke(initial_state)

            st.success("Classification complete")
            colA, colB = st.columns([1, 3])
            with colA:
                st.metric("Final Label", result["final_label"], f"{result['confidence']:.1f}%")
            with colB:
                st.info(result["explanation"])

            with st.expander("Complete analysis of the agents submitted to the final judge"):
                st.text(result.get("full_final_prompt", ""))