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

st.set_page_config(page_title="STM CDW Agentic LLM v3", layout="wide", page_icon="🔬")

# ===================== SIDEBAR =====================
st.sidebar.title("🔬 STM CDW Agentic LLM v3")
st.sidebar.caption("Reviewer-Fixed: Independent agents + Auditable final prompt")

# ===================== LOAD DATASET =====================
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
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED_WEIGHTS = "openai"
CLIP_IMAGE_SIZE = 224
RETRIEVAL_TOP_K = 3
DIDV_SECONDARY_WEIGHT = 0.35

TOPOGRAPH_IMAGE_KEYS = [
    "topograph_image", "topography_image", "topo_image", "stm_image", "image",
    "topograph_path", "topography_path", "topo_path", "stm_path", "image_path",
    "topograph_file", "topography_file", "topo_file", "stm_file", "file",
    "topograph_base64", "topography_base64", "topo_base64", "stm_base64", "image_base64",
    "topograph", "topography", "topo", "stm"
]

DIDV_IMAGE_KEYS = [
    "didv_image", "didv_map", "didv_path", "didv_file", "didv_base64",
    "spectroscopy_image", "spectroscopy_map", "ldos_image", "ldos_map"
]


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
    return str(entry.get("id") or entry.get("sample_id") or entry.get("name") or fallback)


def _normalize_image_data_url(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    if value.startswith("data:image/"):
        return value
    compact = "".join(value.split())
    try:
        base64.b64decode(compact, validate=True)
        return f"data:image/png;base64,{compact}"
    except Exception:
        return None


def _image_bytes_from_reference(reference: Any, dataset_dir: Path) -> Optional[bytes]:
    if reference is None:
        return None

    if isinstance(reference, bytes):
        return reference

    if isinstance(reference, str):
        value = reference.strip()
        if not value:
            return None

        if value.startswith("data:image/"):
            try:
                return base64.b64decode(value.split(",", 1)[1])
            except Exception:
                return None

        maybe_b64 = _normalize_image_data_url(value)
        if maybe_b64:
            try:
                return base64.b64decode(maybe_b64.split(",", 1)[1])
            except Exception:
                return None

        candidate_path = Path(value)
        if not candidate_path.is_absolute():
            candidate_path = (dataset_dir / candidate_path).resolve()

        if candidate_path.exists() and candidate_path.is_file():
            try:
                return candidate_path.read_bytes()
            except Exception:
                return None

    return None


def _pil_from_reference(reference: Any, dataset_dir: Path) -> Optional[Image.Image]:
    raw = _image_bytes_from_reference(reference, dataset_dir)
    if raw is None:
        return None
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return image
    except Exception:
        return None


def _extract_nested_image_reference(payload: Any, candidate_keys: List[str]) -> Optional[Any]:
    if isinstance(payload, dict):
        lowered = {str(key).lower(): key for key in payload.keys()}
        for key in candidate_keys:
            if key in lowered:
                return payload[lowered[key]]

        for nested_key in ["images", "image_paths", "files", "modalities", "maps", "assets"]:
            nested = payload.get(nested_key)
            found = _extract_nested_image_reference(nested, candidate_keys)
            if found is not None:
                return found

        for nested in payload.values():
            if isinstance(nested, (dict, list)):
                found = _extract_nested_image_reference(nested, candidate_keys)
                if found is not None:
                    return found

    if isinstance(payload, list):
        for item in payload:
            found = _extract_nested_image_reference(item, candidate_keys)
            if found is not None:
                return found

    return None


def _find_image_reference(entry: Dict[str, Any], candidate_keys: List[str]) -> Optional[Any]:
    lowered_keys = [key.lower() for key in candidate_keys]
    return _extract_nested_image_reference(entry, lowered_keys)

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

        if didv_ref is None and (
            "didv" in lower
            or "_map" in lower
            or "map.png" in lower
            or "ldos" in lower
        ):
            didv_ref = path

    if topograph_ref is None:
        topograph_ref = _find_image_reference(entry, TOPOGRAPH_IMAGE_KEYS)

    if didv_ref is None:
        didv_ref = _find_image_reference(entry, DIDV_IMAGE_KEYS)

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

        if topograph_ref is not None:
            topo_image = _pil_from_reference(topograph_ref, dataset_dir)
            if topo_image is not None:
                topograph_embedding = _encode_image_with_clip(topo_image, clip_bundle)
                if topograph_embedding is not None:
                    topograph_indexed += 1

        if didv_ref is not None:
            didv_image = _pil_from_reference(didv_ref, dataset_dir)
            if didv_image is not None:
                didv_embedding = _encode_image_with_clip(didv_image, clip_bundle)
                if didv_embedding is not None:
                    didv_indexed += 1

        indexed_items.append({
            "id": item_id,
            "label": entry.get("label"),
            "material": entry.get("material") or entry.get("system") or entry.get("compound"),
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

# ===================== OPENAI CLIENT =====================
api_key = st.text_input("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

st.sidebar.subheader("RAG Status")
st.sidebar.json({
    "dataset_entries": len(dataset),
    "retrieval_backbone_status": clip_backbone.get("status"),
    "retrieval_backbone_model": clip_backbone.get("model_name"),
    "retrieval_backbone_weights": clip_backbone.get("pretrained"),
    "index_status": dataset_image_index.get("status") if isinstance(dataset_image_index, dict) else "unavailable",
    "indexed_candidates": dataset_image_index.get("count", 0) if isinstance(dataset_image_index, dict) else 0,
    "indexed_topographs": dataset_image_index.get("topograph_indexed", 0) if isinstance(dataset_image_index, dict) else 0,
    "indexed_didv_maps": dataset_image_index.get("didv_indexed", 0) if isinstance(dataset_image_index, dict) else 0,
})
# ===================== AGENT STATE =====================
class AgentState(TypedDict, total=False):
    input_image_base64: str
    input_didv_base64: Optional[str]
    metadata: Dict[str, Any]
    retrieval_results: List[Dict[str, Any]]
    retrieval_summary: Dict[str, Any]
    retrieval_audit_log: Dict[str, Any]
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

# ===================== HELPERS =====================
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


def call_json_agent(system_prompt: str, user_content: Any, temperature: float = 0.0, max_output_tokens: int = 700) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("Missing OpenAI API key")

    response = client.responses.create(
        model="gpt-5.4",
        instructions=system_prompt + "\nReturn the response in valid JSON.",
        input=[
            {
                "role": "user",
                "content": _convert_responses_content(user_content),
            }
        ],
        text={"format": {"type": "json_object"}},
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    return json.loads(response.output_text)

# ===================== RETRIEVAL HELPERS =====================

# ===================== RETRIEVAL HELPERS =====================
def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
    ]
    snippet = next((str(value).strip() for value in text_fields if isinstance(value, str) and value.strip()), "")
    if len(snippet) > 220:
        snippet = snippet[:217] + "..."

    return {
        "rank": rank,
        "score": round(combined_score, 4),
        "id": entry.get("id") or entry.get("sample_id") or entry.get("name") or f"entry_{rank}",
        "label": entry.get("label", "Unknown"),
        "material": entry.get("material") or entry.get("system") or entry.get("compound"),
        "topograph_similarity": round(topograph_similarity, 4),
        "didv_similarity": round(didv_similarity, 4) if didv_similarity is not None else None,
        "didv_used": didv_used,
        "retrieval_basis": "topograph_primary_plus_didv_secondary" if didv_used else "topograph_primary_only",
        "snippet": snippet,
    }


def _compute_retrieval_score(
    query_topograph_embedding: Optional[List[float]],
    query_didv_embedding: Optional[List[float]],
    candidate: Dict[str, Any],
) -> Tuple[float, float, Optional[float], bool]:
    topograph_similarity = _cosine_similarity(query_topograph_embedding, candidate.get("topograph_embedding"))
    didv_similarity = None
    didv_used = False
    combined_score = topograph_similarity

    if query_didv_embedding and candidate.get("didv_embedding"):
        didv_similarity = _cosine_similarity(query_didv_embedding, candidate.get("didv_embedding"))
        combined_score += DIDV_SECONDARY_WEIGHT * didv_similarity
        didv_used = True

    return combined_score, topograph_similarity, didv_similarity, didv_used


def _format_retrieved_context(state: AgentState) -> str:
    retrieval_results = state.get("retrieval_results", []) or []
    retrieval_summary = state.get("retrieval_summary", {}) or {}

    if not retrieval_results:
        status = retrieval_summary.get("status", "unavailable")
        message = retrieval_summary.get("message", "No retrieved dataset examples were available.")
        return f"No retrieved dataset examples were available. Status: {status}. Message: {message}"

    lines = [
        "Retrieved dataset examples for grounding:",
        f"retrieval_status: {retrieval_summary.get('status', 'ok')}",
        f"retrieval_backbone: {retrieval_summary.get('retrieval_backbone', 'unknown')}",
        f"ranking_mode: {retrieval_summary.get('ranking_mode', 'unknown')}",
        "retrieval_policy: Image-only retrieval. Topograph similarity is the primary signal; dI/dV similarity is secondary when both query and candidate dI/dV maps are available.",
    ]

    for item in retrieval_results:
        lines.append(
            "- rank={rank}; id={id}; label={label}; material={material}; score={score}; topograph_similarity={topo}; didv_similarity={didv}; didv_used={didv_used}; retrieval_basis={basis}; snippet={snippet}".format(
                rank=item.get("rank"),
                id=item.get("id"),
                label=item.get("label"),
                material=item.get("material"),
                score=item.get("score"),
                topo=item.get("topograph_similarity"),
                didv=item.get("didv_similarity"),
                didv_used=item.get("didv_used"),
                basis=item.get("retrieval_basis"),
                snippet=item.get("snippet") or "",
            )
        )

    return "\n".join(lines)

def retriever_agent(state: AgentState):
    if not dataset:
        state["retrieval_results"] = []
        state["retrieval_summary"] = {
            "status": "dataset_missing_or_empty",
            "message": "curated_dataset.json was not found or contained no retrievable entries.",
            "num_candidates_scored": 0,
            "top_k": RETRIEVAL_TOP_K,
            "retrieval_backbone": "CLIP",
        }
        return state

    if clip_backbone.get("status") != "ok":
        state["retrieval_results"] = []
        state["retrieval_summary"] = {
            "status": "clip_unavailable",
            "message": clip_backbone.get("message", "CLIP retrieval backbone is unavailable."),
            "num_candidates_scored": 0,
            "top_k": RETRIEVAL_TOP_K,
            "retrieval_backbone": "CLIP",
        }
        return state

    indexed_items = dataset_image_index.get("items", []) if isinstance(dataset_image_index, dict) else []
    if not indexed_items:
        state["retrieval_results"] = []
        state["retrieval_summary"] = {
            "status": dataset_image_index.get("status", "no_dataset_images_found") if isinstance(dataset_image_index, dict) else "no_dataset_images_found",
            "message": dataset_image_index.get("message", "No dataset images were available for CLIP-based retrieval.") if isinstance(dataset_image_index, dict) else "No dataset images were available for CLIP-based retrieval.",
            "num_candidates_scored": 0,
            "top_k": RETRIEVAL_TOP_K,
            "retrieval_backbone": "CLIP",
        }
        return state

    dataset_dir = _dataset_base_dir()
    query_topograph = _pil_from_reference(f"data:image/png;base64,{state['input_image_base64']}", dataset_dir)
    query_didv = None
    if state.get("input_didv_base64"):
        query_didv = _pil_from_reference(f"data:image/png;base64,{state['input_didv_base64']}", dataset_dir)

    if query_topograph is None:
        state["retrieval_results"] = []
        state["retrieval_summary"] = {
            "status": "query_topograph_unreadable",
            "message": "The uploaded topograph could not be decoded for CLIP retrieval.",
            "num_candidates_scored": 0,
            "top_k": RETRIEVAL_TOP_K,
            "retrieval_backbone": "CLIP",
        }
        return state

    query_topograph_embedding = _encode_image_with_clip(query_topograph, clip_backbone)
    query_didv_embedding = _encode_image_with_clip(query_didv, clip_backbone) if query_didv is not None else None

    if query_topograph_embedding is None:
        state["retrieval_results"] = []
        state["retrieval_summary"] = {
            "status": "query_embedding_failed",
            "message": "The uploaded topograph could not be embedded by the CLIP retrieval backbone.",
            "num_candidates_scored": 0,
            "top_k": RETRIEVAL_TOP_K,
            "retrieval_backbone": "CLIP",
        }
        return state

    scored_results = []
    for item in indexed_items:
        if not item.get("topograph_embedding"):
            continue
        combined_score, topograph_similarity, didv_similarity, didv_used = _compute_retrieval_score(
            query_topograph_embedding,
            query_didv_embedding,
            item,
        )
        scored_results.append((combined_score, topograph_similarity, didv_similarity, didv_used, item))

    scored_results.sort(key=lambda row: row[0], reverse=True)
    top_results = []
    for index, (combined_score, topograph_similarity, didv_similarity, didv_used, item) in enumerate(scored_results[:RETRIEVAL_TOP_K], start=1):
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
    state["retrieval_summary"] = {
        "status": "ok" if top_results else "no_scored_candidates",
        "message": "Top dataset examples were retrieved using CLIP image similarity. Topograph similarity is the primary signal; dI/dV similarity is a secondary signal when available.",
        "num_candidates_scored": len(scored_results),
        "top_k": RETRIEVAL_TOP_K,
        "ranking_mode": "image_similarity_topograph_primary_didv_secondary",
        "retrieval_backbone": "CLIP",
        "clip_model": clip_backbone.get("model_name"),
        "clip_weights": clip_backbone.get("pretrained"),
        "didv_secondary_weight": DIDV_SECONDARY_WEIGHT,
        "query_has_didv": query_didv_embedding is not None,
    }
    return state

# ===================== AGENTS (INDEPENDENT + TEMPERATURE) =====================
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
Retrieved examples are only supporting references or hard negatives. They must never dominate the decision.

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

3. Use retrieved examples only as reference patterns:
   - they can support a match
   - they can act as hard negatives
   - but they cannot force the current image into their class
   - do not copy their label automatically

4. Distinguish between:
   - real repeated modulation visible in the image
   - isolated defects
   - drift or line-by-line scan artifacts
   - tip distortions
   - random contrast variation
   - pure atomic lattice without larger-scale structure

5. Decision policy:
   - prioritize what a careful human annotator would label from the image appearance
   - if the image weakly but visibly shows the target morphology, prefer “present / likely present” over “absent”
   - only output the opposite label when there is clear visual evidence against the labeled morphology

6. If uncertainty exists, do NOT collapse immediately to the opposite class.
   Instead, state that the morphology is weak/faint/short-ranged but still visually consistent with the class if appropriate.
Return strict JSON with exactly:
{
  "reasoning": "brief explanation grounded only in visible morphology"
}
"""
        },
        {"type": "text", "text": f"Retrieved context:\n{_format_retrieved_context(state)}"},
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    system = (
        "You are an INDEPENDENT STM Visual Morphology Agent. "
        "Analyze the image as the STM topograph. "
        "Use retrieved dataset examples only as grounding context. "
        "Do not rely on metadata or other agents. "
        "Be conservative and return only valid JSON."
    )

    state["visual_analysis"] = call_json_agent(system, prompt, temperature=0.0)
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

Return strict JSON with exactly:
{
  "reasoning": "brief explanation grounded only in visible morphology"
}
"""
        },
        {"type": "text", "text": f"Retrieved context:\n{_format_retrieved_context(state)}"},
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    system = (
        "You are an INDEPENDENT STM fourier and symmetry Agent. "
        "Use retrieved dataset examples only as grounding context. "
        "Do not rely on metadata or other agents. "
        "Be conservative and return only valid JSON."
    )

    state["fourier_analysis"] = call_json_agent(system, prompt, temperature=0.0)
    return state

def chiral_specialist(state: AgentState):
    prompt = [
        {
            "type": "text",
            "text": """
You are the STM Chiral Specialist.

Your task is to decide whether the CURRENT STM topography image supports:
- Chiral CDW
- Not Chiral

Critical rule:
If the CURRENT image visibly shows a handed arrangement of the CDW/superlattice pattern, unequal prominence among symmetry-related modulation directions, or a clockwise/anticlockwise sense in the texture, prefer "Chiral CDW" even if the evidence is not perfectly clean.
Do NOT default to "Not Chiral" merely because the pattern is weak, noisy, or imperfect.

Use retrieved dataset examples only as supporting references or hard negatives.
Never inherit their labels unless the CURRENT image visibly matches them.

Instructions:
1. Inspect only the CURRENT image first.
   Identify whether the visible pattern shows:
   - a handed clockwise or anticlockwise arrangement
   - unequal strength among the symmetry-related CDW directions
   - broken mirror symmetry
   - broken rotational equivalence between symmetry-related directions
   - chiral domain texture
   - anisotropy only
   - nematic/1Q order only
   - drift
   - scan distortion
   - tip asymmetry
   - moire/reconstruction
   - weak random multi-Q noise

2. Base the decision on visible morphology, not on strict proof of microscopic origin.
   - If the pattern visibly has handedness, spiraling sense, or inequivalent symmetry-related directions, this supports Chiral CDW.
   - If the image shows only elongation, one preferred axis, or simple anisotropy without handedness, do not call it chiral.
   - If apparent symmetry breaking is more consistent with drift, scan distortion, tip shape, or edge effects, do not call it chiral.

3. Important decision rule:
   - Visible handedness or clockwise/counterclockwise inequivalence -> favor Chiral CDW.
   - Unequal strengths among three symmetry-related directions, together with broken mirror symmetry -> favor Chiral CDW.
   - Weak but visible chiral texture -> still allow Chiral CDW with lower confidence.
   - Do not collapse weak chiral evidence into Not Chiral unless artifact explanation is clearly stronger.

4. Distinguish chirality from non-chiral alternatives:
   - anisotropy alone is not chirality
   - nematicity alone is not chirality
   - 1Q order alone is not chirality
   - simple multi-Q order with preserved mirror equivalence is not chirality
   - drift, scan-line effects, tip asymmetry, moire, reconstruction, and random noisy contrast are not chirality

5. Be conservative about artifacts, but not overly dismissive.
   The goal is to label the visible morphology as a human STM annotator would.
   If the image looks chiral in its spatial arrangement, do not reject it just because the evidence is imperfect.
Return strict JSON with exactly:
{
  "reasoning": "brief explanation grounded only in visible morphology"
}
"""
        },
        {"type": "text", "text": f"Retrieved context:\n{_format_retrieved_context(state)}"},
        {"type": "text", "text": "Image 1: STM topograph"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{state['input_image_base64']}"},
        },
    ]

    system = (
        "You are an INDEPENDENT Chiral CDW Specialist Agent. "
        "Use retrieved dataset examples only as grounding context. "
        "Do not rely on metadata or other agents. "
        "Be conservative and return only valid JSON."
    )

    state["chiral_analysis"] = call_json_agent(system, prompt, temperature=0.0)
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
- use the retrieved dataset examples as grounding references and hard negatives, but do not copy their labels unless the current dI/dV morphology supports them
- coherent periodic LDOS modulation
- stripe, checkerboard, multi-Q, domain-like, patchy, diffuse, or defect-dominated structure
- whether the pattern is consistent with collective electronic order rather than defects, noise, drift, scan distortion, edge effects, or random inhomogeneity
- whether any symmetry breaking is robust and specific

Call "CDW" only if a reasonably coherent periodic electronic modulation is visible.
Call "Chiral CDW" only if there is convincing CDW-like modulation plus strong, specific chirality evidence beyond ordinary anisotropy.
Prefer None when evidence is weak or non-unique.
Return strict JSON with exactly:
{
  "reasoning": "brief explanation grounded only in visible morphology"
}
"""         },
            {"type": "text", "text": f"Retrieved context:\n{_format_retrieved_context(state)}"},
            {"type": "text", "text": "Image 2: dI/dV map"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{state['input_didv_base64']}"},
            },
    ]
    system = (
        "You are an INDEPENDENT STM spectroscopy correlation agent. "
        "Analyze the image as the dI/dV map. "
        "Use retrieved dataset examples only as grounding context. "
        "Do not rely on metadata or other agents. "
        "Be conservative and return only valid JSON."
    )

    state["spectroscopy_analysis"] = call_json_agent(system, prompt, temperature=0.0)
    return state

def skip_spectroscopy(state: AgentState):
    state["spectroscopy_analysis"] = {
        "reasoning": "No Image 2 (dI/dV map) was provided, so spectroscopy analysis was skipped."
    }
    return state


def spectroscopy_router(state: AgentState) -> str:
    return "run_spectroscopy" if state.get("input_didv_base64") else "skip_spectroscopy"

def _format_retrieval_for_judge(state: AgentState) -> str:
    summary = state.get("retrieval_summary", {}) or {}
    results = state.get("retrieval_results", []) or []
    lines = []

    if summary:
        lines.append("Retrieval summary:")
        for key in ["status", "message", "ranking_mode", "num_candidates_scored", "top_k", "retrieval_backbone", "clip_model", "clip_weights", "didv_secondary_weight"]:
            value = summary.get(key)
            if value not in (None, "", []):
                lines.append(f"- {key}: {value}")
        query_text = summary.get("query_text")
        if query_text:
            lines.append(f"- query_text: {query_text}")

    if not results:
        lines.append("No retrieved precedents were available.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Retrieved precedents to compare against the current sample:")
    for item in results:
        rank = item.get("rank", "?")
        label = item.get("label", "Unknown")
        identifier = item.get("id", f"retrieved_{rank}")
        material = item.get("material") or "Unknown"
        score = item.get("score")
        topograph_similarity = item.get("topograph_similarity")
        didv_similarity = item.get("didv_similarity")
        didv_used = item.get("didv_used")
        lines.append(f"{rank}. id={identifier} | label={label} | material={material} | score={score}")
        lines.append(f"   ranking_details: topograph_similarity={topograph_similarity}, didv_similarity={didv_similarity}, didv_used={didv_used}")
        for field in ["description", "notes", "reason", "reasoning", "summary"]:
            value = item.get(field)
            if value:
                lines.append(f"   {field}: {value}")
        lines.append("   Treat this as precedent only. Do not copy its label unless the agent evidence supports it.")

    return "\n".join(lines)


def _build_retrieval_audit_log(state: AgentState) -> Dict[str, Any]:
    summary = state.get("retrieval_summary", {}) or {}
    results = state.get("retrieval_results", []) or []
    retrieval_status = summary.get("status", "unavailable")
    top_k = summary.get("top_k", len(results))

    agent_usage = {
        "visual_analyst": "Uses retrieved precedents as grounding references and hard negatives during morphology analysis.",
        "fourier_agent": "Uses retrieved precedents to compare expected periodicity, symmetry, and Q-vector style patterns.",
        "chiral_specialist": "Uses CLIP-retrieved image precedents to separate true chirality from anisotropy, drift, or other hard negatives.",
        "Spectroscopy_correlation_agent": "Uses CLIP-retrieved dI/dV-capable precedents to compare spectroscopy-linked CDW signatures when available.",
        "final_judge": "Uses CLIP-retrieved image precedents to weigh support, conflict, or absence of precedent before final confidence calibration.",
    }

    selected = []
    for item in results:
        selected.append({
            "rank": item.get("rank"),
            "id": item.get("id", f"retrieved_{item.get('rank', '?')}"),
            "label": item.get("label"),
            "material": item.get("material"),
            "score": item.get("score"),
            "topograph_similarity": item.get("topograph_similarity"),
            "didv_similarity": item.get("didv_similarity"),
            "didv_used": item.get("didv_used"),
            "reason_selected": item.get("selection_reason") or item.get("reason") or item.get("notes") or item.get("description") or "Selected by retrieval ranking as a relevant precedent.",
            "used_by_agents": list(agent_usage.keys()),
        })

    return {
        "status": retrieval_status,
        "top_k": top_k,
        "ranking_mode": summary.get("ranking_mode"),
        "query_text": summary.get("query_text"),
        "message": summary.get("message"),
        "selected_examples": selected,
        "agent_usage": agent_usage,
    }


def _apply_confidence_penalty(state: AgentState, decision: Dict[str, Any]) -> Dict[str, Any]:
    adjusted = dict(decision)
    try:
        base_confidence = float(adjusted.get("confidence", 0.0))
    except (TypeError, ValueError):
        base_confidence = 0.0

    retrieval_summary = state.get("retrieval_summary", {}) or {}
    retrieval_results = state.get("retrieval_results", []) or []
    final_label = str(adjusted.get("final_label", "")).strip()

    penalties = []
    penalty_points = 0.0

    status = retrieval_summary.get("status")
    if status != "ok":
        penalty_points += 20.0
        penalties.append("retrieval unavailable")
    elif not retrieval_results:
        penalty_points += 15.0
        penalties.append("no retrieved precedents")

    if retrieval_results:
        result_labels = [str(item.get("label", "")).strip().lower() for item in retrieval_results if item.get("label") is not None]
        matching = sum(1 for label in result_labels if label == final_label.lower())
        conflicting = sum(1 for label in result_labels if label and label != final_label.lower())

        top_score = _safe_float(retrieval_results[0].get("score"))
        if top_score is not None and top_score < 0.2:
            penalty_points += 10.0
            penalties.append("weak retrieval match")

        if matching == 0 and conflicting > 0 and final_label != "Inconclusive":
            penalty_points += 18.0
            penalties.append("retrieved precedents conflict with final label")
        elif conflicting > matching and final_label != "Inconclusive":
            penalty_points += 12.0
            penalties.append("retrieved precedents are mixed or mostly conflicting")
        elif matching <= 1 and conflicting >= 1 and final_label != "Inconclusive":
            penalty_points += 8.0
            penalties.append("retrieved precedents weakly support the final label")

    agent_texts = [
        _flatten_text(state.get("visual_analysis", {})).lower(),
        _flatten_text(state.get("fourier_analysis", {})).lower(),
        _flatten_text(state.get("chiral_analysis", {})).lower(),
        _flatten_text(state.get("spectroscopy_analysis", {})).lower(),
    ]
    uncertainty_markers = ["inconclusive", "uncertain", "unclear", "ambiguous", "weak evidence", "no clear"]
    uncertain_agents = sum(1 for text in agent_texts if any(marker in text for marker in uncertainty_markers))
    if uncertain_agents >= 2 and final_label != "Inconclusive":
        penalty_points += 10.0
        penalties.append("multiple agents reported uncertainty")

    adjusted_confidence = max(0.0, min(100.0, base_confidence - penalty_points))
    adjusted["confidence"] = round(adjusted_confidence, 2)
    adjusted["retrieval_audit_log"] = _build_retrieval_audit_log(state)

    if penalties:
        penalty_note = "Confidence was reduced because " + ", ".join(penalties) + "."
        explanation = str(adjusted.get("explanation", "")).strip()
        adjusted["explanation"] = (explanation + " " + penalty_note).strip()
        adjusted["confidence_adjustment"] = {
            "base_confidence": round(base_confidence, 2),
            "adjusted_confidence": round(adjusted_confidence, 2),
            "penalty_points": round(penalty_points, 2),
            "reasons": penalties,
        }
    else:
        adjusted["confidence_adjustment"] = {
            "base_confidence": round(base_confidence, 2),
            "adjusted_confidence": round(adjusted_confidence, 2),
            "penalty_points": 0.0,
            "reasons": [],
        }

    return adjusted

def final_judge(state: AgentState):
    retrieval_context = _format_retrieval_for_judge(state)
    full_prompt = f"""
=== RETRIEVED PRECEDENTS ===
{retrieval_context}

=== VISUAL ANALYSIS ===
{json.dumps(state.get("visual_analysis", {}), indent=2)}

=== Fourier ANALYSIS ===
{json.dumps(state.get("fourier_analysis", {}), indent=2)}

=== CHIRAL ANALYSIS ===
{json.dumps(state.get("chiral_analysis", {}), indent=2)}

=== Spectroscopy ANALYSIS ===
{json.dumps(state.get("spectroscopy_analysis", {}), indent=2)}

You are the **Final Judge**. Integrate the specialist-agent outputs together with the retrieved precedents.
Rules:
- Allowed labels: CDW, Chiral CDW, None, Inconclusive
- Always gives priority on agents (expect Spectroscopy).
Output ONLY JSON: {{"final_label": "...", "confidence": 0-100, "explanation": "..." }}
"""
    state["full_final_prompt"] = full_prompt

    system = (
        "You are a conservative Final Judge. Integrate all evidence transparently, including specialist outputs and retrieved precedents. "
        "Do not copy retrieved labels blindly; use them only to calibrate and support the final decision."
    )
    raw_decision = call_json_agent(system, full_prompt, temperature=0.0)
    state["final_decision"] = _apply_confidence_penalty(state, raw_decision)
    state["retrieval_audit_log"] = state["final_decision"].get("retrieval_audit_log", _build_retrieval_audit_log(state))
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

# ===================== UI =====================
st.title("🔬 STM CDW Pure Agentic LLM Classifier v3")
st.markdown("**Reviewer-Fixed:** Independent agents + CLIP image retrieval + Fully auditable final prompt")

col1, col2 = st.columns([2, 1])
with col1:
    topo_file = st.file_uploader("Topograph Image (required)", type=["png", "jpg", "jpeg", "tif"])
    didv_file = st.file_uploader("dI/dV Map (optional)", type=["png", "jpg", "jpeg", "tif"])
    if topo_file:
        st.image(topo_file, width="stretch")

    if didv_file:
        st.image(didv_file, width="stretch")

with col2:
    v_topo = st.number_input("Bias voltage Topograph (V)", value=-0.1, step=0.001)
    i_topo = st.number_input("Tunneling current Topograph (pA)", value=200.0, step=1.0)
    scale_topo = st.number_input("Scale bar Topograph (nm)", value=2.0, step=0.1)
    v_didv_text = st.text_input("Bias voltage dI/dV (V) [optional]", value="")
    v_didv = float(v_didv_text) if v_didv_text.strip() else None

    run = st.button("🚀 RUN AGENTIC LLM CLASSIFICATION", type="primary", use_container_width=True)

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

            with st.expander("🔍 FULL AUDITABLE PROMPT SENT TO FINAL JUDGE"):
                st.code(result.get("full_final_prompt", ""), language="markdown")

            with st.expander("Retrieved Dataset Context"):
                st.json({
                    "retrieval_summary": result.get("retrieval_summary", {}),
                    "retrieval_results": result.get("retrieval_results", []),
                })

            with st.expander("Retrieval Audit Log"):
                st.json(result.get("retrieval_audit_log", result.get("final_decision", {}).get("retrieval_audit_log", {})))

            with st.expander("Full Agent Trace"):
                st.json(result)

st.caption("**Agents used:** CLIP Image Retriever • Visual • Fourier • Chiral (conditional) • Spectroscopy (conditional) • Final Judge")