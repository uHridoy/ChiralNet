from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple

import streamlit as st
from PIL import Image

if TYPE_CHECKING: 
    from core_pipeline import AgentState

try:
    import torch
except Exception:
    torch = None

try:
    import open_clip
except Exception:
    open_clip = None

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