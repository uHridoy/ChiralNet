import base64
import json
import os
import tempfile
from typing import (TypedDict, Optional, List, Dict, Any, Callable)
from didv_anisotropy import didv_anisotropy_tool
from didv_handedness import didv_handedness_tool
from didv_phase_chirality import didv_phase_chirality_tool
from chiral_metrology import chiral_metrology_tool, _phase_brief
from langgraph.graph import StateGraph, END

import stm_data_io as _SDIO

try:
    import matplotlib
    matplotlib.use("Agg")

    matplotlib.rcParams.update({
        "axes.titlepad": 16.0,
        "axes.titlesize": 11.0,
        "figure.titlesize": 13.0,
        "figure.subplot.hspace": 0.45,
        "figure.subplot.wspace": 0.35,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Arimo",
                            "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    })

    from fft_core import (
        analyze_image_fft,
        plot_bragg_cdw_map,
        plot_peak_diagnostics,
    )
    from fft_peak_analysis import (
        detect_peaks,
        assess_cdw_evidence,
        identify_lattice_peaks,
    )
    FFT_ANALYZER_AVAILABLE = True
    FFT_ANALYZER_IMPORT_ERROR = None
except Exception as _fft_exc:
    FFT_ANALYZER_AVAILABLE = False
    FFT_ANALYZER_IMPORT_ERROR = str(_fft_exc)


LLM_MODELS: List[Dict[str, str]] = [
    {"id": "claude-fable-5-1", "label": "Claude Fable 5.1",
     "provider": "anthropic"},
]

AGENT_MODELS: Dict[str, str] = {
    "moiré_agent": "claude-fable-5-1",
    "topographic_agent": "claude-fable-5-1",
    "spectroscopy_agent": "claude-fable-5-1",
}

AGENT_NAMES: Dict[str, str] = {
    "moiré_agent": "Moiré agent",
    "topographic_agent": "Topographic agent",
    "spectroscopy_agent": "Spectroscopy agent",
    "final_judge": "Judge agent",
}

LLM_MAX_TOKENS = 6000

MAX_TOOL_ROUNDS = 5

_ANTHROPIC_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

clients: Dict[str, Any] = {"anthropic": None}


def agent_model(agent: str) -> Dict[str, str]:
    mid = AGENT_MODELS[agent]
    return next(spec for spec in LLM_MODELS if spec["id"] == mid)


def missing_agent_models() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for spec in LLM_MODELS:
        agents = [AGENT_NAMES.get(a, a) for a, mid in AGENT_MODELS.items()
                  if mid == spec["id"]]
        if agents and clients.get(spec["provider"]) is None:
            out.append({**spec, "agents": agents})
    return out


class AgentState(TypedDict, total=False):
    input_image_base64: str
    input_image_ext: Optional[str]
    input_didv_base64: Optional[str]
    metadata: Dict[str, Any]

    fft_numerical: Dict[str, Any]
    fft_peaks_figure_base64: Optional[str]
    fft_diagnostic_figure_base64: Optional[str]
    fft_labeled_figure_base64: Optional[str]

    didv_fft_numerical: Dict[str, Any]
    didv_fft_peaks_figure_base64: Optional[str]
    didv_fft_labeled_figure_base64: Optional[str]

    moiré_analysis: Dict[str, Any]
    topographic_analysis: Dict[str, Any]
    spectroscopy_analysis: Dict[str, Any]

    final_label: Optional[str]
    confidence: float
    explanation: str

    didv_anisotropy: Dict[str, Any]
    didv_anisotropy_figure_base64: Optional[str]
    anisotropy_gate: Dict[str, Any]
    didv_handedness: Dict[str, Any]
    didv_phase_chirality: Dict[str, Any]
    chiral_metrology: Dict[str, Any]
    chiral_metrology_figure_base64: Optional[str]
    chiral_metrology_section_base64: Optional[str]


def b64_from_upload(file):
    return base64.b64encode(file.getvalue()).decode("utf-8")


def _sniff_media_type(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return "application/octet-stream"


def _image_block(data_url_or_b64: str) -> Optional[Dict[str, Any]]:
    payload = data_url_or_b64
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return None

    media_type = _sniff_media_type(raw)
    if media_type not in _ANTHROPIC_IMAGE_TYPES:
        try:
            import io as _io
            from PIL import Image
            buf = _io.BytesIO()
            with Image.open(_io.BytesIO(raw)) as im:
                im.convert("RGB").save(buf, format="PNG")
            payload = base64.b64encode(buf.getvalue()).decode("ascii")
            media_type = "image/png"
        except Exception:
            return None

    return {"type": "image",
            "source": {"type": "base64", "media_type": media_type,
                       "data": payload}}


def _to_anthropic_content(content: Any) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for item in content:
        if item["type"] == "text":
            converted.append({"type": "text", "text": item.get("text", "")})
        else:
            url = (item.get("image_url") or {}).get("url")
            block = _image_block(url) if url else None
            if block:
                converted.append(block)
    return converted


def _extract_json(text: str) -> Dict[str, Any]:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1] if "```" in body[3:] else body[3:]
        if body[:4].lower() == "json":
            body = body[4:]
        body = body.strip()
    try:
        return json.loads(body)
    except Exception:
        pass

    start = body.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(body[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(body[start:i + 1])
    raise ValueError(f"unterminated JSON object in model reply: {text[:200]!r}")


_JSON_ONLY = ("When you have finished calling tools, your FINAL reply must "
              "be a valid JSON object and nothing else: no prose outside it "
              "and no code fences around it.")

EXPLANATION_STYLE = """
EXPLANATION STYLE
The reader is a CDW / chiral-CDW specialist. Write at most 3 sentences
(about 60 words). Lead with the decisive observable: peak positions as
fractions of |Q_B|, the commensurate cell and rotation, 1Q/3Q organisation,
or the artifact that excludes a feature. Give numbers only when measured or
explicitly marked as visual estimates (e.g. "q/Q_B ~ 1/3, visual"). Name the
main unresolved alternative in a clause. Do not restate the label, these
instructions, tool or key names, or generic caveats.
"""


class ToolSpec:
    def __init__(self, name: str, description: str,
                 input_schema: Dict[str, Any],
                 handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.handler(args or {})
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


_NO_ARGS: Dict[str, Any] = {"type": "object", "properties": {},
                            "additionalProperties": False}


def _anthropic_tools(specs: List[ToolSpec]) -> List[Dict[str, Any]]:
    return [{"name": s.name, "description": s.description,
             "input_schema": s.input_schema} for s in specs]


def _block_to_dict(block: Any) -> Dict[str, Any]:
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if btype == "thinking":
        return {"type": "thinking",
                "thinking": getattr(block, "thinking", ""),
                "signature": getattr(block, "signature", "")}
    if btype == "redacted_thinking":
        return {"type": "redacted_thinking",
                "data": getattr(block, "data", "")}
    if btype == "tool_use":
        return {"type": "tool_use", "id": getattr(block, "id", ""),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", {}) or {}}
    if hasattr(block, "model_dump"):
        try:
            return block.model_dump()
        except Exception:
            pass
    return {"type": "text", "text": str(block)}


def _tool_result_json(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > 20000:
        text = text[:20000] + '... [truncated]"}'
    return text


def _run_tool_loop_anthropic(spec: Dict[str, str], system_prompt: str,
                             user_content: Any, tools: List[ToolSpec],
                             max_tokens: int, max_rounds: int
                             ) -> Dict[str, Any]:
    by_name = {t.name: t for t in tools}
    content = _to_anthropic_content(user_content)
    content.insert(0, {"type": "text", "text": _JSON_ONLY})
    messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]

    for round_index in range(max_rounds + 1):
        kwargs: Dict[str, Any] = dict(
            model=spec["id"], system=system_prompt, messages=messages,
            max_tokens=max_tokens,
            output_config={"effort": "medium"})
        if tools and round_index < max_rounds:
            kwargs["tools"] = _anthropic_tools(tools)

        response = clients["anthropic"].messages.create(**kwargs)
        blocks = list(response.content or [])
        calls = [b for b in blocks if getattr(b, "type", None) == "tool_use"]

        if calls:
            messages.append({"role": "assistant",
                             "content": [_block_to_dict(b) for b in blocks]})
            results: List[Dict[str, Any]] = []
            for call in calls:
                name = getattr(call, "name", "")
                args = getattr(call, "input", {}) or {}
                tool = by_name.get(name)
                payload = (tool.run(args) if tool else
                           {"error": f"no such tool: {name!r}"})
                results.append({"type": "tool_result",
                                "tool_use_id": getattr(call, "id", ""),
                                "content": _tool_result_json(payload)})
            messages.append({"role": "user", "content": results})
            continue

        text = "".join(getattr(b, "text", "") for b in blocks
                       if getattr(b, "type", None) == "text")
        if not text.strip():
            stop = getattr(response, "stop_reason", None)
            raise RuntimeError(
                f"{spec['id']} returned no text block (stop_reason={stop!r}). "
                "If this is 'max_tokens', raise LLM_MAX_TOKENS: adaptive "
                "thinking and tool results both spend from that budget.")
        return _extract_json(text)

    raise RuntimeError(f"{spec['id']} did not finish within "
                       f"{max_rounds} tool rounds")


def call_agent(system_prompt: str, user_content: Any,
               tools: Optional[List[ToolSpec]] = None,
               max_output_tokens: int = LLM_MAX_TOKENS,
               max_rounds: int = MAX_TOOL_ROUNDS,
               *, agent: str) -> Dict[str, Any]:
    spec = agent_model(agent)
    tools = tools or []
    try:
        return _run_tool_loop_anthropic(
            spec, system_prompt, user_content, tools,
            max_output_tokens, max_rounds)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}",
                "final_label": None, "confidence": 0.0,
                "explanation": f"{spec['label']} did not return a result: "
                               f"{exc}"}


def _fft_block(state: AgentState, which: str) -> Dict[str, Any]:
    key = "fft_numerical" if which == "topograph" else "didv_fft_numerical"
    data = state.get(key)
    if not isinstance(data, dict):
        return {"status": "unavailable",
                "message": f"No FFT was computed for the {which}."}
    return data


def _peak_table(state: AgentState, which: str, limit: int) -> Dict[str, Any]:
    fft = _fft_block(state, which)
    if fft.get("status") != "ok":
        return {"status": fft.get("status", "unavailable"),
                "message": fft.get("message", "FFT unavailable.")}
    rows = list(fft.get("strongest_accepted_peaks") or [])[:max(1, limit)]
    return {
        "status": "ok",
        "source": f"{which} FFT",
        "calibrated": fft.get("calibrated"),
        "frequency_units": fft.get("frequency_units"),
        "image_shape_px": fft.get("image_shape_px"),
        "n_accepted_peaks": fft.get("n_accepted_peaks"),
        "n_rejected_peaks": fft.get("n_rejected_peaks"),
        "rejection_reasons": fft.get("rejection_reasons"),
        "peaks": rows,
        "note": ("Peaks are from the mean-subtracted, Hanning-windowed 2D "
                 "FFT. A symmetric +q/-q pair is one modulation, not two. "
                 "An accepted peak is a detection, not an identification: "
                 "which peaks are lattice and which are superlattice is "
                 "answered by the lattice-identification tool."),
    }


def _lattice_identification(state: AgentState, which: str) -> Dict[str, Any]:
    fft = _fft_block(state, which)
    if fft.get("status") != "ok":
        return {"status": fft.get("status", "unavailable"),
                "message": fft.get("message", "FFT unavailable.")}
    return {
        "status": "ok",
        "identification_status": fft.get("identification_status"),
        "bragg_peaks": fft.get("bragg_peaks"),
        "cdw_peaks": fft.get("cdw_peaks"),
        "superlattice_model": fft.get("superlattice_model"),
        "bragg_family": fft.get("bragg_family"),
        "family_ratios": fft.get("family_ratios"),
        "bragg_assignment": fft.get("bragg_assignment"),
        "identification_notes": fft.get("identification_notes"),
        "note": ("EMPTY bragg/cdw lists mean the detection gates were not "
                 "met. That is absence of evidence, not a weak positive: do "
                 "not read an empty cdw_peaks list as a faint CDW. "
                 "`superlattice_model` names the commensurate cell whose "
                 "q_CDW/q_Bragg ratio and rotation best match the measured "
                 "geometry (2x2 = 0.500, 3x3 = 0.333, 4x4 = 0.250, "
                 "sqrt3 = 0.577 at 30 deg, sqrt13 = 0.277 at 13.9 deg)."),
    }


def _periodicity_scales(state: AgentState, which: str) -> Dict[str, Any]:
    fft = _fft_block(state, which)
    if fft.get("status") != "ok":
        return {"status": fft.get("status", "unavailable"),
                "message": fft.get("message", "FFT unavailable.")}

    peaks = list(fft.get("strongest_accepted_peaks") or [])
    bragg = list(fft.get("bragg_peaks") or [])
    b_radii = [p.get("f_radius") for p in bragg
               if isinstance(p.get("f_radius"), (int, float))]
    b_ref = (sum(b_radii) / len(b_radii)) if b_radii else None

    rows = []
    for p in peaks:
        fx, fy = p.get("fx"), p.get("fy")
        if not isinstance(fx, (int, float)) or not isinstance(fy, (int, float)):
            continue
        r = (fx * fx + fy * fy) ** 0.5
        if r <= 0:
            continue
        rows.append({
            "f_radius": round(r, 6),
            "wavelength": round(1.0 / r, 4),
            "ratio_to_bragg": (round(r / b_ref, 4) if b_ref else None),
            "periods_per_lattice_constant": (
                round(b_ref / r, 3) if b_ref else None),
            "snr": p.get("snr"),
        })
    rows.sort(key=lambda d: d["f_radius"])

    return {
        "status": "ok",
        "calibrated": fft.get("calibrated"),
        "wavelength_units": ("physical units per cycle (calibrated)"
                             if fft.get("calibrated")
                             else "PIXELS per cycle (UNCALIBRATED - no field "
                                  "of view was supplied, so no length can be "
                                  "converted to nm)"),
        "bragg_reference_radius": (round(b_ref, 6) if b_ref else None),
        "scales": rows,
        "note": ("`periods_per_lattice_constant` is the modulation "
                 "wavelength in units of the atomic lattice constant, and it "
                 "is the discriminating number. Roughly: 1.0 is the atomic "
                 "lattice itself; 2-5 is the CDW superlattice range; values "
                 "well above 5, and especially above 10, are the moire "
                 "range. It is null when no Bragg family was identified, in "
                 "which case no scale can be referenced to the lattice and "
                 "the distinction cannot be made from these numbers alone."),
    }


def _moire_tools(state: AgentState) -> List[ToolSpec]:
    return [
        ToolSpec("get_periodicity_scales",
                 "Real-space wavelength of every accepted topograph FFT "
                 "peak, and its radius as a fraction of the Bragg radius. "
                 "`periods_per_lattice_constant` is the modulation "
                 "wavelength in lattice constants - the number that "
                 "separates a moire (>5a, usually >10a) from a CDW "
                 "superlattice (2-5a).",
                 _NO_ARGS,
                 lambda a: _periodicity_scales(state, "topograph")),
        ToolSpec("get_lattice_identification",
                 "Which topograph FFT peaks were identified as atomic Bragg "
                 "peaks and which as a CDW superlattice, the best-matching "
                 "commensurate superlattice model, and the measured "
                 "q_CDW/q_Bragg family ratios.",
                 _NO_ARGS,
                 lambda a: _lattice_identification(state, "topograph")),
        ToolSpec("list_fft_peaks",
                 "The strongest accepted peaks of the topograph FFT with "
                 "position, radius, SNR, width and pairing score.",
                 {"type": "object", "additionalProperties": False,
                  "properties": {"limit": {
                      "type": "integer",
                      "description": "How many peaks to return (max 12).",
                      "minimum": 1, "maximum": 12}}},
                 lambda a: _peak_table(state, "topograph",
                                       int(a.get("limit", 12)))),
    ]


def _run_fft(raw_bytes: bytes, ext: str, Lx, Ly,
             channel=None, direction="forward", bias=None,
             prefer="topo") -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "field" + ext)
        with open(img_path, "wb") as f:
            f.write(raw_bytes)
        diag_path = os.path.join(tmpdir, "fft_diagnostic.png")
        peaks_path = os.path.join(tmpdir, "fft_peaks.png")
        labeled_path = os.path.join(tmpdir, "fft_labeled.png")

        results = analyze_image_fft(img_path, Lx=Lx, Ly=Ly,
                                    save_figure=diag_path,
                                    channel=channel, direction=direction,
                                    bias=bias, prefer=prefer)
        detection = detect_peaks(results)
        assessment = assess_cdw_evidence(results, detection)
        plot_peak_diagnostics(results, detection, save_figure=peaks_path)

        identification = identify_lattice_peaks(results)
        plot_bragg_cdw_map(results, identification, save_figure=labeled_path)

        figures = {}
        for name, path in (("diagnostic", diag_path), ("peaks", peaks_path),
                           ("labeled", labeled_path)):
            with open(path, "rb") as f:
                figures[name] = base64.b64encode(f.read()).decode("utf-8")

    accepted = detection["accepted"]
    rejected = detection["rejected"]
    reject_counts: Dict[str, int] = {}
    for p in rejected:
        r = str(p.get("reject_reason"))
        reject_counts[r] = reject_counts.get(r, 0) + 1

    top_peaks = sorted(accepted, key=lambda p: -p["snr"])[:12]
    peak_rows = [{
        "fx": round(p["fx"], 6), "fy": round(p["fy"], 6),
        "qx": round(p["qx"], 6) if p["qx"] is not None else None,
        "qy": round(p["qy"], 6) if p["qy"] is not None else None,
        "snr": round(float(p["snr"]), 1),
        "width_bins": round(float(p["width_bins"]), 2)
        if p["width_bins"] == p["width_bins"] else None,
        "pairing_score": round(float(p["pairing_score"]), 3),
    } for p in top_peaks]

    pair_rows = [{
        "f_radius": round(r["f_radius"], 6),
        "q_radius": round(r["q_radius"], 6)
        if r["q_radius"] is not None else None,
        "angle_deg": round(r["angle_deg"], 1),
        "min_snr": round(r["min_snr"], 1),
        "pairing_score": round(r["pairing_score"], 3),
        "status": r["status"],
        "flags": (r["failures"]
                  + (["axis_aligned"] if r["axis_aligned"] else [])
                  + (["jpeg_frequency"] if r["jpeg_suspect"] else [])),
    } for r in assessment["pairs"]]

    Ny, Nx = results["magnitude"].shape
    numerical = {
        "status": "ok",
        "pipeline": "fft_analyzer (mean-subtracted, 2D-Hanning-windowed "
                    "FFT; conservative peak detection; symmetric-pair "
                    "CDW-evidence assessment)",
        "calibrated": results["calibrated"],
        "frequency_units": ("cycles/unit and rad/unit (calibrated)"
                            if results["calibrated"]
                            else "cycles/pixel (UNCALIBRATED - no physical "
                                 "scale supplied)"),
        "image_shape_px": {"Nx": Nx, "Ny": Ny},
        "field_of_view": {"Lx": Lx, "Ly": Ly},
        "n_accepted_peaks": len(accepted),
        "n_rejected_peaks": len(rejected),
        "rejection_reasons": reject_counts,
        "strongest_accepted_peaks": peak_rows,
        "n_symmetric_pairs": assessment["n_pairs_total"],
        "n_clean_pairs": assessment["n_pairs_clean"],
        "n_suspect_pairs": assessment["n_pairs_suspect"],
        "symmetric_pairs": pair_rows,
        "bragg_peaks": [{
            "index": p["index"],
            "angle_deg": round(p["angle_deg"], 2),
            "f_radius": round(p["f_radius"], 6),
            "q_radius": (round(p["q_radius"], 6)
                         if p.get("q_radius") is not None else None),
            "snr": round(float(p["snr"]), 1),
        } for p in identification["bragg"]],
        "cdw_peaks": [{
            "index": p["index"],
            "angle_deg": round(p["angle_deg"], 2),
            "f_radius": round(p["f_radius"], 6),
            "q_radius": (round(p["q_radius"], 6)
                         if p.get("q_radius") is not None else None),
            "snr": round(float(p["snr"]), 1),
            "shift_from_predicted_bins": round(p["shift_bins"], 2),
        } for p in identification["cdw"]],
        "superlattice_model": identification["model"],
        "bragg_family": identification.get("bragg_family"),
        "family_ratios": identification.get("family_ratios"),
        "identification_status": identification["status"],
        "bragg_assignment": identification.get("bragg_assignment"),
        "scalar_field": results.get("scalar_field"),
        "identification_notes": identification["messages"],
        "possible_q_organization": assessment["q_organization"],
        "artifact_warnings": assessment["warnings"],
        "cdw_evidence_level": assessment["evidence_level"],
        "note": "This is CDW-compatible FFT evidence, NOT a definitive CDW "
                "classification: atomic-lattice, structural, and moire "
                "periodicities produce identical symmetric FFT peak pairs.",
    }
    numerical["data_class"] = results.get("data_class", "rendered")
    numerical["calibration_source"] = results.get("calibration_source")
    return {"numerical": numerical, "figures": figures}


def _normalise_ext(ext: Optional[str]) -> str:
    return _SDIO.normalise_extension(ext)


def _viewable_png_b64(b64: str, ext: Optional[str],
                      **load_kw) -> Optional[str]:
    ext = _normalise_ext(ext)
    if not _SDIO.is_native_extension(ext):
        return b64
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "field" + ext)
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(b64))
            field, info = _SDIO.load_native_field(path, **load_kw)
        png = _SDIO.render_preview_png(field, info, cmap="gray")
        return base64.b64encode(png).decode("utf-8")
    except Exception:
        return None


def fft_tool(state: AgentState) -> AgentState:
    if not FFT_ANALYZER_AVAILABLE:
        state["fft_numerical"] = {
            "status": "unavailable",
            "message": f"fft_analyzer could not be imported: "
                       f"{FFT_ANALYZER_IMPORT_ERROR}. Place fft_core.py "
                       "and fft_peak_analysis.py next to this script and "
                       "install numpy/scipy/matplotlib/Pillow.",
        }
        return state

    try:
        raw_bytes = base64.b64decode(state["input_image_base64"])
        ext = _normalise_ext(state.get("input_image_ext"))
        meta = state.get("metadata") or {}
        Lx, Ly = meta.get("fov_x"), meta.get("fov_y")
        if Lx is None or Ly is None or not Lx or not Ly:
            Lx = Ly = None

        out = _run_fft(raw_bytes, ext, Lx, Ly,
                       channel=meta.get("topo_channel"),
                       direction=meta.get("topo_direction", "forward"),
                       bias=meta.get("topo_bias"), prefer="topo")
        state["fft_numerical"] = out["numerical"]
        state["fft_diagnostic_figure_base64"] = out["figures"]["diagnostic"]
        state["fft_peaks_figure_base64"] = out["figures"]["peaks"]
        state["fft_labeled_figure_base64"] = out["figures"]["labeled"]
    except Exception as exc:
        state["fft_numerical"] = {"status": "failed", "message": str(exc)}
    return state


def didv_fft_tool(state: AgentState) -> AgentState:
    didv_b64 = state.get("input_didv_base64")
    if not didv_b64:
        state["didv_fft_numerical"] = {
            "status": "unavailable",
            "message": "No dI/dV map was provided.",
        }
        return state
    if not FFT_ANALYZER_AVAILABLE:
        state["didv_fft_numerical"] = {
            "status": "unavailable",
            "message": f"fft_analyzer could not be imported: "
                       f"{FFT_ANALYZER_IMPORT_ERROR}.",
        }
        return state

    try:
        raw_bytes = base64.b64decode(didv_b64)
        meta = state.get("metadata") or {}
        ext = _normalise_ext(meta.get("didv_ext"))
        out = _run_fft(raw_bytes, ext, None, None,
                       channel=meta.get("didv_channel"),
                       direction=meta.get("didv_direction", "forward"),
                       bias=meta.get("didv_bias"), prefer="didv")
        state["didv_fft_numerical"] = out["numerical"]
        state["didv_fft_peaks_figure_base64"] = out["figures"]["peaks"]
        state["didv_fft_labeled_figure_base64"] = out["figures"]["labeled"]
    except Exception as exc:
        state["didv_fft_numerical"] = {"status": "failed",
                                       "message": str(exc)}
    return state


def moiré_agent(state: AgentState):
    system = """
Decide whether a moire superlattice is present in the background-subtracted
STM topograph and its FFT. Call the measurement tools and cross-check against
real space. These rules override period ranges in tool descriptions.

Establish the scales
- Identify resolved atomic rows and count rows per envelope repeat. Match the
  envelope's direction and spacing to the corresponding FFT family, not simply
  the strongest low-radius peak. Group directions before comparing radii;
  interleaved families 30° apart must not be averaged.
- Check the Bragg reference behind `periods_per_lattice_constant`; use row
  counts if it is missing or actually a superstructure harmonic.
  `superlattice_model` is a nearest-cell fit, not a detection; rejected
  low-radius peaks do not by themselves disprove a visible envelope.
- Allow about 10% ratio tolerance for drift, creep and cropped-FFT estimates.
  Identify fundamentals by integer-multiple geometry, not brightness: a
  harmonic may be stronger.

Moire evidence
M1: A repeating envelope about >=8 lattice constants across a continuous
lattice: >=2 comparable cells and one dominant wavelength/radius. Strain may
broaden it; evenly spread low-radius power or one bright/dark pair is insufficient.
M2: Two comparably sharp Bragg-like families with their own harmonics, whose
vector difference matches the envelope q and is <=about Q_B/5 (period >=5).
Exclude one lattice plus modulation: simple ratios (1/2, 1/3, 1/4, 1/sqrt(3),
1/sqrt(13), 2/3, 2/sqrt(3), or small multiples of q) along Bragg directions or
at 30°/13.9°, and a family about twice a rotated inner modulation family.
Hexagonal stars 30° apart with radii differing by sqrt(3) indicate
sqrt(3) x sqrt(3) R30°, not a twisted bilayer. Calculate the implied beat
period; a difference giving only 1-2 constants is a coincidence cell.
M3: Smooth, atom-unlocked envelope: straight rows cross bright/dark regions,
maxima change atomic registry, and the period is often non-integer.
M4: Smoothly varying envelope period/shape/direction or curved/split stripes
while atomic rows remain straight.
M5: Envelope symmetry/orientation incompatible with the visible lattice,
such as hexagonal cells on a rectangular lattice, off-row stripes, or rotation
away from 0°, 30° and 13.9°.
Moire may form hexagonal/triangular cells, AA/AB networks, honeycomb holes/rims,
or 1D stripes. Unevenness from strain/twist is allowed; length alone is not proof.

Exclude before calling Moire
N1: Lattice-locked CDW/reconstruction: typically 2-5 constants (2x2, 3x3,
sqrt(3) R30°, sqrt(13) R13.9°, integer-period row stripes), fixed atomic
registry/cells except at sharp domain walls; Star-of-David clusters count.
Simple fractional Bragg radii AND aligned/30°/13.9° orientation with collinear
harmonics establish this lock even if cells are distorted or defective.
A ratio or named cell alone does not. This reciprocal-space lock takes priority
over apparent M3-M5, especially below 5 constants.
N2: Long-spaced discommensuration walls enclosing short-period CDW domains.
N3: Atomic lattice without larger-scale modulation.
N4: Bent/re-spaced atomic rows from drift/creep; tilt/bow/flattening residue
(broad frame-scale undulations, <2 clean repeats, no harmonics); irregular
slow-scan brightness bands leaving lattice/superstructure geometry and contrast
pattern intact; abrupt fast-scan line/tip changes; regular noise stripes
crossing defects/steps; crop/background artifacts; disorder/charge puddles;
defect-centered, decaying Friedel/QPI ripples. Multiple brightness bands alone
do not establish periodicity.
N5: Broad cluster arrays without resolved atoms or a second FFT lattice cannot
establish interference. If neither atomic rows nor a Bragg family is resolved,
apply N5.

Decision
"Moire" requires a periodic envelope with >=2 full repeats, a resolved atomic
lattice or M2 support, and no N1-N5 explanation. Check that field of view divided
by envelope period permits those repeats; strong peaks cannot waive this.
Then apply period requirements:
- About >=8 constants: M1 suffices; do not require hexagonal symmetry, AA/AB
  contrast, uniform cells or a clean FFT.
- About 5-8: require at least one of M2-M5.
- Below about 5: require M2 or two of M3-M5; N1 still takes precedence.
Otherwise return "Non-Moire". Irregularity supports Moire only without the
orientation-and-period lock; do not default to Non-Moire merely from uncertainty.

Confidence: 80-95 for M1/M2 with a clear envelope over resolved atoms; 60-80
for supported 5-8-constant patterns or irregular/faint/obscured long-period
ones; <=55 for Moire below about 5 constants without M2. Non-Moire confidence
is high for clear N1-N4/no envelope, lower for thinly rejected plausible envelopes.

Return JSON only:
{"final_label": "Moire" or "Non-Moire", "confidence": 0-100, "explanation": "..."}
""" + EXPLANATION_STYLE
    prompt: List[Dict[str, Any]] = []
    if state.get("fft_labeled_figure_base64"):
        prompt.append({
            "type": "text",
            "text": "Image 1: the background-subtracted topograph (real "
                    "space, left) and the cropped FFT modulus computed from "
                    "it (right). No markers are drawn on this panel - do not "
                    "describe circles or labels. The peak identification is "
                    "available through the tools, not from this picture."})
        prompt.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,"
                                 + state["fft_labeled_figure_base64"]}})
    else:
        prompt.append({
            "type": "text",
            "text": "Image 1: the raw STM topograph (real space). The "
                    "background-subtracted view and its FFT could not be "
                    "produced for this scan."})
        prompt.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,"
                                 + (_viewable_png_b64(
                                     state["input_image_base64"],
                                     state.get("input_image_ext"),
                                     channel=(state.get("metadata") or {})
                                     .get("topo_channel"),
                                     direction=(state.get("metadata") or {})
                                     .get("topo_direction", "forward"),
                                     bias=(state.get("metadata") or {})
                                     .get("topo_bias"),
                                     prefer="topo")
                                    or state["input_image_base64"])}})
    prompt.append({
        "type": "text",
        "text": "Call the tools to obtain the length scales and the lattice "
                "identification before answering."})

    result = call_agent(system, prompt, _moire_tools(state),
                        agent="moiré_agent")
    state["moiré_analysis"] = result
    return state


def topographic_agent(state: AgentState):
    system = """
Decide whether the topograph FFT shows visual evidence of CDW order. Use only
FFT pixels (right panel in a composite). Judge shape/strength on linear modulus;
use log magnitude only to locate weak features. Do not infer obscured pixels.
Centrosymmetry gives even noise a +/-q partner: pairing alone is not CDW evidence.

Identify the reference
Locate the center and first-order atomic Bragg family that generates the
others, not necessarily the brightest/outermost. Name the chosen family or
state that none is visible. A hexagonal star rotated 30° at ~1.7 times the
inner radius is second order of the same lattice. A weaker rotated ring at
~0.8-1 Q_B may be Bragg satellites: test A2 first. List remaining candidates.

Exclude non-CDW features
E1: Streaks through the origin at any angle, including maxima connected by
continuous intensity toward the center or beyond. Reject diffuse lines and
beads sharing their width; retain superposed compact, Bragg-width maxima
sharper than the streak only if independently anchored below.
E2: Diffuse central glow, granular speckle and scan-axis speckle bands.
Do not discard the whole central region: >=3-period modulations can sit there.
Look for compact spots of roughly Bragg width at a common radius along
equivalent lattice directions, with collinear harmonics; noise lacks this order.
E3: Frame-axis rectangular combs/grids with >=3 similarly strong orders and
no visible Bragg reference (interference/feedback/rectangular arrays).
Axis alignment alone does not invalidate a compact Bragg-referenced pair.
E4: Structural peaks not accounted for by one Bragg lattice plus one modulation
family (q, harmonics, combinations, Bragg +/- q): unrelated radii/angles,
off-axis peaks around only one equivalent Bragg direction, or a second lattice.
Allow shared drift/creep shear, unequal equivalent radii and nonideal angles.
Before calling a second lattice, exclude atomic second order and satellites:
test whether recurring short Bragg-to-nearest-peak offsets reproduce the family
(A2). Collinear modulation harmonics are not extra lattices.
E5: Isolated spots no stronger than same-radius random speckle, crop-edge
artifacts and compression grids.
Brightness or a favorable q/Q_B ratio cannot rescue an excluded candidate.

Require at least one positive anchor
A1: Both q/Q_B and orientation fit a lattice-referenced modulation using the
first-order Bragg reference: e.g. aligned 2x2, 3x3, 4x4 (1/2, 1/3, 1/4),
sqrt(3) x sqrt(3) R30° (1/sqrt(3), 30°), sqrt(13) x sqrt(13) R13.9°
(1/sqrt(13), 13.9°), or incommensurate q along a high-symmetry direction.
Multi-Q requires matching |q| in equivalent directions; two of three suffice
if the third is plainly masked or present as a satellite offset (A2).
1Q: a collinear +/-q pair along a high-symmetry lattice direction, with any
weaker harmonics on that line. Other periods qualify under the same conditions.
A2: Recurring identical q offsets around Bragg peaks. These establish a q-family
member even when its primary peak is absent.
A3: Without visible Bragg peaks, require a compact, internally lattice-like
complete equal-|q| star related by 60°/90° rotations, or a 1Q pair with a weaker
collinear second harmonic. Confidence <=55; missing/cropped Bragg peaks alone
do not exclude CDW.
Anchored peaks may be faint, broadened, unequal or unannotated; these alone
do not disqualify them.

Decision and output
"CDW": a compact non-atomic family survives all exclusions and meets an anchor.
Reduce confidence for weak/partly resolved evidence; state uncertainty and any
unresolved moire/structural alternative, since FFT alone cannot establish origin.
"Non-CDW": only atomic peaks/harmonics, all candidates excluded, or no positive
anchor. Explain the decisive feature/exclusion; lower confidence for a plausible
unanchored candidate, rather than labeling it low-confidence CDW, because the
label drives downstream decisions.
Missing/uninterpretable FFT: "Non-CDW".

Return JSON only:
{"final_label": "CDW" or "Non-CDW", "confidence": 0-100,
 "explanation": "decisive FFT evidence, main alternative"}
""" + EXPLANATION_STYLE
    prompt: List[Dict[str, Any]] = []
    if state.get("fft_labeled_figure_base64"):
        prompt += [
            {"type": "text",
             "text": "Image 1: the FFT computed from the STM topograph "
                     "(background-subtracted field on the left, cropped FFT "
                     "modulus on the right). This panel carries no markers - "
                     "do not describe circles or labels."},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,"
                                  + state["fft_labeled_figure_base64"]}},
        ]
    if state.get("fft_peaks_figure_base64"):
        prompt += [
            {"type": "text",
             "text": "Image 2: the FFT log-magnitude (visualisation only) "
                     "with accepted peaks circled in green, pairing lines in "
                     "cyan, and rejected peaks marked with a red x and their "
                     "rejection reason."},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,"
                                  + state["fft_peaks_figure_base64"]}},
        ]
    if not prompt:
        prompt.append({
            "type": "text",
            "text": "No FFT figure could be produced for this topograph. "
                    "Do not call tools. Return the low-confidence Non-CDW "
                    "fallback and explain that CDW could not be assessed."})
    prompt.append({
        "type": "text",
        "text": "Assess only the visible topograph FFT pixels as instructed. "
                "Ignore real-space panels and automated overlays. "
                "Do not call tools; return the required JSON."})

    result = call_agent(system, prompt, agent="topographic_agent")
    state["topographic_analysis"] = result
    return state


def _third_order_unrotated_star(fft: Dict[str, Any],
                                ratio_tol: float = 0.05,
                                angle_tol_deg: float = 5.0
                                ) -> Optional[Dict[str, Any]]:
    cdw = [p for p in (fft.get("cdw_peaks") or []) if isinstance(p, dict)]
    bragg = [p for p in (fft.get("bragg_peaks") or []) if isinstance(p, dict)]
    if len(cdw) < 3 or len(bragg) < 3:
        return None
    try:
        b_radii = sorted(float(p["f_radius"]) for p in bragg)
        b_angles = [float(p["angle_deg"]) for p in bragg]
        b_med = b_radii[len(b_radii) // 2]
        if b_med <= 0:
            return None
        ratios, offsets = [], []
        for p in cdw:
            ratio = float(p["f_radius"]) / b_med
            if abs(ratio - 1.0 / 3.0) > ratio_tol:
                return None
            ang = float(p["angle_deg"])
            off = min(abs((ang - b + 90.0) % 180.0 - 90.0) for b in b_angles)
            if off > angle_tol_deg:
                return None
            ratios.append(round(ratio, 3))
            offsets.append(round(off, 1))
    except (KeyError, TypeError, ValueError):
        return None
    return {"cdw_to_bragg_radius_ratios": ratios,
            "cdw_to_bragg_angle_offsets_deg": offsets}


def _normalise_spectroscopy_label(
        result: Dict[str, Any], phase: Optional[Dict[str, Any]] = None,
        fft: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
    if not isinstance(result, dict) or result.get("error"):
        return result

    out = dict(result)
    phase = phase if isinstance(phase, dict) else {}
    fft = fft if isinstance(fft, dict) else {}
    phase_verdict = str(phase.get("verdict") or "").lower()
    structural_positive = (
        out.get("final_label") == "Chiral CDW"
        and out.get("establishing_channel") == "structural_enantiomorph"
    )
    try:
        original_confidence = float(out.get("confidence") or 0.0)
    except (TypeError, ValueError):
        original_confidence = 0.0

    third_order = _third_order_unrotated_star(fft)
    if (third_order and not structural_positive
            and phase_verdict != "achiral"):
        out.update(
            final_label="Not Chiral",
            confidence=min(original_confidence or 50.0, 50.0),
            explanation=(
                "C1\u2013C3 at |q| \u2248 |Q_B|/3 (ratios "
                + ", ".join(f"{r:.3f}" for r in
                            third_order["cdw_to_bragg_radius_ratios"])
                + ") along the Bragg directions: unrotated third-order star; "
                "phase-registry verdict not applied to this cell."
            ),
            establishing_channel="none",
            phase_verdict=None,
            chirality_testable=False,
            third_order_star=third_order,
        )
        return out

    if phase_verdict == "phase_chiral" and not structural_positive:
        out.update(
            final_label="Chiral CDW",
            confidence=min(max(original_confidence, 75.0), 95.0),
            explanation=(
                "Phase-registry mirror test: "
                + _phase_brief(phase)
                + "; mirror breaking not accessible to the magnitude FFT."
            ),
            establishing_channel="phase_registry",
            phase_verdict="phase_chiral",
            chirality_testable=True,
        )
        return out

    if phase_verdict == "achiral" and not structural_positive:
        out.update(
            final_label="Not Chiral",
            confidence=min(max(original_confidence, 60.0), 90.0),
            explanation=(
                "Phase-registry mirror test: " + _phase_brief(phase) + "."
            ),
            establishing_channel="none",
            phase_verdict="achiral",
            chirality_testable=True,
        )
        return out

    screen = out.get("fft_screening") or {}
    visual_3q = (
        screen.get("three_pairs_resolved") is True
        and screen.get("opposite_pairs_consistent") is True
        and screen.get("artifact_dominated") is False
        and screen.get("local_background_accounted_for") is not False
    )
    deterministic_3q = (
        len(fft.get("cdw_peaks") or []) >= 3
        and len(fft.get("bragg_peaks") or []) >= 3
    )
    clean_3q = visual_3q or deterministic_3q
    amplitude_only_positive = (
        out.get("final_label") == "Chiral CDW"
        and out.get("establishing_channel") != "structural_enantiomorph"
    )
    if ((out.get("final_label") == "Not Chiral" and clean_3q)
            or amplitude_only_positive):
        out.update(
            final_label="Chance of Chiral CDW",
            confidence=min(original_confidence, 60.0),
            explanation=(
                "3Q CDW resolved (three Bragg and three CDW pairs); "
                "handedness not established by the magnitude FFT. "
                "Phase registry: " + _phase_brief(phase) + "."
            ),
            establishing_channel="none",
            phase_verdict=None,
            chirality_testable=False,
        )
    return out


def Spectroscopy_chirality_agent(state: AgentState):
    system = """
You are the Spectroscopy Agent. Image 1 shows the dI/dV FFT.

1. Position (primary discriminator). For each direction i, compare
   centre-to-C_i with C_i-to-B_i.
   - "half": a peak at the midpoint in all three directions (see 2b for
     peaks on lines). Extra spots often appear midway between adjacent
     Bragg peaks.
   - "third": C_i about 1/3 of the way to B_i, close to the central glow,
     with further spots near 2/3 of the way; the midpoints are dark.
     If C_i-to-B_i is about twice centre-to-C_i,
     it is "third", even when speckle near the midpoint is no brighter than
     its surroundings. Spots at 1/3 and 2/3 in ALL three directions are
     this star, not the one-direction chains of 2; with C_i on the
     centre-to-B_i line the star is unrotated.
   - "other": incommensurate or rotated positions, or no localized peak
     (only glow or speckle no brighter than its
     surroundings at that radius).

2. Streaks, lines and chains are not stripes. Many chiral references carry,
   along ONE direction, extra evenly spaced spots (near 1/4 and 3/4 of the
   way to that B_i), a bright straight streak through the centre, or
   parallel lines of spots that repeat through the Bragg peaks and can tile
   the whole FFT. Spots then look elongated along the lines, and more than
   six spots appear at similar radii. This is an extra 1Q modulation or a
   scan artifact on top of the 3Q star, never by itself a stripe or moire
   verdict and never by itself a reason for "Not Chiral". Moire is not your
   question. When lines are present:
   (a) read the C_i that lie off the lines first;
   (b) a C_i centred on a line counts as a present peak when the
       line is brighter there than in the dark gaps beside the
       line, even if the line is equally bright along its length;
   (c) report 1Q/2Q only after naming a C_i that holds no peak.
   Shear, broad or diffuse C peaks, a strong central glow and a noisy
   background also occur in chiral references. The magnitude FFT is
   centrosymmetric, so +q/-q agreement is automatic and says nothing.

3. Amplitudes (supporting only, never handedness). Unequal C1-C3 brightness
   that does not follow the Bragg brightness raises confidence. Equal or
   saturated brightness does NOT argue against a half-order star.

Decision
- "Chiral CDW": only a rotated CDW star whose rotation relative to Bragg is
  mirror-inequivalent (+30 and -30 degree stars are equivalent; an unrotated
  star is neutral). establishing_channel "structural_enantiomorph",
  chirality_testable true.
- "Chance of Chiral CDW": "half" in all three directions AND all three
  C1-C3 amplitudes are visibly distinct after local-background comparison,
  with a hierarchy that does not track B1-B3. Extra chains may be present,
  but a streak crossing a C peak cannot supply the required difference.
  Never use it when any C_i is "third", whatever the amplitudes.
  Confidence 50-60. establishing_channel "none", chirality_testable false.
- "Not Chiral": "third" or "other" unrotated position; a named C_i
  with no midpoint peak; OR a half-order 3Q star with equal, unresolved,
  saturated, Bragg-tracking or streak-dominated C1-C3 amplitudes. Half-order
  position establishes a CDW, not chirality. Streaks, lines, elongated spots
  or extra spots do not qualify as chiral evidence. Confidence <=50 for a
  clearly resolved non-chiral pattern, <=30 otherwise, <=20 if unreadable.
Set artifact_dominated true only when the C imbalance closely follows the
Bragg imbalance or a streak lies directly on the C peaks.

The explanation must state where C1, C2 and C3 sit (half, third, other or
absent) and, when relevant, whether the C1-C3 amplitude hierarchy tracks
B1-B3. Use null for unassessable checks, always phase_verdict null and
tools_used []. Return JSON only:
{"final_label": "Chiral CDW" or "Chance of Chiral CDW" or "Not Chiral",
 "confidence": 0-100,
 "explanation": "C1-C3 positions, decisive evidence, main limitation"}
""" + EXPLANATION_STYLE
    prompt: List[Dict[str, Any]] = []
    if state.get("didv_fft_labeled_figure_base64"):
        prompt += [
            {"type": "text",
             "text": "Image 1: the FFT modulus computed from the dI/dV "
                     "map (any real-space panel beside it is to be "
                     "ignored). Verify position and intensity from the FFT "
                     "pixels."},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,"
                                  + state["didv_fft_labeled_figure_base64"]}},
        ]
    prompt.append({
        "type": "text",
        "text": "Assess only the visible dI/dV FFT pixels as instructed. "
                "For each of C1, C2 and C3, decide whether it sits at half, "
                "a third or another fraction of the way to its B peak "
                "before judging the overall pattern. Streaks, parallel "
                "lines and chains of extra spots along one direction do "
                "not make the pattern 1Q. Do not call tools and return the "
                "required JSON."})

    result = call_agent(system, prompt, agent="spectroscopy_agent")
    phase = state.get("didv_phase_chirality")
    didv_fft = state.get("didv_fft_numerical")
    state["spectroscopy_analysis"] = _normalise_spectroscopy_label(
        result, phase, didv_fft)
    return state


def skip_spectroscopy(state: AgentState):
    msg = ("No dI/dV map supplied; chirality not assessed (not "
           "determinable from the topograph alone).")
    state["spectroscopy_analysis"] = {
        "status": "skipped",
        "final_label": None,
        "confidence": None,
        "explanation": msg,
        "reasoning": msg,
    }
    return state


def spectroscopy_router(state: AgentState) -> str:
    didv_fft = state.get("didv_fft_numerical") or {}
    if state.get("input_didv_base64") and didv_fft.get("status") == "ok":
        return "run_spectroscopy"
    return "skip_spectroscopy"


def final_judge(state: AgentState):
    def usable(result, labels):
        return (isinstance(result, dict)
                and not result.get("error")
                and not result.get("skipped")
                and str(result.get("status") or "").strip().lower()
                not in ("error", "errored", "failed", "skipped",
                        "missing", "unavailable")
                and result.get("final_label") in labels)

    def confidence_of(result):
        try:
            value = float(result.get("confidence") or 0.0)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return value if float("-inf") < value < float("inf") else 0.0

    def finish(label, confidence, explanation):
        state["final_label"] = label
        state["confidence"] = confidence
        state["explanation"] = explanation
        return state

    moire = state.get("moiré_analysis")
    if usable(moire, ("Moire",)):
        moire_confidence = confidence_of(moire)
        return finish(
            "Non-CDW", moire_confidence,
            f"Moiré Agent classified the sample as Moiré ({moire_confidence:g}%); "
            "the Moiré classification takes absolute priority, so the final "
            "classification is Non-CDW.")
    
    topo = state.get("topographic_analysis")
    invalid = []
    if not usable(moire, ("Moire", "Non-Moire")):
        invalid.append("moiré agent")
    if not usable(topo, ("CDW", "Non-CDW")):
        invalid.append("topographic agent")
    if invalid:
        return finish(
            "Inconclusive", 0.0,
            "No valid label from the " + " and ".join(invalid)
            + "; superlattice origin not assessed.")

    topo_label = topo["final_label"]
    topo_confidence = confidence_of(topo)
    t_txt = f"{topo_confidence:g}%"

    if topo_label == "Non-CDW":
        return finish(
            "Non-CDW", topo_confidence,
            f"No CDW superlattice resolved in the topograph FFT "
            f"({t_txt}); no moiré.")

    accepted = (f"CDW superlattice in the topograph FFT ({t_txt}); no moiré.")

    spectroscopy = state.get("spectroscopy_analysis")
    if not usable(spectroscopy, ("Chiral CDW", "Chance of Chiral CDW",
                                 "Not Chiral")):
        skipped = (isinstance(spectroscopy, dict)
                   and str(spectroscopy.get("status") or "").lower()
                   == "skipped")
        return finish(
            "CDW", topo_confidence,
            accepted + (" No dI/dV map: chirality not assessed." if skipped
                        else " dI/dV analysis unavailable: chirality not "
                             "assessed."))

    detail = " ".join(str(spectroscopy.get("explanation") or "").split())
    if detail and not detail.endswith((".", "!", "?")):
        detail += "."
    if spectroscopy["final_label"] == "Not Chiral":
        return finish(
            "CDW", topo_confidence,
            accepted + " dI/dV FFT: no chirality resolved."
            + (f" {detail}" if detail else ""))

    label = spectroscopy["final_label"]
    spec_conf = confidence_of(spectroscopy)
    return finish(
        label, spec_conf,
        accepted + f" dI/dV: {label} ({spec_conf:g}%)."
        + (f" {detail}" if detail else ""))


def _gate_for_label(screen: Dict[str, Any], verdict: str,
                    judge_label: Any) -> Dict[str, Any]:
    gate: Dict[str, Any] = {
        "verdict": verdict,
        "action": "none",
        "enforced": False,
        "judge_label": judge_label,
        "alpha_cdw": screen.get("alpha_cdw"),
        "alpha_crit": screen.get("alpha_crit"),
        "calibration": screen.get("calibration_name"),
        "aperture_overlap": bool(screen.get("aperture_overlap")),
    }

    gate["verdict_robust_to_aperture"] = screen.get(
        "verdict_robust_to_aperture")
    sens = screen.get("aperture_sensitivity") or {}

    if (screen.get("verdict_robust_to_aperture") is False
            and verdict in ("above", "below")):
        gate.update(
            action="deferred",
            reason=(f"Verdict {verdict!r} is not treated as a decision: "
                    "re-measuring this map with the aperture separation cap "
                    "enabled flips it"
                    + (f" ({sens.get('verdict_primary')} -> "
                       f"{sens.get('verdict_capped')}, alpha "
                       f"{sens.get('alpha_cdw_primary'):.3f} -> "
                       f"{sens.get('alpha_cdw_capped'):.3f})"
                       if sens.get("alpha_cdw_primary") is not None
                       and sens.get("alpha_cdw_capped") is not None else "")
                    + ", so it is an artifact of the aperture procedure and "
                      "cannot be treated as conservative in either "
                      "direction."))
    elif (verdict == "below" and judge_label == "Chiral CDW"
            and screen.get("aperture_overlap")):
        gate.update(
            action="deferred",
            reason=("Verdict 'below' is not treated as a disagreement: the "
                    "measurement flags aperture overlap, which biases alpha "
                    "low, so artifact-consistency cannot be asserted from "
                    "it."))
    elif verdict == "below" and judge_label == "Chiral CDW":
        alpha = screen.get("alpha_cdw")
        crit = screen.get("alpha_crit")
        gate.update(
            action="flagged",
            reason=(
                "The dI/dV map is inside the calibrated domain and its CDW "
                "intensity anisotropy "
                + (f"(alpha = {alpha:.3f}) " if isinstance(alpha, float)
                   else "")
                + "does not exceed the artifact threshold"
                + (f" (alpha_crit = {crit:.3f})" if isinstance(crit, float)
                   else "")
                + f" of the {screen.get('calibration_name') or 'active'} "
                  "calibration. Note that alpha is achiral and could not "
                  "have established the chirality claim in any case; what "
                  "this records is that the intensity anisotropy the claim "
                  "is often accompanied by is absent. The judge agent's "
                  "label is independent and STANDS UNCHANGED; the "
                  "disagreement is recorded for the audit trail and is "
                  "priced into the metrology scorecard, where the artifact "
                  "criterion earns 0 of its 10 points."))
    elif verdict == "above" and judge_label == "Chiral CDW":
        gate["action"] = "consistent"

    return gate


def _apply_anisotropy_gate(state: AgentState):
    aniso = state.get("didv_anisotropy") or {}
    screen = aniso.get("screen") or {}
    verdict = str(screen.get("verdict") or "unscreened")
    state["anisotropy_gate"] = _gate_for_label(screen, verdict,
                                               state.get("final_label"))
    return state


workflow = StateGraph(AgentState)

workflow.add_node("fft_tool", fft_tool)
workflow.add_node("didv_fft_tool", didv_fft_tool)
workflow.add_node("didv_anisotropy_tool", didv_anisotropy_tool)
workflow.add_node("didv_handedness_tool", didv_handedness_tool)
workflow.add_node("didv_phase_chirality_tool", didv_phase_chirality_tool)

workflow.add_node("moiré_agent", moiré_agent)
workflow.add_node("topographic_agent", topographic_agent)
workflow.add_node("spectroscopy_agent", Spectroscopy_chirality_agent)
workflow.add_node("skip_spectroscopy", skip_spectroscopy)
workflow.add_node("final_judge", final_judge)

workflow.add_node("anisotropy_gate", _apply_anisotropy_gate)
workflow.add_node("chiral_metrology_tool", chiral_metrology_tool)

workflow.set_entry_point("fft_tool")
workflow.add_edge("fft_tool", "didv_fft_tool")
workflow.add_edge("didv_fft_tool", "didv_anisotropy_tool")
workflow.add_edge("didv_anisotropy_tool", "didv_handedness_tool")
workflow.add_edge("didv_handedness_tool", "didv_phase_chirality_tool")
workflow.add_edge("didv_phase_chirality_tool", "moiré_agent")
workflow.add_edge("moiré_agent", "topographic_agent")
workflow.add_conditional_edges(
    "topographic_agent",
    spectroscopy_router,
    {
        "run_spectroscopy": "spectroscopy_agent",
        "skip_spectroscopy": "skip_spectroscopy",
    },
)
workflow.add_edge("spectroscopy_agent", "final_judge")
workflow.add_edge("skip_spectroscopy", "final_judge")
workflow.add_edge("final_judge", "anisotropy_gate")
workflow.add_edge("anisotropy_gate", "chiral_metrology_tool")
workflow.add_edge("chiral_metrology_tool", END)
agentic_graph = workflow.compile()
