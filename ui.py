import base64
import html as _html
from pathlib import Path
from typing import Optional, List, Dict, Any

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="ChiralNet", layout="wide")

st.markdown(
    """
    <style>
    .chiralnet-explanation {
        background-color: #2A1B08;
        border-left: 4px solid #6F5AA7;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 0.25rem;
        color: #FFFFFF;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .agent-row {
        display: flex;
        align-items: center;
        margin: 0 0 1.75rem 0;
    }
    .agent-card {
        flex: 1 1 auto;
        min-width: 0;
        background-color: #05060A;
        border: 1px solid #23232E;
        border-right: 4px solid #6F5AA7;
        border-radius: 6px;
        padding: 1.1rem 1.3rem;
    }
    .agent-title {
        color: #C9B8F0;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .agent-json {
        margin: 0;
        padding: 0;
        color: #FFFFFF;
        font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
        font-size: 0.80rem;
        line-height: 1.6;
        word-break: break-word;
    }
    /* Badge sits on the LEFT, so the tail points left into the card. */
    .agent-arrow {
        flex: 0 0 auto;
        width: 0;
        height: 0;
        border-top: 26px solid transparent;
        border-bottom: 26px solid transparent;
        border-right: 34px solid #05060A;
    }
    .agent-badge {
        flex: 0 0 auto;
        width: 86px;
        height: 86px;
        margin-right: 0.4rem;
        border-radius: 50%;
        background-color: #6F5AA7;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    /* Final judge card: same shape, purple accent. */
    .final-card {
        border-right: 4px solid #6F5AA7;
    }
    .final-badge {
        background-color: #6F5AA7;
    }
    .final-eyebrow {
        color: #FFFFFF;
        font-size: 1.2rem;
        margin-bottom: 0.1rem;
    }
    .final-label {
        color: #FFFFFF;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.15;
    }
    .final-conf {
        display: inline-block;
        margin-top: 0.35rem;
        padding: 0.1rem 0.5rem;
        border-radius: 6px;
        background-color: #12341F;
        color: #5AD07A;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import core_pipeline
from core_pipeline import (
    AgentState,
    agentic_graph,
    b64_from_upload,
    _format_retrieval_for_judge,
)

AGENT_PANELS = [
    (lambda result: _format_retrieval_for_judge(result), "Retriever agent analysis"),
    ("moiré_analysis", "Moiré agent analysis"),
    ("visual_analysis", "Visual agent analysis"),
    ("fourier_analysis", "Fourier agent analysis"),
    ("Chiral_topo_analysis", "Motif chirality agent analysis"),
    ("spectroscopy_analysis", "Spectroscopy chirality agent analysis"),
]
 
 
def _preformat(text: str) -> str:
    lines = []
    for line in (text or "").split("\n"):
        stripped = line.lstrip(" ")
        indent = "&nbsp;" * (len(line) - len(stripped))
        lines.append(indent + _html.escape(stripped))
    return "<br>".join(lines)
 
 
def render_final_card(result: Dict[str, Any]) -> None:
    label = _html.escape(str(result.get("final_label", "—")))
    try:
        conf = f"&uarr; {float(result.get('confidence', 0)):.1f}%"
    except (TypeError, ValueError):
        conf = ""
    explanation = _preformat(result.get("explanation", ""))
    st.markdown(
        '<div class="agent-row">'
        '<div class="agent-badge final-badge">ChiralNet</div>'
        '<div class="agent-arrow"></div>'
        '<div class="agent-card final-card">'
        '<div class="final-eyebrow">Final Label</div>'
        f'<div class="final-label">{label}</div>'
        f'<div class="final-conf">{conf}</div>'
        f'<div class="chiralnet-explanation">{explanation}</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

def _format_analysis(value: Any, indent: int = 0) -> List[str]:
    pad = "  " * indent
    lines: List[str] = []
    if isinstance(value, dict):
        for key, val in value.items():
            if isinstance(val, (dict, list)) and val:
                lines.append(f"{pad}{key}:")
                lines.extend(_format_analysis(val, indent + 1))
            else:
                lines.append(f"{pad}{key}: {val}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.extend(_format_analysis(item, indent))
                lines.append("")
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}{value}")
    return lines
 
def _panel_body(result: Dict[str, Any], source) -> Optional[str]:
    if callable(source):
        return source(result) or None
    analysis = result.get(source)
    if not analysis:
        return None
    return "\n".join(_format_analysis(analysis))
 
def render_agent_cards(result: Dict[str, Any]) -> None:
    for source, title in AGENT_PANELS:
        body = _panel_body(result, source)
        if body is None:
            continue
        st.markdown(
            '<div class="agent-row">'
            '<div class="agent-badge">ChiralNet</div>'
            '<div class="agent-arrow"></div>'
            '<div class="agent-card">'
            f'<div class="agent-title">{_html.escape(title)}:</div>'
            f'<div class="agent-json">{_preformat(body)}</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
api_key = st.text_input("OpenAI API Key")

core_pipeline.client = OpenAI(api_key=api_key) if api_key else None

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
    scale_topo = st.number_input("Scale Bar (nm)",  value=None,
                               format="%.2f", placeholder="2.0")
    fov_x = st.number_input("Topograph width (nm)", value=None,
                               format="%.2f", placeholder="10.0",
                               help="Full physical width of the topograph. "
                                    "Both X and Y are required for a "
                                    "calibrated FFT; otherwise the FFT is "
                                    "reported in cycles/pixel.")
    fov_y = st.number_input("Topograph height (nm)", value=None,
                               format="%.2f", placeholder="10.0")

    run = st.button("Run ChiralNet", type="primary", use_container_width=True)

if run:
    if not api_key:
        st.error("Enter your OpenAI API key")
    elif not topo_file:
        st.warning("Upload a Topograph image")
    else:
        with st.spinner("Running ChiralNet..."):
            topo_b64 = b64_from_upload(topo_file)
            didv_b64 = b64_from_upload(didv_file) if didv_file else None

            initial_state: AgentState = {
                "input_image_base64": topo_b64,
                "input_image_ext": Path(topo_file.name).suffix,  
                "input_didv_base64": didv_b64,
                "metadata": {"v_topo": v_topo,
                             "scale_topo": scale_topo,
                             "fov_x": fov_x, "fov_y": fov_y}, 
            }

            result = agentic_graph.invoke(initial_state)

            st.success("Classification complete")
            render_final_card(result)

            with st.expander("Complete analysis of the agents submitted to the final judge"):
                render_agent_cards(result)

            fft_num = result.get("fft_numerical") or {}
            with st.expander("Numerical FFT analysis",
                             expanded=False):
                if fft_num.get("status") == "ok":
                    lvl = fft_num.get("cdw_evidence_level", "?")
                    st.metric("CDW-compatible FFT evidence", lvl.upper())
                    if result.get("fft_diagnostic_figure_base64"):
                        st.image(base64.b64decode(
                            result["fft_diagnostic_figure_base64"]),
                            caption="Windowed FFT diagnostic panels")
                else:
                    st.warning(fft_num.get("message",
                                           "Numerical FFT not available."))
