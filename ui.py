import base64
import html as _html
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

import streamlit as st

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

st.set_page_config(page_title="ChiralNet", layout="wide")

st.markdown(
    """
    <style>
    /* Arial across the whole app.  Streamlit sets its own sans font on
       body and on specific component classes, so a single rule on a high
       enough selector, applied with !important, is what actually reaches
       widget labels, prose, headers and the custom cards alike. */
    html, body, .stApp, [class^="st-"], [class*=" st-"],
    [data-testid] { font-family: Arial, "Helvetica Neue", Helvetica,
                    "Liberation Sans", Arimo, sans-serif !important; }
    /* ...but NOT the Material-icon glyphs.  Streamlit draws the expander
       arrow (and other icons) as a <span data-testid="stIconMaterial">
       whose text is a ligature name ("keyboard_arrow_right") that only
       renders as an arrow while the element uses the Material Symbols
       font.  The rule above matches that span through [data-testid] and
       forces Arial onto it, so the ligature falls back to literal text and
       overlaps the label.  Re-assert the icon font here (same specificity,
       later in the cascade, so it wins) for every Material-icon element,
       however Streamlit names the class. */
    [data-testid="stIconMaterial"],
    [class*="material-symbols"], [class*="material-icons"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                     "Material Symbols Sharp", "Material Icons" !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-feature-settings: "liga" !important;
    }
    .chiralnet-explanation {
        background-color: #2A1B08;
        border-left: 4px solid #6F5AA7;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-top: 0.5rem;
    }
    .agent-row {
        display: flex;
        align-items: center;
        margin: 0 0 1.75rem 0;
    }
    .agent-card {
        flex: 1 1 auto;
        min-width: 0;
        font-family: Arial, "Helvetica Neue", Helvetica, "Liberation Sans",
                     Arimo, sans-serif;
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
    /* One body-text style for every agent card, the judge explanation
       included, so all agent writing renders in the same face, size,
       weight and line height. */
    .agent-json,
    .agent-fields dt,
    .agent-fields dd,
    .chiralnet-explanation {
        font-family: Arial, "Helvetica Neue", Helvetica, "Liberation Sans",
                     Arimo, sans-serif !important;
        font-size: 0.90rem;
        font-weight: 400;
        font-style: normal;
        line-height: 1.55;
        letter-spacing: normal;
        color: #FFFFFF;
    }
    .agent-json {
        margin: 0;
        padding: 0;
        word-break: break-word;
    }
    .agent-fields {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr);
        gap: 0.35rem 0.9rem;
        margin: 0;
    }
    .agent-fields dt { color: #9FA3B5; white-space: nowrap; }
    .agent-fields dd { margin: 0; word-break: break-word; }
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
        color: #C9B8F0;
        font-family: Arial, "Helvetica Neue", Helvetica,
                    "Liberation Sans", Arimo, sans-serif !important;
        font-size: 1.15rem;
        font-weight: 600;
        font-style: normal;
        letter-spacing: 0.2px;
        margin-bottom: 0.3rem;
    }

    .final-label {
        color: #FFFFFF;
        font-family: Arial, "Helvetica Neue", Helvetica,
                    "Liberation Sans", Arimo, sans-serif !important;
        font-size: 1.5rem;
        font-weight: 700;
        font-style: normal;
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

    /* ---------------- Anisotropy tool card ---------------- */
    /* The calibrated verdict is the headline of this card, not the number.
       It renders in every state, including a successful measurement that
       the decision rule refuses to interpret. */
    .aniso-verdict {
        border: 1px solid;
        border-radius: 6px;
        padding: 0.7rem 0.95rem;
        margin-bottom: 0.95rem;
    }
    .aniso-verdict .vt {
        font-size: 0.98rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        margin-bottom: 0.25rem;
    }
    .aniso-verdict .vr {
        color: #D6D8E3;
        font-size: 0.85rem;
        line-height: 1.55;
    }
    .v-above   { background-color: #12281B; border-color: #2F6B41; }
    .v-above   .vt { color: #7FD79A; }
    .v-below   { background-color: #2A2410; border-color: #6B5A2F; }
    .v-below   .vt { color: #E9D68F; }
    .v-refused { background-color: #2C1414; border-color: #7A3434; }
    .v-refused .vt { color: #F0A5A5; }
    .v-none    { background-color: #14151C; border-color: #33364A; }
    .v-none    .vt { color: #B9BDCE; }

    .aniso-notes {
        margin: 0 0 0.95rem 0;
        padding-left: 1.1rem;
        color: #C8CBD9;
        font-size: 0.85rem;
        line-height: 1.55;
    }
    .aniso-notes li { margin-bottom: 0.3rem; }
    .aniso-caveat {
        border-left: 3px solid #6F5AA7;
        padding-left: 0.75rem;
        margin-bottom: 0.95rem;
        color: #C8CBD9;
        font-size: 0.85rem;
        line-height: 1.55;
    }
    .aniso-provenance {
        color: #7E8296;
        font-size: 0.74rem;
        line-height: 1.5;
        margin-top: 0.2rem;
    }
    .aniso-title {
        color: #C9B8F0;
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        margin-bottom: 0.9rem;
    }
    .aniso-alphas {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem 2.75rem;
        margin-bottom: 0.9rem;
    }
    .aniso-alpha { display: flex; align-items: baseline; gap: 0.6rem; }
    .aniso-alpha .k { color: #9FA3B5; font-size: 0.95rem; }
    .aniso-alpha .v {
        color: #FFFFFF;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1;
        font-variant-numeric: tabular-nums;
    }
    .aniso-alpha .v.missing {
        color: #6E7288;
        font-size: 1rem;
        font-weight: 500;
        font-style: italic;
    }
    /* A measured value the calibrated rule does not license interpreting.
       It stays on the card for the audit trail, at the size of a footnote
       rather than the size of a result. */
    .aniso-alpha .v.suppressed {
        color: #8A8FA6;
        font-size: 1rem;
        font-weight: 500;
    }
    .aniso-alpha .v.suppressed .q {
        color: #6E7288;
        font-size: 0.8rem;
        font-style: italic;
        margin-left: 0.35rem;
    }
    .aniso-empty {
        color: #9FA3B5;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-bottom: 0.5rem;
    }

    .aniso-box {
        border: 1px solid #2E2E3C;
        border-radius: 6px;
        padding: 0.85rem 1rem 0.6rem 1rem;
        margin-bottom: 1rem;
        overflow-x: auto;
    }
    .aniso-box-title {
        color: #C9B8F0;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.55rem;
    }
    .aniso-table {
        width: 100%;
        border-collapse: collapse;
        color: #FFFFFF;
        /* Arial, with tabular-nums so the numeric columns still align even
           though the face is proportional. */
        font-family: Arial, "Helvetica Neue", Helvetica, "Liberation Sans",
                     Arimo, sans-serif;
        font-size: 0.80rem;
        font-variant-numeric: tabular-nums;
    }
    .aniso-table th {
        color: #9FA3B5;
        font-weight: 600;
        text-align: right;
        padding: 0.2rem 0.55rem 0.4rem 0.55rem;
        border-bottom: 1px solid #2E2E3C;
        white-space: nowrap;
    }
    .aniso-table td {
        text-align: right;
        padding: 0.32rem 0.55rem;
        white-space: nowrap;
    }
    .aniso-table th:first-child,
    .aniso-table td:first-child { text-align: left; color: #9FA3B5; }

    .aniso-facts {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr);
        gap: 0.5rem 1.25rem;
        margin: 0 0 1rem 0;
        font-size: 0.9rem;
    }
    .aniso-facts dt { color: #9FA3B5; white-space: nowrap; }
    .aniso-facts dd { color: #FFFFFF; margin: 0; }
    .aniso-facts dd .sub { color: #9FA3B5; font-size: 0.82rem; }

    /* ---------------- Chiral CDW metrology card ---------------- */
    /* This card is set in Times, larger than the agent cards: it is the
       summary panel of the run and the one a reader takes a number from,
       so it is typeset rather than styled like a debug dump. */
    .metro-title {
        color: #C9B8F0;
        font-family: Arial, "Helvetica Neue", Helvetica, "Liberation Sans",
                     Arimo, sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        /* Tight: the meter figure follows immediately, and the gap between
           the two was dead space. */
        margin-bottom: 0.3rem;
    }
    .metro-crit {
        display: grid;
        grid-template-columns: auto auto minmax(0, 1fr);
        gap: 0.55rem 0.8rem;
        margin: 0;
        align-items: start;
        font-family: Arial, "Helvetica Neue", Helvetica, "Liberation Sans",
                     Arimo, sans-serif;
    }
    .metro-crit .n { color: #7E8296; font-size: 1.0rem; }
    .metro-crit .mark { font-size: 1.1rem; font-weight: 700; }
    .metro-crit .mark.ok { color: #7FD79A; }
    .metro-crit .mark.pt { color: #E9D68F; }
    .metro-crit .mark.no { color: #F0A5A5; }
    .metro-crit .mark.na { color: #9FA3B5; }
    .metro-crit .pts { color: #9FA3B5; font-size: 0.95rem; }
    .metro-crit .name { color: #FFFFFF; font-size: 1.08rem; font-weight: 700; }
    .metro-crit .val { color: #C9B8F0; font-size: 1.0rem; }
    /* Subsections of the merged alpha criterion: same grammar, indented
       and a step down in weight, so they read as parts of one row rather
       than as criteria in their own right. */
    .metro-sub {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr);
        gap: 0.35rem 0.7rem;
        margin: 0.3rem 0 0.2rem 0.9rem;
        align-items: start;
    }
    .metro-sub .subname {
        color: #E6E8F2;
        font-size: 1.0rem;
        font-weight: 600;
    }
    .metro-crit .det {
        color: #C8CBD9;
        font-size: 0.98rem;
        line-height: 1.55;
        margin-top: 0.15rem;
    }
    /* The meter is rendered on #05060A, the card's own background, so it
       needs no plate and no border: it reads as part of the card. */
    .metro-fig {
        margin: 0 0 0.55rem 0;
        padding: 0;
        background-color: transparent;
    }
    .metro-fig img { width: 100%; display: block; }

    .aniso-figs { display: flex; flex-wrap: wrap; gap: 0.75rem; }
    .aniso-fig {
        flex: 1 1 320px;
        min-width: 0;
        margin: 0;
        border: 1px solid #2E2E3C;
        border-radius: 6px;
        background-color: #0B0B12;
        padding: 0.5rem;
    }
    .aniso-fig img { width: 100%; display: block; border-radius: 4px; }
    .aniso-fig figcaption {
        color: #9FA3B5;
        font-size: 0.75rem;
        text-align: center;
        margin-top: 0.4rem;
    }

    .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6 { color: #000000; }
    /* st.markdown prose, but only at the top level - the cards are also
       rendered through st.markdown and must keep their own colours. */
    [data-testid="stMarkdownContainer"] > p { color: #000000; }
    /* widget labels: "Anthropic API Key", "Topograph Bias (V)", ... */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] * { color: #000000; }
    /* st.caption and st.image(caption=...).  Streamlit ships its own muted
       grey for these, at a specificity this block cannot beat, so the
       override is forced.  The cards' own <figcaption>s live under
       .aniso-fig and are not matched here, so they stay light on dark. */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    [data-testid="stImageCaption"],
    [data-testid="stImage"] figcaption { color: #000000 !important; }
    /* names and sizes of the files already uploaded: these render BELOW the
       dark drop zone, on the page background, not inside it. */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] * { color: #000000 !important; }
    /* the "Running ChiralNet..." spinner label */
    [data-testid="stSpinner"],
    [data-testid="stSpinner"] * { color: #000000 !important; }

    /* Expander headers ("Complete analysis of the agents...", "Numerical FFT
       analysis").  Streamlit still paints these with the dark theme's
       secondary background, so black text on them is invisible.  Clearing
       the fill lets the page background show through and the black read. */
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpanderDetails"] {
        background-color: transparent !important;
    }
    [data-testid="stExpander"] details {
        border: 1px solid #B4B7C0;
        border-radius: 6px;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * { color: #000000 !important; }
    [data-testid="stExpander"] summary svg { fill: #000000 !important; }

    .stApp { background-color: #FFFFFF; }
    [data-testid="stHeader"] { background-color: transparent; }

    </style>
    """,
    unsafe_allow_html=True,
)

import core_pipeline
from core_pipeline import (
    AgentState,
    agentic_graph,
    b64_from_upload,
)

try:
    import matplotlib

    matplotlib.rcParams.update({
        "text.color": "#000000",
        "axes.titlecolor": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.labelcolor": "#000000",
        "ytick.labelcolor": "#000000",
        "axes.titleweight": "medium",
        "axes.labelweight": "medium",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Arimo",
                            "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
except Exception:
    pass

AGENT_PANELS = [
    ("moiré_analysis", "Moiré agent"),
    ("topographic_analysis", "Topographic agent"),
    ("spectroscopy_analysis", "Spectroscopy agent"),
]
 
 
def _preformat(text: str) -> str:
    lines = []
    for line in (text or "").split("\n"):
        stripped = line.lstrip(" ")
        indent = "&nbsp;" * (len(line) - len(stripped))
        lines.append(indent + _html.escape(stripped))
    return "<br>".join(lines)
 
 
def _confidence_text(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if f != f:
        return "—"
    return f"{f:.0f}%" if abs(f - round(f)) < 0.05 else f"{f:.1f}%"


def render_final_card(result: Dict[str, Any]) -> None:
    label = _html.escape(str(result.get("final_label") or "—"))
    conf_txt = _confidence_text(result.get("confidence"))
    explanation = _preformat(str(result.get("explanation") or ""))
    html = (
        '<div class="agent-row">'
        '<div class="agent-badge final-badge">ChiralNet</div>'
        '<div class="agent-arrow"></div>'
        '<div class="agent-card final-card">'
        '<div class="final-eyebrow">Judge agent</div>'
        f'<div class="final-label">{label}</div>'
        + (f'<div class="final-conf">Confidence {conf_txt}</div>'
           if conf_txt != "—" else "")
        + (f'<div class="chiralnet-explanation">{explanation}</div>'
           if explanation else "")
        + '</div></div>')
    st.markdown(html, unsafe_allow_html=True)


def _num(value: Any, spec: str = "{:.3f}", dash: str = "&mdash;") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return dash
    if f != f:
        return dash
    return spec.format(f)


def _split_panels(png_bytes: bytes) -> Optional[List[str]]:
    try:
        import io
        from PIL import Image

        im = Image.open(io.BytesIO(png_bytes))
        w, h = im.size
        mid = w // 2
        halves = []
        for box in ((0, 0, mid, h), (mid, 0, w, h)):
            buf = io.BytesIO()
            im.crop(box).save(buf, format="PNG")
            halves.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        return halves
    except Exception:
        return None


_VERDICT_PRESENTATION = {
    "above":          ("v-above",   "Exceeds the artifact threshold", True),
    "below":          ("v-below",   "Consistent with artifacts",      True),
    "refused":        ("v-refused", "Outside the analysable domain",  False),
    "no_measurement": ("v-none",    "No measurement",                 False),
    "unscreened":     ("v-refused", "Not screened",                   False),
}


def _verdict_of(aniso: Dict[str, Any]) -> str:
    screen = aniso.get("screen") or {}
    if screen.get("verdict"):
        return str(screen["verdict"])
    if aniso.get("status") == "ok":
        return "unscreened"
    return "no_measurement"


def render_anisotropy_card(result: Dict[str, Any]) -> None:
    aniso = result.get("didv_anisotropy") or {}
    if not aniso:
        return

    scr = aniso.get("screen") or {}
    ok = aniso.get("status") == "ok"
    verdict = _verdict_of(aniso)
    css, fallback_headline, interpretable = _VERDICT_PRESENTATION.get(
        verdict, ("v-none", "Unknown verdict", False))

    parts: List[str] = [
        '<div class="agent-row">',
        '<div class="agent-badge">ChiralNet</div>',
        '<div class="agent-arrow"></div>',
        '<div class="agent-card">',
        '<div class="aniso-title">Anisotropy tool</div>',
    ]

    headline = str(scr.get("headline") or fallback_headline)
    if verdict == "unscreened":
        reason = ("The measurement completed but the B2 decision rule did "
                  "not run, so the value below has not been compared to the "
                  "artifact null and cannot be interpreted.")
    else:
        reason = str(scr.get("reason") or aniso.get("message") or "")
    parts.append(
        f'<div class="aniso-verdict {css}">'
        f'<div class="vt">{_html.escape(headline)}</div>'
        + (f'<div class="vr">{_html.escape(reason)}</div>' if reason else "")
        + '</div>')

    parts.append('<div class="aniso-alphas">')
    for key, label in (("alpha_cdw", "&alpha; (CDW)"),
                       ("alpha_bragg", "&alpha; (Bragg)")):
        shown = _num(scr.get(key, aniso.get(key)), dash="")
        if not shown:
            cell = '<span class="v missing">not measured</span>'
        elif interpretable:
            cell = f'<span class="v">{shown}</span>'
        else:
            cell = (f'<span class="v suppressed">{shown}'
                    f'<span class="q">not interpretable</span></span>')
        parts.append(f'<div class="aniso-alpha"><span class="k">{label}</span>'
                     f'{cell}</div>')
    parts.append('</div>')

    if not ok:
        why = (scr.get("reason") or aniso.get("message")
               or "The dI/dV map did not yield a measurement.")
        parts.append(f'<div class="aniso-empty">{_html.escape(str(why))}</div>')

    if ok:
        parts += ['<div class="aniso-box">',
                  '<div class="aniso-box-title">CDW and Bragg peak '
                  'directions</div>',
                  '<table class="aniso-table"><thead><tr>'
                  '<th>Direction</th><th>angle</th><th>C<sub>i</sub></th>'
                  '<th>B<sub>i</sub></th><th>R<sub>i</sub> = C/B</th>'
                  '<th>SNR</th></tr></thead><tbody>']
        angles = aniso.get("cdw_angles_deg") or []
        C = aniso.get("C") or []
        B = aniso.get("B") or []
        R = aniso.get("R") or []
        peaks = aniso.get("cdw_peaks") or []
        for k in range(3):
            snr = peaks[k].get("snr") if k < len(peaks) else None
            parts.append(
                f'<tr><td>{k + 1}</td>'
                f'<td>{_num(angles[k] if k < len(angles) else None, "{:.1f}&deg;")}</td>'
                f'<td>{_num(C[k] if k < len(C) else None, "{:.4g}")}</td>'
                f'<td>{_num(B[k] if k < len(B) else None, "{:.4g}")}</td>'
                f'<td>{_num(R[k] if k < len(R) else None, "{:.4f}")}</td>'
                f'<td>{_num(snr, "{:.1f}")}</td></tr>')
        parts += ['</tbody></table></div>']

        cm = aniso.get("cdw_model") or {}
        fam = aniso.get("bragg_family") or {}
        method = aniso.get("method") or {}

        model_bits = _html.escape(str(cm.get("name", "?")))
        extra = []
        if cm.get("q_ratio_to_bragg") is not None:
            extra.append("|q|/|q<sub>B</sub>| = "
                         f"{_num(cm['q_ratio_to_bragg'], '{:.4f}')}")
        if cm.get("rotation_deg") is not None:
            extra.append(f"rot = {_num(cm['rotation_deg'], '{:+.2f}')}&deg;")
        if cm.get("selection"):
            extra.append(_html.escape(str(cm["selection"])))
        if extra:
            model_bits += f'<span class="sub"> &middot; {" &middot; ".join(extra)}</span>'

        aperture_bits = f"r = {_num(method.get('aperture_radius_bins'), '{:.2f}')} bins"
        ap_extra = ["identical for every peak"]
        if aniso.get("bragg_source"):
            ap_extra.append("Bragg anchor "
                            + _html.escape(str(aniso["bragg_source"])))
        if fam.get("power_frac_of_strongest") is not None:
            ap_extra.append("anchor holds "
                            f"{_num(fam['power_frac_of_strongest'] * 100, '{:.1f}')}% "
                            "of peak power")
        aperture_bits += f'<span class="sub"> &middot; {" &middot; ".join(ap_extra)}</span>'

        facts = [
            ("&alpha; (normalized)",
             f'{_num(aniso.get("alpha_normalized"), "{:.4f}")}'
             '<span class="sub"> from R<sub>i</sub> = C<sub>i</sub> / '
             'B<sub>i</sub></span>'),
            ("CDW peaks resolved",
             f'{_html.escape(str(aniso.get("n_directions_resolved", "—")))}/3'),
            ("Aperture", aperture_bits),
            ("CDW model", model_bits),
        ]

        ref = scr.get("reference_screen") or {}
        if ref.get("verdict"):
            ref_bits = _html.escape(str(ref["verdict"]))
            ref_extra = []
            if ref.get("alpha") is not None:
                ref_extra.append(
                    f"Original calibration &alpha; = {_num(ref['alpha'])} vs "
                    f"&alpha;<sub>crit</sub> = {_num(ref.get('alpha_crit'))}")
            if scr.get("rule_sensitive"):
                ref_extra.append("above the recalibrated threshold only")
            elif scr.get("concordant") is True:
                ref_extra.append("concordant with the operational verdict")
            elif scr.get("concordant") is False:
                ref_extra.append("discordant with the operational verdict")
            if ref_extra:
                ref_bits += (f'<span class="sub"> &middot; '
                             f'{" &middot; ".join(ref_extra)}</span>')
            facts.append(("Original calibration (&alpha;<sub>crit</sub> = 0.374)", ref_bits))
        parts.append('<dl class="aniso-facts">')
        for term, value in facts:
            parts.append(f'<dt>{term}</dt><dd>{value}</dd>')
        parts.append('</dl>')

    parts += ['</div>', '</div>']
    st.markdown("".join(parts), unsafe_allow_html=True)

    fig_b64 = result.get("didv_anisotropy_figure_base64")
    if ok and fig_b64:
        halves = _split_panels(base64.b64decode(fig_b64))

        if halves:
            titles = ("dI/dV map", "FFT with peak apertures")
            captions = (
                "dI/dV map (recovered scalar field)",
                "log(1+|FFT|²) with the apertures used",
            )

            image_parts = [
                '<div class="agent-row">',
                '<div class="agent-badge" '
                'style="visibility:hidden;" aria-hidden="true"></div>',
                '<div class="agent-arrow" '
                'style="visibility:hidden;" aria-hidden="true"></div>',
                '<div style="display:grid;'
                'grid-template-columns:repeat(2,minmax(0,1fr));'
                'gap:1rem;flex:1;min-width:0;">',
            ]

            for data, title, caption in zip(halves, titles, captions):
                image_parts.append(
                    '<div class="agent-card">'
                    f'<div class="aniso-title">{title}</div>'
                    f'<img src="data:image/png;base64,{data}" '
                    'style="width:100%;display:block;border-radius:4px;" '
                    f'alt="{_html.escape(caption, quote=True)}">'
                    '<div style="text-align:center;color:#9FA3B5;'
                    'font-size:0.85rem;margin-top:0.6rem;">'
                    f'{_html.escape(caption)}</div>'
                    '</div>'
                )

            image_parts += ['</div>', '</div>']
            st.markdown("".join(image_parts), unsafe_allow_html=True)
        else:
            st.warning(
                "The diagnostic figure could not be split into two images."
            )

_HANDEDNESS_PRESENTATION = {
    "indeterminate":     ("v-below",
                          "Indeterminate: intensity hierarchy not robust"),
    "not_licensed":      ("v-refused",
                          "Nematic intensity anisotropy; handedness not "
                          "determinable"),
    "no_measurement":    ("v-none", "No measurement"),
}


def render_handedness_card(result: Dict[str, Any]) -> None:
    h = result.get("didv_handedness") or {}
    if not h:
        return
    verdict = str(h.get("verdict") or "no_measurement")
    css, fallback = _HANDEDNESS_PRESENTATION.get(
        verdict, ("v-none", "Unknown verdict"))
    headline = str(h.get("headline") or fallback)
    reason = str(h.get("reason") or "")

    parts: List[str] = [
        '<div class="agent-row">',
        '<div class="agent-badge">ChiralNet</div>',
        '<div class="agent-arrow"></div>',
        '<div class="agent-card">',
        '<div class="aniso-title">Handedness tool</div>',
        f'<div class="aniso-verdict {css}">'
        f'<div class="vt">{_html.escape(headline)}</div>'
        + (f'<div class="vr">{_html.escape(reason)}</div>' if reason else "")
        + '</div>',
    ]

    if h.get("sense"):
        stab = h.get("stability") or {}
        facts = [
            ("Sense (raw)", _html.escape(str(h["sense"]))),
            ("Sense (normalized R<sub>i</sub>)",
             _html.escape(str(h.get("sense_normalized") or "—"))),
            ("Sense (tip-corrected)",
             _html.escape(str(h.get("sense_tip_corrected") or "—"))),
            ("Stability",
             (f"p(ccw) = {_num(stab.get('p_ccw'))} &middot; "
              f"p(cw) = {_num(stab.get('p_cw'))} &middot; "
              f"p(non-mono) = {_num(stab.get('p_non_monotonic'))}"
              f'<span class="sub"> &middot; {stab.get("n_draw", 0)} '
              'draws</span>') if stab else "—"),
            ("Hierarchy margin", _num(h.get("hierarchy_margin"), "{:.4f}")),
            ("Anisotropy screen",
             _html.escape(str(h.get("screen_verdict") or "—"))),
        ]
        parts.append('<dl class="aniso-facts">')
        for term, value in facts:
            parts.append(f'<dt>{term}</dt><dd>{value}</dd>')
        parts.append('</dl>')

    notes = [n for n in (h.get("notes") or []) if n]
    if notes:
        parts.append('<ul class="aniso-notes">')
        for note in notes:
            parts.append(f'<li>{_html.escape(str(note))}</li>')
        parts.append('</ul>')
    parts += ['</div>', '</div>']
    st.markdown("".join(parts), unsafe_allow_html=True)


_METRO_MARK = {"satisfied": ("&check;", "ok"),
               "partial": ("&#189;", "pt"),
               "not_satisfied": ("&#10007;", "no"),
               "unavailable": ("&ndash;", "na")}


def render_metrology_card(result: Dict[str, Any]) -> None:
    metro = result.get("chiral_metrology") or {}
    if not metro:
        return

    parts: List[str] = [
        '<div class="agent-row">',
        '<div class="agent-badge">ChiralNet</div>',
        '<div class="agent-arrow"></div>',
        '<div class="agent-card">',
        '<div class="metro-title">Chiral CDW metrology</div>',
    ]

    fig_b64 = result.get("chiral_metrology_figure_base64")
    if fig_b64:
        parts.append(
            '<figure class="metro-fig">'
            f'<img src="data:image/png;base64,{fig_b64}" '
            'alt="Chiral CDW metrology scorecard"></figure>')

    criteria = metro.get("criteria") or []
    numbers = list(range(1, len(criteria) + 1))

    if criteria:
        parts.append('<div class="metro-crit">')
        for i, c in zip(numbers, criteria):
            mark, mark_css = _METRO_MARK.get(
                str(c.get("status")), ("&ndash;", "na"))
            pts = (f'<span class="pts"> &middot; '
                   f'{c.get("points_display") or c.get("points", 0)}/'
                   f'{c.get("weight", 0)}%</span>'
                   if c.get("weight") is not None else "")
            subs = c.get("subcriteria") or []
            if subs:
                body = ['<div class="metro-sub">']
                for sc in subs:
                    s_mark, s_css = _METRO_MARK.get(
                        str(sc.get("status")), ("&ndash;", "na"))
                    body.append(
                        f'<div class="mark {s_css}">{s_mark}</div>'
                        f'<div><span class="subname">'
                        f'{_html.escape(str(sc.get("title", "")))}</span> '
                        f'<span class="val">&middot; '
                        f'{_html.escape(str(sc.get("value", "")))}</span>'
                        f'<span class="pts"> &middot; '
                        f'{sc.get("points_display") or 0}/'
                        f'{sc.get("weight", 0)}%</span>'
                        f'<div class="det">'
                        f'{_html.escape(str(sc.get("detail", "")))}</div>'
                        f'</div>')
                body.append('</div>')
                inner = "".join(body)
            else:
                inner = (f'<div class="det">'
                         f'{_html.escape(str(c.get("detail", "")))}</div>')
            parts.append(f'<div class="n">{i}.</div>'
                         f'<div class="mark {mark_css}">{mark}</div>'
                         f'<div><span class="name">'
                         f'{_html.escape(str(c.get("title", "")))}</span>'
                         f'{pts}'
                         f'{inner}'
                         f'</div>')
        parts.append('</div>')

    parts += ['</div>', '</div>']
    st.markdown("".join(parts), unsafe_allow_html=True)


def _agent_fields(result: Dict[str, Any], source: str
                  ) -> Optional[Dict[str, str]]:
    analysis = result.get(source)
    if not isinstance(analysis, dict) or not analysis:
        return None
    label = analysis.get("final_label")
    if not label and str(analysis.get("status") or "").lower() == "skipped":
        label = "Not run"
    explanation = (analysis.get("explanation") or analysis.get("reasoning")
                   or analysis.get("error") or "")
    return {"label": str(label or "—"),
            "confidence": _confidence_text(analysis.get("confidence")),
            "explanation": " ".join(str(explanation).split()) or "—"}


def render_agent_cards(result: Dict[str, Any]) -> None:
    for source, title in AGENT_PANELS:
        f = _agent_fields(result, source)
        if f is None:
            continue
        st.markdown(
            '<div class="agent-row">'
            '<div class="agent-badge">ChiralNet</div>'
            '<div class="agent-arrow"></div>'
            '<div class="agent-card">'
            f'<div class="agent-title">{_html.escape(title)}</div>'
            '<dl class="agent-fields">'
            f'<dt>Final label:</dt><dd>{_html.escape(f["label"])}</dd>'
            f'<dt>Confidence:</dt><dd>{_html.escape(f["confidence"])}</dd>'
            f'<dt>Explanation:</dt><dd>{_html.escape(f["explanation"])}</dd>'
            '</dl>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


anthropic_key = st.text_input("Anthropic API Key", type="password")

core_pipeline.clients["anthropic"] = (
    Anthropic(api_key=anthropic_key) if (anthropic_key and Anthropic) else None)

_missing_models = core_pipeline.missing_agent_models()

import stm_data_io as _sdio

st.title("ChiralNet")

col1, col2 = st.columns([2, 1])
with col1:
    _IMG_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]
    _NATIVE_TYPES = ["sxm", "3ds"]
    _ALL_TYPES = _NATIVE_TYPES + _IMG_TYPES

    topo_file = st.file_uploader(
        "Topograph (required)", type=_ALL_TYPES,
        help="Image (.png/.jpg/.tif) as before, or Nanonis .sxm / .3ds "
             "for the measured data. For a .3ds grid the topograph is "
             "the per-point tip height, 'Z (m)'.")
    didv_file = st.file_uploader(
        "dI/dV Map (optional)", type=_ALL_TYPES,
        help="Image as before, or Nanonis .3ds / .sxm for the measured "
             "lock-in signal. A .3ds grid also lets you pick the bias.")

    def _inspect(f):
        if not f or not _sdio.is_native_extension(Path(f.name).suffix):
            return {}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, "in" + Path(f.name).suffix.lower())
                with open(p, "wb") as fh:
                    fh.write(f.getvalue())
                field, info = _sdio.load_native_field(p)
            return {"info": info, "shape": field.shape,
                    "channels": info.get("available_channels") or []}
        except Exception as exc:
            return {"error": str(exc)}

    def _preview(f, insp, cmap="gray", channel=None, direction="forward",
                 bias=None, prefer="auto"):
        if not f or not insp.get("info"):
            return None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, "in" + Path(f.name).suffix.lower())
                with open(p, "wb") as fh:
                    fh.write(f.getvalue())
                field, info = _sdio.load_native_field(
                    p, channel=channel, direction=direction, bias=bias,
                    prefer=prefer)
            return _sdio.render_preview_png(field, info, cmap=cmap)
        except Exception:
            return None

    def _default_index(channels, prefer):
        try:
            return int(_sdio._pick_channel(list(channels), None,
                                           prefer=prefer))
        except Exception:
            return 0

    topo_native = _inspect(topo_file)
    didv_native = _inspect(didv_file)

    for label, insp in (("Topograph", topo_native), ("dI/dV map", didv_native)):
        if insp.get("error"):
            st.error(f"{label}: could not read this file - {insp['error']}")
        elif insp.get("info"):
            i = insp["info"]
            size = ""
            if i.get("lx") and i.get("ly"):
                size = (f" - {i['lx']:.3g} x {i['ly']:.3g} "
                        f"{i.get('length_unit') or ''}".rstrip()
                        + " from the file header, FFT will be calibrated")
            st.caption(
                f"{label}: {i.get('source_format')} - "
                f"{insp['shape'][1]}x{insp['shape'][0]} px{size}")

    topo_channel = didv_channel = None
    topo_direction = didv_direction = "forward"
    didv_bias = None
    if len(topo_native.get("channels") or []) > 1:
        topo_channel = st.selectbox(
            "Topograph channel", topo_native["channels"],
            index=_default_index(topo_native["channels"], "topo"),
            help="The tip height, 'Z (m)'. On a .3ds grid the swept "
                 "channels (Current, Bias, LI X) are spectra, not a "
                 "topograph: the current is held at the setpoint by the "
                 "feedback, so its map is noise.")
    if len(didv_native.get("channels") or []) > 1:
        didv_channel = st.selectbox(
            "dI/dV channel", didv_native["channels"],
            index=_default_index(didv_native["channels"], "didv"),
            help="The lock-in channel (LIX / dI/dV), not the current.")
    if topo_file and Path(topo_file.name).suffix.lower() == ".sxm":
        topo_direction = st.radio(
            "Topograph scan direction", ["forward", "backward"],
            horizontal=True,
            help="Comparing the two frames is the cheapest drift and "
                 "tip-stability check available.")
    _dinfo = didv_native.get("info") or {}
    _sweep = _dinfo.get("sweep_range_v")
    if _sweep:
        _lo = round(float(min(_sweep)), 4)
        _hi = round(float(max(_sweep)), 4)
        _sp = _dinfo.get("setpoint_bias_v")
        _default = (float(_sp) if _sp is not None
                    and _lo - 1e-6 <= _sp <= _hi + 1e-6
                    else float(sum(_sweep) / 2.0))
        _default = min(max(0.0, _lo), _hi)
        didv_bias = st.slider(
            "dI/dV bias slice (V)", _lo, _hi,
            _default, step=0.001, format="%.3f",
            help="Which constant-bias slice of the grid is measured. For a "
                 "CDW this is normally the gap-edge energy.")

    if topo_file or didv_file:
        st.divider()
        st.markdown("#### Upload previews")

        prev_topo, prev_didv = st.columns(2, gap="medium")

        with prev_topo:
            st.markdown("**Topograph**")
            if topo_file:
                st.image(_preview(topo_file, topo_native, cmap="gray",
                                  channel=topo_channel,
                                  direction=topo_direction, prefer="topo")
                         or topo_file,
                         width="stretch", caption=f"{topo_file.name}")
            else:
                st.info("No topograph uploaded (required)")

        with prev_didv:
            st.markdown("**dI/dV map**")
            if didv_file:
                st.image(_preview(didv_file, didv_native, cmap="magma",
                                  channel=didv_channel,
                                  direction=didv_direction, bias=didv_bias,
                                  prefer="didv")
                         or didv_file,
                         width="stretch", caption=f"{didv_file.name}")
            else:
                st.caption("No dI/dV map uploaded")

with col2:
    fov_x = st.number_input("Topograph width (nm)", value=None,
                               format="%.2f", placeholder="10.0",
                               help="Full physical width of the topograph. "
                                    "Both X and Y are required for a "
                                    "calibrated FFT; otherwise the FFT is "
                                    "reported in cycles/pixel.")
    fov_y = st.number_input("Topograph height (nm)", value=None,
                               format="%.2f", placeholder="10.0")

    CDW_MODELS = {
        "Auto-detect": None,
        "2x2": "2x2",
        "3x3": "3x3",
        "sqrt13 x sqrt13": "sqrt13xsqrt13",
        "sqrt3 x sqrt3": "sqrt3xsqrt3",
        "4x4": "4x4",
    }
    cdw_model_label = st.selectbox(
        "CDW superlattice", list(CDW_MODELS) + ["Custom…"],
        help="Pin the superlattice when the crystal structure is known. "
             "Auto-detection is a fallback and can pick the wrong shell "
             "when the atomic Bragg peaks are weak.")

    if cdw_model_label == "Custom…":
        cq1, cq2 = st.columns(2)
        custom_ratio = cq1.number_input(
            "|q_CDW| / |q_Bragg|", min_value=0.01, max_value=1.0,
            value=0.50, step=0.01, format="%.4f",
            help="1/n for an n×n cell (2×2 → 0.5, 3×3 → 0.3333); "
                 "1/√n for a √n×√n cell (√13 → 0.2774).")
        custom_rot = cq2.number_input(
            "Rotation (deg)", min_value=0.0, max_value=180.0,
            value=0.0, step=0.1, format="%.2f",
            help="Rotation of the CDW wavevectors relative to the Bragg "
                 "directions. 0 for n×n cells; 13.9° for √13×√13; 30° "
                 "for √3×√3.")
        cdw_model = {"ratio": float(custom_ratio),
                     "rotation_deg": float(custom_rot),
                     "name": f"custom r={custom_ratio:.4f}"}
        st.caption("The geometry checks still apply: a ratio that does not "
                   "land on real peaks is rejected rather than fitted.")
    else:
        cdw_model = CDW_MODELS[cdw_model_label]

    run = st.button("Run ChiralNet", type="primary", use_container_width=True)

if run:
    if _missing_models:
        st.error("Missing Anthropic API key (required for "
                 + "; ".join(f"{m['label']}: {', '.join(m['agents'])}"
                             for m in _missing_models) + ").")
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
                "metadata": {"fov_x": fov_x, "fov_y": fov_y,
                             "cdw_model": cdw_model,
                             "didv_ext": (Path(didv_file.name).suffix
                                          if didv_file else None),
                             "topo_channel": topo_channel,
                             "topo_direction": topo_direction,
                             "didv_channel": didv_channel,
                             "didv_direction": didv_direction,
                             "didv_bias": didv_bias},
            }

            st.session_state["chiralnet_result"] = \
                agentic_graph.invoke(initial_state)

result = st.session_state.get("chiralnet_result")
if result:
    st.success("Classification complete")

    render_metrology_card(result)

    render_final_card(result)

    render_anisotropy_card(result)

    render_handedness_card(result)

    with st.expander("Specialist agent reports"):
        render_agent_cards(result)

    fft_num = result.get("fft_numerical") or {}
    with st.expander("Numerical FFT analysis", expanded=False):
        if fft_num.get("status") == "ok":
            if result.get("fft_labeled_figure_base64"):
                st.image(base64.b64decode(
                    result["fft_labeled_figure_base64"]))
        else:
            st.warning(fft_num.get("message",
                                   "Numerical FFT not available."))
