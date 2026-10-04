from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Tuple


WEIGHTS: Dict[str, int] = {
    "alpha": 40,
    "judge": 40,
    "handedness": 20,
}
assert sum(WEIGHTS.values()) == 100

ALPHA_SUBWEIGHTS: Dict[str, int] = {
    "artifact": 10,
    "anisotropy": 30,
}

ARTIFACT_POINTS_V2_ONLY = 5

ANISOTROPY_DELTA_MIN = 0.2

from didv_handedness import STABILITY_THRESHOLD as HANDEDNESS_STABILITY_MIN

_SENSES = ("counter-clockwise", "clockwise")


CHANNEL_ORDER = ("structural", "phase", "amplitude")

STRUCTURAL_MIRROR_FIXED_POINT_PERIOD_DEG = 30.0

STRUCTURAL_MODEL_ROTATION_TOL_DEG = 0.05

STRUCTURAL_MIN_OFFSET_DEG = 2.0

STRUCTURAL_MAX_SCATTER_DEG = 4.0

STRUCTURAL_BRAGG_POWER_FRAC_MAX = 0.35

STRUCTURAL_ANCHOR_RANK_MAX = 0


PHASE_SENSE_CONVENTION = "sin(Phi) > 0 -> counter-clockwise (display frame)"

PHASE_SENSE_MIN_SIGMA = 3.0

PHASE_SENSE_MIN_DEG = 2.0

HANDEDNESS_POINTS_ESTABLISHED_UNSIGNED = 16.0
HANDEDNESS_POINTS_AMPLITUDE_ONLY = 10.4
HANDEDNESS_POINTS_AMPLITUDE_UNREPRODUCED = 5.6

DETERMINISTIC_KEYS = ("alpha", "handedness")
JUDGE_KEYS = ("judge",)


JUDGE_ZERO_LABELS = ("moire", "noncdw", "notcdw", "inconclusive")

JUDGE_CHIRAL_LABEL = "Chiral CDW"

JUDGE_CHANCE_LABEL = "Chance of Chiral CDW"

JUDGE_CDW_LABEL = "CDW"

JUDGE_CHANCE_CONFIDENCE_PIVOT = 50.0
JUDGE_POINTS_CHANCE_ABOVE = 30.0
JUDGE_POINTS_CHANCE_AT = 25.0
JUDGE_POINTS_CHANCE_BELOW = 20.0
JUDGE_CHANCE_CONFIDENCE_TOL = 1e-6


BANDS: Tuple[Tuple[float, str, str, str, str], ...] = (
    (75.0, "green",  "Chiral CDW",           "#3FA45B", "#7FD79A"),
    (50.0, "yellow", "Chance of Chiral CDW", "#D4A72C", "#F2D16B"),
    (0.0,  "red",    "Not Chiral CDW",       "#C4483F", "#F0A5A5"),
)
_BAND_BY_KEY: Dict[str, Tuple[float, str, str, str, str]] = {
    row[1]: row for row in BANDS}


def _band_key(score: float) -> str:
    s = float(score)
    if s > _BAND_BY_KEY["green"][0]:
        return "green"
    if s >= _BAND_BY_KEY["yellow"][0]:
        return "yellow"
    return "red"


_BG = "#05060A"
_INK = "#FFFFFF"
_MUTED = "#9FA3B5"
_RULE = "#2E2E3C"
_UNFILLED = "#1B1C26"
_EDGE = "#3A3B4A"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _sense(value: Any) -> Optional[str]:
    s = str(value or "").strip().lower()
    return s if s in _SENSES else None


def _fmt_points(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:g}"


def _criterion(key: str, title: str, satisfied: Optional[bool],
               value: str, detail: str,
               points: Optional[int] = None,
               weight: Optional[int] = None) -> Dict[str, Any]:
    weight = int(WEIGHTS[key] if weight is None else weight)
    if satisfied is None:
        pts = 0.0
    elif satisfied:
        pts = float(weight) if points is None \
            else max(0.0, min(float(weight), float(points)))
    else:
        pts = max(0.0, min(float(weight), float(points or 0.0)))
    pts = round(pts, 4)
    return {
        "key": key,
        "title": title,
        "satisfied": bool(satisfied),
        "available": satisfied is not None,
        "weight": weight,
        "points": pts,
        "points_display": _fmt_points(pts),
        "status": ("unavailable" if satisfied is None
                   else "satisfied" if pts >= weight - 1e-9
                   else "partial" if pts > 0
                   else "not_satisfied"),
        "value": value,
        "detail": detail,
    }


def _wrap_signed(x: float, period: float) -> float:
    return (float(x) + 0.5 * period) % period - 0.5 * period


def _circ_mean_mod(angles: List[float], period: float) -> Optional[float]:
    if not angles:
        return None
    k = 2.0 * math.pi / period
    s = sum(math.sin(k * a) for a in angles)
    c = sum(math.cos(k * a) for a in angles)
    if abs(s) < 1e-15 and abs(c) < 1e-15:
        return None
    return (math.atan2(s, c) / k) % period


def structural_enantiomorph(aniso: Dict[str, Any],
                            hand: Optional[Dict[str, Any]] = None
                            ) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "channel": "structural",
        "sense": None,
        "resolved": False,
        "offset_deg": None,
        "offset_from_fixed_point_deg": None,
        "mirror_fixed_point_deg": None,
        "enantiomorph_separation_deg": None,
        "scatter_deg": None,
        "threshold_deg": None,
        "model": None,
        "model_is_mirror_fixed_point": None,
        "reason": "",
    }
    aniso = _as_dict(aniso)
    if aniso.get("status") != "ok":
        out["reason"] = "No anisotropy measurement; no peak positions."
        return out

    b = [_as_float(a) for a in (aniso.get("bragg_angles_deg") or [])]
    c = [_as_float(a) for a in (aniso.get("cdw_angles_deg") or [])]
    if len(b) != 3 or len(c) != 3 or any(x is None for x in b + c):
        out["reason"] = "Three Bragg and three CDW directions not available."
        return out

    bragg_family = _as_dict(aniso.get("bragg_family"))
    model_d = _as_dict(aniso.get("cdw_model"))
    frac = _as_float(bragg_family.get("power_frac_of_strongest"))
    rank = _as_float(bragg_family.get("anchor_rank"))
    shell_seen = model_d.get("cdw_shell_independently_detected")
    if shell_seen is None:
        shell_seen = aniso.get("cdw_shell_independently_detected")
    harmonic = aniso.get("anchor_harmonic_suspect")

    out["anchor_quality"] = {
        "power_frac_of_strongest": frac,
        "anchor_rank": (int(rank) if rank is not None else None),
        "cdw_shell_independently_detected": (None if shell_seen is None
                                             else bool(shell_seen)),
        "anchor_harmonic_suspect": bool(harmonic),
        "power_frac_is_refusal": False,
    }

    if rank is not None and rank > STRUCTURAL_ANCHOR_RANK_MAX:
        out["reason"] = (
            f"Bragg anchor is hexagonal family rank {int(rank)}, not the "
            "outermost; it may be a CDW shell, so the offset could be "
            "CDW-to-CDW.")
        return out

    if (shell_seen is False and frac is not None
            and frac > STRUCTURAL_BRAGG_POWER_FRAC_MAX):
        out["reason"] = (
            "Single resolved periodicity (CDW shell not independently "
            f"detected; anchor holds {frac:.0%} of peak power): lattice and "
            "superlattice roles cannot be separated.")
        return out

    notes: List[str] = []
    if frac is not None and frac > STRUCTURAL_BRAGG_POWER_FRAC_MAX:
        notes.append(f"Anchor holds {frac:.0%} of peak power (recorded, "
                     "not gated: the outermost family fixes the lattice "
                     "directions).")
    if harmonic:
        notes.append("Anchor may be a lattice harmonic; directions, and "
                     "hence the offset, are unaffected.")
    out["anchor_strength_note"] = " ".join(notes) or None

    b_ref = _circ_mean_mod([float(x) for x in b], 60.0)
    if b_ref is None:
        out["reason"] = "Bragg directions do not define a mean direction."
        return out

    raw = [float(x) - b_ref for x in c]
    mean_circ = _circ_mean_mod(raw, 60.0)
    if mean_circ is None:
        out["reason"] = "CDW-to-lattice offsets do not define a mean rotation."
        return out
    mean_off = _wrap_signed(mean_circ, 60.0)
    devs = [_wrap_signed(r - mean_circ, 60.0) for r in raw]
    scatter = max(devs) - min(devs)
    offsets = [mean_off + d for d in devs]

    period = STRUCTURAL_MIRROR_FIXED_POINT_PERIOD_DEG
    from_fixed = _wrap_signed(mean_off, period)
    fixed_point = mean_off - from_fixed

    out["offset_deg"] = round(mean_off, 3)
    out["offset_from_fixed_point_deg"] = round(from_fixed, 3)
    out["mirror_fixed_point_deg"] = round(fixed_point, 3)
    out["enantiomorph_separation_deg"] = round(abs(2.0 * from_fixed), 3)
    out["scatter_deg"] = round(scatter, 3)
    out["offsets_deg"] = [round(o, 3) for o in offsets]

    model = _as_dict(aniso.get("cdw_model"))
    out["model"] = model.get("name")
    rot = _as_float(model.get("rotation_deg"))
    model_fixed = None
    if rot is not None:
        out["model_rotation_deg_display"] = round(-rot, 3)
        out["model_rotation_agrees"] = bool(
            abs(_wrap_signed(-rot - mean_off, 60.0)) <= 3.0
            + STRUCTURAL_MAX_SCATTER_DEG)
        model_fixed = bool(
            abs(_wrap_signed(rot, period)) <= STRUCTURAL_MODEL_ROTATION_TOL_DEG)
        out["model_is_mirror_fixed_point"] = model_fixed

    if scatter > STRUCTURAL_MAX_SCATTER_DEG:
        out["reason"] = (
            f"CDW-to-lattice offsets scatter by {scatter:.1f}\u00b0 (limit "
            f"{STRUCTURAL_MAX_SCATTER_DEG:g}\u00b0): not a single rigid "
            "rotation (shear, mis-assigned anchor or mixed domains).")
        return out

    star = _as_dict(_as_dict(_as_dict(hand).get("geometry_control"))
                    .get("bragg_star"))
    dev = _as_float(star.get("angle_deviation_deg")) or 0.0
    thresh = max(STRUCTURAL_MIN_OFFSET_DEG, 1.5 * scatter, dev)
    out["threshold_deg"] = round(thresh, 3)

    if abs(from_fixed) <= thresh:
        out["sense"] = "achiral_superlattice"
        out["resolved"] = True
        at_zero = abs(fixed_point) < 1e-9
        cell = ("unrotated n\u00d7n cell" if at_zero
                else "\u221a3\u00d7\u221a3 R30\u00b0-type cell")
        out["reason"] = (
            f"CDW star at {mean_off:+.2f}\u00b0, within {thresh:.2f}\u00b0 "
            f"of the {fixed_point:+.0f}\u00b0 mirror-fixed point: {cell}, "
            "which has no enantiomorph.")
        return out

    if model_fixed:
        out["reason"] = (
            f"Assigned cell {out['model']!r} is mirror-fixed (rotation "
            f"{rot:+.3f}\u00b0) but the measured star sits "
            f"{from_fixed:+.2f}\u00b0 from the fixed point; model and "
            "geometry disagree, no sense assigned.")
        return out

    out["sense"] = "counter-clockwise" if mean_off > 0 else "clockwise"
    out["resolved"] = True
    out["reason"] = (
        f"CDW star rotated {mean_off:+.2f}\u00b0 from the lattice (scatter "
        f"{scatter:.2f}\u00b0, threshold {thresh:.2f}\u00b0): "
        f"{out['sense']} enantiomorph.")
    return out


def _first_sentence(text: Any, max_len: int = 200) -> str:
    t = " ".join(str(text or "").split())
    if not t:
        return ""
    import re as _re
    m = _re.search(r"(?<=[.;])\s+(?=[A-Z(])", t)
    head = t[:m.start()] if m else t
    if len(head) > max_len:
        head = head[:max_len - 1].rstrip() + "\u2026"
    return head


def _lc(text: str) -> str:
    t = str(text or "").strip().rstrip(".")
    if len(t) > 1 and t[0].isupper() and t[1].islower():
        t = t[0].lower() + t[1:]
    return t


def _phase_brief(phase: Dict[str, Any]) -> str:
    phase = _as_dict(phase)
    v = str(phase.get("verdict") or "")
    cp = _as_float(phase.get("chirality_phase"))
    ca = _as_float(phase.get("chirality_amplitude"))
    fl = _as_float(phase.get("phase_significance_floor"))
    pu = _as_float(phase.get("p_value_used_for_verdict"))
    if pu is None:
        pu = _as_float(phase.get("p_value_phase_achiral"))
    lattice = bool(phase.get("lattice_referenced"))

    def num(x, spec=".3f"):
        return "n/a" if x is None else format(x, spec)

    if not phase or v in ("", "no_measurement"):
        why = _first_sentence(phase.get("reason")) if phase else ""
        return "not run" + (f" ({_lc(why)})" if why else "")
    if v == "phase_chiral":
        return (f"all mirror planes broken (phase residual "
                f"{num(cp)} \u2265 floor {num(fl)}, p = {num(pu, '.2g')})")
    if v == "achiral":
        return (f"consistent with a mirror-symmetric 3Q state (phase "
                f"residual {num(cp)} < floor {num(fl)})")
    if v == "amplitude_only":
        return (f"amplitudes break the mirrors (residual {num(ca)}) without "
                f"phase support ({num(cp)} < {num(fl)})"
                + ("" if lattice else "; no lattice reference"))
    head = str(phase.get("headline") or "")
    if "leakage" in head:
        return "residual attributable to amplitude-to-phase leakage"
    if "reproduce" in head:
        return "residual not reproducible across sub-windows"
    if "winds" in head:
        w = _as_float(_as_dict(phase.get("registry_winding"))
                      .get("winding_cycles"))
        return (f"near-commensurate registry ({num(w, '.2f')} cycles across "
                "the frame)")
    if "could not be tested" in head:
        return (f"underpowered (residual {num(cp)} below method floor "
                f"{num(fl)})")
    snr = [x for x in (_as_float(y) for y in (phase.get("peak_snr") or []))
           if x is not None]
    if snr and min(snr) < 5.0:
        return f"weakest peak |F|/\u03c3 = {min(snr):.1f} < 5"
    cr = _as_float(phase.get("closure_residual_bins_cdw"))
    if cr is not None and cr > 0.75:
        return f"CDW triple does not close ({cr:.2f} bins)"
    if phase.get("hexagonal_star_supported") is False:
        return "peaks not a hexagonal star under a plausible distortion"
    n_req = phase.get("n_mirror_axes_requested")
    n_got = phase.get("n_mirror_axes_matched")
    if n_req and n_got is not None and n_got < n_req:
        return f"only {n_got}/{n_req} mirror axes matchable"
    why = _first_sentence(phase.get("reason"))
    return "indeterminate" + (f" ({_lc(why)})" if why else "")


def phase_registry_sense(phase: Dict[str, Any]) -> Dict[str, Any]:
    phase = _as_dict(phase)
    out: Dict[str, Any] = {
        "sense": None,
        "resolved": False,
        "source": "phase_sum_cdw_deg (closed-triple phase sum Phi)",
        "convention": PHASE_SENSE_CONVENTION,
        "phi_deg": None,
        "distance_to_mirror_even_deg": None,
        "sigma_deg": None,
        "margin_sigma": None,
        "reason": "",
    }

    if str(phase.get("verdict") or "") != "phase_chiral":
        out["reason"] = ("Mirror breaking not established in the phase "
                         "channel; \u03a6 not used as a sign.")
        return out

    phi = _as_float(phase.get("phase_sum_cdw_deg"))
    if phi is None:
        out["reason"] = "No closed-triple phase sum \u03a6; sign unavailable."
        return out

    folded = abs(_wrap_signed(phi, 360.0))
    dist = min(folded, 180.0 - folded)
    out["phi_deg"] = round(float(_wrap_signed(phi, 360.0)), 3)
    out["distance_to_mirror_even_deg"] = round(float(dist), 3)

    sigmas = [_as_float(s) for s in (phase.get("phase_sigma_deg") or [])]
    cdw_sigmas = [s for s in sigmas[:3] if s is not None and s >= 0]
    sigma = (math.sqrt(sum(s * s for s in cdw_sigmas))
             if len(cdw_sigmas) == 3 else None)
    out["sigma_deg"] = (round(float(sigma), 4) if sigma is not None else None)

    if dist < PHASE_SENSE_MIN_DEG:
        out["reason"] = (
            f"\u03a6 = {out['phi_deg']:+.2f}\u00b0 lies {dist:.2f}\u00b0 "
            f"from a mirror-even value (0\u00b0/180\u00b0), below the "
            f"{PHASE_SENSE_MIN_DEG:g}\u00b0 floor; enantiomorph unresolved.")
        return out

    if sigma is not None and sigma > 0:
        margin = dist / sigma
        out["margin_sigma"] = round(float(margin), 2)
        if margin < PHASE_SENSE_MIN_SIGMA:
            out["reason"] = (
                f"\u03a6 = {out['phi_deg']:+.2f}\u00b0 is {margin:.1f}\u03c3 "
                "from a mirror-even value (required "
                f"{PHASE_SENSE_MIN_SIGMA:g}\u03c3); enantiomorph unresolved.")
            return out

    sense = ("counter-clockwise" if math.sin(math.radians(phi)) > 0
             else "clockwise")
    out.update(
        sense=sense, resolved=True,
        reason=(
            f"\u03a6 = {out['phi_deg']:+.2f}\u00b0 ({dist:.2f}\u00b0"
            + (f", {out['margin_sigma']:.1f}\u03c3"
               if out["margin_sigma"] is not None else "")
            + f" from mirror-even): {sense} (convention sin \u03a6 > 0 "
              "\u2192 counter-clockwise)."))
    return out


def phase_channel(phase: Dict[str, Any]) -> Dict[str, Any]:
    phase = _as_dict(phase)
    verdict = str(phase.get("verdict") or "")
    lattice = bool(phase.get("lattice_referenced"))
    sense_info = phase_registry_sense(phase)

    out: Dict[str, Any] = {
        "channel": "phase",
        "verdict": verdict or None,
        "lattice_referenced": lattice,
        "status": "untested",
        "p_value": _as_float(phase.get("p_value_phase_achiral")),
        "chirality_phase": _as_float(phase.get("chirality_phase")),
        "chirality_amplitude": _as_float(phase.get("chirality_amplitude")),
        "sense": sense_info["sense"],
        "sense_resolved": bool(sense_info["resolved"]),
        "sense_source": sense_info["source"],
        "sense_convention": sense_info["convention"],
        "sense_detail": sense_info,
        "amplitude_sense_diagnostic": _sense(phase.get("amplitude_sense")),
        "reason": "",
    }

    brief = _phase_brief(phase)
    if verdict == "phase_chiral":
        out["status"] = "establishes"
    elif verdict == "achiral" and lattice:
        out["status"] = "contradicts"
    elif verdict == "amplitude_only" and lattice:
        out["status"] = "tested_negative"
    out["reason"] = f"Phase registry: {brief}."
    return out


def chirality_channels(aniso: Dict[str, Any], hand: Dict[str, Any],
                       phase: Dict[str, Any]) -> Dict[str, Any]:
    struct = structural_enantiomorph(aniso, hand)
    ph = phase_channel(phase)
    hand = _as_dict(hand)
    amp = {
        "channel": "amplitude",
        "sense": _sense(hand.get("sense")),
        "sense_normalized": _sense(hand.get("sense_normalized")),
        "sense_tip_corrected": _sense(hand.get("sense_tip_corrected")),
        "stability": _as_float(hand.get("stability_of_sense")),
        "verdict": hand.get("verdict"),
    }
    return {"order": list(CHANNEL_ORDER), "structural": struct,
            "phase": ph, "amplitude": amp}


def _screen_refusal_text(screen: Dict[str, Any], aniso: Dict[str, Any]
                         ) -> str:
    screen = _as_dict(screen)
    verdict = str(screen.get("verdict") or "")
    cal = _as_dict(screen.get("calibration"))
    if verdict == "unscreened":
        return "\u03b1 measured, but the calibrated screen did not run"
    if verdict == "no_measurement":
        why = _first_sentence(screen.get("reason") or aniso.get("message"))
        return "no \u03b1 measurement" + (f" ({_lc(why)})"
                                           if why else "")
    parts: List[str] = []
    ap, need = screen.get("aperture_mode"), cal.get("aperture")
    if ap is not None and need and str(ap) != str(need):
        parts.append(f"aperture mode \u2018{ap}\u2019 \u2260 calibrated "
                     f"\u2018{need}\u2019")
    snr, snr_floor = (_as_float(screen.get("min_snr")),
                      _as_float(cal.get("snr_floor")))
    if snr is not None and snr_floor is not None and snr < snr_floor:
        parts.append(f"min CDW SNR {snr:.1f} < {snr_floor:g}")
    w, w_floor = (_as_float(screen.get("width_bins")),
                  _as_float(cal.get("width_floor")))
    if w is not None and w_floor is not None and w < w_floor:
        parts.append(f"CDW peak width {w:.2f} < {w_floor:g} bins")
    if (w is None and screen.get("width_source")
            and "width" in str(screen.get("reason") or "")):
        parts.append("corrected CDW peak width not measurable")
    if not parts:
        why = _first_sentence(screen.get("reason"))
        if why:
            parts.append(why.rstrip("."))
    return ("outside the calibrated domain"
            + (f" ({'; '.join(parts)})" if parts else ""))


def _criterion_artifact(aniso: Dict[str, Any]) -> Dict[str, Any]:
    title = "Artifact assessment"
    w_art = ALPHA_SUBWEIGHTS["artifact"]
    screen = _as_dict(aniso.get("screen"))
    verdict = str(screen.get("verdict") or
                  ("unscreened" if aniso.get("status") == "ok"
                   else "no_measurement"))

    alpha = _as_float(screen.get("alpha_cdw", aniso.get("alpha_cdw")))
    crit = _as_float(screen.get("alpha_crit"))

    def threshold_comparison(name: str, value: Optional[float],
                            threshold: Optional[float]) -> str:
        if value is None or threshold is None:
            return ""

        relation = (
            ">" if value > threshold
            else "<" if value < threshold
            else "="
        )

        return (
            f"{name}: α_CDW = {value:.3f} {relation} "
            f"α_crit = {threshold:.3f}"
        )

    op = threshold_comparison("Recalibration", alpha, crit)

    if verdict == "above":
        ref = _as_dict(screen.get("reference_screen"))
        ref_verdict = str(ref.get("verdict") or "")
        ref_alpha = _as_float(ref.get("alpha"))
        ref_crit = _as_float(ref.get("alpha_crit"))
        ref_text = threshold_comparison(
        "Original calibration", ref_alpha, ref_crit
)

        if ref_verdict == "above":
            return _criterion(
                "artifact", title, True, "above both calibration thresholds",
                (op + "; " if op else "")
                + (ref_text + ". " if ref_text else "")
                + "Anisotropy exceeds the modelled artifact null.",
                weight=w_art)

        if ref_verdict == "below":
            return _criterion(
                "artifact", title, False, "above recalibrated threshold only",
                (op + ", " if op else "")
                + (ref_text + "." if ref_text else
                    "Original calibration could not be evaluated."),
                points=ARTIFACT_POINTS_V2_ONLY, weight=w_art)

        why = _first_sentence(ref.get("reason"))
        return _criterion(
            "artifact", title, False, "above recalibrated threshold; original calibration unavailable",
            (op + "; " if op else "")
            + "original calibration not evaluable"
            + (f" ({_lc(why)})" if why else "")
            + ".",
            points=ARTIFACT_POINTS_V2_ONLY, weight=w_art)

    if verdict == "below":
        overlap = bool(screen.get("aperture_overlap"))
        ref_v = str(_as_dict(screen.get("reference_screen"))
                    .get("verdict") or "")
        return _criterion(
            "artifact", title, False, "within artifact null",
            (op + ": " if op else "")
            + "anisotropy within the modelled artifact null."
            + (" The original calibration rule would pass it." if ref_v == "above"
               else "")
            + (" Aperture geometry affects \u03b1 on this map; treat as "
               "weak evidence." if overlap else ""),
            weight=w_art)

    value = {"refused": "outside calibrated domain",
             "no_measurement": "no measurement",
             "unscreened": "not screened"}.get(verdict, f"undecided ({verdict})")
    why = _screen_refusal_text(screen, aniso)
    if "no cdw superlattice model is supported by this map" in why.lower():
        why = "No α measurement. No CDW superlattice model is supported by this map"
    return _criterion(
        "artifact", title, None, value,
        (why[:1].upper() + why[1:] + ".") if why else "No screen verdict.",
        weight=w_art)


def _criterion_anisotropy(aniso: Dict[str, Any]) -> Dict[str, Any]:
    title = "Anisotropy measurement"
    w_ani = ALPHA_SUBWEIGHTS["anisotropy"]
    a_cdw = _as_float(aniso.get("alpha_cdw"))
    a_bragg = _as_float(aniso.get("alpha_bragg"))
    delta = _as_float(aniso.get("alpha_excess_over_bragg"))
    if delta is None:
        delta = _as_float(aniso.get("alpha_corrected"))
    if delta is None and a_cdw is not None and a_bragg is not None:
        delta = a_cdw - a_bragg

    if delta is None:
        missing = ("\u03b1_CDW" if a_cdw is None else "\u03b1_Bragg")
        return _criterion(
            "anisotropy", title, None, "not measured",
            f"\u0394\u03b1 not formed: {missing} unavailable.",
            weight=w_ani)

    delta = round(delta, 6)
    ok = delta > ANISOTROPY_DELTA_MIN
    pieces = ("" if a_cdw is None or a_bragg is None
              else f" = {a_cdw:.3f} \u2212 {a_bragg:.3f}")
    tip = _as_float(aniso.get("alpha_excess_tipscaled"))
    rel = ">" if ok else "\u2264"
    return _criterion(
        "anisotropy", title, ok, f"\u0394\u03b1 = {delta:+.3f}",
        f"\u0394\u03b1 = \u03b1_CDW \u2212 \u03b1_Bragg{pieces} = "
        f"{delta:+.3f} {rel} {ANISOTROPY_DELTA_MIN:g}. "
        "Raw excess over the Bragg control, not q\u00b2-tip-corrected"
        + (f" (tip-scaled excess {tip:+.3f})" if tip is not None else "")
        + ".",
        weight=w_ani)


def _criterion_alpha(aniso: Dict[str, Any]) -> Dict[str, Any]:
    title = "Alpha measurement"
    art = _criterion_artifact(aniso)
    ani = _criterion_anisotropy(aniso)
    subs = [art, ani]

    pts = round(sum(float(c["points"]) for c in subs), 4)
    any_available = any(c["available"] for c in subs)
    all_satisfied = all(c["satisfied"] for c in subs)

    def _short(c: Dict[str, Any]) -> str:
        return f"{c['key']} {c['points_display']}/{c['weight']}"

    if not any_available:
        value = "not measured"
        satisfied: Optional[bool] = None
        detail = " ".join(str(c.get("detail") or "") for c in subs)
    else:
        satisfied = all_satisfied
        value = "\u03b1 thresholds: " + " + ".join(_short(c) for c in subs)
        detail = ("Two thresholds on one \u03b1 measurement. "
                  + " ".join(str(c.get("detail") or "") for c in subs))

    c = _criterion("alpha", title, satisfied, value, detail,
                   points=pts if satisfied is not None else None)
    c["subcriteria"] = subs
    c["subweights"] = dict(ALPHA_SUBWEIGHTS)
    c["shared_measurement"] = (
        "alpha_cdw (and alpha_bragg for the excess), from one "
        "didv_anisotropy measurement")
    c["n_subcriteria_satisfied"] = sum(1 for x in subs
                                       if x["status"] == "satisfied")
    c["n_subcriteria_unavailable"] = sum(1 for x in subs
                                         if not x["available"])
    return c


def _amplitude_channel_check(hand: Dict[str, Any]) -> Dict[str, Any]:
    hand = _as_dict(hand)
    raw = _sense(hand.get("sense"))
    norm = _sense(hand.get("sense_normalized"))
    tip = _sense(hand.get("sense_tip_corrected"))

    if raw is None:
        why = _first_sentence(hand.get("reason"))
        return {"state": "unavailable", "sense": None,
                "value": "no raw sense",
                "detail": ("No cyclic intensity hierarchy"
                           + (f" ({_lc(why)})" if why else "")
                           + ".")}

    p_win = _as_float(hand.get("stability_of_sense"))
    if p_win is None:
        stab = _as_dict(hand.get("stability"))
        p_win = _as_float(stab.get("p_ccw" if raw == "counter-clockwise"
                                   else "p_cw"))
    if p_win is None:
        return {"state": "unavailable", "sense": raw,
                "value": "no stability estimate",
                "detail": (f"Intensity sense {raw} not tested against "
                           "photometric noise.")}
    if p_win < HANDEDNESS_STABILITY_MIN:
        return {"state": "fail", "sense": raw, "stability": p_win,
                "value": f"unstable (p = {p_win:.0%})",
                "detail": (f"Intensity sense {raw} survives {p_win:.0%} of "
                           f"redraws (< {HANDEDNESS_STABILITY_MIN:.0%}).")}

    rep = _as_dict(hand.get("reproducibility"))
    rep_status = str(rep.get("status") or "missing")
    if rep_status in ("missing", "unavailable"):
        why = _first_sentence(rep.get("reason"))
        return {"state": "pass_unreproduced", "sense": raw,
                "stability": p_win,
                "value": "amplitude only, reproducibility untested",
                "detail": (f"Intensity sense {raw}, stable in {p_win:.0%} of "
                           "redraws; sub-window reproducibility not tested"
                           + (f" ({_lc(why)})" if why else "")
                           + ".")}
    if rep_status == "not_significant":
        return {"state": "fail", "sense": raw, "stability": p_win,
                "value": "not reproducible",
                "detail": (f"Intensity sense {raw} does not reproduce across "
                           f"{rep.get('n_tiles', '?')} sub-windows "
                           f"(p = {rep.get('p_value')}): finite-window "
                           "fluctuation.")}
    rep_sense = _sense(rep.get("sense"))
    if rep_sense is not None and rep_sense != raw:
        return {"state": "fail", "sense": raw, "stability": p_win,
                "value": "sub-windows oppose",
                "detail": (f"Full-window sense {raw} opposes the sub-window "
                           f"sense {rep_sense}.")}

    compared = [("R\u1d62-normalized", norm), ("tip-corrected", tip)]
    usable = [(name, sn) for name, sn in compared if sn is not None]
    if not usable:
        return {"state": "unavailable", "sense": raw, "stability": p_win,
                "value": "no corrected sense",
                "detail": ("No R\u1d62-normalized or tip-corrected sense "
                           "to cross-check.")}

    opposed = [name for name, sn in usable if sn != raw]
    ok = not opposed

    if opposed:
        detail = (f"Intensity sense {raw} is reversed by the "
                  f"{' and '.join(opposed)} sense: instrumental response "
                  "may set the hierarchy.")
    else:
        detail = (f"Intensity sense {raw}: stable in {p_win:.0%} of redraws, "
                  "unchanged by " + " and ".join(n for n, _ in usable)
                  + f" analysis, reproducible over {rep.get('n_tiles', '?')} "
                  f"sub-windows (p = {rep.get('p_value')}).")

    return {"state": "pass" if ok else "fail", "sense": raw,
            "stability": p_win, "opposed": opposed,
            "value": "opposed" if opposed else "all agree",
            "detail": detail}


def _structural_brief(struct: Dict[str, Any]) -> str:
    struct = _as_dict(struct)
    off = _as_float(struct.get("offset_deg"))
    sense = struct.get("sense")
    if struct.get("resolved") and sense in _SENSES and off is not None:
        return f"CDW star rotated {off:+.2f}\u00b0 from the lattice ({sense})"
    if sense == "achiral_superlattice":
        fp = _as_float(struct.get("mirror_fixed_point_deg")) or 0.0
        return ("cell on a mirror-fixed point"
                + (f" ({off:+.2f}\u00b0 \u2248 {fp:+.0f}\u00b0)"
                   if off is not None else "")
                + "; no enantiomorph")
    why = _first_sentence(struct.get("reason"))
    return "not resolved" + (f" ({_lc(why)})" if why else "")


def _criterion_handedness(hand: Dict[str, Any], aniso: Dict[str, Any],
                          channels: Dict[str, Any]) -> Dict[str, Any]:
    title = "Handedness determination tool"

    struct = _as_dict(channels.get("structural"))
    ph = _as_dict(channels.get("phase"))
    amp = _amplitude_channel_check(hand)

    struct_sense = struct.get("sense") if struct.get("resolved") else None
    struct_establishes = struct_sense in _SENSES
    phase_status = str(ph.get("status") or "untested")

    establishing: List[str] = []
    if struct_establishes:
        establishing.append("structural")
    if phase_status == "establishes":
        establishing.append("phase")

    contradicting = ["phase"] if phase_status == "contradicts" else []
    tested_negative = ["phase"] if phase_status == "tested_negative" else []

    def _mk(satisfied, value, detail, points=None):
        c = _criterion("handedness", title, satisfied, value, detail,
                       points=points)
        c["channels"] = {
            "structural": {"sense": struct.get("sense"),
                           "resolved": bool(struct.get("resolved")),
                           "offset_deg": struct.get("offset_deg"),
                           "reason": struct.get("reason")},
            "phase": {"status": phase_status,
                      "verdict": ph.get("verdict"),
                      "p_value": ph.get("p_value"),
                      "reason": ph.get("reason")},
            "amplitude": {"state": amp.get("state"),
                          "sense": amp.get("sense"),
                          "stability": amp.get("stability"),
                          "detail": amp.get("detail")},
        }
        c["evidence_class"] = c.get("evidence_class") or "amplitude_only"
        return c

    struct_txt = f"Structural enantiomorph: {_structural_brief(struct)}."
    phase_txt = str(ph.get("reason") or "Phase registry: not run.")

    if contradicting:
        c = _mk(False, "phase registry: mirror-symmetric",
                phase_txt + " No chirality.")
        c["evidence_class"] = "contradicted"
        return c

    if establishing:
        bits = []
        if struct_establishes:
            bits.append(struct_txt)
        if phase_status == "establishes":
            bits.append(phase_txt)
        base = " ".join(bits)

        phase_sense = ph.get("sense") if ph.get("sense_resolved") else None
        resolved: List[Tuple[str, str]] = []
        if struct_sense in _SENSES:
            resolved.append(("structural", struct_sense))
        if phase_sense in _SENSES:
            resolved.append(("phase", phase_sense))

        if len(resolved) == 2 and resolved[0][1] != resolved[1][1]:
            c = _mk(False, "established, senses disagree",
                    base + f" Structural ({resolved[0][1]}) and phase "
                    f"({resolved[1][1]}) senses disagree: enantiomorph "
                    "unresolved.",
                    points=HANDEDNESS_POINTS_ESTABLISHED_UNSIGNED)
            c["evidence_class"] = "established_sense_conflict"
            c["resolved_senses"] = {k: v for k, v in resolved}
            return c

        if not resolved:
            c = _mk(False, "established (mirror breaking), sense unresolved",
                    base + _phase_sense_note(ph)
                    + " Mirror breaking without a resolved enantiomorph.",
                    points=HANDEDNESS_POINTS_ESTABLISHED_UNSIGNED)
            c["evidence_class"] = "established_unsigned"
            return c

        chan, sense_final = resolved[0]
        concord = ""
        if len(resolved) == 2:
            concord = (f" Structural and phase senses agree "
                       f"({sense_final}).")
        elif chan == "phase":
            concord = _phase_sense_note(ph)
        c = _mk(True,
                f"established: {' + '.join(establishing)} ({sense_final})",
                base + concord)
        c["evidence_class"] = "robust"
        c["sense"] = sense_final
        c["sense_source"] = chan
        c["resolved_senses"] = {k: v for k, v in resolved}
        return c

    robust_note = " " + struct_txt + " " + phase_txt
    screen = _as_dict(_as_dict(aniso).get("screen"))
    screen_verdict = str(screen.get("verdict") or "")
    if (screen_verdict == "below"
            and amp["state"] in ("pass", "pass_unreproduced")):
        a_val = _as_float(screen.get("alpha_cdw"))
        a_crit = _as_float(screen.get("alpha_crit"))
        pair = ("" if a_val is None or a_crit is None
                else f" (\u03b1_CDW = {a_val:.3f} \u2264 {a_crit:.3f})")
        c = _mk(False, "screened out: spread is artifact-consistent",
                f"Intensity sense {amp.get('sense')} passes its controls, "
                f"but the hierarchy spread is within the artifact null{pair}; "
                "not credited." + robust_note)
        c["evidence_class"] = "amplitude_screened_out"
        return c

    if amp["state"] == "unavailable":
        c = _mk(None, amp["value"], amp["detail"] + robust_note)
        c["evidence_class"] = "untested"
        return c

    if amp["state"] == "fail":
        c = _mk(False, amp["value"], amp["detail"] + robust_note)
        c["evidence_class"] = "amplitude_failed"
        return c

    unreproduced = amp["state"] == "pass_unreproduced"
    pts = float(HANDEDNESS_POINTS_AMPLITUDE_UNREPRODUCED if unreproduced
                else HANDEDNESS_POINTS_AMPLITUDE_ONLY)
    cap_note = (" Corroborating only: three intensities at 60\u00b0 are a "
                "mean plus a nematic director (achiral), so this channel "
                "cannot establish handedness.")

    if tested_negative:
        c = _mk(False, "amplitude only, not corroborated",
                amp["detail"] + " " + phase_txt + cap_note, points=pts)
        c["evidence_class"] = "amplitude_only_uncorroborated"
        return c

    c = _mk(False, amp["value"], amp["detail"] + cap_note + robust_note,
            points=pts)
    c["evidence_class"] = ("amplitude_only_unreproduced" if unreproduced
                           else "amplitude_only")
    return c


def _phase_sense_note(ph: Dict[str, Any]) -> str:
    detail = _as_dict(_as_dict(ph).get("sense_detail"))
    why = str(detail.get("reason") or "").strip()
    return (" " + why) if why else ""


def _normalise_judge_label(label: Any) -> str:
    return "".join(ch for ch in str(label or "").lower() if ch.isalpha())


def judge_award(judge_label: Any, confidence: Any = None) -> Dict[str, Any]:
    w = float(WEIGHTS["judge"])
    norm = _normalise_judge_label(judge_label)
    conf = _as_float(confidence)
    shown = str(judge_label)

    def _out(points: float, branch: str, explanation: str) -> Dict[str, Any]:
        return {
            "points": round(float(points), 4),
            "base": round(float(points), 4),
            "adjustment": 0.0,
            "branch": branch,
            "shortfall": None,
            "confidence": conf,
            "rule": ("Chiral CDW 40; Chance of Chiral CDW 30 / 25 / 20 for "
                     "confidence > / = / < 50%; otherwise 0"),
            "explanation": explanation,
        }

    if norm == _normalise_judge_label(JUDGE_CHIRAL_LABEL):
        return _out(
            w,
            "chiral",
            "Integrated moiré, topographic, and spectroscopic evidence "
            "supports CDW order with a chiral signature."
        )

    if norm == _normalise_judge_label(JUDGE_CHANCE_LABEL):
        pivot = JUDGE_CHANCE_CONFIDENCE_PIVOT

        if conf is None:
            return _out(
                JUDGE_POINTS_CHANCE_BELOW,
                "chance_no_confidence",
                "A provisional chiral-CDW classification was returned, "
                "but its evidence strength cannot be assessed because "
                "confidence is unavailable."
            )

        if abs(conf - pivot) <= JUDGE_CHANCE_CONFIDENCE_TOL:
            return _out(
                JUDGE_POINTS_CHANCE_AT,
                "chance_at",
                "Evidence for chiral CDW is equivocal and does not support a "
                "definitive chiral CDW classification."
            )

        if conf > pivot:
            return _out(
                JUDGE_POINTS_CHANCE_ABOVE,
                "chance_above",
                "Integrated evidence favors chiral CDW but does not establish a "
                "definitive chiral CDW classification."
            )

        return _out(
            JUDGE_POINTS_CHANCE_BELOW,
            "chance_below",
            "Evidence provides limited support for chiral CDW and is insufficient "
            "for a definitive chiral CDW classification."
        )

    if norm == _normalise_judge_label(JUDGE_CDW_LABEL):
        return _out(
            0.0,
            "cdw",
            "CDW order is supported, but the combined evidence does not "
            "establish a chiral signature."
        )

    if norm in ("noncdw", "notcdw"):
        return _out(
            0.0,
            "zero_label",
            "CDW order is not supported; therefore, a chiral-CDW "
            "interpretation is not applicable."
        )

    if norm == "moire":
        return _out(
            0.0,
            "zero_label",
            "The observed periodic modulation is attributed to moiré "
            "interference rather than chiral CDW order."
        )

    if norm == "inconclusive":
        return _out(
            0.0,
            "zero_label",
            "The agent evidence is insufficient or conflicting, preventing "
            "a defensible CDW or chirality assignment."
        )

    if norm in JUDGE_ZERO_LABELS:
        return _out(
            0.0,
            "zero_label",
            "The final classification does not support a chiral-CDW "
            "interpretation."
        )

    return _out(
        0.0,
        "unrecognised_label",
        f"The returned classification “{shown}” is not recognized by the "
        "metrology decision rule."
    )

def _criterion_judge(state: Dict[str, Any]) -> Dict[str, Any]:
    title = "Judge agent decision"
    label = str(state.get("final_label"))
    ok = label == JUDGE_CHIRAL_LABEL
    award = judge_award(label, state.get("confidence"))

    gate = _as_dict(state.get("anisotropy_gate"))
    note = ""
    if gate.get("action") == "flagged":
        note = (" The calibrated \u03b1 screen is below \u03b1_crit for this "
                "map; this is scored in the artifact sub-criterion and does "
                "not alter the label.")
    elif gate.get("action") == "deferred":
        note = (" The \u03b1-screen comparison is deferred: the verdict "
                "depends on aperture geometry.")

    c = _criterion("judge", title, ok, label, award["explanation"] + note,
                   points=award["points"])
    c["award"] = award
    return c


def _split_scores(criteria: List[Dict[str, Any]]) -> Dict[str, Any]:
    det = [c for c in criteria if c.get("key") in DETERMINISTIC_KEYS]
    jud = [c for c in criteria if c.get("key") in JUDGE_KEYS]

    det_pts = round(sum(float(c["points"]) for c in det), 4)
    det_weight = sum(int(c["weight"]) for c in det)
    det_evaluable = sum(int(c["weight"]) for c in det if c.get("available"))
    jud_pts = round(sum(float(c["points"]) for c in jud), 4)
    jud_weight = sum(int(c["weight"]) for c in jud)

    return {
        "deterministic_points": det_pts,
        "deterministic_weight": det_weight,
        "deterministic_weight_evaluable": det_evaluable,
        "deterministic_score_percent": (
            int(round(100.0 * det_pts / det_evaluable))
            if det_evaluable else None),
        "deterministic_satisfied": any(c.get("status") == "satisfied"
                                       for c in det),
        "judge_points": jud_pts,
        "judge_weight": jud_weight,
        "judge_score_percent": (int(round(100.0 * jud_pts / jud_weight))
                                if jud_weight else None),
        "judge_carries_more_than_measurements": bool(jud_pts > det_pts),
    }


def _reachable_points(criteria: List[Dict[str, Any]]) -> int:
    return int(sum(int(c["weight"]) for c in criteria if c.get("available")))


def _display_score(value: float) -> Any:
    v = round(float(value), 4)
    return int(round(v)) if abs(v - round(v)) < 1e-9 else round(v, 1)


def score_band(score_percent: float,
               reachable_points: Optional[float] = None) -> Dict[str, Any]:
    s = max(0.0, min(100.0, float(score_percent)))
    key = _band_key(s)

    _, band, label, fill, text = _BAND_BY_KEY[key]
    out = {
        "score_percent": _display_score(s),
        "band": band,
        "band_label": label,
        "color": fill,
        "text_color": text,
    }

    if reachable_points is not None:
        rp = max(0.0, min(100.0, float(reachable_points)))
        out["reachable_points"] = int(round(rp))
        out["band_ceiling_percent"] = int(round(rp))
        top = _BAND_BY_KEY[_band_key(rp)]
        out["band_ceiling"] = top[1]
        out["band_ceiling_label"] = top[2]
        if top[1] != "green":
            out["band_ceiling_reason"] = (
                f"At most {rp:.0f}/100 is attainable because unevaluated "
                "criteria earn nothing; the highest reachable band is "
                f"\u201c{top[2]}\u201d. A lower score here is missing "
                "evidence, not evidence against chirality.")
    return out


def evaluate_chiral_metrology(state: Dict[str, Any]) -> Dict[str, Any]:
    state = _as_dict(state)
    aniso = _as_dict(state.get("didv_anisotropy"))
    hand = _as_dict(state.get("didv_handedness"))
    phase = _as_dict(state.get("didv_phase_chirality"))

    channels = chirality_channels(aniso, hand, phase)

    chirality = _criterion_handedness(hand, aniso, channels)
    shared = [
        _criterion_alpha(aniso),
        chirality,
    ]
    evidence = str(chirality.get("evidence_class") or "amplitude_only")

    criteria = [*shared, _criterion_judge(state)]
    total = round(sum(float(c["points"]) for c in criteria), 4)

    n_sat = sum(1 for c in criteria if c["status"] == "satisfied")
    n_partial = sum(1 for c in criteria if c["status"] == "partial")
    n_unavail = sum(1 for c in criteria if not c["available"])
    split = _split_scores(criteria)
    out: Dict[str, Any] = score_band(
        total, reachable_points=_reachable_points(criteria))
    out.update(split)
    out.update({
        "score": out["score_percent"],
        "points_max": sum(int(c["weight"]) for c in criteria),
        "criteria": criteria,
        "chirality_channels": channels,
        "chirality_evidence_class": evidence,
        "n_criteria": len(criteria),
        "n_satisfied": n_sat,
        "n_partial": n_partial,
        "n_unavailable": n_unavail,
        "weights": dict(WEIGHTS),
        "anisotropy_delta_min": ANISOTROPY_DELTA_MIN,
        "headline": f"{out['score_percent']}% \u2014 {out['band_label']}",
        "summary": (f"{_fmt_points(total)}/100 points \u2014 {n_sat} satisfied"
                    + (f", {n_partial} partial" if n_partial else "")
                    + (f", {n_unavail} could not be evaluated"
                       if n_unavail else "")
                    + f" of {len(criteria)} weighted criteria"),
    })
    if n_unavail:
        out["caveat"] = (
            f"{n_unavail} of {len(criteria)} criteria could not be evaluated "
            "and earn 0; on the bar, missing evidence and contrary evidence "
            "look the same.")
    else:
        out["caveat"] = (
            "Weights: alpha 40 (artifact 10 + anisotropy 30, two thresholds "
            "on one \u03b1 measurement), judge 40, handedness 20. The total "
            "is an ordinal score, not a probability; only the handedness "
            "criterion addresses chirality.")

    out["weights_provenance"] = (
        "Weights (alpha 40, judge 40, handedness 20) and band edges (50, 75) "
        "are an operating point chosen on the development corpus, not "
        "derived from a null distribution. The judge criterion is an LLM "
        "reading of the specialist reports; 'deterministic_score_percent' "
        "covers the alpha and handedness criteria only, renormalized over "
        "those that could be evaluated.")
    if out.get("judge_carries_more_than_measurements"):
        out["caveat"] = out["caveat"] + (
            f" The judge supplies {out.get('judge_points')} of {total} "
            f"points against {out.get('deterministic_points')} from the "
            "deterministic criteria; see 'deterministic_score_percent' "
            f"({out.get('deterministic_score_percent')}"
            + ("%)." if out.get("deterministic_score_percent") is not None
               else ")."))

    out["chirality_note"] = {
        "robust": "The chirality claim rests on a channel that tip response "
                  "and affine drift cannot alter.",
        "amplitude_only": "The chirality claim rests on the intensity "
                          "hierarchy alone; no stronger channel was "
                          "testable on this map.",
        "amplitude_only_uncorroborated":
            "The chirality claim rests on the intensity hierarchy, and the "
            "phase registry was tested and did not corroborate it.",
        "amplitude_screened_out":
            "The only chirality evidence was the ordering of three "
            "intensities whose spread the calibrated screen found "
            "consistent with artifacts, so no channel supports the claim.",
        "contradicted": "A stronger channel contradicts the chirality "
                        "claim.",
        "amplitude_failed": "The intensity hierarchy did not survive its "
                            "own checks and no stronger channel spoke.",
        "untested": "No chirality channel reached a determination.",
    }.get(evidence, "")
    return out


_ARIAL_PREFERENCE = ("Arial", "Liberation Sans", "Arimo", "Helvetica",
                     "Nimbus Sans", "DejaVu Sans")


def _arial_family() -> str:
    try:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in _ARIAL_PREFERENCE:
            if name in installed:
                return name
    except Exception:
        pass
    return "sans-serif"


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _draw_meter(ax, metro: Dict[str, Any], show_title: bool) -> None:
    from matplotlib.patches import Rectangle

    score = max(0.0, min(100.0, float(metro.get("score_percent", 0))))
    score_txt = _fmt_points(score)
    color = str(metro.get("color", _UNFILLED))
    text_color = str(metro.get("text_color", _INK))
    fam = _arial_family()

    top = 1.25 if show_title else 1.0
    ax.set_xlim(-14, 114)
    ax.set_ylim(0, top)
    ax.axis("off")
    ax.set_facecolor(_BG)

    if show_title:
        ax.text(0, top - 0.01, "Chiral CDW Metrology", ha="left", va="top",
                fontsize=17.0, fontweight="bold", color=_INK, family=fam)

    y0, y1 = 0.36, 0.64
    ax.add_patch(Rectangle((0, y0), 100, y1 - y0, facecolor=_UNFILLED,
                           edgecolor="none", zorder=2))

    if score > 0:
        ax.add_patch(Rectangle((0, y0), float(score), y1 - y0,
                               facecolor=color, edgecolor="none",
                               zorder=2.5))

    ax.add_patch(Rectangle((0, y0), 100, y1 - y0, facecolor="none",
                           edgecolor=_EDGE, linewidth=1.1, zorder=3))

    ax.plot([score], [y1], marker="v", markersize=12,
            markerfacecolor=color, markeredgecolor=_INK,
            markeredgewidth=1.0, zorder=4, clip_on=False)

    label = str(metro.get("band_label", ""))
    ax.annotate(
        label, xy=(score, y1), xytext=(score, 0.90),
        ha="center", va="center", fontsize=13.0, fontweight="bold",
        color=text_color, family=fam, annotation_clip=False,
        bbox=dict(boxstyle="square,pad=0.45", facecolor=_BG,
                  edgecolor=text_color, linewidth=1.5),
        arrowprops=dict(arrowstyle="-", color=text_color, linewidth=2.6,
                        shrinkA=5, shrinkB=3))

    ax.text(score, y0 - 0.05, f"{score_txt}%", ha="center", va="top",
            fontsize=13.5, fontweight="bold", color=text_color, family=fam,
            clip_on=False)


def _draw_criteria(ax, metro: Dict[str, Any]) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(_BG)
    ax.plot([0, 100], [0.97, 0.97], color=_RULE, linewidth=0.9)

    fam = _arial_family()
    glyph = {"satisfied": ("\u2713", "#7FD79A"),
             "partial": ("\u00bd", "#E9D68F"),
             "not_satisfied": ("\u2717", "#F0A5A5"),
             "unavailable": ("\u2013", _MUTED)}
    for i, c in enumerate(metro.get("criteria") or []):
        y = 0.90 - i * 0.30
        mark, mark_col = glyph.get(c.get("status", "unavailable"),
                                   ("\u2013", _MUTED))
        ax.text(0.5, y, f"{i + 1}.", ha="left", va="top", fontsize=11.5,
                color=_MUTED, family=fam)
        ax.text(4.5, y, mark, ha="left", va="top", fontsize=13.5,
                fontweight="bold", color=mark_col)
        ax.text(9.0, y, str(c.get("title", "")), ha="left", va="top",
                fontsize=12.0, fontweight="bold", color=_INK, family=fam)
        pts_txt = (f"{c.get('points_display') or _fmt_points(c.get('points', 0))}"
                   f"/{c.get('weight', 0)}%")
        head = (pts_txt if c.get("subcriteria")
                else f"{c.get('value', '')}  \u00b7  {pts_txt}")
        ax.text(46.0, y, head, ha="left", va="top", fontsize=11.5,
                color=mark_col, family=fam)
        tag = {
            "robust": ("tip-/drift-immune channel", "#7FD79A"),
            "amplitude_only": ("intensity hierarchy only", "#E9D68F"),
            "amplitude_only_uncorroborated": ("phase channel says no",
                                              "#F0A5A5"),
            "contradicted": ("contradicted by phase channel", "#F0A5A5"),
            "amplitude_screened_out": ("spread is artifact-consistent",
                                       "#F0A5A5"),
            "amplitude_failed": ("no channel established it", _MUTED),
            "untested": ("no channel tested", _MUTED),
            "established_unsigned": ("mirror broken, enantiomorph unresolved",
                                     "#E9D68F"),
            "established_sense_conflict": ("robust channels disagree on sense",
                                           "#E9D68F"),
        }.get(str(c.get("evidence_class") or ""))
        if tag:
            ax.text(99.5, y - 0.066 - 2 * 0.054, tag[0], ha="right",
                    va="top", fontsize=9.4, color=tag[1], family=fam)
        subs = c.get("subcriteria") or []
        if subs:
            for j, sc in enumerate(subs[:2]):
                s_mark, s_col = glyph.get(sc.get("status", "unavailable"),
                                          ("\u2013", _MUTED))
                yy = y - 0.066 - j * 0.058
                ax.text(11.0, yy, s_mark, ha="left", va="top", fontsize=10.5,
                        fontweight="bold", color=s_col)
                ax.text(15.0, yy, str(sc.get("title", "")), ha="left",
                        va="top", fontsize=10.0, color=_INK, family=fam)
                ax.text(45.0, yy,
                        f"{sc.get('value', '')}  \u00b7  "
                        f"{sc.get('points_display') or 0}/"
                        f"{sc.get('weight', 0)}%",
                        ha="left", va="top", fontsize=9.8, color=s_col,
                        family=fam)
            ax.text(15.0, y - 0.066 - 2 * 0.058,
                    "both are thresholds on the same \u03b1 measurement",
                    ha="left", va="top", fontsize=9.0, color=_MUTED,
                    family=fam)
        else:
            for j, line in enumerate(_wrap(str(c.get("detail", "")), 78)[:2]):
                ax.text(9.0, y - 0.066 - j * 0.054, line, ha="left",
                        va="top", fontsize=9.6, color=_MUTED, family=fam)


def render_metrology_figure(metro: Dict[str, Any], fmt: str = "png",
                            dpi: int = 300,
                            variant: str = "meter") -> Optional[bytes]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    try:
        section = variant == "section"
        meter_h = 1.9
        fig = plt.figure(figsize=(8.6, (4.1 + meter_h) if section
                                 else meter_h))
        fig.patch.set_facecolor(_BG)

        if section:
            gs = fig.add_gridspec(2, 1, height_ratios=[meter_h, 4.1],
                                  left=0.10, right=0.955, top=0.94,
                                  bottom=0.04, hspace=0.08)
            _draw_meter(fig.add_subplot(gs[0]), metro, show_title=True)
            _draw_criteria(fig.add_subplot(gs[1]), metro)
        else:
            ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
            _draw_meter(ax, metro, show_title=False)

        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, facecolor=_BG,
                    bbox_inches="tight", pad_inches=0.10)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        return None


def format_metrology_report(metro: Dict[str, Any]) -> str:
    L = ["=" * 72,
         "CHIRAL CDW METROLOGY  (three-criterion scorecard)",
         "=" * 72,
         f"Score   : {metro.get('score_percent')}%  "
         f"({metro.get('band', '?')})  {metro.get('band_label', '')}",
         "          measurements only : "
         + (f"{metro['deterministic_score_percent']}%  "
            f"({metro.get('deterministic_points')}/"
            f"{metro.get('deterministic_weight_evaluable')} of the "
            "evaluable alpha + chirality weight)"
            if metro.get("deterministic_score_percent") is not None
            else "no deterministic criterion could be evaluated")
         + (f"   |  judge agent: {metro['judge_score_percent']}%"
            if metro.get("judge_score_percent") is not None else ""),
         f"Criteria: {metro.get('summary', '')}"]

    _reach = metro.get("reachable_points")
    if _reach is not None and int(_reach) < 100:
        _ceil = metro.get("band_ceiling_label") or metro.get("band_ceiling")
        L.append(
            f"Ceiling : {int(_reach)}/100 attainable on this map "
            f"(highest reachable band: {_ceil}). "
            f"{100 - int(_reach)} points of weight belong to criteria that "
            "could not be evaluated, so the score is bounded from above "
            "before any evidence is weighed.")
    L.append("-" * 72)
    for i, c in enumerate(metro.get("criteria") or [], start=1):
        mark = {"satisfied": "[x]", "partial": "[/]", "not_satisfied": "[ ]",
                "unavailable": "[-]"}.get(c.get("status"), "[?]")
        L.append(f"{mark} {i}. {c.get('title')}: {c.get('value')} "
                 f"({c.get('points_display') or _fmt_points(c.get('points', 0))}"
                 f"/{c.get('weight', 0)}%)")
        for sc in (c.get("subcriteria") or []):
            s_mark = {"satisfied": "[x]", "partial": "[/]",
                      "not_satisfied": "[ ]",
                      "unavailable": "[-]"}.get(sc.get("status"), "[?]")
            L.append(f"      {s_mark} {sc.get('title')}: {sc.get('value')} "
                     f"({sc.get('points_display')}/{sc.get('weight')}%)")
            L.append(f"          {sc.get('detail')}")
        if not c.get("subcriteria"):
            L.append(f"      {c.get('detail')}")

    ch = metro.get("chirality_channels") or {}
    if ch:
        L.append("-" * 72)
        L.append("Chirality channels (strongest first; the chirality "
                 "criterion reads them in this order)")
        st = _as_dict(ch.get("structural"))
        ph = _as_dict(ch.get("phase"))
        am = _as_dict(ch.get("amplitude"))
        off = st.get("offset_deg")
        L.append("  S structural enantiomorph : "
                 + (str(st.get("sense")) if st.get("resolved")
                    else "not resolved")
                 + (f"   (CDW star {off:+.2f} deg from the lattice)"
                    if off is not None else ""))
        L.append("  P phase registry         : "
                 + str(ph.get("status"))
                 + (f"   (verdict {ph.get('verdict')}"
                    + (f", p = {ph['p_value']:.3g}"
                       if ph.get("p_value") is not None else "") + ")"
                    if ph.get("verdict") else ""))
        L.append("  A amplitude hierarchy    : "
                 + str(am.get("sense") or "none")
                 + (f"   (stable in {am['stability']:.0%} of redraws)"
                    if am.get("stability") is not None else ""))
        L.append("  claim rests on           : "
                 + str(metro.get("chirality_evidence_class", "?"))
                 + " \u2014 " + str(metro.get("chirality_note", "")))
    L.append("-" * 72)
    L.append(f"Note    : {metro.get('caveat', '')}")
    L.append("=" * 72)
    return "\n".join(L)


def chiral_metrology_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    import base64

    try:
        metro = evaluate_chiral_metrology(state)
        metro["report"] = format_metrology_report(metro)
        state["chiral_metrology"] = metro

        meter = render_metrology_figure(metro, fmt="png", dpi=300,
                                        variant="meter")
        if meter:
            state["chiral_metrology_figure_base64"] = \
                base64.b64encode(meter).decode("utf-8")
        section = render_metrology_figure(metro, fmt="png", dpi=300,
                                          variant="section")
        if section:
            state["chiral_metrology_section_base64"] = \
                base64.b64encode(section).decode("utf-8")
    except Exception as exc:                     # pragma: no cover
        state["chiral_metrology"] = {
            **score_band(0),
            "n_satisfied": 0, "n_partial": 0, "n_criteria": 3,
            "n_unavailable": 3, "criteria": [], "weights": dict(WEIGHTS),
            "headline": "Scorecard unavailable",
            "summary": f"metrology scoring failed: {exc}",
            "caveat": "The scorecard could not be built for this run.",
        }
    return state
