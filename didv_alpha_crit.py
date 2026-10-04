from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

TARGET_FPR = 0.05

OPERATING_POINT_RULE = (
    "smallest calibrated threshold whose measured false-positive rate has "
    "a Wilson 95% upper bound at or below TARGET_FPR")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    n = int(n)
    p = float(k) / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = (z / d) * float(np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)))
    return (max(0.0, centre - half), min(1.0, centre + half))


OPERATING_POINT_CANDIDATES = (
    {"alpha_crit": 0.204, "n_null": 699, "n_false_positive": 35,
     "power_at_0.5": 1.000, "origin": "tau = 0.95 quantile refit on the "
                                      "deployed-path null (attains the "
                                      "target BY CONSTRUCTION)"},
    {"alpha_crit": 0.310, "n_null": 699, "n_false_positive": 9,
     "power_at_0.5": 0.988, "origin": "v2 width-factorial refit, revalidated "
                                      "end to end on the deployed path"},
    {"alpha_crit": 0.374, "n_null": 699, "n_false_positive": 1,
     "power_at_0.5": 0.973, "origin": "v1 published threshold, remeasured on "
                                      "the deployed-path null"},
)


def select_operating_point(candidates=OPERATING_POINT_CANDIDATES,
                           target_fpr: float = TARGET_FPR) -> Dict[str, Any]:
    ladder: List[Dict[str, Any]] = []
    for c in sorted(candidates, key=lambda d: float(d["alpha_crit"])):
        lo, hi = wilson_interval(int(c["n_false_positive"]), int(c["n_null"]))
        row = dict(c)
        row.update(fpr=float(c["n_false_positive"]) / float(c["n_null"]),
                   fpr_ci95=(round(lo, 5), round(hi, 5)),
                   upper_bound_at_or_below_target=bool(hi <= target_fpr))
        ladder.append(row)

    passing = [r for r in ladder if r["upper_bound_at_or_below_target"]]

    chosen = passing[0]
    rejected = [r for r in ladder
                if not r["upper_bound_at_or_below_target"]]
    out = dict(chosen)
    out.update(
        selected=True, ladder=ladder, target_fpr=target_fpr,
        rule=OPERATING_POINT_RULE,
        power_cost_vs_smallest=round(
            float(ladder[0]["power_at_0.5"]) - float(chosen["power_at_0.5"]),
            4),
        reason=(
            f"alpha_crit = {chosen['alpha_crit']:.3f} is the smallest "
            f"candidate whose measured false-positive rate "
            f"({chosen['n_false_positive']}/{chosen['n_null']} = "
            f"{100 * chosen['fpr']:.2f}%) has a Wilson 95% upper bound "
            f"({100 * chosen['fpr_ci95'][1]:.2f}%) at or below the "
            f"{100 * target_fpr:.0f}% target."
            + (" Rejected below it: "
               + "; ".join(
                   f"{r['alpha_crit']:.3f} (upper bound "
                   f"{100 * r['fpr_ci95'][1]:.2f}%)" for r in rejected)
               + "." if rejected else "")))
    return out


OPERATING_POINT = select_operating_point()


CALIBRATION_V1 = {
    "name": "v1 (published)",
    "estimator_version": 1,
    "alpha_crit": 0.374,
    "snr_floor": 21.0,
    "width_floor": 2.0,
    "width_field": "width_bins_cdw",
    "width_ceiling": None,
    "aperture": "auto",
    "tau": 0.95,
    "n_null": 4700,
    "validated_fpr": 0.050,
    "validated_power_at_0.5": 0.879,
    "source": "B2 pooled: width factorial + noise layer + defects/steps "
              "(27,088 runs)",
    "layers": ["white / isotropic 1-f / scan-correlated 1-f noise",
               "finite correlation length", "elliptical tip",
               "drift, creep, aspect error",
               "point defects", "step edges"],
    "known_defect": "The width floor gates on the legacy width, which "
                    "inflates as SNR falls; 64% of imposed-width-1.5 maps "
                    "pass a floor meant to refuse them.",
    "not_modelled": "Whatever causes 69% of real published maps to fail "
                    "measurement entirely; the realism gate found a 65-point "
                    "gap between real and synthetic failure rates.",
}

CALIBRATION_V2 = {
    "name": "v2 (provisional)",
    "estimator_version": 2,
    "alpha_crit": 0.310,
    "snr_floor": 21.0,
    "width_floor": 1.2,
    "width_field": "width_bins_cdw_corrected",
    "width_ceiling": None,
    "aperture": "auto",
    "tau": 0.95,
    "n_null": 6549,
    "nominal_fpr": 0.050,
    "nominal_power_at_0.5": 0.891,
    "source": "B2 width factorial refit (21,600 runs, domain cut applied, "
              "aperture >= 5)",
    "layers": ["white noise", "finite correlation length", "elliptical tip"],
    "provisional": True,
    "not_modelled": "Drift, creep, aspect error, point defects, step edges "
                    "and 1/f noise - all present in the v1 pooled null and "
                    "absent here. Also the (anchor x model x rotation) "
                    "search, which no B2 null has ever included, so the "
                    "nominal 5% is understated for auto-anchored maps. "
                    "Also the deployed adaptive-aperture procedure itself: "
                    "this refit pooled FIXED apertures >= 5 bins, while "
                    "deployment sizes the aperture iteratively from the "
                    "measured width. And whatever causes 69% of real "
                    "published maps to fail measurement entirely.",
}

CALIBRATION_V3 = dict(CALIBRATION_V2)
CALIBRATION_V3.update({
    "name": "v3 (deployed-path validated)",
    "n_null": int(OPERATING_POINT["n_null"]),
    "alpha_crit": float(OPERATING_POINT["alpha_crit"]),
    "n_false_positive": int(OPERATING_POINT["n_false_positive"]),
    "validated_fpr": float(OPERATING_POINT["fpr"]),
    "validated_fpr_ci95": tuple(OPERATING_POINT["fpr_ci95"]),
    "validated_power_at_0.5": float(OPERATING_POINT["power_at_0.5"]),
    "target_fpr": TARGET_FPR,
    "operating_point_rule": OPERATING_POINT_RULE,
    "operating_point_reason": OPERATING_POINT["reason"],
    "operating_point_ladder": OPERATING_POINT["ladder"],
    "fitted_alpha_crit_at_tau": 0.204,
    "fitted_tau": 0.95,
    "fitted_alpha_crit_rejected_because": (
        "It attains the 5% target BY CONSTRUCTION on this null "
        "(35/699 = 5.01%), so its Wilson 95% interval [3.62%, 6.88%] "
        "straddles the target. A rule advertised at 5% cannot be operated "
        "at a threshold whose own validation is consistent with 6.9%."),
    "power_cost_vs_fitted": OPERATING_POINT.get("power_cost_vs_smallest"),
    "source": ("Deployed-path null: 1300 achiral realizations, auto "
               "aperture, (anchor x model x rotation) search enabled, "
               "domain cut applied (920 measured, 699 in domain)"),
    "layers": ["white noise", "isotropic 1-f", "scan-correlated 1-f",
               "finite correlation length", "elliptical tip",
               "drift", "creep", "aspect error",
               "point defects", "step edges",
               "adaptive aperture (deployed procedure)",
               "(anchor x model x rotation) search"],
    "provisional": False,
    "not_modelled": ("Whatever causes 69% of real published maps to fail "
                     "measurement entirely: this null failed 29%, so it is "
                     "cleaner than reality in the dimension the realism "
                     "gate identified, and its FPR is a floor on the real "
                     "one rather than an estimate of it. The artifact "
                     "severity ranges are the harness's choice and are "
                     "reported with it."),
    "validation_harness": "Re-calibration/b2_deployed_null.py",
})
CALIBRATION_V3.pop("nominal_fpr", None)
CALIBRATION_V3.pop("nominal_power_at_0.5", None)

ACTIVE_CALIBRATION = CALIBRATION_V3

CALIBRATION = ACTIVE_CALIBRATION

ALPHA_CRIT_FITTED = ACTIVE_CALIBRATION.get("fitted_alpha_crit_at_tau")

INTEGRATED_SNR_FLOOR = 3.0


def _mean3(values: Optional[List[Any]]) -> Optional[float]:
    vals = [float(v) for v in (values or []) if v is not None]
    return float(np.mean(vals)) if len(vals) == 3 else None


def _v1_reference_verdict(res: Dict[str, Any],
                          min_snr: Optional[float]) -> Dict[str, Any]:
    v1 = CALIBRATION_V1
    a = res.get("alpha_cdw_legacy")
    w = _mean3(res.get(v1["width_field"]))

    out: Dict[str, Any] = {
        "name": v1["name"],
        "alpha_crit": float(v1["alpha_crit"]),
        "validated_fpr": v1["validated_fpr"],
        "alpha": (float(a) if isinstance(a, (int, float))
                  and a == a else None),
        "width_bins": w,
        "min_snr": min_snr,
        "notes": [],
    }

    if out["alpha"] is None:
        out.update(verdict="no_measurement",
                   reason="The legacy (v1) alpha is unavailable for this "
                          "measurement.")
        return out
    if min_snr is None or w is None:
        out.update(verdict="no_measurement",
                   reason="SNR or legacy peak width unavailable, so the v1 "
                          "domain cut cannot be evaluated.")
        return out

    reasons = []
    if min_snr < float(v1["snr_floor"]):
        reasons.append(f"min SNR {min_snr:.1f} below {v1['snr_floor']:g}")
    if w < float(v1["width_floor"]):
        reasons.append(f"legacy width {w:.2f} bins below "
                       f"{v1['width_floor']:g}")
    if reasons:
        out.update(verdict="refused", reason="; ".join(reasons))
        return out

    above = out["alpha"] > float(v1["alpha_crit"])
    out.update(verdict="above" if above else "below",
               reason=(f"alpha_legacy = {out['alpha']:.3f} vs "
                       f"alpha_crit = {v1['alpha_crit']:.3f} "
                       "(v1, validated 5% FPR on the pooled null)."))
    if res.get("legacy_clamp_applied"):
        out["notes"].append(
            "The v1 clamp fired on this map (a CDW aperture integrated to "
            "<= 0), so the legacy alpha is an artifact of the clamp; the "
            "v1 verdict is reported for the record but should not be "
            "trusted here.")
    if bool(v1.get("known_defect")):
        out["notes"].append(
            "The v1 domain cut gates on the legacy width, which is known "
            "to admit sub-floor maps; the operational rule's domain cut is "
            "the corrected one.")
    return out


def screen_result(res: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"calibration": CALIBRATION}

    if not res or res.get("status") != "ok":
        out.update(verdict="no_measurement",
                   headline="No measurement",
                   reason=(res or {}).get(
                       "message", "B1 could not measure the peaks."))
        return out

    _dc = str(res.get("data_class") or "unknown")
    out["data_class"] = _dc
    out["calibration_domain_matched"] = (_dc == "native")

    version = int(res.get("estimator_version", 1) or 1)
    cal = ACTIVE_CALIBRATION
    out["calibration"] = cal
    ap_mode = (res.get("method") or {}).get("aperture_mode")

    if res.get("alpha_measurable") is False:
        out.update(verdict="no_measurement", headline="No measurement",
                   estimator_version=version,
                   alpha_cdw_lower_bound=res.get("alpha_cdw_lower_bound"),
                   reason=("At least one CDW aperture integrates to <= 0 "
                           "against an unbiased background, so the peak is at "
                           "or below the noise floor and alpha is undefined. "
                           "A missing peak is a failure to measure alpha, not "
                           "a measurement of alpha ~ 1."))
        return out

    snrs = [p["snr"] for p in res.get("cdw_peaks", [])
            if p.get("snr") is not None]
    min_snr = min(snrs) if len(snrs) == 3 else None

    alpha_crit = float(cal["alpha_crit"])
    snr_floor = float(cal["snr_floor"])
    width_floor = float(cal["width_floor"])

    width_field = cal.get("width_field", "width_bins_cdw")
    width = _mean3(res.get(width_field))

    if width is None and width_field != "width_bins_cdw":
        w_legacy = _mean3(res.get("width_bins_cdw"))
        out.update(verdict="refused",
                   headline="Outside the analysable domain",
                   estimator_version=version, min_snr=min_snr,
                   width_bins=None, width_source=width_field,
                   reason=("The bias-corrected peak width could not be "
                           "measured: the flux enclosed by the moment disc "
                           "is not significantly positive, which means the "
                           "peak is not cleanly separable from the local "
                           "background."
                           + (f"  The legacy width of {w_legacy:.2f} bins is "
                              "not a substitute - it inflates as a peak "
                              "weakens, which is why it is not the gating "
                              "statistic." if w_legacy is not None else "")))
        return out

    alpha_field = "alpha_cdw"
    alpha = res.get(alpha_field)
    alpha_corrected = res.get("alpha_cdw")
    alpha_source = alpha_field

    unc = res.get("alpha_uncertainty") or {}
    out.update(aperture_mode=(str(ap_mode) if ap_mode is not None else None),
               aperture_overlap=bool(res.get("aperture_overlap")),
               alpha_cdw=alpha,
               alpha_cdw_corrected=alpha_corrected,
               alpha_cdw_lower_bound=res.get("alpha_cdw_lower_bound"),
               alpha_source=alpha_source,
               estimator_version=version,
               calibration_name=cal.get("name"),
               calibration_provisional=bool(cal.get("provisional")),
               calibration_stale=False,
               alpha_crit=alpha_crit,
               width_source=width_field,
               alpha_ci68=unc.get("alpha_ci68"),
               alpha_ci95=unc.get("alpha_ci95"),
               p_value_equal_intensity=unc.get("p_value_equal_intensity"),
               alpha_bias_equal_null=unc.get("alpha_bias_equal_null"),
               alpha_bragg=res.get("alpha_bragg"),
               min_snr=min_snr, width_bins=width,
               n_resolved=res.get("n_directions_resolved"),
               hierarchy_sense=res.get("hierarchy_sense_cdw"))

    if alpha is None or (isinstance(alpha, float) and alpha != alpha):
        out.update(verdict="no_measurement", headline="No measurement",
                   reason=("At least one CDW aperture integrates to <= 0 "
                           "after background subtraction, so alpha is "
                           "undefined for this map. A missing peak is a "
                           "failure to measure alpha, not a measurement of "
                           "alpha ~ 1."))
        return out

    if min_snr is None or width is None:
        out.update(verdict="no_measurement", headline="No measurement",
                   reason="SNR or peak width unavailable.")
        return out

    reasons = []
    if min_snr < snr_floor:
        reasons.append(f"min SNR {min_snr:.1f} is below {snr_floor:g}")
    if width < width_floor:
        reasons.append(f"peak width {width:.2f} bins is below "
                       f"{width_floor:g} (too few independent domains)")

    int_snrs = [p["snr_integrated"] for p in res.get("cdw_peaks", [])
                if p.get("snr_integrated") is not None]
    weak_int = [p["index"] for p in res.get("cdw_peaks", [])
                if p.get("snr_integrated") is not None
                and p["snr_integrated"] < INTEGRATED_SNR_FLOOR]
    out["integrated_snr_min"] = (round(float(min(int_snrs)), 3)
                                 if int_snrs else None)
    out["integrated_snr_floor"] = INTEGRATED_SNR_FLOOR
    out["weak_integrated_directions"] = list(weak_int)
    out["statistic_disagreement"] = bool(weak_int)
    out["integrated_snr_enforced"] = False

    if reasons:
        out.update(
            verdict="refused", headline="Outside the analysable domain",
            reason="; ".join(reasons) + ".  In this regime the artifact "
                   "null alone reaches alpha ~ 1, so the measured value "
                   "cannot be interpreted.")
        out["verdict_if_enforced"] = "refused"
        return out

    above = alpha > alpha_crit
    notes = []

    a_fitted = cal.get("fitted_alpha_crit_at_tau")
    if a_fitted is not None:
        a_fitted = float(a_fitted)
        fitted_above = alpha > a_fitted
        out["alpha_crit_fitted"] = a_fitted
        out["verdict_at_fitted_threshold"] = "above" if fitted_above else "below"
        out["fitted_threshold_sensitive"] = bool(fitted_above and not above)
        out["operating_point_rule"] = cal.get("operating_point_rule")
        out["operating_point_reason"] = cal.get("operating_point_reason")
        out["target_fpr"] = cal.get("target_fpr")
        if out["fitted_threshold_sensitive"]:
            notes.append(
                f"OPERATING-POINT SENSITIVE: alpha = {alpha:.3f} falls in "
                f"the ({a_fitted:.3f}, {alpha_crit:.3f}] window, so this map "
                "is 'below' only because the shipped threshold is the "
                "conservative one. The tau = 0.95 fit on the same null puts "
                f"alpha_crit at {a_fitted:.3f}, which would call it 'above'. "
                "The shipped value is not that fit: it is the smallest "
                "candidate whose measured false-positive rate has a Wilson "
                "95% upper bound at or below the "
                f"{100 * float(cal.get('target_fpr', TARGET_FPR)):.0f}% "
                "target, and the fitted value fails that bound (its own "
                "validation is consistent with 6.9%). This verdict is a "
                "deliberate false negative bought with that confidence, not "
                "a measurement that the anisotropy is artifact-consistent.")
    if res.get("aperture_overlap"):
        annulus_hit = bool(res.get("annulus_reaches_neighbour"))
        notes.append(
            "Adjacent apertures overlap on this map, sharing power between "
            "peaks.  Shared power pulls the three intensities toward each "
            "other and biases alpha LOW (by up to a factor of five on the "
            "reference corpus), so a 'below' outcome here is weak evidence "
            "of artifact-consistency and is not enforced by the pipeline's "
            "deterministic gate."
            + (" A neighbouring peak also lies inside the background "
               "annulus on this map, which over-estimates the background "
               "and biases alpha HIGH. The two mechanisms act in OPPOSITE "
               "directions, so the net bias is not determined and an "
               "'above' outcome CANNOT be assumed conservative either."
               if annulus_hit else
               " The background annulus is clear of neighbouring peaks on "
               "this map, so the shared-power (alpha-low) mechanism is the "
               "one in play here; note that this is a per-map finding, not "
               "a general property of overlap."))
    if cal.get("validated_fpr") is not None:
        ci = cal.get("validated_fpr_ci95")
        notes.append(
            f"alpha_crit = {alpha_crit:.3f} carries a MEASURED "
            f"false-positive rate of {100 * float(cal['validated_fpr']):.2f}%"
            + (f" (95% CI {100 * ci[0]:.2f}-{100 * ci[1]:.2f}%)"
               if ci else "")
            + f" on n = {cal.get('n_null')} in-domain achiral realizations"
            + (f", power {100 * float(cal['validated_power_at_0.5']):.1f}% at "
               "alpha_true = 0.5"
               if cal.get("validated_power_at_0.5") is not None else "")
            + ". This is the rate for the rule AS DEPLOYED, including the "
              "adaptive aperture and the (anchor x model x rotation) "
              "search. It is measured on a SYNTHETIC null whose maps fail "
              "measurement 29% of the time against 69% for real published "
              "maps, so treat it as a floor on the real rate rather than an "
              "estimate of it.")
        if cal.get("operating_point_reason"):
            notes.append(
                "Operating point: " + str(cal["operating_point_reason"])
                + " The threshold is therefore DERIVED from a stated rule ("
                + str(cal.get("operating_point_rule", "")) + ") rather than "
                "inherited, and it is deliberately more conservative than "
                "the quantile fit: quote "
                f"{100 * float(cal['validated_fpr']):.2f}% as this rule's "
                "false-positive rate, never the "
                f"{100 * float(cal.get('target_fpr', TARGET_FPR)):.0f}% "
                "target it is bounded by."
                + (f" The conservatism costs {100 * float(cal['power_cost_vs_fitted']):.1f} "
                   "points of power at alpha_true = 0.5; maps it turns into "
                   "false negatives are flagged individually by "
                   "`fitted_threshold_sensitive`."
                   if cal.get("power_cost_vs_fitted") else ""))
    if res.get("width_saturated_cdw") and any(res["width_saturated_cdw"]):
        notes.append(
            "At least one peak width is still at the moment estimator's "
            "ceiling after the disc was expanded, so the width is a lower "
            "bound and the aperture sized from it may be too small.")
    p_eq = out.get("p_value_equal_intensity")
    if p_eq is not None:
        notes.append(
            f"Against a same-map null in which the three intensities are "
            f"equal, p = {p_eq:.3g}. This accounts for photometric (speckle) "
            "noise only - no tip, drift or scan-artifact model - and on "
            "synthetic nulls it is mildly anti-conservative (14% of null "
            "maps give p < 0.05). Treat it as a floor on the evidence, not "
            "as a false-positive rate.")
    ci = out.get("alpha_ci68")
    if isinstance(ci, (list, tuple)) and len(ci) == 2 \
            and ci[0] <= alpha_crit <= ci[1]:
        notes.append(
            f"The 68% confidence interval on alpha [{ci[0]:.3f}, {ci[1]:.3f}] "
            f"straddles alpha_crit = {alpha_crit:.3f}, so this map does not "
            "separate the two hypotheses at 1 sigma whichever side the point "
            "estimate falls.")
    if weak_int:
        notes.append(
            "Direction(s) " + ", ".join(str(i) for i in weak_int) +
            " pass the single-pixel SNR floor but their INTEGRATED "
            f"aperture power is below {INTEGRATED_SNR_FLOOR:g} sigma. "
            "The domain cut and alpha are built from different "
            "statistics, and for this map they disagree: the peak is "
            "speckle-dominated. alpha is formed from the integrated "
            "power, so the domain cut did not screen the quantity being "
            "thresholded, and this verdict"
            + " WOULD BE REFUSED with the integrated-SNR floor "
              "enforced (see `verdict_if_enforced`).")
    if any(res.get("width_saturated_cdw") or []):
        notes.append(
            "At least one peak width is at the second-moment estimator's "
            "saturation ceiling (r_max/sqrt(2) = 5.66 bins), so the "
            "measured width is a lower bound and the width-dependent "
            "parts of the calibration do not apply.")
    if res.get("n_directions_resolved", 3) < 3:
        notes.append(
            f"Only {res.get('n_directions_resolved')}/3 CDW peaks resolved: "
            "this points to unidirectional (1Q / stripe) order, which is "
            "not chiral.")
    fam = res.get("bragg_family") or {}
    frac = fam.get("power_frac_of_strongest")
    if frac is not None:
        if frac < 0.05:
            notes.append(
                f"The Bragg family carries only {frac:.1%} of the strongest "
                "peak in the spectrum, so alpha_Bragg is measured on peaks "
                "close to the noise floor and should not be used as an "
                "artifact control for this map.")
        elif frac > 0.35:
            notes.append(
                f"The Bragg anchor carries {frac:.0%} of the strongest peak. "
                "In a dI/dV map the atomic Bragg peaks are usually weaker "
                "than the CDW peaks, so this shell may itself be the CDW and "
                "alpha_Bragg may not describe the lattice.")

    if res.get("warnings"):
        notes.append(f"{len(res['warnings'])} measurement warning(s) from B1.")

    op_verdict = "above" if above else "below"
    ref = _v1_reference_verdict(res, min_snr)
    ref_verdict = str(ref.get("verdict"))
    concordant = (ref_verdict == op_verdict
                  if ref_verdict in ("above", "below") else None)
    rule_sensitive = bool(op_verdict == "above" and ref_verdict == "below")
    out.update(reference_screen=ref, concordant=concordant,
               rule_sensitive=rule_sensitive)
    if rule_sensitive:
        notes.append(
            "RULE-SENSITIVE: the anisotropy exceeds the operational "
            f"threshold ({alpha_crit:.3f}, {cal['name']}) but not the "
            f"validated v1 reference ({ref['alpha_crit']:.3f} on the legacy "
            "alpha). The claim exists only between the two rules; treat it "
            "as weaker evidence than a concordant 'above'.")
    elif op_verdict == "below" and ref_verdict == "above":
        notes.append(
            "The v1 reference rule returns 'above' on the legacy alpha "
            "while the operational rule returns 'below'. The corrected "
            "estimator is the operational one; the discordance is recorded "
            "for the audit trail.")
    elif ref_verdict not in ("above", "below"):
        notes.append(
            f"The v1 reference rule reached no decision ({ref_verdict}): "
            f"{ref.get('reason', '')}")

    out.update(
        verdict="above" if above else "below",
        headline=("Exceeds the artifact threshold" if above
                  else "Consistent with artifacts"),
        reason=f"Recalibration: α_CDW = {alpha:.3f} vs α_crit = {alpha_crit:.3f}.",
        notes=notes,
        caveat=("Exceeding the threshold means the anisotropy is larger than "
                "modelled artifacts produce.  It is not by itself evidence "
                "of chirality: alpha is a scalar and cannot express "
                "handedness, which lies in the cyclic sense of the "
                "intensity hierarchy."))
    out["verdict_if_enforced"] = (
        "refused" if out.get("statistic_disagreement") else out["verdict"])
    return out
