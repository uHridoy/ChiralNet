from __future__ import annotations

import base64
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as _stats


AMPLITUDE_CHANNEL_LICENSES_HANDEDNESS = False

NEMATIC_NULL_DRAWS = 20000

NEMATIC_SENSE_NULL_BASELINE = 0.5

STABILITY_THRESHOLD = 0.60

N_DRAW = 20000

SENSE_NULL_BASELINE = 0.5

SENSE_RESOLUTION_KAPPA = 0.25

PLACEMENT_SYSTEMATIC_FLOOR = 0.02

DISTORTION_SYSTEMATIC_MAX = 0.60


REPRODUCIBILITY_ALPHA = 0.05

REPRODUCIBILITY_TILES_PER_SIDE: Tuple[int, ...] = (3, 2)

REPRODUCIBILITY_MIN_TILES = 4

REPRODUCIBILITY_MIN_TILE_PX = 64
REPRODUCIBILITY_MIN_TILE_PERIODS = 6.0

REPRODUCIBILITY_MIN_APERTURE_BINS = 3.0

DISPLAY_FRAME = "display (x right, y up)"

_CAVEAT = (
    "The sense is read in DISPLAY coordinates - x to the right, y UP, the "
    "map as it appears on screen with its first row at the top - so "
    "'clockwise' means clockwise to a reader looking at the panel. (The "
    "conversion happens once, in didv_anisotropy: the array's second "
    "frequency axis is conjugate to the row index and points DOWN, so an "
    "unconverted sense would be mirrored.) What the convention cannot fix "
    "is the provenance of the image itself: a source figure that was "
    "mirrored or flipped in reproduction still reverses the sense, and "
    "scan/display conventions differ between instruments, so comparing "
    "handedness across maps - or to a crystallographic axis - requires the "
    "raw scan orientation to be known. More fundamentally, the sense CANNOT "
    "establish chirality at all, in this or any frame: three intensities at "
    "60 degrees are exactly a mean plus a nematic director, a director is "
    "achiral, and the sense is a step function of its axis that is a fair "
    "coin under any achiral director. It is reported as a description of "
    "the anisotropy, never as a handedness. See section 0 of the module "
    "docstring, and didv_phase_chirality for a channel that can carry "
    "chirality.")


def _wrap90(x: np.ndarray) -> np.ndarray:
    return (x + 90.0) % 180.0 - 90.0


def _sense_of(angles_deg: np.ndarray, values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[None, :]
    order = np.argsort(-vals, axis=1)
    a = np.asarray(angles_deg, dtype=np.float64) % 180.0
    a_ord = a[order]
    s1 = _wrap90(a_ord[:, 1] - a_ord[:, 0])
    s2 = _wrap90(a_ord[:, 2] - a_ord[:, 1])
    out = np.zeros(vals.shape[0], dtype=np.int64)
    out[(s1 > 0) & (s2 > 0)] = +1
    out[(s1 < 0) & (s2 < 0)] = -1
    return out


_SENSE_NAME = {+1: "counter-clockwise", -1: "clockwise", 0: "non-monotonic"}
_SENSE_TO_INT = {v: k for k, v in _SENSE_NAME.items()}


def nematic_decomposition(angles_deg: Sequence[float],
                          intensities: Sequence[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False,
        "space": None, "I0": None, "A": None, "B": None,
        "nematic_magnitude": None, "nematic_contrast": None,
        "axis_deg": None, "residual_max": None, "residual_dof": 0,
        "note": (
            "Three intensities at 60 degrees carry exactly two degrees of "
            "freedom beyond their mean: the magnitude and axis of a nematic "
            "director. The inversion below is exact (zero residual, zero "
            "remaining degrees of freedom), so there is no third quantity "
            "in which a handedness could hide. A director is invariant "
            "under reflection about its own axis; the cyclic sense read "
            "from the intensity ordering is a step function of `axis_deg` "
            "alone and is independent of `nematic_magnitude`."),
    }
    try:
        th = np.radians(np.asarray(angles_deg, dtype=np.float64) % 180.0)
        I = np.asarray(intensities, dtype=np.float64)
        if th.size != 3 or I.size != 3 or not np.all(np.isfinite(I)):
            return out

        use_log = bool(np.all(I > 0))
        y = np.log(I) if use_log else I.copy()

        M = np.column_stack([np.ones(3), np.cos(2 * th), np.sin(2 * th)])
        if abs(float(np.linalg.det(M))) < 1e-12:
            out["note"] = ("The three directions are degenerate, so the "
                           "nematic inversion is singular.")
            return out
        c = np.linalg.solve(M, y)
        I0, A, B = (float(v) for v in c)

        mag = float(np.hypot(A, B))
        axis = float(np.degrees(0.5 * np.arctan2(B, A)) % 180.0)

        out.update(
            available=True,
            space="log" if use_log else "linear",
            I0=round(I0, 8), A=round(A, 8), B=round(B, 8),
            nematic_magnitude=round(mag, 8),
            nematic_contrast=(round(float(np.exp(2.0 * mag) - 1.0), 6)
                              if use_log else
                              (round(float(2.0 * mag / I0), 6)
                               if I0 > 0 else None)),
            axis_deg=round(axis, 4),
            residual_max=float(np.max(np.abs(M @ c - y))),
            residual_dof=0)
        return out
    except Exception:                                   # pragma: no cover
        return out


def nematic_null_test(angles_deg: Sequence[float],
                      observed_sense: int,
                      nematic_magnitude: Optional[float] = None,
                      n_draw: int = NEMATIC_NULL_DRAWS,
                      seed: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "null": "achiral 3Q state carrying a nematic director of unknown "
                "axis (uniaxial strain / elliptical tip / scan aspect "
                "error); axis uniform on [0, 180), magnitude held at the "
                "measured value",
        "available": False,
        "p_ccw": None, "p_cw": None, "p_non_monotonic": None,
        "p_value_observed_sense": None,
        "max_attainable_significance": NEMATIC_SENSE_NULL_BASELINE,
        "discriminating": False,
        "n_draw": int(n_draw),
        "note": (
            "The sense is a step function of the director's axis and is "
            "independent of its magnitude, so under an achiral director of "
            "unknown axis the two senses are equiprobable and the best "
            "p-value the statistic can attain is 0.5. It cannot support a "
            "chirality claim at any SNR, field of view or hierarchy "
            "strength."),
    }
    try:
        a = np.asarray(angles_deg, dtype=np.float64) % 180.0
        if a.size != 3:
            return out
        mag = float(nematic_magnitude) if nematic_magnitude is not None \
            else 0.25
        if not np.isfinite(mag) or mag <= 0:
            mag = 0.25

        rng = np.random.default_rng(seed)
        th0 = rng.uniform(0.0, 180.0, int(n_draw))
        th = np.radians(a)[None, :]
        draws = np.exp(mag * np.cos(2.0 * (th - np.radians(th0)[:, None])))
        senses = _sense_of(a, draws)

        p_ccw = float(np.mean(senses == +1))
        p_cw = float(np.mean(senses == -1))
        p_non = float(np.mean(senses == 0))
        obs = int(observed_sense)
        p_obs = (p_ccw if obs == +1 else p_cw if obs == -1 else p_non)

        out.update(available=True,
                   p_ccw=round(p_ccw, 4), p_cw=round(p_cw, 4),
                   p_non_monotonic=round(p_non, 4),
                   p_value_observed_sense=round(float(p_obs), 4),
                   discriminating=bool(p_obs < 0.05))
        return out
    except Exception:                                   # pragma: no cover
        return out


def _mean_q(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    vals = [_num(r.get("q_cycles_per_px")) for r in rows or []]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    return float(np.mean(vals)) if vals else None


def _cdw_uv(rows: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for r in rows or []:
        q = _num(r.get("q_cycles_per_px"))
        a = r.get("angle_deg_array")
        if a is None or not np.isfinite(q) or q <= 0:
            return []
        th = np.radians(float(a))
        out.append((float(q * np.cos(th)), float(q * np.sin(th))))
    return out if len(out) == 3 else []


def _aperture_cpp(res: Dict[str, Any]) -> float:
    method = res.get("method") or {}
    shape = (res.get("image") or {})
    nx, ny = _num(shape.get("Nx")), _num(shape.get("Ny"))
    bins = _num(method.get("aperture_radius_bins"))
    if not np.isfinite(bins) or bins <= 0:
        bins = 3.0
    if np.isfinite(nx) and np.isfinite(ny) and nx > 0 and ny > 0:
        return float(bins / np.sqrt(nx * ny))
    return float(bins / 512.0)


def _rows_ok(rows: Optional[Sequence[Dict[str, Any]]]) -> bool:
    return bool(rows) and len(rows) == 3 and all(
        r.get("angle_deg") is not None and r.get("intensity") is not None
        for r in rows)


def _tip_corrected_intensities(cdw_rows: Sequence[Dict[str, Any]],
                               bragg_rows: Sequence[Dict[str, Any]]
                               ) -> Optional[Dict[str, Any]]:
    C = np.array([r["intensity"] for r in cdw_rows], dtype=np.float64)
    B = np.array([r["intensity"] for r in bragg_rows], dtype=np.float64)
    if not (np.all(np.isfinite(C)) and np.all(np.isfinite(B))
            and np.all(C > 0) and np.all(B > 0)):
        return None
    qc = np.array([r.get("q_cycles_per_px") for r in cdw_rows],
                  dtype=np.float64)
    qb = np.array([r.get("q_cycles_per_px") for r in bragg_rows],
                  dtype=np.float64)
    if not (np.all(np.isfinite(qc)) and np.all(np.isfinite(qb))
            and qb.mean() > 0):
        return None
    s2 = float((qc.mean() / qb.mean()) ** 2)

    thb = np.radians(np.array([r["angle_deg"] for r in bragg_rows],
                              dtype=np.float64))
    thc = np.radians(np.array([r["angle_deg"] for r in cdw_rows],
                              dtype=np.float64))
    M = np.column_stack([np.ones(3), np.cos(2 * thb), np.sin(2 * thb)])
    try:
        coef = np.linalg.solve(M, np.log(B))
    except np.linalg.LinAlgError:
        return None
    _, A2, B2 = (float(c) for c in coef)
    tip_at_cdw = s2 * (A2 * np.cos(2 * thc) + B2 * np.sin(2 * thc))
    return {
        "intensities": np.exp(np.log(C) - tip_at_cdw),
        "q_ratio_sq": s2,
        "quadrupole_logpower": [float(A2), float(B2)],
        "tip_logspread_at_cdw": float(np.ptp(tip_at_cdw)),
    }


def _num(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _star_geometry(rows: Optional[Sequence[Dict[str, Any]]]
                   ) -> Optional[Dict[str, Any]]:
    if not rows or len(rows) != 3:
        return None
    try:
        a = np.array([float(r["angle_deg"]) for r in rows],
                     dtype=np.float64) % 180.0
        r_ = np.array([float(r["q_cycles_per_px"]) for r in rows],
                      dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(r_))
            and float(np.min(r_)) > 0):
        return None

    order = np.argsort(a)
    a, r_ = a[order], r_[order]
    r0 = float(np.mean(r_))

    gaps = np.diff(np.concatenate([a, [a[0] + 180.0]]))
    out: Dict[str, Any] = {
        "radial_spread": round(float((r_.max() - r_.min()) / r0), 5),
        "angle_deviation_deg": round(float(np.max(np.abs(gaps - 60.0))), 3),
        "mean_radius_cycles_per_px": round(r0, 6),
        "affine_anisotropy": None,
        "hex_residual_frac": None,
    }

    th = np.radians(a)
    q = np.stack([r_ * np.cos(th), r_ * np.sin(th)], axis=1)
    a0 = float(np.mean(a - np.array([0.0, 60.0, 120.0])))
    ideal = np.stack([
        r0 * np.array([np.cos(np.radians(a0 + 60.0 * k)),
                       np.sin(np.radians(a0 + 60.0 * k))])
        for k in range(3)])

    try:
        if abs(float(np.linalg.det(q[:2].T))) < 1e-30:
            return out
        A = ideal[:2].T @ np.linalg.inv(q[:2].T)
        sv = np.linalg.svd(A, compute_uv=False)
        if not np.all(np.isfinite(sv)) or float(np.min(sv)) <= 0:
            return out
        out["affine_anisotropy"] = round(float(np.max(sv) / np.min(sv)), 5)
        out["hex_residual_frac"] = round(
            float(np.max(np.hypot(*((q @ A.T) - ideal).T)) / r0), 5)
    except np.linalg.LinAlgError:
        return out
    return out


def _distortion_systematic(geom: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    parts: List[float] = [PLACEMENT_SYSTEMATIC_FLOOR]
    drivers: List[str] = []
    if geom:
        k = geom.get("affine_anisotropy")
        if k is not None and np.isfinite(k):
            parts.append(abs(float(k) - 1.0))
            drivers.append(f"affine anisotropy {float(k):.3f}")
        resid = geom.get("hex_residual_frac")
        if resid is not None and np.isfinite(resid):
            parts.append(float(resid))
            drivers.append(f"hexagonal residual {float(resid):.1%}")
        spread = geom.get("radial_spread")
        if spread is not None and np.isfinite(spread):
            parts.append(float(spread))
            drivers.append(f"radial spread {float(spread):.1%}")

    raw = max(parts)
    eps = float(min(raw, DISTORTION_SYSTEMATIC_MAX))
    return {
        "systematic_frac": round(eps, 5),
        "floor": PLACEMENT_SYSTEMATIC_FLOOR,
        "at_floor": bool(eps <= PLACEMENT_SYSTEMATIC_FLOOR + 1e-12),
        "capped": bool(raw > DISTORTION_SYSTEMATIC_MAX),
        "drivers": drivers,
        "censored_by_detection": True,
        "censoring_note": (
            "eps is bounded above by the family-search tolerances that had "
            "to be satisfied for this star to be detected at all "
            "(radius_rtol = 0.12, angle_tol = 14 deg in didv_anisotropy), "
            "so it is a lower bound on the distortion present in the map "
            "population rather than an estimate of it. Maps distorted "
            "beyond those tolerances are refused upstream and are absent "
            "from every eps ever measured. Do not read a small eps as "
            "evidence that the scan grid is clean."),
        "detector_tolerance_radius_rtol": 0.12,
        "detector_tolerance_angle_deg": 14.0,
    }


def _student_sf(t: float, df: int) -> float:
    t = abs(float(t))
    if not np.isfinite(t) or df < 1:
        return 1.0
    return float(_stats.t.sf(t, df=df))


def _tile_bounds(shape: Sequence[int], q_cycles_per_px: Optional[float],
                 k: int) -> Optional[List[Tuple[int, int, int, int]]]:
    ny, nx = int(shape[0]), int(shape[1])
    hy, hx = ny // k, nx // k
    if min(hy, hx) < REPRODUCIBILITY_MIN_TILE_PX:
        return None
    if q_cycles_per_px and q_cycles_per_px > 0:
        if min(hy, hx) * float(q_cycles_per_px) \
                < REPRODUCIBILITY_MIN_TILE_PERIODS:
            return None
    return [(a * hy, (a + 1) * hy, b * hx, (b + 1) * hx)
            for a in range(k) for b in range(k)]


def reproducibility_control(field: Optional[np.ndarray],
                            uv: Sequence[Sequence[float]],
                            r_aperture_cpp: float,
                            q_cycles_per_px: Optional[float],
                            angles_deg: Sequence[float],
                            alpha: float = REPRODUCIBILITY_ALPHA
                            ) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": "unavailable", "sense": None, "p_value": None,
        "n_tiles": 0, "tiles_per_side": None, "alpha": float(alpha),
        "reason": "",
    }
    if field is None:
        out["reason"] = "dI/dV field unavailable; sub-windows not measured."
        return out

    try:
        from didv_anisotropy import (power_spectrum, _uv_grid,
                                     _aperture_photometry)
    except Exception as exc:                         # pragma: no cover
        out["reason"] = f"Anisotropy photometry unavailable: {exc}"
        return out

    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        out["reason"] = f"Field is not 2D (shape {field.shape})."
        return out
    uv = [(float(u), float(v)) for u, v in uv]
    if len(uv) != 3:
        out["reason"] = "Three CDW wavevectors required."
        return out

    patterns: List[np.ndarray] = []
    used_k: Optional[int] = None
    n_attempted = 0
    for k in REPRODUCIBILITY_TILES_PER_SIDE:
        bounds = _tile_bounds(field.shape, q_cycles_per_px, k)
        if bounds is None:
            continue
        patterns, n_attempted = [], len(bounds)
        for (y0, y1, x0, x1) in bounds:
            sub = field[y0:y1, x0:x1]
            try:
                spec = power_spectrum(sub)
            except Exception:
                continue
            power = spec["power"]
            ug, vg = _uv_grid(spec["shape"], spec["center"])
            df = 1.0 / np.sqrt(float(sub.shape[0]) * float(sub.shape[1]))
            r = max(float(r_aperture_cpp),
                    REPRODUCIBILITY_MIN_APERTURE_BINS * df)
            vals: List[float] = []
            for (u, v) in uv:
                m = _aperture_photometry(power, ug, vg, u, v,
                                         r, 1.8 * r, 3.2 * r)
                val = m.get("intensity")
                if val is None or not np.isfinite(val) or val <= 0:
                    vals = []
                    break
                vals.append(float(val))
            if len(vals) == 3:
                lg = np.log(np.asarray(vals, dtype=np.float64))
                patterns.append(lg - lg.mean())
        if len(patterns) >= REPRODUCIBILITY_MIN_TILES:
            used_k = k
            break

    if used_k is None or len(patterns) < REPRODUCIBILITY_MIN_TILES:
        out["n_tiles"] = len(patterns)
        out["reason"] = (
            f"Fewer than {REPRODUCIBILITY_MIN_TILES} sub-windows of "
            f"\u2265 {REPRODUCIBILITY_MIN_TILE_PX} px and "
            f"\u2265 {REPRODUCIBILITY_MIN_TILE_PERIODS:g} CDW periods with "
            "three positive CDW intensities"
            + (f" ({len(patterns)}/{n_attempted} measured)"
               if n_attempted else "")
            + ".")
        return out

    D = np.vstack(patterns)
    n = int(D.shape[0])
    mean = D.mean(axis=0)
    sem = D.std(axis=0, ddof=1) / np.sqrt(n)
    out.update(n_tiles=n, tiles_per_side=int(used_k),
               tile_log_hierarchy=[round(float(x), 5) for x in mean],
               tile_log_hierarchy_sem=[round(float(x), 5) for x in sem],
               tile_log_scatter=round(
                   float(np.sqrt(np.mean(D.std(axis=0, ddof=1) ** 2))), 5))

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.abs(mean) / sem
    finite = tstat[np.isfinite(tstat)]
    t_max = float(np.max(finite)) if finite.size else 0.0
    p = float(min(1.0, 2.0 * 2.0 * _student_sf(t_max, n - 1)))
    out["t_statistic"] = round(t_max, 3)
    out["p_value"] = round(p, 6) if p >= 1e-6 else 1e-6
    out["p_value_underflow"] = bool(p < 1e-6)
    out["degrees_of_freedom"] = n - 1

    sense_tile = int(_sense_of(np.asarray(angles_deg, dtype=np.float64),
                               np.exp(mean))[0])
    out["sense"] = _SENSE_NAME[sense_tile]
    per_tile = _sense_of(np.asarray(angles_deg, dtype=np.float64), np.exp(D))
    out["tile_senses"] = [_SENSE_NAME[int(s)] for s in per_tile]
    out["tile_concordance"] = round(
        float(np.mean(per_tile == sense_tile)), 4) if sense_tile != 0 else None

    if p < float(alpha):
        out["status"] = "significant"
        p_rel = "<" if p < 1e-6 else "="
        out["reason"] = (
            f"Log-intensity hierarchy reproduces over {n} sub-windows "
            f"(t = {t_max:.2f}, {n - 1} df, p {p_rel} {max(p, 1e-6):.2g}, "
            f"Bonferroni); sub-window sense {out['sense']}"
            + (f", {out['tile_concordance']:.0%} tile concordance."
               if out["tile_concordance"] is not None else "."))
    else:
        out["status"] = "not_significant"
        out["reason"] = (
            f"Log-intensity hierarchy consistent with zero over {n} "
            f"sub-windows (t = {t_max:.2f}, {n - 1} df, p = {p:.2g}): a "
            "finite-window fluctuation of broadened peaks.")
    return out


def _stability(angles_deg: np.ndarray, rows: Sequence[Dict[str, Any]],
               fallback_nap: float, n_draw: int = N_DRAW,
               seed: int = 0, sys_frac: float = 0.0,
               resolution_kappa: float = SENSE_RESOLUTION_KAPPA
               ) -> Optional[Dict[str, Any]]:
    I = np.array([_num(r.get("intensity")) for r in rows], dtype=np.float64)
    S = np.array([_num(r.get("intensity_sigma")) for r in rows],
                 dtype=np.float64)
    Bg = np.array([_num(r.get("background")) for r in rows],
                  dtype=np.float64)
    N = np.array([fallback_nap if r.get("n_aperture") is None
                  else _num(r.get("n_aperture")) for r in rows],
                 dtype=np.float64)
    if not (np.all(np.isfinite(I)) and np.all(np.isfinite(S))
            and np.all(np.isfinite(Bg)) and np.all(np.isfinite(N))
            and np.all(S > 0) and np.all(N > 0)):
        return None

    rng = np.random.default_rng(seed)
    offset = N * Bg
    var = S ** 2

    sys_frac = float(sys_frac) if np.isfinite(sys_frac) else 0.0
    if sys_frac > 0.0:
        means = I[None, :] * np.exp(
            rng.normal(0.0, sys_frac, size=(n_draw, 3)))
    else:
        means = np.broadcast_to(I[None, :], (n_draw, 3))

    totals = np.maximum(means + offset[None, :], 1e-30)
    shape = np.maximum(totals ** 2 / var[None, :], 1e-6)
    scale = var[None, :] / totals
    t = rng.gamma(shape, scale)
    draws = t - offset[None, :]
    senses = _sense_of(angles_deg, draws)
    p_geom_nonmono = float(np.mean(senses == 0))

    kappa = float(resolution_kappa) if np.isfinite(resolution_kappa) else 0.0
    p_unresolved = 0.0
    if kappa > 0.0:
        order = np.argsort(-draws, axis=1)
        vals = np.take_along_axis(draws, order, axis=1)
        spread = np.asarray(draws.std(axis=0), dtype=np.float64)
        sg = spread[order]
        gap_hi = vals[:, 0] - vals[:, 1]
        gap_lo = vals[:, 1] - vals[:, 2]
        sc_hi = np.hypot(sg[:, 0], sg[:, 1])
        sc_lo = np.hypot(sg[:, 1], sg[:, 2])
        with np.errstate(invalid="ignore"):
            resolved = ((gap_hi >= kappa * sc_hi)
                        & (gap_lo >= kappa * sc_lo))
        resolved = np.where(np.isfinite(resolved), resolved, False)
        p_unresolved = float(np.mean(~resolved))
        senses = np.where(resolved, senses, 0)

    return {
        "p_ccw": round(float(np.mean(senses == +1)), 4),
        "p_cw": round(float(np.mean(senses == -1)), 4),
        "p_non_monotonic": round(float(np.mean(senses == 0)), 4),
        "p_unresolved": round(p_unresolved, 4),
        "p_geometrically_non_monotonic": round(p_geom_nonmono, 4),
        "n_draw": int(n_draw),
        "systematic_frac": round(sys_frac, 5),
        "resolution_kappa": round(kappa, 4),
        "null_baseline_p_win": SENSE_NULL_BASELINE,
    }


def assess_handedness(res: Optional[Dict[str, Any]],
                      field: Optional[np.ndarray] = None,
                      n_draw: int = N_DRAW,
                      stability_threshold: float = STABILITY_THRESHOLD,
                      seed: int = 0
                      ) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "verdict": "no_measurement",
        "headline": "No measurement",
        "caveat": _CAVEAT,
        "notes": [],
        "source": "didv_anisotropy (B1) Fourier measurement; "
                  "nothing re-measured",
    }

    if not res or res.get("status") != "ok":
        out["reason"] = ("No anisotropy measurement; the CDW intensity "
                         "hierarchy cannot be read.")
        return out

    frame = str((res.get("method") or {}).get("angle_frame")
                or DISPLAY_FRAME)
    out["frame"] = frame

    cdw_rows = res.get("cdw_peaks") or []
    bragg_rows = res.get("bragg_peaks") or []
    if not _rows_ok(cdw_rows):
        out["reason"] = "Three CDW peak pairs with intensities not available."
        return out

    C = np.array([r["intensity"] for r in cdw_rows], dtype=np.float64)
    angles = np.array([r["angle_deg"] for r in cdw_rows], dtype=np.float64)
    measurable = res["alpha_measurable"]
    if not (np.all(np.isfinite(C)) and bool(measurable)):
        out["reason"] = ("A CDW peak integrates to \u2264 0 above background; "
                         "no three-peak hierarchy.")
        return out
    n_res = res["n_directions_resolved"]
    if int(n_res) < 3:
        out["reason"] = (f"Only {int(n_res)}/3 CDW directions resolved "
                         "(1Q/2Q-like); no cyclic hierarchy.")
        return out

    sense_raw = int(_sense_of(angles, C)[0])
    out["sense"] = _SENSE_NAME[sense_raw]
    R = res.get("R") or []
    if len(R) == 3 and all(r is not None and np.isfinite(r) for r in R):
        out["sense_normalized"] = _SENSE_NAME[
            int(_sense_of(angles, np.asarray(R, dtype=np.float64))[0])]
    else:
        out["sense_normalized"] = None
    out["angles_deg"] = [float(a) for a in angles]
    out["intensities"] = [float(c) for c in C]
    srt = np.sort(C)
    out["hierarchy_margin"] = (
        round(float((srt[1] - srt[0]) / srt[2]), 4) if srt[2] > 0 else None)
    out["hierarchy_log_spread"] = (
        round(float(np.log(srt[2] / srt[0])), 5)
        if srt[0] > 0 and np.isfinite(srt[2] / srt[0]) else None)

    nem = nematic_decomposition(angles, C)
    out["nematic"] = nem
    out["nematic_null"] = nematic_null_test(
        angles, sense_raw,
        nematic_magnitude=nem.get("nematic_magnitude"),
        n_draw=NEMATIC_NULL_DRAWS, seed=seed)
    out["amplitude_sense"] = out["sense"]
    out["amplitude_channel_licensed"] = bool(
        AMPLITUDE_CHANNEL_LICENSES_HANDEDNESS)

    geom_bragg = _star_geometry(bragg_rows) if _rows_ok(bragg_rows) else None
    geom_cdw = _star_geometry(cdw_rows)
    geom = geom_bragg or geom_cdw
    dist = _distortion_systematic(geom)
    eps = float(dist["systematic_frac"])
    out["geometry_control"] = {
        "reference": ("bragg" if geom_bragg else
                      "cdw" if geom_cdw else "unavailable"),
        "bragg_star": geom_bragg,
        "cdw_star": geom_cdw,
        "systematic_frac": round(eps, 5),
        "at_floor": bool(dist["at_floor"]),
        "capped": bool(dist["capped"]),
        "drivers": list(dist["drivers"]),
        "censored_by_detection": bool(dist.get("censored_by_detection", True)),
        "censoring_note": dist.get("censoring_note"),
        "detector_tolerance_radius_rtol":
            dist.get("detector_tolerance_radius_rtol"),
        "detector_tolerance_angle_deg":
            dist.get("detector_tolerance_angle_deg"),
        "note": (
            "A fixed circular aperture applied to a star this far from an "
            "ideal hexagon truncates each of the three peaks by a different "
            "fraction, which perturbs the intensity hierarchy "
            "multiplicatively with no broken symmetry behind it. The "
            "fraction is carried into the stability draws as a "
            "per-direction log-normal systematic; it is absent from "
            "intensity_sigma, which measures the aperture sum against the "
            "local background only."),
    }

    method = res.get("method") or {}
    rep = reproducibility_control(
        field,
        uv=_cdw_uv(cdw_rows),
        r_aperture_cpp=_aperture_cpp(res),
        q_cycles_per_px=_mean_q(cdw_rows),
        angles_deg=angles)
    out["reproducibility"] = rep
    rep_status = str(rep.get("status") or "unavailable")
    rep_sense = _SENSE_TO_INT.get(str(rep.get("sense") or ""), 0)

    if rep_status == "unavailable":
        out["reproducibility_untested"] = True
        out["notes"].append(
            "Sub-window reproducibility not tested"
            + (f": {str(rep.get('reason')).rstrip('.')}."
               if rep.get("reason") else "."))

    ap_bins = float(method.get("aperture_radius_bins") or 3.0)
    fallback_nap = np.pi * ap_bins ** 2
    stab_phot = _stability(angles, cdw_rows, fallback_nap=fallback_nap,
                           n_draw=n_draw, seed=seed, sys_frac=0.0)
    stab = _stability(angles, cdw_rows, fallback_nap=fallback_nap,
                      n_draw=n_draw, seed=seed, sys_frac=eps)
    stab_unres = _stability(angles, cdw_rows, fallback_nap=fallback_nap,
                            n_draw=n_draw, seed=seed, sys_frac=eps,
                            resolution_kappa=0.0)
    out["stability"] = stab
    out["stability_diagnostic"] = stab
    out["stability_photometric_only"] = stab_phot
    out["stability_unresolved_sense"] = stab_unres

    sense_corr: Optional[int] = None
    if _rows_ok(bragg_rows):
        tip = _tip_corrected_intensities(cdw_rows, bragg_rows)
        if tip is not None:
            sense_corr = int(_sense_of(angles, tip["intensities"])[0])
            out["sense_tip_corrected"] = _SENSE_NAME[sense_corr]
            out["tip_control"] = {
                "q_ratio_sq": round(tip["q_ratio_sq"], 4),
                "tip_logspread_at_cdw": round(
                    tip["tip_logspread_at_cdw"], 4),
            }
        else:
            out["sense_tip_corrected"] = None
            out["notes"].append(
                "Tip-quadrupole control unavailable: non-positive Bragg or "
                "CDW intensity.")
    else:
        out["sense_tip_corrected"] = None
        out["notes"].append(
            "Tip-quadrupole control unavailable: incomplete Bragg triple.")
    fam = res.get("bragg_family") or {}
    frac = fam.get("power_frac_of_strongest")
    if frac is not None and frac > 0.35:
        out["notes"].append(
            f"Bragg anchor holds {frac:.0%} of the strongest peak and may be "
            "a CDW shell; tip control then not referenced to the lattice.")

    log_spread = out.get("hierarchy_log_spread")
    drift_attributable = bool(log_spread is not None and log_spread < eps)
    out["drift_attributable"] = drift_attributable
    if drift_attributable:
        out["notes"].append(
            f"Hierarchy spread I_max/I_min = "
            f"{float(np.exp(log_spread)):.3f} is below the {eps:.1%} "
            "per-peak systematic implied by the star distortion"
            + (f" ({'; '.join(dist['drivers'])})" if dist["drivers"] else "")
            + ": explainable by scan-grid distortion and aperture "
              "truncation.")

    screen = res.get("screen") or {}
    out["screen_verdict"] = str(screen.get("verdict") or "unscreened")

    reasons: List[str] = []
    if sense_raw == 0:
        reasons.append("non-monotonic hierarchy (tied intensities or "
                       "irregular angles)")
    if stab is None:
        reasons.append("no photometric stability estimate")
    elif sense_raw != 0:
        key = "p_ccw" if sense_raw == +1 else "p_cw"
        p_win = stab[key]
        out["stability_of_sense"] = p_win
        out["stability_of_sense_diagnostic"] = p_win
        p_win_phot = stab_phot[key] if stab_phot is not None else None
        out["stability_of_sense_photometric_only"] = p_win_phot
        p_win_unres = (stab_unres[key] if stab_unres is not None else None)
        out["stability_of_sense_unresolved"] = p_win_unres
        out["p_unresolved"] = stab.get("p_unresolved")
        if p_win < stability_threshold:
            detail = (f"sense survives {p_win:.0%} of redraws "
                      f"(< {stability_threshold:.0%})")
            if (p_win_unres is not None
                    and p_win_unres >= stability_threshold
                    and stab.get("p_unresolved", 0.0) > 0.25):
                detail += (
                    f"; {stab['p_unresolved']:.0%} of draws unresolved "
                    "against their own scatter")
            elif (p_win_phot is not None
                    and p_win_phot >= stability_threshold):
                detail += (
                    f" once the {eps:.1%} geometric systematic is included "
                    f"({p_win_phot:.0%} with photometric noise alone)")
            reasons.append(detail)
    if sense_corr is not None and sense_corr != sense_raw:
        reasons.append(
            "sense reversed by the q\u00b2-scaled tip-quadrupole correction")

    if rep_status == "not_significant":
        reasons.append(
            f"not reproducible over {rep.get('n_tiles')} sub-windows "
            f"(p = {rep.get('p_value')})")
    elif rep_status == "significant" and sense_raw != 0 \
            and rep_sense != 0 and rep_sense != sense_raw:
        reasons.append(
            f"sub-window sense {_SENSE_NAME[rep_sense]} opposes the "
            f"full-window sense {_SENSE_NAME[sense_raw]}")

    if reasons:
        out.update(verdict="indeterminate",
                   headline="Indeterminate: intensity hierarchy not robust",
                   reason="No robust hierarchy: " + "; ".join(reasons) + ".")
        return out

    nn = out.get("nematic_null") or {}
    p_nem = nn.get("p_value_observed_sense")
    axis_txt = ""
    if nem.get("available") and nem.get("axis_deg") is not None:
        contrast = nem.get("nematic_contrast")
        axis_txt = (
            f" The three intensities are exactly a mean plus a nematic "
            f"director along {nem['axis_deg']:.1f}\u00b0"
            + (f" (director I_max/I_min = {float(contrast) + 1.0:.2f})"
               if isinstance(contrast, (int, float)) else "")
            + ", which is achiral.")

    rep_clause = ""
    if rep_status == "significant":
        rep_clause = f", reproducible over {rep.get('n_tiles')} sub-windows"

    p_txt = f"{p_nem:.2f}" if isinstance(p_nem, float) else "0.50"
    out.update(
        verdict="not_licensed",
        headline="Nematic intensity anisotropy; handedness not determinable",
        reason=(
            f"Intensity hierarchy {_SENSE_NAME[sense_raw]}: stable in "
            f"{out['stability_of_sense_diagnostic']:.0%} of redraws "
            f"(photometric + {eps:.1%} geometric), invariant under tip-"
            f"quadrupole correction{rep_clause}."
            + axis_txt
            + f" Against an achiral director of unknown axis the sense is a "
              f"fair coin (p = {p_txt}; 0.5 by symmetry at any SNR), so "
              "handedness requires the structural enantiomorph or the phase "
              "registry."))
    return out


def format_handedness_report(h: Dict[str, Any]) -> str:
    L = ["=" * 72,
         "dI/dV HANDEDNESS  (cyclic sense of the CDW intensity hierarchy)",
         "  CORROBORATING ONLY - this channel cannot ESTABLISH chirality;",
         "  see section 0 of didv_handedness and didv_phase_chirality.",
         "=" * 72,
         f"Verdict : {h.get('headline', h.get('verdict', '?'))}",
         f"Reason  : {h.get('reason', '-')}"]
    if h.get("sense"):
        L.append(f"Sense   : raw {h['sense']}   (DIAGNOSTIC - not a "
                 f"handedness)"
                 + (f"\n          normalized {h['sense_normalized']}"
                    if h.get("sense_normalized") else "")
                 + (f" | tip-corrected {h['sense_tip_corrected']}"
                    if h.get("sense_tip_corrected") else ""))
    nem = h.get("nematic") or {}
    if nem.get("available"):
        L.append(f"Nematic : director axis = {nem['axis_deg']:.1f} deg, "
                 f"peak-to-peak contrast = {nem.get('nematic_contrast')} "
                 f"({nem.get('space')} space)")
        L.append(f"          exact inversion: residual = "
                 f"{nem['residual_max']:.2e}, {nem['residual_dof']} degrees "
                 f"of freedom remain -> the three intensities ARE a mean "
                 f"plus one director, which is achiral")
    nn = h.get("nematic_null") or {}
    if nn.get("available"):
        L.append(f"Achiral null: p(this sense | achiral director of unknown "
                 f"axis) = {nn['p_value_observed_sense']:.3f}   "
                 f"[best attainable = "
                 f"{nn['max_attainable_significance']:.2f}]")
        L.append("          the sense cannot reach significance against an "
                 "achiral state at ANY SNR")
    stab = h.get("stability") or h.get("stability_diagnostic")
    if stab:
        L.append(f"Stability: p(ccw) = {stab['p_ccw']:.3f}, "
                 f"p(cw) = {stab['p_cw']:.3f}, "
                 f"p(non-monotonic) = {stab['p_non_monotonic']:.3f} "
                 f"({stab['n_draw']} draws, photometric noise + "
                 f"{stab.get('systematic_frac', 0.0):.1%} geometric)")
    phot = h.get("stability_photometric_only")
    if phot and stab and phot != stab:
        L.append(f"           photometric noise alone: p(ccw) = "
                 f"{phot['p_ccw']:.3f}, p(cw) = {phot['p_cw']:.3f}")
    geo = h.get("geometry_control")
    if geo:
        L.append(f"Geometry : the measured star departs from an ideal "
                 f"hexagon by {geo['systematic_frac']:.1%} "
                 f"({geo['reference']} reference"
                 + (", at floor" if geo.get("at_floor") else "")
                 + (", capped" if geo.get("capped") else "") + ")")
        if geo.get("drivers"):
            L.append("           " + "; ".join(geo["drivers"]))
    rep = h.get("reproducibility") or {}
    if rep:
        status = str(rep.get("status") or "unavailable")
        head = {"significant": "reproduces across the map",
                "not_significant": "DOES NOT reproduce across the map",
                "unavailable": "not evaluated"}.get(status, status)
        line = f"Repro.   : {head}"
        if rep.get("n_tiles"):
            line += (f" ({rep['n_tiles']} sub-windows"
                     + (f", p {'<' if rep.get('p_value_underflow') else '='} "
                        f"{rep['p_value']:.3g}"
                        if rep.get("p_value") is not None else "")
                     + (f", sub-window sense {rep['sense']}"
                        if rep.get("sense") else "") + ")")
        L.append(line)
        if status == "unavailable" and rep.get("reason"):
            L.append(f"           {rep['reason']}")
    if h.get("hierarchy_log_spread") is not None:
        L.append(f"Hierarchy: ln(I_max / I_min) = "
                 f"{h['hierarchy_log_spread']:.3f}"
                 + ("   <- smaller than the geometric systematic"
                    if h.get("drift_attributable") else ""))
    for n in h.get("notes", []):
        L.append(f"  ! {n}")
    if h.get("frame"):
        L.append(f"Frame   : {h['frame']} - senses are as the map is "
                 f"displayed")
    L.append(f"Caveat  : {h.get('caveat', '')}")
    L.append("=" * 72)
    return "\n".join(L)


def didv_handedness_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        res = state.get("didv_anisotropy") or None
        field = None
        didv_b64 = state.get("input_didv_base64")
        if didv_b64:
            try:
                from didv_anisotropy import load_didv_scalar
                meta = state.get("metadata") or {}
                from stm_data_io import normalise_extension
                ext = normalise_extension(meta.get("didv_ext"))
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "didv" + ext)
                    with open(path, "wb") as fh:
                        fh.write(base64.b64decode(didv_b64))
                    field, _info = load_didv_scalar(
                        path, colormap=meta.get("didv_colormap"),
                        channel=meta.get("didv_channel"),
                        direction=meta.get("didv_direction", "forward"),
                        bias=meta.get("didv_bias"))
            except Exception:
                field = None
        h = assess_handedness(res, field=field)
        h["report"] = format_handedness_report(h)
        state["didv_handedness"] = h
    except Exception as exc:                     # pragma: no cover
        state["didv_handedness"] = {
            "verdict": "no_measurement", "headline": "No measurement",
            "reason": f"Handedness assessment failed: {exc}",
            "caveat": _CAVEAT, "notes": [],
        }
    return state
