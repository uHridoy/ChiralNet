from __future__ import annotations

import base64
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from didv_anisotropy import _remove_plane, _sanitize, load_didv_scalar


PHASE_SNR_FLOOR = 5.0

CLOSURE_TOL_BINS = 0.75

MAX_COMMENSURATE_N = 16

COMMENSURATE_TOL = 0.08

REGISTRY_WINDING_NOTE = 0.25

REGISTRY_WINDING_MAX = 1.0

MIRROR_TOL_FRAC = 0.20

MIRROR_TOL_SEPARATION_FRAC = 0.45

AFFINE_ANISOTROPY_NOTE = 1.25

AFFINE_ANISOTROPY_MAX = 2.0

REQUIRE_ALL_MIRROR_AXES = True

PHASE_REPRO_TILES = 2

PHASE_REPRO_MIN_PERIODS = 6.0

PHASE_REPRO_MIN_FRACTION = 0.75

PHASE_REPRO_SIGN_UNANIMOUS = True

HEX_RESIDUAL_NOTE = 0.05
HEX_RESIDUAL_MAX = 0.15

REFINE_HALF_BINS = 0.6
REFINE_STEP_BINS = 0.1

N_DRAW = 4000


PHASE_EFFECT_FLOOR = 0.02

PHASE_NULL_MARGIN = 1.0

BRAGG_CONTROL_MARGIN = 3.0

REQUIRE_CONSERVATIVE_P = True

NULL_JITTER_FROM_BRAGG_CONTROL = 2.0


LEAKAGE_CONTROL_MIN_N = 3

LEAKAGE_MIN_SURVIVING_FRACTION = 0.6

_T_GRID = 192
_T_REFINE = 24

_FRAME_NOTE = (
    "Geometry is in display coordinates (x right, y UP), i.e. the image as "
    "shown with row 0 at the top, so 'counter-clockwise' means "
    "counter-clockwise to a reader looking at the panel. didv_anisotropy now "
    "converts to this frame at the point of measurement and declares it in "
    "method['angle_frame'], so this module, the anisotropy card and the "
    "handedness card all report senses in one convention; 'amplitude_sense' "
    "below and didv_handedness's verdict should agree. A measurement that "
    "predates the declaration is converted here instead - see "
    "'source_angle_frame'.")

_CAVEAT = (
    "A non-zero chirality_phase means the CDW-to-lattice registry breaks "
    "every mirror of the measured star, which a real multiplicative tip "
    "response and an affine drift distortion cannot do. It does not by "
    "itself establish a bulk chiral CDW, for three reasons that are "
    "reported per map rather than assumed away. (1) DOMAINS: the "
    "measurement is over one field of view at one bias, so it reports the "
    "net registry of whatever domains the frame contains, and a frame "
    "spanning two domains of opposite handedness averages toward zero. "
    "(2) NEAR-COMMENSURATION: if the CDW is only nearly commensurate the "
    "registry is not a single number - it winds through discommensurations "
    "across the frame - and the residual that produces is a property of "
    "where those sit relative to the window rather than of the crystal, "
    "changing when the window moves. Most real systems are on that side of "
    "the line (the NC phase of 1T-TaS2, 2H-NbSe2, CsV3Sb5 under some "
    "conditions), so `registry_winding` is measured on every map and "
    "`significance_gates['commensurate_registry']` withholds the chirality "
    "reading above REGISTRY_WINDING_MAX cycles. (3) TIP PHASE: the "
    "invariance argument holds for a tip whose transfer function is real "
    "and positive, i.e. a centrosymmetric apex; a non-centrosymmetric tip "
    "(a double tip, an adsorbate on one side) has a complex T(q) and does "
    "move psi_i. Handedness comparison across maps, or to a "
    "crystallographic axis, requires the stated frame convention above.")


def _dict_get(d: Any, key: str) -> Any:
    return d.get(key) if isinstance(d, dict) else None


def _as_num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _wrap_pi(x):
    return (np.asarray(x, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _fold180(a_deg: float) -> float:
    return float(a_deg % 180.0)


def _to_display_deg(angle_deg: float) -> float:
    return _fold180(float(angle_deg))


def _rot(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([np.cos(t), np.sin(t)])


def _reflect(q: np.ndarray, beta_deg: float) -> np.ndarray:
    b = np.radians(2.0 * beta_deg)
    M = np.array([[np.cos(b), np.sin(b)], [np.sin(b), -np.cos(b)]])
    return np.atleast_2d(q) @ M.T


def _prepare(field: np.ndarray) -> np.ndarray:
    img = _sanitize(np.asarray(field, dtype=np.float64))
    img = _remove_plane(img)
    Ny, Nx = img.shape
    win = np.outer(np.hanning(Ny), np.hanning(Nx))
    wsum = float(win.sum())
    wmean = float((img * win).sum() / wsum) if wsum > 0 else float(img.mean())
    return (img - wmean) * win


def _project(g: np.ndarray, q: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    Ny, Nx = g.shape
    x = np.arange(Nx, dtype=np.float64) - 0.5 * (Nx - 1)
    y_up = -(np.arange(Ny, dtype=np.float64) - 0.5 * (Ny - 1))
    ex = np.exp(-2j * np.pi * np.outer(x, q[:, 0]))
    ey = np.exp(-2j * np.pi * np.outer(y_up, q[:, 1]))
    return np.einsum("yn,yn->n", g.astype(np.complex128) @ ex, ey)


def _refine(g: np.ndarray, q0: np.ndarray, df: float) -> np.ndarray:
    steps = np.arange(-REFINE_HALF_BINS, REFINE_HALF_BINS + 1e-9,
                      REFINE_STEP_BINS) * df
    dq = np.stack(np.meshgrid(steps, steps, indexing="ij"), -1).reshape(-1, 2)
    cand = q0[None, :] + dq
    amp = np.abs(_project(g, cand))
    return cand[int(np.argmax(amp))]


def _noise_sigma(g: np.ndarray, q: np.ndarray, df: float,
                 r_in: float = 4.0, r_out: float = 10.0) -> float:
    Ny, Nx = g.shape
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    cy, cx = Ny // 2, Nx // 2
    ix = cx + int(round(q[0] * Nx))
    iy = cy + int(round(-q[1] * Ny))
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    rr = np.hypot((yy - iy) * (1.0 / Ny), (xx - ix) * (1.0 / Nx)) / df
    sel = (rr > r_in) & (rr <= r_out) & np.isfinite(P)
    vals = P[sel]
    if vals.size < 24:
        sel = (rr > r_in) & (rr <= r_out * 2.0) & np.isfinite(P)
        vals = P[sel]
    if vals.size < 24:
        return float("nan")
    med = float(np.median(vals))
    kept = vals[vals <= 6.0 * max(med, 1e-300)]
    if kept.size < 12:
        kept = vals
    return float(np.sqrt(max(float(kept.mean()), 1e-300)))


def _close(qs: np.ndarray) -> Tuple[np.ndarray, float]:
    d = qs[0] - qs[1] + qs[2]
    out = qs.copy()
    out[0] -= d / 3.0
    out[1] += d / 3.0
    out[2] -= d / 3.0
    return out, float(np.hypot(*d))


def _coords(q: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.linalg.solve(basis.T, np.atleast_2d(q).T).T


_ANG_FOLD_EPS_DEG = 1e-6


def _fold_dirs_deg(q: np.ndarray) -> np.ndarray:
    ang = np.degrees(np.arctan2(np.asarray(q)[:, 1],
                                np.asarray(q)[:, 0])) % 180.0
    return np.where(ang > 180.0 - _ANG_FOLD_EPS_DEG, ang - 180.0, ang)


def _hex_residual(q: np.ndarray) -> Optional[float]:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (3, 2):
        return None
    ang = _fold_dirs_deg(q)
    order = np.argsort(ang)
    ang, qs = ang[order], q[order]
    rad = np.hypot(qs[:, 0], qs[:, 1])
    r0 = float(np.mean(rad))
    if not (r0 > 0):
        return None
    a0 = float(np.mean(ang - np.array([0.0, 60.0, 120.0])))
    ideal = np.stack([r0 * _rot(a0 + 60.0 * k) for k in range(3)])
    sign = np.sign(np.einsum("ij,ij->i", qs, ideal))
    sign[sign == 0.0] = 1.0
    ideal = ideal * sign[:, None]
    return float(np.max(np.hypot(*((qs - ideal).T))) / r0)


def _deshear(ref: np.ndarray, other: Optional[np.ndarray] = None
             ) -> Tuple[np.ndarray, Dict[str, Any]]:
    ang = _fold_dirs_deg(ref)
    order = np.argsort(ang)
    ang, ref = ang[order], ref[order]
    rad = np.hypot(ref[:, 0], ref[:, 1])
    a0 = float(np.mean(ang - np.array([0.0, 60.0, 120.0])))
    r0 = float(np.mean(rad))
    ideal = np.stack([r0 * _rot(a0 + 60.0 * k) for k in range(3)])
    A = ideal[:2].T @ np.linalg.inv(ref[:2].T)
    fitted_resid = float(
        np.max(np.hypot(*((ref @ A.T) - ideal).T)) / max(r0, 1e-30))

    cross = None
    if other is not None:
        try:
            cross = _hex_residual(np.asarray(other, dtype=np.float64) @ A.T)
        except Exception:                                # pragma: no cover
            cross = None

    sv = np.linalg.svd(A, compute_uv=False)
    det = float(np.linalg.det(A))
    info = {
        "ideal_orientation_deg": round(a0 % 60.0, 3),
        "determinant": round(det, 6),
        "orientation_preserving": bool(det > 0),
        "ideal_radius_cycles_per_px": round(r0, 6),
        "affine_anisotropy": round(float(max(sv) / max(min(sv), 1e-30)), 4),
        "cross_family_residual_frac": (None if cross is None
                                       else round(float(cross), 5)),
        "residual_frac": (None if cross is None else round(float(cross), 5)),
        "fitted_family_residual_frac": round(fitted_resid, 8),
        "fitted_family_residual_is_vacuous": True,
        "note": ("An affine distortion of the scan grid moves the peaks but "
                 "leaves the complex amplitude at each corresponding "
                 "wavevector unchanged, so removing it is exact and the "
                 "phases are untouched. affine_anisotropy is 1.000 for an "
                 "undistorted star and grows with drift, creep or an aspect "
                 "error. fitted_family_residual_frac is identically 0 by "
                 "a counting argument (an affine map has four degrees of "
                 "freedom and a closed triple has four, so any closed "
                 "triple can be carried exactly onto a hexagon) and is NOT "
                 "a test; residual_frac carries the cross-family residual, "
                 "which is one."),
    }
    return A, info


def _commensuration(q_cdw: np.ndarray, q_bragg: np.ndarray
                    ) -> Optional[Dict[str, Any]]:
    basis = q_bragg[:2]
    if abs(float(np.linalg.det(basis))) < 1e-12:
        return None
    ab = _coords(q_cdw, basis)
    for N in range(1, MAX_COMMENSURATE_N + 1):
        scaled = N * ab
        err = np.abs(scaled - np.round(scaled))
        if err.max() <= COMMENSURATE_TOL:
            return {"N": int(N),
                    "coeffs": np.round(scaled).astype(int),
                    "max_error": float(err.max())}
    return None


def _registry_winding(q_cdw: np.ndarray, q_bragg: np.ndarray,
                      comm: Dict[str, Any], shape: Tuple[int, int]
                      ) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False, "winding_cycles": None,
        "residual_q_cycles_per_px": None, "field_of_view_px": None,
        "commensuration_max_error": None,
        "note": ("Cycles of registry-phase advance across the field of "
                 "view. Near zero the registry is one number and psi_i "
                 "means what it says; approaching or above one cycle the "
                 "frame contains a phason texture and psi_i is an average "
                 "over a varying quantity."),
    }
    try:
        N = int(comm["N"])
        coeffs = np.asarray(comm["coeffs"], dtype=np.float64)
        b = np.asarray(q_bragg, dtype=np.float64)[:2]
        qc = np.asarray(q_cdw, dtype=np.float64)
        dq = N * qc - coeffs @ b
        dq_mag = float(np.max(np.hypot(dq[:, 0], dq[:, 1])))
        L = float(math.sqrt(float(shape[0]) * float(shape[1])))
        out.update(
            available=True,
            residual_q_cycles_per_px=round(dq_mag, 8),
            field_of_view_px=round(L, 1),
            commensuration_max_error=round(float(comm.get("max_error", 0.0)),
                                           5),
            winding_cycles=round(dq_mag * L, 4))
        return out
    except Exception:                                    # pragma: no cover
        return out


_PH_CACHE: Dict[Any, np.ndarray] = {}


def _coarse_phase(coords: np.ndarray) -> np.ndarray:
    u = coords[:, 0].astype(np.float64)
    v = coords[:, 1].astype(np.float64)
    key = (u.tobytes(), v.tobytes(), len(u), _T_GRID)
    ph = _PH_CACHE.get(key)
    if ph is None:
        t = np.linspace(0.0, 2.0 * np.pi, _T_GRID, endpoint=False)
        ph = np.exp(1j * (u[:, None, None] * t[None, :, None]
                          + v[:, None, None] * t[None, None, :]))
        if len(_PH_CACHE) > 8:
            _PH_CACHE.clear()
        _PH_CACHE[key] = ph
    return ph


def _mirror_tolerances(peaks_q: np.ndarray) -> np.ndarray:
    q = np.asarray(peaks_q, dtype=np.float64)
    n = len(q)
    full_q = np.vstack([q, -q])
    r = np.hypot(q[:, 0], q[:, 1])
    tol = MIRROR_TOL_FRAC * r
    for p in range(n):
        d = np.hypot(*(full_q - q[p][None, :]).T)
        d[p] = np.inf
        d_min = float(np.min(d)) if d.size else np.inf
        if np.isfinite(d_min) and d_min > 0:
            tol[p] = min(tol[p], MIRROR_TOL_SEPARATION_FRAC * d_min)
    return tol


def _mirror_plan(peaks_q: np.ndarray, beta_deg: float,
                 tol: Any) -> Optional[np.ndarray]:
    n = len(peaks_q)
    tol_arr = np.broadcast_to(np.asarray(tol, dtype=np.float64), (n,))
    full_q = np.vstack([peaks_q, -peaks_q])
    idx = np.empty(n, dtype=np.int64)
    for p in range(n):
        mq = _reflect(peaks_q[p], beta_deg)[0]
        d = np.hypot(*(full_q - mq[None, :]).T)
        order = np.argsort(d)
        j = int(order[0])
        if d[j] > tol_arr[p]:
            return None
        if len(order) > 1 and d[int(order[1])] <= tol_arr[p]:
            return None
        idx[p] = j
    return idx


def _mirror_residual(peaks_q: np.ndarray, amps: np.ndarray,
                     coords: np.ndarray, beta_deg: float,
                     tol: Any,
                     plan: Optional[np.ndarray] = None) -> Optional[float]:
    if plan is None:
        plan = _mirror_plan(peaks_q, beta_deg, tol)
    if plan is None:
        return None

    full_a = np.concatenate([amps, np.conj(amps)])
    mirrored = full_a[plan]

    u = coords[:, 0].astype(np.float64)
    v = coords[:, 1].astype(np.float64)
    denom = float(np.sum(np.abs(amps) ** 2))
    if not (denom > 0):
        return None

    t = np.linspace(0.0, 2.0 * np.pi, _T_GRID, endpoint=False)
    ph = _coarse_phase(coords)
    r = np.sum(np.abs(mirrored[:, None, None]
                      - amps[:, None, None] * ph) ** 2, axis=0)
    k = int(np.argmin(r))
    i, j = np.unravel_index(k, r.shape)
    r_best = float(r[i, j])

    step = 2.0 * np.pi / _T_GRID
    fine = np.linspace(-step, step, _T_REFINE)
    t1, t2 = t[i] + fine, t[j] + fine
    phf = np.exp(1j * (u[:, None, None] * t1[None, :, None]
                       + v[:, None, None] * t2[None, None, :]))
    rf = np.sum(np.abs(mirrored[:, None, None]
                       - amps[:, None, None] * phf) ** 2, axis=0)
    r_best = min(r_best, float(np.min(rf)))
    return float(np.sqrt(max(r_best, 0.0) / denom))


def _chirality(peaks_q: np.ndarray, amps: np.ndarray, coords: np.ndarray,
               axes: Sequence[float], families: np.ndarray,
               tol: float) -> Dict[str, Any]:
    eq = amps.copy()
    for f in np.unique(families):
        m = families == f
        gm = float(np.exp(np.mean(np.log(np.maximum(np.abs(amps[m]),
                                                    1e-300)))))
        eq[m] = gm * np.exp(1j * np.angle(amps[m]))

    per_axis: List[Dict[str, Any]] = []
    for beta in axes:
        plan = _mirror_plan(peaks_q, beta, tol)
        if plan is None:
            continue
        d_tot = _mirror_residual(peaks_q, amps, coords, beta, tol, plan=plan)
        if d_tot is None:
            continue
        d_pha = _mirror_residual(peaks_q, eq, coords, beta, tol, plan=plan)
        full_m = np.concatenate([np.abs(amps), np.abs(amps)])
        mm = full_m[plan]
        d_amp = float(np.sqrt(np.sum((mm - np.abs(amps)) ** 2)
                              / max(float(np.sum(np.abs(amps) ** 2)), 1e-300)))
        per_axis.append({"axis_deg": round(float(beta % 180.0), 2),
                         "total": round(float(d_tot), 5),
                         "amplitude": round(float(d_amp), 5),
                         "phase": round(float(d_pha), 5)})

    if not per_axis:
        return {"available": False, "axes": [],
                "n_axes_requested": int(len(axes)), "n_axes_matched": 0}
    return {
        "available": True,
        "axes": per_axis,
        "n_axes_requested": int(len(axes)),
        "n_axes_matched": int(len(per_axis)),
        "chirality_total": min(a["total"] for a in per_axis),
        "chirality_amplitude": min(a["amplitude"] for a in per_axis),
        "chirality_phase": min(a["phase"] for a in per_axis),
        "nearest_mirror_axis_deg": min(per_axis, key=lambda a: a["total"])[
            "axis_deg"],
    }


def _phase_min_over_axes(peaks_q: np.ndarray, amps: np.ndarray,
                         coords: np.ndarray, families: np.ndarray,
                         live: Sequence[Tuple[float, np.ndarray]],
                         tol: Any) -> Optional[float]:
    eq = _equalise(amps, families)
    vals = [_mirror_residual(peaks_q, eq, coords, b, tol, plan=pl)
            for b, pl in live]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _phase_reproducibility(field: np.ndarray, q_meas: np.ndarray,
                           peaks_q: np.ndarray, coords: np.ndarray,
                           families: np.ndarray,
                           live: Sequence[Tuple[float, np.ndarray]],
                           tol: Any, obs: float, phi_full: float,
                           n_side: int = PHASE_REPRO_TILES) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False, "passed": None, "n_tiles": 0,
        "tile_residuals": [], "tile_phi_deg": [],
        "fraction_residual_reproduced": None,
        "fraction_sign_reproduced": None,
        "min_fraction_required": PHASE_REPRO_MIN_FRACTION,
        "reason": "",
    }
    try:
        arr = np.asarray(field, dtype=np.float64)
        if arr.ndim != 2:
            out["reason"] = "the field is not a 2-D map."
            return out
        Ny, Nx = arr.shape
        ty, tx = Ny // n_side, Nx // n_side
        r = np.hypot(q_meas[:, 0], q_meas[:, 1])
        r_min = float(np.min(r[r > 0])) if np.any(r > 0) else 0.0
        periods = min(ty, tx) * r_min
        out["periods_per_tile"] = round(float(periods), 2)
        if r_min <= 0 or periods < PHASE_REPRO_MIN_PERIODS:
            out["reason"] = (
                f"a {n_side}x{n_side} split gives only {periods:.1f} CDW "
                f"periods across a sub-window (minimum "
                f"{PHASE_REPRO_MIN_PERIODS:g}), so the control cannot be "
                "run on this map. It is reported as UNRUN, not as passed.")
            return out

        if not (float(obs) >= PHASE_EFFECT_FLOOR):
            out["reason"] = (
                f"The full-window phase residual ({float(obs):.5f}) is "
                f"below the effect-size floor ({PHASE_EFFECT_FLOOR:g}), so "
                "there is no mirror breaking for this control to reproduce. "
                "Not applicable rather than failed.")
            out["not_applicable"] = True
            return out

        sign_full = float(np.sign(math.sin(phi_full)))
        res: List[Optional[float]] = []
        phis: List[Optional[float]] = []
        for iy in range(n_side):
            for ix in range(n_side):
                sub = arr[iy * ty:(iy + 1) * ty, ix * tx:(ix + 1) * tx]
                g_t = _prepare(sub)
                a_t = _project(g_t, q_meas)
                r_t = _phase_min_over_axes(peaks_q, a_t, coords, families,
                                           live, tol)
                res.append(None if r_t is None else float(r_t))
                if len(a_t) >= 3:
                    phi_t = float(_wrap_pi(np.angle(a_t[0])
                                           - np.angle(a_t[1])
                                           + np.angle(a_t[2])))
                    phis.append(phi_t)
                else:
                    phis.append(None)

        good_r = [v for v in res if v is not None]
        n = len(good_r)
        if n == 0:
            out["reason"] = ("no sub-window yielded a residual, so the "
                             "control could not be run.")
            return out
        f_res = float(np.mean([v >= PHASE_EFFECT_FLOOR for v in good_r]))
        good_p = [v for v in phis if v is not None]
        f_sgn = (float(np.mean([np.sign(math.sin(v)) == sign_full
                                for v in good_p])) if good_p else 0.0)
        out.update(
            available=True, n_tiles=n,
            tile_residuals=[None if v is None else round(v, 5) for v in res],
            tile_phi_deg=[None if v is None else round(math.degrees(v), 1)
                          for v in phis],
            fraction_residual_reproduced=round(f_res, 3),
            fraction_sign_reproduced=round(f_sgn, 3),
            sign_required=(1.0 if PHASE_REPRO_SIGN_UNANIMOUS
                           else PHASE_REPRO_MIN_FRACTION),
            passed=bool(
                f_res >= PHASE_REPRO_MIN_FRACTION
                and f_sgn >= (1.0 if PHASE_REPRO_SIGN_UNANIMOUS
                              else PHASE_REPRO_MIN_FRACTION)))
        sign_req = (1.0 if PHASE_REPRO_SIGN_UNANIMOUS
                    else PHASE_REPRO_MIN_FRACTION)
        out["reason"] = (
            f"The phase residual clears the effect-size floor in "
            f"{f_res:.0%} of the {n} sub-windows (required "
            f"{PHASE_REPRO_MIN_FRACTION:.0%}) and the sign of Phi matches "
            f"the full-window sign in {f_sgn:.0%} of them (required "
            f"{sign_req:.0%})."
            + ("" if f_sgn >= sign_req else
               " A sub-window whose Phi has the opposite sign holds the "
               "OTHER enantiomorph, so this frame spans both and its "
               "full-window residual measures which domain is larger, not "
               "the handedness of the sample."))
        return out
    except Exception as exc:                             # pragma: no cover
        out["reason"] = f"the control could not be run: {exc}"
        return out


def assess_phase_chirality(res: Optional[Dict[str, Any]],
                           field: Optional[np.ndarray],
                           n_draw: int = N_DRAW,
                           seed: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "verdict": "no_measurement",
        "headline": "No measurement",
        "frame": "display (x right, y up)",
        "frame_note": _FRAME_NOTE,
        "caveat": _CAVEAT,
        "notes": [],
        "source": "complex FFT of the dI/dV field, re-derived; peak geometry "
                  "from didv_anisotropy (B1). Nothing in B1 is altered.",
    }

    if not res or res.get("status") != "ok":
        out["reason"] = (res or {}).get(
            "message", "The anisotropy tool produced no measurement.")
        return out
    if field is None:
        out["reason"] = "The dI/dV field was not available to re-derive the "\
                        "complex spectrum."
        return out

    cdw_rows = res.get("cdw_peaks") or []
    bragg_rows = res.get("bragg_peaks") or []

    src_frame = str((res.get("method") or {}).get("angle_frame")
                    or "array (x right, y down)")
    out["source_angle_frame"] = src_frame

    def geom(rows):
        if len(rows) != 3:
            return None
        try:
            a = [_to_display_deg(r["angle_deg"]) for r in rows]
            r_ = [float(r["q_cycles_per_px"]) for r in rows]
        except (KeyError, TypeError, ValueError):
            return None
        if not all(np.isfinite(r_)) or min(r_) <= 0:
            return None
        order = np.argsort(a)
        return np.stack([r_[i] * _rot(a[i]) for i in order])

    q_c = geom(cdw_rows)
    if q_c is None:
        out["reason"] = ("The measurement does not carry three CDW peaks with "
                         "angles and radii.")
        return out
    q_b = geom(bragg_rows)

    try:
        g = _prepare(np.asarray(field, dtype=np.float64))
    except Exception as exc:                            # pragma: no cover
        out["reason"] = f"the dI/dV field could not be prepared: {exc}"
        return out
    Ny, Nx = g.shape
    df = 1.0 / math.sqrt(float(Nx) * float(Ny))

    q_c = np.stack([_refine(g, q, df) for q in q_c])
    q_c, res_c = _close(q_c)
    out["closure_residual_bins_cdw"] = round(res_c / df, 4)
    if q_b is not None:
        q_b = np.stack([_refine(g, q, df) for q in q_b])
        q_b, res_b = _close(q_b)
        out["closure_residual_bins_bragg"] = round(res_b / df, 4)
        if res_b / df > CLOSURE_TOL_BINS:
            out["notes"].append(
                f"The Bragg triple closes only to {res_b / df:.2f} bins "
                f"(tolerance {CLOSURE_TOL_BINS:g}); the lattice reference is "
                "correspondingly uncertain.")
            q_b = None
    if res_c / df > CLOSURE_TOL_BINS:
        out.update(verdict="indeterminate", headline="Indeterminate",
                   reason=(f"The three CDW wavevectors close to only "
                           f"{res_c / df:.2f} bins (tolerance "
                           f"{CLOSURE_TOL_BINS:g}). Without an exactly closed "
                           "triple the phase invariants depend on the choice "
                           "of origin and carry no information."))
        return out

    c_c = _project(g, q_c)
    sig_c = np.array([_noise_sigma(g, q, df) for q in q_c])
    peaks_q, amps, sigs, families = q_c, c_c, sig_c, np.zeros(3, dtype=int)
    if q_b is not None:
        c_b = _project(g, q_b)
        sig_b = np.array([_noise_sigma(g, q, df) for q in q_b])
        peaks_q = np.vstack([q_c, q_b])
        amps = np.concatenate([c_c, c_b])
        sigs = np.concatenate([sig_c, sig_b])
        families = np.array([0, 0, 0, 1, 1, 1])

    snr = np.abs(amps) / np.where(np.isfinite(sigs) & (sigs > 0),
                                  sigs, np.nan)
    out["peak_snr"] = [None if not np.isfinite(s) else round(float(s), 1)
                       for s in snr]
    out["phase_sigma_deg"] = [
        None if not np.isfinite(s) or s <= 0
        else round(float(np.degrees(1.0 / (math.sqrt(2.0) * s))), 2)
        for s in snr]
    if not np.all(np.isfinite(snr)) or float(np.nanmin(snr)) < PHASE_SNR_FLOOR:
        worst = (float(np.nanmin(snr)) if np.any(np.isfinite(snr))
                 else float("nan"))
        out.update(
            verdict="indeterminate", headline="Indeterminate",
            reason=(f"The weakest peak reaches |F|/sigma = {worst:.1f}, below "
                    f"the floor of {PHASE_SNR_FLOOR:g} required for its phase "
                    "to be a measurement. Phase error scales as "
                    "1/(sqrt(2)|F|/sigma), so below this the registry is "
                    "noise."))
        return out

    phi_c = np.angle(c_c)
    Phi_c = float(_wrap_pi(phi_c[0] - phi_c[1] + phi_c[2]))
    out["phase_sum_cdw_deg"] = round(float(np.degrees(Phi_c)), 2)
    out["phase_sum_note"] = (
        "Phi is translation-invariant, but on its own it is NOT a chirality "
        "measure: an equal-amplitude 3Q pattern is mirror-symmetric for every "
        "Phi once a translation is allowed. It is reported as the shape "
        "parameter of the 3Q superposition and enters chirality only through "
        "the mirror conditions below.")

    commensurate = None
    if q_b is not None:
        phi_b = np.angle(c_b)
        out["phase_sum_bragg_deg"] = round(
            float(np.degrees(_wrap_pi(phi_b[0] - phi_b[1] + phi_b[2]))), 2)
        commensurate = _commensuration(q_c, q_b)
        if commensurate is None:
            out["notes"].append(
                "The CDW wavevectors are not an integer fraction of the "
                "measured lattice at this precision, so the registry phases "
                "are undefined; the mirror test below runs on the CDW family "
                "alone and is therefore amplitude-only.")
        else:
            N = commensurate["N"]
            m = commensurate["coeffs"]
            psi = _wrap_pi(N * phi_c - m[:, 0] * phi_b[0]
                           - m[:, 1] * phi_b[1])
            out["superlattice_denominator"] = N
            out["registry_phases_deg"] = [round(float(np.degrees(p)), 2)
                                          for p in psi]
            out["registry_mirror_residuals_deg"] = [
                round(float(np.degrees(_wrap_pi(psi[(i + 1) % 3]
                                                + psi[(i + 2) % 3]))), 2)
                for i in range(3)]
            out["registry_note"] = (
                "psi_i = N phi_C,i - m_i phi_B1 - n_i phi_B2 is exactly "
                "invariant under translation of the field of view and under "
                "any real positive tip transfer function, so it is untouched "
                "by an elliptical tip, by contrast stretch, and by affine "
                "drift. The mirror through direction i is a symmetry only if "
                "the i-th residual above vanishes AND the other two "
                "amplitudes are equal.")

            wind = _registry_winding(q_c, q_b, commensurate, g.shape)
            out["registry_winding"] = wind
            w_cycles = wind.get("winding_cycles")
            out["registry_is_commensurate"] = (
                None if w_cycles is None
                else bool(w_cycles <= REGISTRY_WINDING_NOTE))
            if w_cycles is not None and w_cycles > REGISTRY_WINDING_NOTE:
                out["notes"].append(
                    f"NEAR-COMMENSURATE, NOT COMMENSURATE: the registry "
                    f"phase winds through {w_cycles:.2f} cycles across this "
                    f"field of view, because the measured CDW wavevectors "
                    f"miss exact commensuration by "
                    f"{wind['residual_q_cycles_per_px']:.2e} cycles/px. The "
                    "frame therefore contains a phason texture - "
                    "commensurate patches separated by discommensurations - "
                    "and psi_i is an average over a registry that varies "
                    "within it, not the registry of the crystal. A mirror "
                    "residual built on that depends on where the "
                    "discommensurations sit relative to the window and is "
                    "produced by an achiral texture as readily as by a "
                    "chiral one."
                    + (f" Above {REGISTRY_WINDING_MAX:g} cycles the phase "
                       "channel may not establish chirality; this map is "
                       "above that limit."
                       if w_cycles > REGISTRY_WINDING_MAX else ""))

            out["registry_usable"] = bool(
                w_cycles is None or w_cycles <= REGISTRY_WINDING_MAX)
            if not out["registry_usable"]:
                commensurate = None

    if commensurate is not None:
        basis = q_c[:2]
        coords = np.round(_coords(peaks_q, basis)).astype(int)
        if np.abs(_coords(peaks_q, basis) - coords).max() > COMMENSURATE_TOL:
            coords = None
    else:
        peaks_q, amps, sigs = q_c, c_c, sig_c
        families = np.zeros(3, dtype=int)
        coords = np.array([[1, 0], [0, 1], [-1, 1]])

    if q_b is not None:
        A, affine = _deshear(q_b, other=q_c)
    else:
        A, affine = _deshear(q_c)
    out["affine_correction"] = affine

    cross = affine.get("cross_family_residual_frac")
    aniso_ratio = float(affine.get("affine_anisotropy") or 1.0)
    orient_ok = bool(affine.get("orientation_preserving", True))
    out["hexagonal_star_supported"] = bool(
        orient_ok
        and aniso_ratio <= AFFINE_ANISOTROPY_MAX
        and (cross is None or cross <= HEX_RESIDUAL_MAX))

    if not orient_ok:
        out.update(
            verdict="indeterminate", headline="Indeterminate",
            reason=("The affine map that de-shears the measured star has a "
                    f"negative determinant ({affine.get('determinant')}), "
                    "i.e. it is a REFLECTION rather than a scan distortion. "
                    "Applying it would invert every chirality sign this "
                    "module reports, so no verdict is issued. This "
                    "indicates a degenerate or mis-ordered peak triple."))
        return out

    if cross is not None and cross > HEX_RESIDUAL_NOTE:
        out["notes"].append(
            f"The CDW star departs from a hexagon by {cross:.1%} of |q| "
            "after the affine map that de-shears the LATTICE star, so the "
            "two are not one hexagonal lattice and one hexagonal "
            "superlattice under a single scan distortion. A mixed-domain "
            "field of view, a mis-assigned anchor and genuinely "
            "non-hexagonal order all look like this, and the mirror axes "
            "below are correspondingly approximate.")
    if aniso_ratio > AFFINE_ANISOTROPY_NOTE:
        out["notes"].append(
            f"De-shearing this star onto a hexagon required an anisotropic "
            f"stretch of {aniso_ratio:.2f}:1, which is large for a scan "
            "distortion (drift, creep and aspect error are typically a few "
            "percent to ~20%). The star may not be a distorted hexagonal "
            "star at all.")

    if not out["hexagonal_star_supported"]:
        out.update(
            verdict="indeterminate", headline="Indeterminate",
            reason=(
                "The measured peaks are not a hexagonal star under a "
                "plausible scan distortion"
                + (f" (CDW star {cross:.1%} from a hexagon after de-shearing "
                   f"the lattice; limit {HEX_RESIDUAL_MAX:.0%})"
                   if cross is not None and cross > HEX_RESIDUAL_MAX else "")
                + (f" (anisotropic stretch {aniso_ratio:.2f}:1; limit "
                   f"{AFFINE_ANISOTROPY_MAX:g}:1)"
                   if aniso_ratio > AFFINE_ANISOTROPY_MAX else "")
                + ". The six mirror axes of a hexagonal star are not "
                  "symmetry axes of this geometry, so a residual measured "
                  "against them is not a test of mirror symmetry."))
        return out

    peaks_q_meas = np.array(peaks_q, dtype=np.float64, copy=True)
    peaks_q = peaks_q @ A.T
    a0 = affine["ideal_orientation_deg"]
    axes = [a0 + 30.0 * k for k in range(6)]
    tol = _mirror_tolerances(peaks_q)
    out["mirror_tolerance_cycles_per_px"] = [round(float(t), 8) for t in tol]

    if coords is None:
        out.update(verdict="indeterminate", headline="Indeterminate",
                   reason=("The measured peaks are not integer combinations "
                           "of the CDW basis, so the translation search "
                           "underlying the mirror test is not well posed."))
        return out

    ch = _chirality(peaks_q, amps, coords, axes, families, tol)
    if not ch.get("available"):
        out.update(verdict="indeterminate", headline="Indeterminate",
                   reason=("No mirror axis of the measured star could be "
                           "matched onto the peak set."))
        return out
    n_req = int(ch.get("n_axes_requested") or len(axes))
    n_got = int(ch.get("n_axes_matched") or 0)
    out["n_mirror_axes_requested"] = n_req
    out["n_mirror_axes_matched"] = n_got
    if REQUIRE_ALL_MIRROR_AXES and 0 < n_got < n_req:
        out.update(
            verdict="indeterminate", headline="Indeterminate",
            mirror_test=ch,
            reason=(
                f"Only {n_got} of the measured star's {n_req} mirror axes "
                "could be matched onto the peak set. chirality_phase is the "
                "MINIMUM of the residual over those axes, so a minimum over "
                f"{n_got} of them is not the 'breaks every mirror' quantity "
                "the verdict would assert - it is a systematically larger "
                "statistic, and the axes that drop out are the ones whose "
                "reflected peaks land furthest from their partners, which "
                "on a distorted star are often the ones that would have "
                "given the smallest residual. A clean geometry matches all "
                "six axes or none; a proper subset means the star is "
                "distorted or mis-assigned, and the surviving residuals "
                "cannot be read as mirror breaking."))
        return out

    out["mirror_test"] = ch
    for key in ("chirality_total", "chirality_amplitude", "chirality_phase",
                "nearest_mirror_axis_deg"):
        out[key] = ch[key]
    out["lattice_referenced"] = bool(commensurate is not None)

    ang_c = np.degrees(np.arctan2(q_c[:, 1], q_c[:, 0])) % 180.0
    order = np.argsort(-np.abs(c_c) ** 2)
    s1 = float((ang_c[order[1]] - ang_c[order[0]] + 90.0) % 180.0 - 90.0)
    out["amplitude_sense"] = ("counter-clockwise" if s1 > 0 else
                              "clockwise" if s1 < 0 else None)

    rng = np.random.default_rng(seed)
    obs = float(ch["chirality_phase"])
    axes_tested = [float(a["axis_deg"]) for a in ch["axes"]]
    axis_single = float(ch["nearest_mirror_axis_deg"])
    null_amp = np.abs(amps)
    draws = np.empty(n_draw)
    draws_single = np.empty(n_draw)
    plans = {b: _mirror_plan(peaks_q, b, tol) for b in axes_tested}
    plan_single = plans.get(axis_single)
    if plan_single is None:
        plan_single = _mirror_plan(peaks_q, axis_single, tol)
    live = [(b, plans[b]) for b in axes_tested if plans[b] is not None]

    repro = _phase_reproducibility(field, peaks_q_meas, peaks_q, coords,
                                   families, live, tol, obs, Phi_c)
    out["reproducibility"] = repro
    reproducible: Optional[bool] = repro.get("passed")
    if repro.get("not_applicable"):
        reproducible = None
    if reproducible is False:
        out["notes"].append(
            "Phase residual not reproducible across the field of view: "
            + str(repro.get("reason") or ""))
    elif reproducible is None and repro.get("reason"):
        out["notes"].append(
            "Phase reproducibility control UNRUN: "
            + str(repro.get("reason")))

    bragg_control: Optional[float] = None
    try:
        fam_ids = np.unique(families)
        if out.get("lattice_referenced") and fam_ids.size >= 2:
            bmask = families == fam_ids[0]
            if int(np.sum(bmask)) >= 3:
                ctrl = _chirality(peaks_q[bmask], amps[bmask],
                                  coords[bmask], axes_tested,
                                  families[bmask], tol)
                if ctrl.get("available"):
                    bragg_control = float(ctrl["chirality_phase"])
    except Exception:                                   # pragma: no cover
        bragg_control = None
    out["phase_residual_bragg_control"] = (
        None if bragg_control is None else round(bragg_control, 6))
    out["bragg_control_note"] = (
        "Phase mirror residual of the BRAGG family alone. The atomic "
        "lattice is achiral, so this number is entirely estimator floor "
        "(peak-position refinement, closure correction, leakage, "
        "discretization) measured on this map. It calibrates the null's "
        "phase jitter and sets one of the three significance gates: the CDW "
        f"residual must exceed it by {BRAGG_CONTROL_MARGIN:g}x. It is a "
        "lower bound on the floor - the translation search is freer for a "
        "single family - so it is used alongside the effect-size and "
        "null-margin gates, never alone.")

    sigma_psi = (0.0 if bragg_control is None
                 else float(NULL_JITTER_FROM_BRAGG_CONTROL * bragg_control))
    out["null_phase_jitter_rad"] = round(sigma_psi, 8)
    out["null_jitter_source"] = (
        "calibrated from the Bragg achiral control" if bragg_control
        else "no lattice reference; photometric noise only")

    for d in range(n_draw):
        noise = (rng.normal(0.0, sigs / math.sqrt(2.0))
                 + 1j * rng.normal(0.0, sigs / math.sqrt(2.0)))
        a_null = null_amp.astype(np.complex128) + noise
        if sigma_psi > 0:
            a_null = a_null * np.exp(
                1j * rng.normal(0.0, sigma_psi, size=a_null.shape))
        eq_null = _equalise(a_null, families)
        per = [_mirror_residual(peaks_q, eq_null, coords, b, tol, plan=pl)
               for b, pl in live]
        per = [r for r in per if r is not None]
        draws[d] = min(per) if per else np.nan
        r1 = (None if plan_single is None else
              _mirror_residual(peaks_q, eq_null, coords, axis_single, tol,
                               plan=plan_single))
        draws_single[d] = np.nan if r1 is None else r1
    out["null_axes_deg"] = [round(b, 2) for b in axes_tested]
    out["null_is_min_over_axes"] = True
    ds = draws_single[np.isfinite(draws_single)]
    if ds.size:
        out["p_value_phase_achiral_single_axis"] = round(
            float((np.sum(ds >= obs) + 1) / (ds.size + 1)), 5)
    draws = draws[np.isfinite(draws)]
    if draws.size:
        p = float((np.sum(draws >= obs) + 1) / (draws.size + 1))
        out["p_value_phase_achiral"] = round(p, 5)
        out["null_phase_residual_p95"] = round(float(np.percentile(draws, 95)),
                                               5)
        out["p_value_note"] = (
            "The null is the minimum of the phase residual over the same "
            f"{len(axes_tested)} mirror axes the observed statistic "
            "minimises over, so the selection over axes is inside the null "
            "rather than only in the observation. "
            "`p_value_phase_achiral_single_axis` reproduces the older, "
            "uncalibrated single-axis comparison for audit; the VERDICT uses "
            "the more conservative of the two (`p_value_used_for_verdict`), "
            "because the min-over-axes null is the calibrated one but also "
            "the anti-conservative one. The null now carries the "
            "registry-phase jitter implied by this map's peak-position error "
            "(`null_phase_jitter_rad`) in addition to photometric noise, but "
            "it still models no tip, drift or scan-artifact mechanism, so it "
            "remains a floor on the evidence rather than a false-positive "
            "rate. That is why the verdict is gated on effect size and on "
            "the Bragg achiral control as well as on p - see "
            "`significance_gates`.")
    else:
        p = float("nan")
        out["p_value_phase_achiral"] = None

    amp_break = float(ch["chirality_amplitude"])
    phase_break = obs

    p_single = out.get("p_value_phase_achiral_single_axis")
    p_used = p
    if REQUIRE_CONSERVATIVE_P and isinstance(p_single, float) \
            and np.isfinite(p_single):
        p_used = max(float(p), float(p_single)) if np.isfinite(p) else p
    out["p_value_used_for_verdict"] = (
        None if not np.isfinite(p_used) else round(float(p_used), 5))

    null_p95 = out.get("null_phase_residual_p95")
    floor_terms: Dict[str, Optional[float]] = {
        "effect_size_floor": PHASE_EFFECT_FLOOR,
        "null_p95_margin": (None if null_p95 is None
                            else round(PHASE_NULL_MARGIN * float(null_p95), 6)),
        "bragg_control_margin": (
            None if bragg_control is None
            else round(BRAGG_CONTROL_MARGIN * bragg_control, 6)),
    }
    floor = max(v for v in floor_terms.values() if v is not None)

    _wind = _as_num(_dict_get(out.get("registry_winding"), "winding_cycles"))
    commensurate_registry: Optional[bool] = (
        None if _wind is None else bool(_wind <= REGISTRY_WINDING_MAX))

    gates = {
        "lattice_referenced": bool(out.get("lattice_referenced")),
        "commensurate_registry": (commensurate_registry,
                                  None if _wind is None else round(_wind, 4)),
        "reproducible": (reproducible,
                         repro.get("fraction_residual_reproduced")),
        "p_value": (bool(np.isfinite(p_used) and p_used < 0.05),
                    None if not np.isfinite(p_used) else round(float(p_used), 5)),
        "effect_size": (bool(phase_break >= PHASE_EFFECT_FLOOR),
                        round(phase_break, 6)),
        "null_margin": (
            None if floor_terms["null_p95_margin"] is None
            else bool(phase_break >= floor_terms["null_p95_margin"]),
            floor_terms["null_p95_margin"]),
        "bragg_control": (
            None if floor_terms["bragg_control_margin"] is None
            else bool(phase_break >= floor_terms["bragg_control_margin"]),
            floor_terms["bragg_control_margin"]),
    }
    out["significance_gates"] = gates
    out["phase_significance_floor"] = round(float(floor), 6)
    out["floor_terms"] = floor_terms

    significant = bool(
        out.get("lattice_referenced")
        and commensurate_registry is not False
        and reproducible is not False
        and np.isfinite(p_used) and p_used < 0.05
        and phase_break >= floor)

    leak_blocked = False
    _N = int(commensurate["N"]) if commensurate is not None else None
    if significant and _N is not None and _N >= LEAKAGE_CONTROL_MIN_N:
        c_corr, leak = _amplitude_leakage_correction(g, q_c, c_c, df)
        amps_corr = amps.astype(np.complex128).copy()
        amps_corr[families == 0] = c_corr
        corrected = _phase_min_over_axes(peaks_q, amps_corr, coords, families,
                                         live, tol)
        leak_passed = bool(
            corrected is not None and corrected >= floor
            and corrected >= LEAKAGE_MIN_SURVIVING_FRACTION * phase_break)
        leak.update(
            superlattice_denominator=_N,
            phase_residual_leakage_corrected=(
                None if corrected is None else round(corrected, 6)),
            surviving_fraction=(
                None if corrected is None or not phase_break > 0
                else round(corrected / phase_break, 4)),
            min_surviving_fraction=LEAKAGE_MIN_SURVIVING_FRACTION,
            passed=leak_passed,
            note=("Nonlinear contrast mixes CDW components onto one another "
                  "(q_j - q_k is a first-order peak), so an achiral "
                  "amplitude hierarchy rotates the registry phases. The "
                  "mixing term on each C_i follows from one real mixing "
                  "coefficient fitted to the partner peaks at q_j + q_k; it "
                  "is removed and the phase residual recomputed on the "
                  "corrected amplitudes."))
        out["amplitude_leakage"] = leak
        gates["amplitude_leakage"] = (leak_passed,
                                      leak["phase_residual_leakage_corrected"])
        if not leak_passed:
            leak_blocked = True
            significant = False

    underpowered = bool(not significant
                        and floor > PHASE_EFFECT_FLOOR
                        and phase_break < floor)

    if significant:
        out.update(
            verdict="phase_chiral",
            headline="Chirality corroborated in the phase channel",
            reason=(f"The CDW-to-lattice registry breaks every mirror of the "
                    f"measured star (phase residual {phase_break:.5f}, above "
                    f"the {floor:.5f} floor set by "
                    f"{max(floor_terms, key=lambda k: floor_terms[k] or -1.0)}"
                    f"; p = {p_used:.3g} against an achiral registry at this "
                    f"SNR, with the null carrying the "
                    f"{out.get('null_phase_jitter_rad'):.2e} rad "
                    f"registry-phase jitter calibrated from this map's Bragg "
                    f"achiral control). A real multiplicative tip response "
                    "and an affine drift distortion leave the registry "
                    "phases exactly unchanged, so this is mirror breaking "
                    "those artifacts cannot produce."))
    elif leak_blocked:
        _lk = out["amplitude_leakage"]
        out.update(
            verdict="indeterminate",
            headline="Phase residual explained by amplitude-to-phase leakage",
            reason=(
                f"The phase residual is {phase_break:.5f}, but removing the "
                f"registry-phase pull that nonlinear mixing of the unequal "
                f"CDW amplitudes produces (up to "
                f"{max(abs(v) for v in _lk['phase_pull_removed_rad']):.3f} "
                f"rad, from one mixing coefficient fitted to the q_j + q_k "
                f"peaks) brings it "
                f"to "
                f"{_lk['phase_residual_leakage_corrected']} "
                f"({(_lk['surviving_fraction'] or 0.0):.0%} of it survives; "
                f"the claim "
                f"needs {LEAKAGE_MIN_SURVIVING_FRACTION:.0%} and at least "
                f"the {floor:.5f} floor). On this N = {_N} registry an "
                "achiral intensity hierarchy - strain or the tip, a nematic "
                "director - is enough to produce the residual, so it is not "
                "evidence of a broken mirror. Reported as untested, not as "
                "achiral."))
        out["notes"].append(
            "Phase channel not conclusive: the residual lies within the "
            "amplitude-to-phase leakage bound measured on this map.")
    elif reproducible is False:
        out.update(
            verdict="indeterminate",
            headline="Phase residual does not reproduce across the map",
            reason=(
                f"The full-window phase residual is {phase_break:.5f}, but "
                "it does not survive re-windowing: "
                + str(repro.get("reason") or "")
                + " A broken mirror is a property of the crystal and "
                  "reproduces in every sub-window, in the residual AND in "
                  "the sign of Phi. A residual that appears in one framing "
                  "of the same data and not in another is a property of the "
                  "window - a phason texture, a domain boundary inside the "
                  "field of view, or an edge effect - so it is reported as "
                  "untested rather than as chirality. This is the control "
                  "didv_handedness already applies to the amplitude "
                  "channel."))
    elif commensurate_registry is False:
        out.update(
            verdict="indeterminate",
            headline="Registry winds across the field of view",
            reason=(
                f"The registry phase advances through {_wind:.2f} cycles "
                f"across this field of view (limit "
                f"{REGISTRY_WINDING_MAX:g}), because the measured CDW "
                "wavevectors are only NEARLY commensurate with the lattice. "
                "The frame therefore contains a phason texture rather than "
                "a single registry, psi_i is an average over a quantity "
                f"that varies within it, and the {phase_break:.5f} mirror "
                "residual measured against it is a property of where the "
                "discommensurations sit relative to the window - it changes "
                "when the window moves, which a broken symmetry does not. "
                "An achiral texture produces such a residual as readily as "
                "a chiral one, so this is neither evidence for chirality "
                "nor against it. A smaller field of view inside one "
                "commensurate patch, or a map of the commensurate phase, "
                "would be testable."))
        out["notes"].append(
            "Phase channel not applicable: near-commensurate registry with "
            f"{_wind:.2f} cycles of winding across the frame.")
    elif underpowered:
        out.update(
            verdict="indeterminate",
            headline="Phase channel could not be tested on this map",
            reason=(f"The registry phase residual is {phase_break:.5f}, below "
                    f"the {floor:.5f} floor this map's own systematics set "
                    f"(effect-size floor {PHASE_EFFECT_FLOOR:.3f}; "
                    f"{PHASE_NULL_MARGIN:g}x null p95 = "
                    f"{floor_terms['null_p95_margin']}; "
                    f"{BRAGG_CONTROL_MARGIN:g}x the Bragg achiral control = "
                    f"{floor_terms['bragg_control_margin']}). The test "
                    "therefore had no power here, and this is NOT evidence "
                    "that the state is mirror-symmetric - it is absence of a "
                    "usable measurement. Reported as untested."))
        out["notes"].append(
            "Phase channel underpowered: the estimator floor on this map "
            "exceeds the residual, so neither chirality nor achirality can "
            "be asserted from it.")
    elif amp_break > 0.05:
        out.update(
            verdict="amplitude_only",
            headline="Chirality not corroborated in the phase channel",
            reason=(f"The intensity hierarchy breaks the mirrors (amplitude "
                    f"residual {amp_break:.5f}) but the phase structure does "
                    f"not (phase residual {phase_break:.5f}, floor "
                    f"{floor:.5f}"
                    + (f", p = {p_used:.3g}" if np.isfinite(p_used) else "")
                    + "). Uniaxial strain, an asymmetric tip and aperture "
                      "truncation under drift all produce exactly this "
                      "signature, so the amplitude claim stands on its own "
                      "evidence and gains no independent support here. Note "
                      "that the amplitude residual is itself a nematic "
                      "director and is achiral - see section 0 of "
                      "didv_handedness."))
        if not out.get("lattice_referenced"):
            out["notes"].append(
                "No usable lattice reference, so the phase channel could only "
                "be tested within the CDW family, where it carries no "
                "chirality information. Treat 'not corroborated' here as "
                "'not tested' rather than as evidence against.")
    else:
        out.update(
            verdict="achiral",
            headline="Consistent with a mirror-symmetric 3Q state",
            reason=(f"Neither the intensity hierarchy ({amp_break:.5f}) nor "
                    f"the registry phases ({phase_break:.5f}, floor "
                    f"{floor:.5f}) break the mirror at "
                    f"{ch['nearest_mirror_axis_deg']:.1f} degrees. The "
                    "estimator floor here is below the effect size that "
                    "would count as physical, so the test had power and "
                    "this is evidence for a mirror-symmetric 3Q state "
                    "rather than a missing measurement."))

    out["corroboration"] = {
        "amplitude_claim": out.get("amplitude_sense"),
        "phase_supports": bool(significant),
        "intended_use": (
            "REPORTED, NOT SCORED. This module writes no label and feeds no "
            "threshold: didv_alpha_crit, didv_handedness and chiral_metrology "
            "are untouched and their numbers are identical with or without "
            "it. It is the independent cross-check a handedness read off "
            "three intensities does not otherwise have."),
    }
    return out


def _amplitude_leakage_correction(g: np.ndarray, q_c: np.ndarray,
                                  c_c: np.ndarray, df: float
                                  ) -> Tuple[np.ndarray, Dict[str, Any]]:
    q_c = np.asarray(q_c, dtype=np.float64)
    c = np.asarray(c_c, dtype=np.complex128)
    partners = np.stack([q_c[1] + q_c[2], q_c[0] - q_c[2], q_c[0] + q_c[1]])
    products = np.array([c[1] * c[2], c[0] * np.conj(c[2]), c[0] * c[1]])
    leak_basis = np.array([c[1] * np.conj(c[2]), c[0] * c[2],
                           c[1] * np.conj(c[0])])
    f_s = np.array([complex(_project(g, _refine(g, s, df))[0])
                    for s in partners])
    denom = float(np.sum(np.abs(products) ** 2))
    k = (float(np.real(np.sum(np.conj(products) * f_s))) / denom
         if denom > 0 else 0.0)
    fit_resid = (float(np.sqrt(np.sum(np.abs(f_s - k * products) ** 2)
                               / max(float(np.sum(np.abs(f_s) ** 2)), 1e-300)))
                 if denom > 0 else None)
    leak = k * leak_basis
    corrected = c - leak
    info = {
        "mixing_wavevectors_cycles_per_px": [
            [round(float(v), 6) for v in s] for s in partners],
        "mixing_coefficient": round(k, 8),
        "mixing_fit_residual_frac": (None if fit_resid is None
                                     else round(fit_resid, 4)),
        "leakage_amplitude_ratio": [
            round(float(abs(l) / abs(ci)), 4) if abs(ci) > 0 else None
            for l, ci in zip(leak, c)],
        "phase_pull_removed_rad": [
            round(float(_wrap_pi(np.angle(cc) - np.angle(ci))), 5)
            for cc, ci in zip(corrected, c)],
    }
    return corrected, info


def _equalise(amps: np.ndarray, families: np.ndarray) -> np.ndarray:
    eq = amps.astype(np.complex128).copy()
    for f in np.unique(families):
        m = families == f
        gm = float(np.exp(np.mean(np.log(np.maximum(np.abs(amps[m]),
                                                    1e-300)))))
        eq[m] = gm * np.exp(1j * np.angle(amps[m]))
    return eq


def format_phase_chirality_report(d: Dict[str, Any]) -> str:
    L = ["=" * 72,
         "dI/dV PHASE CHIRALITY  (mirror test on the complex 3Q structure "
         "factor)",
         "=" * 72,
         f"Verdict : {d.get('headline', d.get('verdict', '?'))}",
         f"Reason  : {d.get('reason', '-')}"]
    if d.get("mirror_test", {}).get("available"):
        L.append(f"Mirror residuals (0 = a mirror exists, so achiral):")
        L.append(f"  amplitude channel : {d['chirality_amplitude']:.6f}"
                 "   (a nematic director - achiral)")
        L.append(f"  phase channel     : {d['chirality_phase']:.6f}"
                 + (f"   (p = {d['p_value_phase_achiral']:.3g} vs achiral)"
                    if d.get("p_value_phase_achiral") is not None else ""))
        L.append(f"  full complex      : {d['chirality_total']:.6f}")
        L.append(f"  nearest mirror at : "
                 f"{d['nearest_mirror_axis_deg']:.1f} deg")
    if d.get("phase_significance_floor") is not None:
        ft = d.get("floor_terms") or {}
        L.append(f"Significance floor  : {d['phase_significance_floor']:.6f}"
                 f"   (effect size {ft.get('effect_size_floor')}, "
                 f"{ft.get('null_p95_margin')} from the null, "
                 f"{ft.get('bragg_control_margin')} from the Bragg control)")
        if d.get("phase_residual_bragg_control") is not None:
            L.append(f"  Bragg achiral ctrl: "
                     f"{d['phase_residual_bragg_control']:.6f}   (lattice is "
                     "achiral, so this is pure estimator floor)")
        if d.get("null_phase_jitter_rad") is not None:
            L.append(f"  null jitter       : "
                     f"{d['null_phase_jitter_rad']:.2e} rad "
                     f"({d.get('null_jitter_source')})")
        gates = d.get("significance_gates") or {}
        if gates:
            def _g(v):
                s = v[0] if isinstance(v, (tuple, list)) else v
                return {True: "PASS", False: "FAIL", None: "n/a"}.get(s, "n/a")
            L.append("  gates             : "
                     + ", ".join(f"{k}={_g(v)}" for k, v in gates.items()))
    if d.get("registry_phases_deg"):
        L.append("Registry psi_i (deg) : "
                 + ", ".join(f"{v:+.1f}" for v in d["registry_phases_deg"]))
        L.append("  mirror residuals   : "
                 + ", ".join(f"{v:+.1f}"
                             for v in d["registry_mirror_residuals_deg"]))
    lk = d.get("amplitude_leakage") or {}
    if lk:
        L.append(f"Amplitude leakage    : "
                 f"{'PASS' if lk.get('passed') else 'FAIL'}   residual after "
                 f"removing mixing pull = "
                 f"{lk.get('phase_residual_leakage_corrected')}   (max pull "
                 f"{max([abs(v) for v in lk.get('phase_pull_removed_rad') or [0.0]]):.3f} rad, "
                 f"N = {lk.get('superlattice_denominator')})")
    rp = d.get("reproducibility") or {}
    if rp.get("reason"):
        state = {True: "PASS", False: "FAIL", None: "not run"}.get(
            rp.get("passed"), "?")
        L.append(f"Reproducibility      : {state}"
                 + (f"   residual in {rp['fraction_residual_reproduced']:.0%} "
                    f"of {rp.get('n_tiles')} sub-windows, sign of Phi in "
                    f"{rp['fraction_sign_reproduced']:.0%}"
                    if rp.get("fraction_residual_reproduced") is not None
                    else ""))
        if rp.get("tile_phi_deg"):
            L.append("  per-tile Phi (deg) : "
                     + ", ".join("n/a" if v is None else f"{v:+.1f}"
                                 for v in rp["tile_phi_deg"]))
    if d.get("n_mirror_axes_requested") is not None:
        L.append(f"Mirror axes matched  : {d.get('n_mirror_axes_matched')}"
                 f"/{d.get('n_mirror_axes_requested')}")
    rw = d.get("registry_winding") or {}
    if rw.get("winding_cycles") is not None:
        w = float(rw["winding_cycles"])
        state = ("commensurate" if w <= REGISTRY_WINDING_NOTE
                 else "NEAR-commensurate" if w <= REGISTRY_WINDING_MAX
                 else "NEAR-commensurate, registry unusable")
        L.append(f"Registry winding     : {w:.3f} cycles across the "
                 f"{rw.get('field_of_view_px')} px field of view   "
                 f"({state})")
        L.append(f"  residual |dq|      : "
                 f"{rw.get('residual_q_cycles_per_px')} cycles/px "
                 f"(commensuration coord error "
                 f"{rw.get('commensuration_max_error')}, selection "
                 f"tolerance {COMMENSURATE_TOL:g})")
    af = d.get("affine_correction") or {}
    if af:
        L.append(f"Affine de-shear      : anisotropy "
                 f"{af.get('affine_anisotropy')}:1, det "
                 f"{af.get('determinant')}"
                 + (f", CDW star {af['cross_family_residual_frac']:.2%} from "
                    "a hexagon after de-shearing the lattice"
                    if af.get("cross_family_residual_frac") is not None
                    else ", no second family to cross-check against"))
    if d.get("phase_sum_cdw_deg") is not None:
        L.append(f"Phi_CDW : {d['phase_sum_cdw_deg']:+.1f} deg"
                 + (f"   Phi_Bragg : {d['phase_sum_bragg_deg']:+.1f} deg"
                    if d.get("phase_sum_bragg_deg") is not None else ""))
    if d.get("peak_snr"):
        L.append("Peak |F|/sigma : "
                 + ", ".join("n/a" if s is None else f"{s:.0f}"
                             for s in d["peak_snr"]))
    if d.get("amplitude_sense"):
        L.append(f"Amplitude sense (this frame) : {d['amplitude_sense']}")
    for n in d.get("notes", []):
        L.append(f"  ! {n}")
    L.append(f"Frame   : {d.get('frame_note', '')}")
    L.append(f"Caveat  : {d.get('caveat', '')}")
    L.append("=" * 72)
    return "\n".join(L)


def didv_phase_chirality_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        res = state.get("didv_anisotropy") or None
        didv_b64 = state.get("input_didv_base64")
        field, info = None, {}
        if didv_b64:
            meta = state.get("metadata") or {}
            from stm_data_io import normalise_extension
            ext = normalise_extension(meta.get("didv_ext"))
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "didv" + ext)
                with open(path, "wb") as fh:
                    fh.write(base64.b64decode(didv_b64))
                field, info = load_didv_scalar(
                    path, colormap=meta.get("didv_colormap"),
                    channel=meta.get("didv_channel"),
                    direction=meta.get("didv_direction", "forward"),
                    bias=meta.get("didv_bias"))
        d = assess_phase_chirality(res, field)
        if field is not None:
            d["data_class"] = (info or {}).get("data_class", "rendered")
        if field is not None and str(
                (info or {}).get("method")) == "luminance_fallback":
            d.setdefault("notes", []).append(
                "The scalar field came from a luminance fallback rather than "
                "an identified colormap. Luminance is a monotone-per-channel "
                "remapping, so it distorts magnitudes more than phases, but "
                "a non-monotonic colormap folds the range and corrupts both.")
        d["report"] = format_phase_chirality_report(d)
        state["didv_phase_chirality"] = d
    except Exception as exc:                             # pragma: no cover
        state["didv_phase_chirality"] = {
            "verdict": "no_measurement", "headline": "No measurement",
            "reason": f"phase-chirality assessment failed: {exc}",
            "caveat": _CAVEAT, "frame_note": _FRAME_NOTE, "notes": [],
        }
    return state
