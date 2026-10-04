from __future__ import annotations

import math
import os
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

from scipy.spatial import cKDTree
from scipy.special import gammaincc as _gammaincc

ESTIMATOR_VERSION = 2

_LN2 = math.log(2.0)

_BG_CLIP_Q = 0.995

_WIDTH_DISC_FACTOR = 2.2
_WIDTH_CONVERGE_TOL = 0.03
_WIDTH_MAX_EXPANSIONS = 5

_APERTURE_MAX_PASSES = 3

APERTURE_MAX_BINS = 12.0

SEPARATION_FRACTION_RECOMMENDED = 0.45


import stm_data_io as _sdio
SUPPORTED_EXTENSIONS = set(_sdio.SUPPORTED_EXTENSIONS)

PEAK_LABEL_FONTSIZE = 17
PEAK_LABEL_FAMILY_PREFERENCE = ("Arial", "Liberation Sans", "Arimo",
                                "Helvetica", "Nimbus Sans", "DejaVu Sans")

FFT_VIEW_MAX_ORDER = 2
FFT_VIEW_MARGIN = 1.0
def _peak_label_family() -> str:
    try:
        from matplotlib import font_manager

        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in PEAK_LABEL_FAMILY_PREFERENCE:
            if name in installed:
                return name
    except Exception:
        pass
    return "sans-serif"

ANGLE_FRAME = "display (x right, y up)"


def _display_angle_deg(u: float, v: float) -> float:
    return float(np.degrees(np.arctan2(-v, u)) % 180.0)


CDW_MODELS: Dict[str, Tuple[float, float]] = {
    "2x2": (1.0 / 2.0, 0.0),
    "3x3": (1.0 / 3.0, 0.0),
    "4x4": (1.0 / 4.0, 0.0),
    "sqrt3xsqrt3": (1.0 / np.sqrt(3.0), 30.0),
    "sqrt13xsqrt13": (1.0 / np.sqrt(13.0), 13.898),
}

MIRROR_FIXED_POINT_PERIOD_DEG = 30.0

ROTATION_DEGENERACY_TOL_DEG = 0.05


def _wrap_signed(x: float, period: float) -> float:
    return (float(x) + 0.5 * period) % period - 0.5 * period


def _rotation_candidates(rot: float,
                         bragg_dirs: Optional[Sequence[Any]] = None,
                         ratio: Optional[float] = None,
                         df: Optional[float] = None
                         ) -> Tuple[List[float], Dict[str, Any]]:
    rot = float(rot)
    info: Dict[str, Any] = {"nominal_rotation_deg": round(rot, 4)}

    if abs(_wrap_signed(rot, MIRROR_FIXED_POINT_PERIOD_DEG)) \
            <= ROTATION_DEGENERACY_TOL_DEG:
        info.update(
            two_variants=False, reason="mirror_fixed_point",
            detail=(
                f"A rotation of {rot:+.3f} deg is a multiple of "
                f"{MIRROR_FIXED_POINT_PERIOD_DEG:g} deg, so -rot and +rot "
                "describe the SAME hexagonal star and the cell is its own "
                "mirror image. One candidate is evaluated; there is no "
                "second variant to choose between."))
        return [rot], info

    sep_bins = None
    if bragg_dirs and ratio and df and float(df) > 0:
        try:
            seps = []
            for r_b, _ang in bragg_dirs:
                r_c = float(r_b) * float(ratio)
                seps.append(2.0 * r_c
                            * abs(float(np.sin(np.radians(rot))))
                            / float(df))
            sep_bins = float(max(seps)) if seps else None
        except Exception:                                # pragma: no cover
            sep_bins = None

    if sep_bins is not None:
        info["separation_bins"] = round(sep_bins, 3)
        if sep_bins < 1.0:
            info.update(
                two_variants=False, reason="unresolvable",
                detail=(
                    f"The +{rot:.3f} and -{rot:.3f} deg stars are formally "
                    f"distinct but their predicted CDW positions differ by "
                    f"only {sep_bins:.2f} frequency bins, which this map "
                    "cannot resolve. One candidate is evaluated rather than "
                    "letting the model sort break the tie on noise."))
            return [rot], info

    info.update(
        two_variants=True, reason="enantiomorphic",
        detail=(
            f"A rotation of {rot:+.3f} deg is not a multiple of "
            f"{MIRROR_FIXED_POINT_PERIOD_DEG:g} deg, so +rot and -rot are "
            "mirror-related but DISTINCT stars. Both are evaluated and the "
            "data chooses."))
    return [rot, -rot], info

_CMAP_CANDIDATES = (
    "viridis", "plasma", "inferno", "magma", "cividis", "jet", "turbo",
    "hot", "afmhot", "gist_heat", "gray", "bone", "copper", "cubehelix",
    "coolwarm", "seismic", "rainbow", "nipy_spectral", "terrain",
    "Blues", "Greens", "Reds", "Oranges", "Purples",
    "BuGn", "GnBu", "YlGnBu", "YlOrRd", "RdYlBu",
)


def _cmap_lut(name: str, n: int) -> np.ndarray:
    import matplotlib
    try:
        cmap = matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        from matplotlib import cm
        cmap = cm.get_cmap(name)
    return np.asarray(cmap(np.linspace(0.0, 1.0, n)))[:, :3] * 255.0


def _fit_colormap(rgb: np.ndarray, n_sample: int = 20000,
                  lut_size: int = 256) -> Tuple[Optional[str], float]:
    flat = rgb.reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(0)
    idx = rng.choice(flat.shape[0], size=min(n_sample, flat.shape[0]),
                     replace=False)
    pts = flat[idx]

    best_name, best_resid = None, np.inf
    for name in _CMAP_CANDIDATES:
        try:
            lut = _cmap_lut(name, lut_size)
        except Exception:
            continue
        dist, _ = cKDTree(lut).query(pts, k=1)
        resid = float(np.median(dist))
        if resid < best_resid:
            best_name, best_resid = name, resid
    return best_name, best_resid


def _invert_colormap(rgb: np.ndarray, name: str, lut_size: int = 1024
                     ) -> np.ndarray:
    from scipy.spatial import cKDTree
    lut = _cmap_lut(name, lut_size)
    flat = rgb.reshape(-1, 3).astype(np.float64)
    _, idx = cKDTree(lut).query(flat, k=1)
    return (idx / (lut_size - 1.0)).reshape(rgb.shape[:2])


def load_didv_scalar(path: str, colormap: Optional[str] = None,
                     colormap_resid_tol: float = 20.0,
                     channel: Optional[str] = None,
                     direction: str = "forward",
                     bias: Optional[float] = None
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _sdio.load_field(
        path, colormap=colormap, colormap_resid_tol=colormap_resid_tol,
        channel=channel, direction=direction, prefer="didv", bias=bias,
        image_loader=_load_rendered_didv_scalar)


def _load_rendered_didv_scalar(path: str, colormap: Optional[str] = None,
                               colormap_resid_tol: float = 20.0
                               ) -> Tuple[np.ndarray, Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    image_exts = (set(_sdio.IMAGE_EXTENSIONS) if _sdio is not None
                  else SUPPORTED_EXTENSIONS)
    if ext not in image_exts:
        raise ValueError(f"Unsupported extension {ext!r} for the "
                         f"rendered-image loader; image formats: "
                         f"{sorted(image_exts)}. Native data formats are "
                         f"handled by load_didv_scalar.")

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
        lum = np.asarray(im.convert("F"), dtype=np.float64) / 255.0

    info: Dict[str, Any] = {"requested_colormap": colormap}

    if colormap is not None:
        try:
            field = _invert_colormap(rgb, colormap)
            info.update(method="colormap_inversion", colormap=colormap,
                        colormap_residual=None,
                        note="colormap supplied by caller")
            return field, info
        except Exception as exc:                       # pragma: no cover
            info["colormap_error"] = str(exc)

    best, resid = _fit_colormap(rgb)
    info.update(best_colormap_fit=best,
                colormap_residual=(None if not np.isfinite(resid)
                                   else round(float(resid), 2)))

    if best is not None and resid <= colormap_resid_tol:
        info.update(method="colormap_inversion", colormap=best,
                    note=f"colormap identified as {best!r} "
                         f"(median RGB residual {resid:.1f}/255)")
        return _invert_colormap(rgb, best), info

    info.update(method="luminance_fallback", colormap=None,
                note="no colormap matched within tolerance; using luminance. "
                     "Relative peak intensities may be distorted, especially "
                     "for non-monotonic colormaps such as jet.")
    return lum, info


def _sanitize(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    finite = np.isfinite(img)
    if finite.all():
        return img
    if not finite.any():
        return np.zeros_like(img)
    from scipy.ndimage import distance_transform_edt
    nearest = distance_transform_edt(~finite, return_distances=False,
                                     return_indices=True)
    return img[tuple(nearest)]


def _remove_plane(img: np.ndarray) -> np.ndarray:
    Ny, Nx = img.shape
    y, x = np.mgrid[0:Ny, 0:Nx]
    A = np.column_stack([np.ones(img.size), x.ravel(), y.ravel()])
    coef, *_ = np.linalg.lstsq(A, img.ravel(), rcond=None)
    return img - (A @ coef).reshape(Ny, Nx)


def power_spectrum(img: np.ndarray, remove_plane: bool = True
                   ) -> Dict[str, Any]:
    img = _sanitize(img)
    if remove_plane:
        img = _remove_plane(img)

    Ny, Nx = img.shape
    win = np.outer(np.hanning(Ny), np.hanning(Nx))
    wsum = win.sum()
    wmean = float((img * win).sum() / wsum) if wsum > 0 else float(img.mean())
    windowed = (img - wmean) * win

    fft = np.fft.fftshift(np.fft.fft2(windowed))
    return {
        "power": np.abs(fft) ** 2,
        "shape": (Ny, Nx),
        "center": (Ny // 2, Nx // 2),
        "window": win,
        "fx": np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0)),
        "fy": np.fft.fftshift(np.fft.fftfreq(Ny, d=1.0)),
    }


def _rho_kernel(window: np.ndarray, half: int = 4) -> np.ndarray:
    w2 = np.asarray(window, dtype=np.float64) ** 2
    W2 = np.fft.fft2(w2)
    dc = W2.flat[0]
    if not np.isfinite(dc) or dc == 0:
        k = np.zeros((2 * half + 1, 2 * half + 1))
        k[half, half] = 1.0
        return k
    rho = np.fft.fftshift(np.abs(W2 / dc) ** 2)
    cy, cx = rho.shape[0] // 2, rho.shape[1] // 2
    h = min(half, cy, cx)
    out = np.zeros((2 * half + 1, 2 * half + 1))
    out[half - h:half + h + 1, half - h:half + h + 1] = \
        rho[cy - h:cy + h + 1, cx - h:cx + h + 1]
    out[half, half] = 1.0
    return out


def _pair_correlation_sum(mask: np.ndarray, rho: np.ndarray) -> float:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return 0.0
    half = rho.shape[0] // 2
    y0, x0 = idx.min(axis=0) - half
    y1, x1 = idx.max(axis=0) + half + 1
    y0, x0 = max(int(y0), 0), max(int(x0), 0)
    y1 = min(int(y1), mask.shape[0])
    x1 = min(int(x1), mask.shape[1])
    sub = mask[y0:y1, x0:x1]

    total = 0.0
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            r = float(rho[dy + half, dx + half])
            if r == 0.0:
                continue
            shifted = np.roll(np.roll(sub, dy, axis=0), dx, axis=1)
            if dy > 0:
                shifted[:dy, :] = False
            elif dy < 0:
                shifted[dy:, :] = False
            if dx > 0:
                shifted[:, :dx] = False
            elif dx < 0:
                shifted[:, dx:] = False
            total += r * float(np.count_nonzero(shifted & sub))
    return total


def _background_stats(values: np.ndarray) -> Optional[Dict[str, float]]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 12:
        return None

    med = float(np.median(v))
    if not (med > 0):
        mean_raw = float(v.mean())
        if not (mean_raw > 0):
            return None
        return {"mean": mean_raw, "median": med, "sigma_pixel": mean_raw,
                "n": int(v.size), "exponentiality": float("nan"),
                "method": "raw_mean", "clip_correction": 1.0}

    t_over_mean = -math.log(1.0 - _BG_CLIP_Q)
    thresh = (med / _LN2) * t_over_mean
    kept = v[v <= thresh]
    if kept.size < 12:
        kept = v

    e_t = math.exp(-t_over_mean)
    corr = (1.0 - e_t) / (1.0 - (1.0 + t_over_mean) * e_t)
    mean_clipped = float(kept.mean()) * corr

    mean_from_median = med / _LN2
    ratio = (mean_clipped / mean_from_median
             if mean_from_median > 0 else float("nan"))

    return {
        "mean": mean_clipped,
        "median": med,
        "sigma_pixel": mean_clipped,
        "n": int(kept.size),
        "exponentiality": float(ratio),
        "method": "clipped_mean",
        "clip_correction": float(corr),
    }


@lru_cache(maxsize=8)
def hann_noise_model(Ny: int, Nx: int) -> Tuple[np.ndarray, float, float]:
    win = np.outer(np.hanning(Ny), np.hanning(Nx))
    rho = _rho_kernel(win)
    rho.flags.writeable = False
    return rho, float(rho.sum()), _window_response_width_bins(
        win, 1.0 / float(Nx), 1.0 / float(Ny))


def _r6(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, 6) if np.isfinite(f) else None


def _gamma_sf(x: float, shape: float, scale: float) -> float:
    if not (np.isfinite(x) and np.isfinite(shape) and np.isfinite(scale)) \
            or shape <= 0 or scale <= 0:
        return float("nan")
    if x <= 0:
        return 1.0
    return float(_gammaincc(shape, x / scale))


def _uv_grid(shape: Tuple[int, int], center: Tuple[int, int]
             ) -> Tuple[np.ndarray, np.ndarray]:
    Ny, Nx = shape
    cy, cx = center
    y, x = np.mgrid[0:Ny, 0:Nx]
    return (x - cx) / float(Nx), (y - cy) / float(Ny)


def _peak_width_bins(power: np.ndarray, u_grid: np.ndarray,
                     v_grid: np.ndarray, u0: float, v0: float,
                     background: float, df: float,
                     r_max_bins: float = 8.0) -> float:
    r_max = r_max_bins * df
    du = u_grid - u0
    dv = v_grid - v0
    rr2 = du ** 2 + dv ** 2
    sel = rr2 <= r_max ** 2
    if sel.sum() < 9:
        return float("nan")

    w = np.clip(power[sel] - background, 0.0, None)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        return float("nan")

    moment = float((w * rr2[sel]).sum() / total)
    return float(np.sqrt(max(moment, 0.0)) / df)


def _window_response_width_bins(window: np.ndarray, df_x: float, df_y: float,
                                r_max_bins: float = 8.0) -> float:
    W = np.fft.fftshift(np.fft.fft2(np.asarray(window, dtype=np.float64)))
    P = np.abs(W) ** 2
    Ny, Nx = P.shape
    cy, cx = Ny // 2, Nx // 2
    y, x = np.mgrid[0:Ny, 0:Nx]
    du = (x - cx) / float(Nx)
    dv = (y - cy) / float(Ny)
    rr2 = du ** 2 + dv ** 2
    r_max = r_max_bins * max(df_x, df_y)
    sel = rr2 <= r_max ** 2
    tot = float(P[sel].sum())
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")
    moment = float((P[sel] * rr2[sel]).sum() / tot)
    return float(np.sqrt(max(moment, 0.0)) / max(df_x, df_y))


def _peak_width_corrected(power: np.ndarray, u_grid: np.ndarray,
                          v_grid: np.ndarray, u0: float, v0: float,
                          bg_mean: float, df: float,
                          rho: Optional[np.ndarray] = None,
                          r_max_bins: float = 8.0,
                          instrument_bins: Optional[float] = None,
                          max_r_max_bins: float = 20.0,
                          ) -> Dict[str, Any]:
    ceiling_of = lambda r: float(r / math.sqrt(2.0))
    out: Dict[str, Any] = {"raw_bins": float("nan"),
                           "deconvolved_bins": float("nan"),
                           "instrument_bins": instrument_bins,
                           "ceiling_bins": ceiling_of(r_max_bins),
                           "r_max_bins_used": float(r_max_bins),
                           "saturated": False,
                           "significant": False,
                           "expansions": 0}
    if not np.isfinite(bg_mean):
        return out

    du = u_grid - u0
    dv = v_grid - v0
    rr2 = du ** 2 + dv ** 2

    def measure_at(r_bins: float) -> Optional[Dict[str, Any]]:
        r_max = r_bins * df
        sel = rr2 <= r_max ** 2
        n_sel = int(sel.sum())
        if n_sel < 9:
            return None
        w = power[sel] - bg_mean
        total = float(w.sum())
        pair = (_pair_correlation_sum(sel, rho) if rho is not None
                else float(n_sel))
        sigma_total = math.sqrt(max(pair, 1.0)) * abs(bg_mean)
        if not np.isfinite(total) or sigma_total <= 0 \
                or total < 3.0 * sigma_total:
            return None
        moment = float((w * rr2[sel]).sum() / total)
        if not np.isfinite(moment) or moment <= 0:
            return None
        return {"raw": float(np.sqrt(moment) / df), "r_bins": float(r_bins)}

    best = measure_at(r_max_bins)
    if best is None:
        return out

    expansions = 0
    r_cur = float(r_max_bins)
    while (best["raw"] * _WIDTH_DISC_FACTOR > r_cur
           and expansions < _WIDTH_MAX_EXPANSIONS
           and r_cur < max_r_max_bins):
        r_next = min(r_cur * 1.6, max_r_max_bins)
        if r_next <= r_cur * 1.001:
            break
        trial = measure_at(r_next)
        if trial is None:
            break
        gain = (trial["raw"] - best["raw"]) / max(best["raw"], 1e-12)
        expansions += 1
        r_cur, best = r_next, trial
        if gain < _WIDTH_CONVERGE_TOL:
            break

    raw = best["raw"]
    out.update(raw_bins=raw, significant=True,
               r_max_bins_used=r_cur, expansions=expansions,
               ceiling_bins=ceiling_of(r_cur),
               saturated=bool(raw > 0.6 * r_cur))
    if instrument_bins is not None and np.isfinite(instrument_bins):
        out["deconvolved_bins"] = float(
            np.sqrt(max(raw ** 2 - float(instrument_bins) ** 2, 0.0)))
    else:
        out["deconvolved_bins"] = raw
    return out


def _aperture_photometry(power: np.ndarray, u_grid: np.ndarray,
                         v_grid: np.ndarray, u0: float, v0: float,
                         r_ap: float, r_in: float, r_out: float,
                         rho: Optional[np.ndarray] = None,
                         kappa: Optional[float] = None
                         ) -> Dict[str, Any]:
    du = u_grid - u0
    dv = v_grid - v0
    rr = np.hypot(du, dv)

    ap = rr <= r_ap
    ann = (rr > r_in) & (rr <= r_out)
    n_ap, n_ann = int(ap.sum()), int(ann.sum())

    nan = float("nan")
    if n_ap == 0 or n_ann < 12:
        return {"intensity": nan, "intensity_legacy": nan, "background": nan,
                "background_median": nan, "sigma": nan, "snr": nan,
                "snr_integrated": nan, "sigma_intensity": nan,
                "peak_power": nan, "n_aperture": n_ap, "n_annulus": n_ann,
                "n_eff": nan, "pair_sum": nan, "log10_p": nan,
                "exponentiality": nan, "valid": False}

    ann_vals = power[ann]

    bg_med = float(np.median(ann_vals))
    mad = float(np.median(np.abs(ann_vals - bg_med)))
    sigma = 1.4826 * mad
    if sigma > 0:
        keep = ann_vals <= bg_med + 3.0 * sigma
        if keep.sum() >= 12:
            bg_med = float(np.median(ann_vals[keep]))
            mad = float(np.median(np.abs(ann_vals[keep] - bg_med)))
            sigma = 1.4826 * mad
    peak = float(power[ap].max())
    snr = float((peak - bg_med) / sigma) if sigma > 0 else nan
    integrated_legacy = float((power[ap] - bg_med).sum())

    stats = _background_stats(ann_vals)
    if stats is None:
        return {"intensity": nan, "intensity_legacy": integrated_legacy,
                "background": nan, "background_median": bg_med,
                "sigma": sigma, "snr": snr, "snr_integrated": nan,
                "sigma_intensity": nan, "peak_power": peak,
                "n_aperture": n_ap, "n_annulus": n_ann, "n_eff": nan,
                "pair_sum": nan, "log10_p": nan, "exponentiality": nan,
                "valid": False}

    b = stats["mean"]
    integrated = float((power[ap] - b).sum())

    pair = _pair_correlation_sum(ap, rho) if rho is not None else float(n_ap)
    pair = max(pair, 1.0)
    n_eff = float(n_ap ** 2 / pair)

    k = float(kappa) if kappa else 1.0
    var_null = pair * b * b
    var_bg = (n_ap ** 2) * k * b * b / max(float(stats["n"]), 1.0)
    var_sig = 2.0 * b * max(integrated, 0.0) * (pair / max(n_ap, 1))
    var_meas = var_null + var_bg + var_sig

    denom_null = math.sqrt(var_null + var_bg)
    snr_integrated = float(integrated / denom_null) if denom_null > 0 else nan

    total_power = float(power[ap].sum())
    log10_p = nan
    if b > 0 and var_null > 0:
        shape = (n_ap * b) ** 2 / var_null
        scale = var_null / (n_ap * b)
        p = _gamma_sf(total_power, shape, scale)
        if np.isfinite(p):
            log10_p = float(math.log10(max(p, 1e-300)))

    return {"intensity": integrated,
            "intensity_legacy": integrated_legacy,
            "background": b,
            "background_median": bg_med,
            "sigma": sigma,
            "snr": snr,
            "snr_integrated": snr_integrated,
            "sigma_intensity": float(math.sqrt(var_meas)),
            "var_null": float(var_null + var_bg),
            "peak_power": peak,
            "n_aperture": n_ap,
            "n_annulus": n_ann,
            "n_eff": n_eff,
            "pair_sum": float(pair),
            "log10_p": log10_p,
            "exponentiality": float(stats["exponentiality"]),
            "valid": True}


def _refine_position(power: np.ndarray, shape: Tuple[int, int],
                     center: Tuple[int, int], u0: float, v0: float,
                     search_bins: int = 3, max_shift: Optional[float] = None
                     ) -> Tuple[float, float, float]:
    Ny, Nx = shape
    cy, cx = center
    iy = int(round(v0 * Ny)) + cy
    ix = int(round(u0 * Nx)) + cx
    s = int(search_bins)

    y0, y1 = max(0, iy - s), min(Ny, iy + s + 1)
    x0, x1 = max(0, ix - s), min(Nx, ix + s + 1)
    if y1 <= y0 or x1 <= x0:
        return u0, v0, 0.0
    sub = power[y0:y1, x0:x1]
    dy, dx = np.unravel_index(int(np.argmax(sub)), sub.shape)
    iy, ix = y0 + dy, x0 + dx

    def _off(vm: float, v_0: float, vp: float) -> float:
        denom = vm - 2.0 * v_0 + vp
        if denom >= 0 or not np.isfinite(denom):
            return 0.0
        return float(np.clip(0.5 * (vm - vp) / denom, -0.5, 0.5))

    ox = (_off(power[iy, ix - 1], power[iy, ix], power[iy, ix + 1])
          if 1 <= ix < Nx - 1 else 0.0)
    oy = (_off(power[iy - 1, ix], power[iy, ix], power[iy + 1, ix])
          if 1 <= iy < Ny - 1 else 0.0)

    u = (ix + ox - cx) / float(Nx)
    v = (iy + oy - cy) / float(Ny)
    shift = float(np.hypot(u - u0, v - v0))

    if max_shift is not None and shift > max_shift:
        return u0, v0, shift

    return u, v, shift


def _find_candidates(power: np.ndarray, u_grid: np.ndarray,
                     v_grid: np.ndarray, dc_radius_bins: float = 5.0,
                     neighborhood: int = 5, n_sigma: float = 4.0,
                     n_radial_bins: int = 40, max_candidates: int = 400
                     ) -> List[Dict[str, Any]]:
    Ny, Nx = power.shape
    r = np.hypot(u_grid, v_grid)
    df = 1.0 / np.sqrt(float(Nx) * float(Ny))
    valid = r > dc_radius_bins * df

    is_max = power == maximum_filter(power, size=neighborhood, mode="nearest")

    r_max = float(r.max())
    r_idx = np.minimum((r / max(r_max, 1e-12) * n_radial_bins).astype(int),
                       n_radial_bins - 1)
    thr = np.full(power.shape, np.inf)
    for b in range(n_radial_bins):
        m = (r_idx == b) & valid
        vals = power[m]
        if vals.size >= 32:
            med = float(np.median(vals))
            sig = 1.4826 * float(np.median(np.abs(vals - med)))
            thr[m] = med + n_sigma * max(sig, 1e-300)

    iy, ix = np.nonzero(is_max & valid & (power > thr))
    cand = [{"iy": int(a), "ix": int(b),
             "u": float(u_grid[a, b]), "v": float(v_grid[a, b]),
             "r": float(r[a, b]),
             "angle": float(np.degrees(np.arctan2(v_grid[a, b],
                                                  u_grid[a, b])) % 360.0),
             "power": float(power[a, b])}
            for a, b in zip(iy, ix)]
    cand.sort(key=lambda p: -p["power"])
    return cand[:max_candidates]


def _hex_families(cand: Sequence[Dict[str, Any]], radius_rtol: float = 0.12,
                  angle_tol_deg: float = 14.0, max_directions: int = 8
                  ) -> List[Dict[str, Any]]:
    from itertools import combinations

    families: List[Dict[str, Any]] = []
    for seed in cand:
        shell = [p for p in cand
                 if abs(p["r"] - seed["r"]) <= radius_rtol * seed["r"]]
        if len(shell) < 4:
            continue

        dirs: Dict[float, List[Dict[str, Any]]] = {}
        for p in sorted(shell, key=lambda z: -z["power"]):
            a = p["angle"] % 180.0
            key = None
            for k in dirs:
                d = abs(a - k)
                if min(d, 180.0 - d) <= angle_tol_deg:
                    key = k
                    break
            dirs.setdefault(a if key is None else key, []).append(p)

        paired = {k: max(v, key=lambda z: z["power"])
                  for k, v in dirs.items() if len(v) >= 2}
        if len(paired) < 3:
            continue

        keys = sorted(paired, key=lambda k: -paired[k]["power"])[:max_directions]

        best: Optional[Dict[str, Any]] = None
        for combo in combinations(keys, 3):
            angs = sorted(combo)
            seps = [angs[1] - angs[0], angs[2] - angs[1],
                    180.0 - (angs[2] - angs[0])]
            if not all(abs(s - 60.0) <= angle_tol_deg for s in seps):
                continue
            peaks = [paired[a] for a in angs]
            min_power = float(min(p["power"] for p in peaks))
            if best is not None and min_power <= best["min_power"]:
                continue
            best = {
                "radius": float(np.mean([p["r"] for p in peaks])),
                "angles_deg": [float(a) for a in angs],
                "peaks": peaks,
                "min_power": min_power,
                "max_power": float(max(p["power"] for p in peaks)),
                "hex_deviation_deg": float(np.mean([abs(s - 60.0)
                                                    for s in seps])),
            }

        if best is None:
            continue
        if not any(abs(f["radius"] - best["radius"])
                   <= radius_rtol * best["radius"] for f in families):
            families.append(best)

    families.sort(key=lambda f: f["radius"])
    return families


def _alpha_uncertainty(intensities: Sequence[float],
                       sigmas: Sequence[float],
                       backgrounds: Sequence[float],
                       n_ap: Sequence[float],
                       n_draw: int = 20000,
                       seed: int = 0) -> Dict[str, Any]:
    I = np.asarray(intensities, dtype=np.float64)
    S = np.asarray(sigmas, dtype=np.float64)
    Bg = np.asarray(backgrounds, dtype=np.float64)
    N = np.asarray(n_ap, dtype=np.float64)
    out: Dict[str, Any] = {"alpha_ci68": None, "alpha_ci95": None,
                           "alpha_bias_equal_null": None,
                           "p_value_equal_intensity": None,
                           "n_draw": int(n_draw)}
    if I.size != 3 or not (np.all(np.isfinite(I)) and np.all(np.isfinite(S))
                           and np.all(np.isfinite(Bg))
                           and np.all(np.isfinite(N)) and np.all(S > 0)):
        return out

    rng = np.random.default_rng(seed)
    offset = N * Bg
    var = S ** 2

    def draw(means: np.ndarray) -> np.ndarray:
        totals = np.maximum(means + offset, 1e-30)
        shape = np.maximum(totals ** 2 / var, 1e-6)
        scale = var / totals
        t = rng.gamma(np.broadcast_to(shape, (n_draw, 3)),
                      np.broadcast_to(scale, (n_draw, 3)))
        vals = t - offset[None, :]
        hi, lo = vals.max(axis=1), vals.min(axis=1)
        tot = hi + lo
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(tot > 0, (hi - lo) / tot, np.nan)

    obs = draw(I)
    obs = obs[np.isfinite(obs)]
    if obs.size:
        out["alpha_ci68"] = [round(float(np.percentile(obs, 16)), 4),
                             round(float(np.percentile(obs, 84)), 4)]
        out["alpha_ci95"] = [round(float(np.percentile(obs, 2.5)), 4),
                             round(float(np.percentile(obs, 97.5)), 4)]

    w = 1.0 / np.maximum(var, 1e-300)
    common = float(np.sum(w * I) / np.sum(w))
    null = draw(np.full(3, common))
    null = null[np.isfinite(null)]
    if null.size:
        alpha_obs = anisotropy([float(x) for x in I])
        out["alpha_bias_equal_null"] = round(float(np.median(null)), 4)
        if np.isfinite(alpha_obs):
            out["p_value_equal_intensity"] = round(
                float((np.sum(null >= alpha_obs) + 1) / (null.size + 1)), 5)
    return out


def anisotropy(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return float("nan")
    total = float(arr.max() + arr.min())
    if total <= 0:
        return float("nan")
    return float((arr.max() - arr.min()) / total)


def _tipscaled_excess(cdw_rows: Sequence[Dict[str, Any]],
                      bragg_rows: Sequence[Dict[str, Any]],
                      alpha_cdw: float) -> Dict[str, Any]:
    blank = {"alpha_cdw_tipcorrected": None, "alpha_excess_tipscaled": None,
             "q_ratio_sq": None, "tip_logspread_at_cdw": None,
             "available": False}
    try:
        if len(cdw_rows) != 3 or len(bragg_rows) != 3:
            return blank
        C = np.array([r.get("intensity") for r in cdw_rows], dtype=np.float64)
        B = np.array([r.get("intensity") for r in bragg_rows],
                     dtype=np.float64)
        if not (np.all(np.isfinite(C)) and np.all(np.isfinite(B))
                and np.all(C > 0) and np.all(B > 0)):
            return blank
        qc = np.array([r.get("q_cycles_per_px") for r in cdw_rows],
                      dtype=np.float64)
        qb = np.array([r.get("q_cycles_per_px") for r in bragg_rows],
                      dtype=np.float64)
        if not (np.all(np.isfinite(qc)) and np.all(np.isfinite(qb))
                and qb.mean() > 0):
            return blank
        s2 = float((qc.mean() / qb.mean()) ** 2)

        thb = np.radians(np.array([r["angle_deg"] for r in bragg_rows],
                                  dtype=np.float64))
        thc = np.radians(np.array([r["angle_deg"] for r in cdw_rows],
                                  dtype=np.float64))
        M = np.column_stack([np.ones(3), np.cos(2 * thb), np.sin(2 * thb)])
        coef = np.linalg.solve(M, np.log(B))
        _, A2, B2 = (float(c) for c in coef)
        tip_at_cdw = s2 * (A2 * np.cos(2 * thc) + B2 * np.sin(2 * thc))
        a_tc = anisotropy(np.exp(np.log(C) - tip_at_cdw))
        if not np.isfinite(a_tc):
            return blank
        excess = (float(a_tc - alpha_cdw) if np.isfinite(alpha_cdw)
                  else None)
        return {
            "alpha_cdw_tipcorrected": round(float(a_tc), 6),
            "alpha_excess_tipscaled": (round(excess, 6)
                                       if excess is not None else None),
            "q_ratio_sq": round(s2, 6),
            "tip_logspread_at_cdw": round(float(np.ptp(tip_at_cdw)), 6),
            "available": True,
        }
    except (np.linalg.LinAlgError, KeyError, TypeError, ValueError):
        return blank


def _conjugate_consistency(m_p, m_m):
    i_p, i_m = m_p["intensity"], m_m["intensity"]
    int_resid = (float(abs(i_p - i_m) / (0.5 * (i_p + i_m)))
                 if (np.isfinite(i_p) and np.isfinite(i_m)
                     and (i_p + i_m) > 0) else None)
    n_p, n_m = m_p.get("n_aperture"), m_m.get("n_aperture")
    mismatch = (abs(int(n_p) - int(n_m)) if (n_p and n_m) else None)
    return int_resid, mismatch


def _hierarchy_sense(angles_deg: Sequence[float],
                     intensities: Sequence[float]) -> Optional[str]:
    a = np.asarray(angles_deg, dtype=np.float64) % 180.0
    inten = np.asarray(intensities, dtype=np.float64)
    if inten.size != 3 or not np.all(np.isfinite(inten)):
        return None
    order = np.argsort(-inten)
    a1, a2, a3 = a[order]
    step = (a2 - a1 + 90.0) % 180.0 - 90.0
    step2 = (a3 - a2 + 90.0) % 180.0 - 90.0
    if step == 0 or step2 == 0:
        return None
    if step > 0 and step2 > 0:
        return "counter-clockwise"
    if step < 0 and step2 < 0:
        return "clockwise"
    return "non-monotonic"


def analyze_didv_field(
    field: np.ndarray,
    source_label: str = "<array>",
    scalar_field_info: Optional[Dict[str, Any]] = None,
    cdw_model: Optional[str] = None,
    aperture_bins: Any = 3.0,
    separation_fraction: Optional[float] = None,
    annulus_in_factor: float = 1.8,
    annulus_out_factor: float = 3.2,
    search_bins: int = 3,
    dc_radius_bins: float = 5.0,
    min_bragg_snr: float = 4.0,
    min_bragg_power_frac: float = 0.01,
    min_cdw_snr: float = 5.0,
    min_cdw_power_frac: float = 5e-3,
    refine_tol_bins: float = 2.0,
    remove_plane: bool = True,
    alpha_mc_draws: int = 20000,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": "failed",
        "source_path": source_label,
        "warnings": [],
        "method": {
            "intensity": "background-subtracted integrated |FFT|^2",
            "aperture": "circular, identical in cycles/pixel for all peaks",
            "peak_identification": "Bragg-anchored",
            "definition": "alpha = (I_max - I_min) / (I_max + I_min)",
        },
    }

    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        out["message"] = f"expected a 2D field, got shape {field.shape}"
        return out

    load_info = scalar_field_info or {"method": "array_supplied",
                                      "colormap": None}
    out["scalar_field"] = load_info
    if load_info.get("method") == "luminance_fallback":
        out["warnings"].append(
            "RGB->scalar used a luminance fallback; intensity ratios (and "
            "therefore alpha) may be biased.")

    spec = power_spectrum(field, remove_plane=remove_plane)
    power = spec["power"]
    Ny, Nx = spec["shape"]
    u_grid, v_grid = _uv_grid(spec["shape"], spec["center"])
    df = 1.0 / np.sqrt(float(Nx) * float(Ny))

    aperture_mode = "fixed"
    if isinstance(aperture_bins, str) and aperture_bins == "auto":
        aperture_mode = "auto"
        aperture_bins = 3.0
    aperture_bins = float(aperture_bins)

    if aperture_mode == "auto" and separation_fraction is not None:
        aperture_mode = "auto+separation_cap"

    aperture_overlap = False

    r_ap = aperture_bins * df
    r_in = annulus_in_factor * r_ap
    r_out = annulus_out_factor * r_ap

    out["image"] = {"Nx": Nx, "Ny": Ny, "aspect": round(Nx / Ny, 4)}
    out["method"].update(
        angle_frame=ANGLE_FRAME,
        aperture_mode=aperture_mode,
        aperture_radius_cycles_per_px=round(float(r_ap), 6),
        aperture_radius_bins=float(aperture_bins),
        annulus_cycles_per_px=[round(float(r_in), 6), round(float(r_out), 6)],
    )
    if abs(Nx - Ny) > 0.02 * max(Nx, Ny):
        out["warnings"].append(
            f"Map is not square ({Nx}x{Ny}px).  Reciprocal space is handled "
            "in cycles/pixel so apertures stay circular, but if the physical "
            "field of view is not proportional to the pixel counts the "
            "measured directions are geometrically distorted.")

    dc_radius = dc_radius_bins * df
    r_grid = np.hypot(u_grid, v_grid)
    outside_dc = r_grid > dc_radius
    ref_power = (float(power[outside_dc].max()) if outside_dc.any()
                 else float(power.max()))

    rho_kernel, kappa, instrument_width_bins = hann_noise_model(Ny, Nx)
    out["noise_model"] = {
        "estimator_version": ESTIMATOR_VERSION,
        "background": "clipped mean, exponential-debiased (was: median)",
        "bin_correlation_kappa": round(kappa, 4),
        "instrument_width_bins": (round(float(instrument_width_bins), 4)
                                  if np.isfinite(instrument_width_bins)
                                  else None),
        "note": "Periodogram power is exponential, not Gaussian, and the "
                "Hann window correlates neighbouring bins; an "
                "independent-pixel model understates aperture variance by "
                f"about {kappa:.1f}x.",
    }

    def _width_disc_cap(u: float, v: float) -> float:
        q_r = float(np.hypot(u, v))
        room_dc = (q_r - dc_radius) / df
        room_mirror = q_r / df
        return float(max(6.0, min(25.0, 0.9 * room_dc, 0.9 * room_mirror)))

    def measure(u: float, v: float) -> Dict[str, Any]:
        return _aperture_photometry(power, u_grid, v_grid, u, v,
                                    r_ap, r_in, r_out,
                                    rho=rho_kernel, kappa=kappa)

    def snr_at(u: float, v: float) -> float:
        return measure(u, v)["snr"]

    model_table = dict(CDW_MODELS)
    if cdw_model is not None and not isinstance(cdw_model, str):
        try:
            ratio = float(cdw_model["ratio"])
            rot = float(cdw_model.get("rotation_deg", 0.0))
            label = str(cdw_model.get("name") or "custom")
        except (TypeError, ValueError, KeyError) as exc:
            out["message"] = (f"could not read custom cdw_model "
                              f"{cdw_model!r}: {exc}")
            return out
        if not (0.0 < ratio <= 1.0):
            out["message"] = (f"custom cdw_model ratio {ratio} must lie in "
                              "(0, 1]: the CDW wavevector cannot exceed the "
                              "Bragg wavevector")
            return out
        model_table[label] = (ratio, float(rot) % 180.0)
        cdw_model = label
        out.setdefault("warnings", []).append(
            f"Using a custom superlattice: |q_CDW|/|q_Bragg| = {ratio:.4f}, "
            f"rotation {rot:g} deg.  The geometry checks still apply, so a "
            "ratio that does not land on real peaks will be rejected.")

    if cdw_model is not None and cdw_model not in model_table:
        out["message"] = (f"unknown cdw_model {cdw_model!r}; "
                          f"choose from {sorted(CDW_MODELS)} or supply "
                          "{'ratio': r, 'rotation_deg': theta}")
        return out
    models = ({cdw_model: model_table[cdw_model]} if cdw_model
              else dict(CDW_MODELS))

    herm_res: List[float] = []

    cand = _find_candidates(power, u_grid, v_grid,
                            dc_radius_bins=dc_radius_bins)
    families = _hex_families(cand)
    out["n_candidate_peaks"] = len(cand)
    out["n_hex_families"] = len(families)
    max_cand_power = max((c["power"] for c in cand), default=0.0)

    anchors = []
    for fam in sorted(families, key=lambda f: -f["radius"]):
        frac = (fam["min_power"] / max_cand_power
                if max_cand_power > 0 else 0.0)
        if frac < min_bragg_power_frac:
            continue
        snrs = [snr_at(p["u"], p["v"]) for p in fam["peaks"]]
        if not all(np.isfinite(s) and s >= min_bragg_snr for s in snrs):
            continue
        anchors.append({
            "dirs": sorted([(float(p["r"]), float(p["angle"] % 180.0))
                            for p in fam["peaks"]], key=lambda t: t[1]),
            "radius": float(fam["radius"]),
            "source": "auto-detected",
            "power_frac": float(frac),
            "hex_deviation_deg": float(fam["hex_deviation_deg"]),
        })

    if not anchors:
        out["status"] = "insufficient"
        out["message"] = (
            "No hexagonal family of three symmetry-related peaks cleared "
            "the Bragg gates, so there is no anchor.  Anisotropy is "
            "undefined for this map (it may be 1Q/stripe-ordered, "
            "non-hexagonal, or too blurred to resolve peaks).  Supply "
            "bragg_radius and bragg_angles_deg to override.")
        return out
    family_radii = [f["radius"] for f in families]

    def measure_family(dirs: Sequence[Tuple[float, float]]
                       ) -> Tuple[List[float], List[Dict[str, Any]], List[float]]:
        vals, rows, herm = [], [], []
        for k, (r_b, ang) in enumerate(dirs, start=1):
            th = np.radians(ang)
            u0, v0 = r_b * np.cos(th), r_b * np.sin(th)
            u_p, v_p, _ = _refine_position(power, spec["shape"],
                                           spec["center"], u0, v0,
                                           search_bins,
                                           max_shift=refine_tol_bins * df)
            m_plus = measure(u_p, v_p)
            m_minus = measure(-u_p, -v_p)
            i_p, i_m = m_plus["intensity"], m_minus["intensity"]
            _ir, _pr = _conjugate_consistency(m_plus, m_minus)
            if _pr is not None:
                herm.append(float(_pr))
            inten = float(np.nanmean([i_p, i_m]))
            vals.append(inten)
            fwhm = _peak_width_bins(power, u_grid, v_grid, u_p, v_p,
                                    m_plus["background_median"], df)
            wc = _peak_width_corrected(
                power, u_grid, v_grid, u_p, v_p, m_plus["background"], df,
                rho=rho_kernel, instrument_bins=instrument_width_bins,
                max_r_max_bins=_width_disc_cap(u_p, v_p))
            rows.append({
                "index": k,
                "width_bins": (round(float(fwhm), 3)
                              if np.isfinite(fwhm) else None),
                "width_bins_corrected": (
                    round(float(wc["deconvolved_bins"]), 3)
                    if np.isfinite(wc["deconvolved_bins"]) else None),
                "width_raw_bins": (round(float(wc["raw_bins"]), 3)
                                   if np.isfinite(wc["raw_bins"]) else None),
                "width_saturated": bool(wc["saturated"]),
                "width_r_max_bins_used": round(float(wc["r_max_bins_used"]), 1),
                "width_expansions": int(wc["expansions"]),
                "angle_deg": round(_display_angle_deg(u_p, v_p), 2) % 180.0,
                "angle_deg_array": round(
                    float(np.degrees(np.arctan2(v_p, u_p)) % 180.0), 2) % 180.0,
                "q_cycles_per_px": round(float(np.hypot(u_p, v_p)), 6),
                "intensity": inten,
                "intensity_legacy": float(np.nanmean(
                    [m_plus["intensity_legacy"], m_minus["intensity_legacy"]])),
                "intensity_sigma": _r6(m_plus["sigma_intensity"]),
                "intensity_var_null": _r6(m_plus.get("var_null")),
                "intensity_plus_q": i_p,
                "intensity_minus_q": i_m,
                "snr": (round(float(m_plus["snr"]), 2)
                        if np.isfinite(m_plus["snr"]) else None),
                "snr_integrated": (round(float(m_plus["snr_integrated"]), 2)
                                   if np.isfinite(m_plus["snr_integrated"])
                                   else None),
                "log10_p": (round(float(m_plus["log10_p"]), 2)
                            if np.isfinite(m_plus["log10_p"]) else None),
                "n_eff": (round(float(m_plus["n_eff"]), 1)
                          if np.isfinite(m_plus["n_eff"]) else None),
                "n_aperture": (int(m_plus["n_aperture"])
                               if m_plus.get("n_aperture") else None),
                "background": m_plus["background"],
                "background_median": m_plus["background_median"],
            })
        return vals, rows, herm

    def evaluate(bragg_dirs: Sequence[Tuple[float, float]],
                 ratio: float, rot_deg: float) -> Dict[str, Any]:
        max_shift = refine_tol_bins * df
        rows, ints, herm = [], [], []
        valid, reasons, notes = True, [], []

        for k, (r_b, ang) in enumerate(bragg_dirs, start=1):
            th = np.radians(ang + rot_deg)
            r_c = r_b * ratio
            if r_c - r_ap <= dc_radius:
                valid = False
                reasons.append("aperture overlaps the DC exclusion zone")
            elif r_c - r_out <= dc_radius:
                notes.append("background annulus clips the DC skirt")
            u0, v0 = r_c * np.cos(th), r_c * np.sin(th)
            u_p, v_p, shift = _refine_position(
                power, spec["shape"], spec["center"], u0, v0, search_bins,
                max_shift=max_shift)
            if shift > max_shift:
                valid = False
                reasons.append(
                    f"direction {k}: nearest maximum is {shift / df:.1f} bins "
                    f"from the predicted position (tolerance "
                    f"{refine_tol_bins:g})")

            m_plus = measure(u_p, v_p)
            m_minus = measure(-u_p, -v_p)

            prominence = m_plus["peak_power"] - m_plus["background_median"]
            if not (np.isfinite(m_plus["snr"])
                    and m_plus["snr"] >= min_cdw_snr):
                valid = False
                reasons.append(f"direction {k}: SNR below {min_cdw_snr:g}")
            elif prominence < min_cdw_power_frac * ref_power:
                valid = False
                reasons.append(
                    f"direction {k}: peak holds only "
                    f"{prominence / ref_power:.2e} of the strongest peak "
                    f"(gate {min_cdw_power_frac:g})")

            i_p, i_m = m_plus["intensity"], m_minus["intensity"]
            _ir, _pr = _conjugate_consistency(m_plus, m_minus)
            if _pr is not None:
                herm.append(float(_pr))
            inten = float(np.nanmean([i_p, i_m]))
            ints.append(inten)
            rows.append({
                "index": k,
                "u": float(u_p), "v": float(v_p),
                "bg": float(m_plus["background"]),
                "angle_deg": round(_display_angle_deg(u_p, v_p), 2) % 180.0,
                "angle_deg_array": round(
                    float(np.degrees(np.arctan2(v_p, u_p)) % 180.0), 2) % 180.0,
                "q_cycles_per_px": round(float(np.hypot(u_p, v_p)), 6),
                "predicted_q_cycles_per_px": round(float(r_c), 6),
                "intensity": inten,
                "intensity_legacy": float(np.nanmean(
                    [m_plus["intensity_legacy"],
                     m_minus["intensity_legacy"]])),
                "intensity_sigma": _r6(m_plus["sigma_intensity"]),
                "intensity_var_null": _r6(m_plus.get("var_null")),
                "intensity_plus_q": i_p,
                "intensity_minus_q": i_m,
                "snr": (round(float(m_plus["snr"]), 2)
                        if np.isfinite(m_plus["snr"]) else None),
                "snr_integrated": (round(float(m_plus["snr_integrated"]), 2)
                                   if np.isfinite(m_plus["snr_integrated"])
                                   else None),
                "log10_p": (round(float(m_plus["log10_p"]), 2)
                            if np.isfinite(m_plus["log10_p"]) else None),
                "n_eff": (round(float(m_plus["n_eff"]), 1)
                          if np.isfinite(m_plus["n_eff"]) else None),
                "n_aperture": (int(m_plus["n_aperture"])
                               if m_plus.get("n_aperture") else None),
                "background": m_plus["background"],
                "background_median": m_plus["background_median"],
            })

        angs = [r["angle_deg"] for r in rows]
        for a, b in ((0, 1), (1, 2), (0, 2)):
            d = abs(angs[a] - angs[b])
            if min(d, 180.0 - d) < 20.0:
                valid = False
                reasons.append("directions collapsed")
                break

        mean_r = float(np.mean([r["q_cycles_per_px"] for r in rows]))
        family_match = any(abs(fr - mean_r) <= 0.12 * mean_r
                           for fr in family_radii)

        snrs = [r["snr"] for r in rows if r["snr"] is not None]
        min_snr = min(snrs) if len(snrs) == 3 else float("-inf")
        return {"rows": rows, "intensities": ints, "min_snr": min_snr,
                "valid": valid, "family_match": family_match,
                "reasons": sorted(set(reasons)),
                "notes": sorted(set(notes)), "hermitian": herm}

    scored = []
    rotation_info: Dict[str, Dict[str, Any]] = {}
    for a_idx, anchor in enumerate(anchors):
        for name, (ratio, rot) in models.items():
            rotations, rinfo = _rotation_candidates(
                rot, bragg_dirs=anchor["dirs"], ratio=ratio, df=df)
            rotation_info.setdefault(name, rinfo)
            for rot_try in rotations:
                ev = evaluate(anchor["dirs"], ratio, rot_try)
                scored.append({
                    "valid": bool(ev["valid"]),
                    "family_match": bool(ev["family_match"]),
                    "anchor_rank": a_idx,
                    "min_snr": ev["min_snr"],
                    "name": name, "rot": rot_try,
                    "anchor": anchor, "ev": ev,
                })

    scored.sort(key=lambda s: (not s["valid"], not s["family_match"],
                               s["anchor_rank"], -s["min_snr"]))
    chosen = scored[0]
    best, best_name, best_rot = chosen["ev"], chosen["name"], chosen["rot"]
    best_valid, best_snr = chosen["valid"], chosen["min_snr"]
    anchor = chosen["anchor"]

    out["bragg_source"] = anchor["source"]
    if anchor["source"] == "auto-detected":
        out["bragg_family"] = {
            "radius_cycles_per_px": round(anchor["radius"], 6),
            "hex_deviation_deg": round(anchor["hex_deviation_deg"], 2),
            "power_frac_of_strongest": round(anchor["power_frac"], 5),
            "n_anchors_considered": len(anchors),
            "anchor_rank": chosen["anchor_rank"],
        }
        _a_r = float(anchor.get("radius") or 0.0)
        _a_ang = [float(a) % 180.0 for _r, a in (anchor.get("dirs") or [])]
        _a_pow = next((float(f["min_power"]) for f in families
                       if abs(float(f["radius"]) - _a_r)
                       <= 0.02 * max(_a_r, 1e-30)), None)
        _suspects = []
        if _a_r > 0 and len(_a_ang) == 3 and _a_pow:
            for _f in families:
                _fr = float(_f["radius"])
                if _fr <= 0 or _fr >= _a_r * 0.95:
                    continue
                _n = _a_r / _fr
                if round(_n) < 2 or abs(_n - round(_n)) > 0.06 * round(_n):
                    continue
                _fa = [float(a) % 180.0 for a in _f["angles_deg"]]
                if not all(min(min(abs(x - y), 180.0 - abs(x - y))
                               for y in _a_ang) <= 12.0 for x in _fa):
                    continue
                _ratio = float(_f["min_power"]) / _a_pow
                if _ratio >= 0.5:
                    _suspects.append({
                        "radius_cycles_per_px": round(_fr, 6),
                        "anchor_is_order": int(round(_n)),
                        "power_ratio_to_anchor": round(_ratio, 3)})
        out["anchor_harmonic_suspect"] = _suspects or None
        if _suspects:
            _w = max(_suspects, key=lambda d: d["power_ratio_to_anchor"])
            _n = _w["anchor_is_order"]
            out["warnings"].append(
                f"ANCHOR MAY BE A HARMONIC: an independently detected "
                f"hexagonal family sits at 1/{_n} of the anchor radius "
                f"({_w['radius_cycles_per_px']:g} cycles/px), shares its "
                f"directions, and carries "
                f"{_w['power_ratio_to_anchor']:g}x the anchor's power. A "
                f"harmonic is always weaker than its fundamental, so that "
                f"inner family is probably the atomic lattice and this "
                f"anchor is its order-{_n} harmonic. If so |q_B| is {_n}x "
                f"too large: the reciprocal-space view reaches {_n}x further "
                f"than it states (which fills the panel with unlabelled "
                f"peaks), and the CDW apertures are placed off the real "
                f"peaks. Re-run with "
                f"bragg_radius={_w['radius_cycles_per_px']:g} to resolve.")

        if anchor["power_frac"] is not None and anchor["power_frac"] > 0.35:
            out["warnings"].append(
                f"The Bragg anchor carries {anchor['power_frac']:.0%} of the "
                "strongest peak in the spectrum.  In a dI/dV map the atomic "
                "Bragg peaks are usually weaker than the CDW peaks, so this "
                "shell may itself be the CDW - in which case the reported "
                "'CDW' is a sub-harmonic and the labels are swapped.  Pin "
                "bragg_radius / bragg_angles_deg to resolve.")
        if chosen["anchor_rank"] > 0:
            out["warnings"].append(
                f"The outermost {chosen['anchor_rank']} hexagonal "
                "family(ies) did not yield a consistent superlattice, so a "
                "smaller-|q| shell was used as the Bragg anchor.  Verify "
                "that it really is the atomic lattice and not a CDW shell.")
    if not chosen["family_match"]:
        out["warnings"].append(
            "The CDW shell was not independently detected as a hexagonal "
            "family; peaks were measured only at the positions predicted "
            "from the Bragg anchor.")

    def _fill_widths(rows: List[Dict[str, Any]]) -> None:
        for r in rows:
            if "u" not in r:
                continue
            w = _peak_width_bins(power, u_grid, v_grid,
                                 r["u"], r["v"],
                                 r.get("background_median", r["bg"]), df)
            r["width_bins"] = round(float(w), 3) if np.isfinite(w) else None
            wc = _peak_width_corrected(
                power, u_grid, v_grid, r["u"], r["v"], r["bg"], df,
                rho=rho_kernel, instrument_bins=instrument_width_bins,
                max_r_max_bins=_width_disc_cap(r["u"], r["v"]))
            r["width_bins_corrected"] = (
                round(float(wc["deconvolved_bins"]), 3)
                if np.isfinite(wc["deconvolved_bins"]) else None)
            r["width_raw_bins"] = (round(float(wc["raw_bins"]), 3)
                                   if np.isfinite(wc["raw_bins"]) else None)
            r["width_saturated"] = bool(wc["saturated"])
            r["width_r_max_bins_used"] = round(float(wc["r_max_bins_used"]), 1)
            r["width_expansions"] = int(wc["expansions"])
            r.pop("u", None); r.pop("v", None); r.pop("bg", None)

    bragg_dirs = list(anchor["dirs"])
    B, bragg_rows, bragg_herm = measure_family(bragg_dirs)
    _fill_widths(best["rows"])

    if aperture_mode == "auto":
        def _w(row: Dict[str, Any]) -> Optional[float]:
            w = row.get("width_bins_corrected")
            return w if w is not None else row.get("width_bins")

        for _aperture_pass in range(_APERTURE_MAX_PASSES):
            widths = [w for w in (_w(r) for r in bragg_rows) if w is not None]
            widths += [w for w in (_w(r) for r in best["rows"]) if w is not None]
            if widths:
                target = max(3.0, min(APERTURE_MAX_BINS, float(np.max(widths))))

                pts: List[Tuple[float, float]] = []
                for row in list(bragg_rows) + list(best["rows"]):
                    ang = np.radians(row["angle_deg"])
                    q_b = row["q_cycles_per_px"] / df
                    pts.append((q_b * np.cos(ang), q_b * np.sin(ang)))
                    pts.append((-q_b * np.cos(ang), -q_b * np.sin(ang)))
                min_sep = np.inf
                for a_i in range(len(pts)):
                    for b_i in range(a_i + 1, len(pts)):
                        d_ab = float(np.hypot(pts[a_i][0] - pts[b_i][0],
                                              pts[a_i][1] - pts[b_i][1]))
                        if d_ab < min_sep:
                            min_sep = d_ab

                if np.isfinite(min_sep):
                    out["method"]["min_peak_separation_bins"] = round(
                        float(min_sep), 3)
                    out["method"]["separation_capped_target_bins"] = round(
                        float(SEPARATION_FRACTION_RECOMMENDED * min_sep), 3)
                    out["method"]["separation_cap_enabled"] = False
                    severity = (float(target / (0.5 * min_sep))
                                if min_sep > 0 else float("inf"))
                    out["method"]["aperture_overlap_severity"] = (
                        round(severity, 3) if np.isfinite(severity) else None)
                    annulus_reach = annulus_out_factor * target
                    out["method"]["annulus_outer_reach_bins"] = round(
                        float(annulus_reach), 3)
                    annulus_hit = bool(min_sep <= annulus_reach)
                    out["method"]["annulus_reaches_neighbour"] = annulus_hit
                    if target > 0.5 * min_sep:
                        aperture_overlap = True
                        out["warnings"].append(
                            f"Aperture ({target:.2f} bins) exceeds half the "
                            f"nearest peak separation ({min_sep:.2f} bins), so "
                            "adjacent apertures overlap and share power. "
                            "Shrinking the aperture instead truncates the peaks, "
                            "so neither choice is clean for this map; treat the "
                            "intensities as provisional.")
                    if annulus_hit:
                        out["warnings"].append(
                            f"A neighbouring peak lies within this map's "
                            f"background annulus (outer reach "
                            f"{annulus_reach:.2f} bins vs nearest separation "
                            f"{min_sep:.2f} bins), so the background is "
                            "over-estimated and alpha is biased HIGH. This "
                            "acts OPPOSITE to the shared-power bias from "
                            "aperture overlap: the net direction for this "
                            "map is not determined, so neither an 'above' "
                            "nor a 'below' outcome can be assumed "
                            "conservative.")
                target = max(2.0, target)

                if abs(target - aperture_bins) > 0.25:
                    aperture_bins = target
                    r_ap = aperture_bins * df
                    r_in = annulus_in_factor * r_ap
                    r_out = annulus_out_factor * r_ap
                    B, bragg_rows, bragg_herm = measure_family(bragg_dirs)
                    best = evaluate(bragg_dirs,
                                    model_table[best_name][0], best_rot)
                    _fill_widths(best["rows"])
                    out["warnings"].append(
                        f"Aperture resized to {aperture_bins:.1f} bins from the "
                        "measured peak width.")
                    continue
            break
        else:
            out["warnings"].append(
                f"The aperture did not settle in {_APERTURE_MAX_PASSES} "
                "passes. Aperture and width are coupled - the background "
                "annulus is placed from the aperture, and the width is "
                "measured against that background - so a map whose peak "
                "wings reach into the annulus can oscillate. The reported "
                "aperture is the last one tried; treat the width and the "
                "intensities as provisional.")

        out["method"]["aperture_radius_bins"] = float(aperture_bins)
        out["method"]["aperture_radius_cycles_per_px"] = round(float(r_ap), 6)
        out["method"]["aperture_passes"] = _aperture_pass + 1

    herm_res.extend(bragg_herm)
    herm_res.extend(best["hermitian"])

    if not all(np.isfinite(B)) or min(B) <= 0:
        out["status"] = "insufficient"
        out["message"] = ("Bragg photometry produced a non-positive or "
                          "undefined intensity; cannot normalize.")
        out["bragg_peaks"] = bragg_rows
        return out

    bragg_snrs = [r["snr"] for r in bragg_rows if r["snr"] is not None]
    if bragg_snrs and min(bragg_snrs) < min_bragg_snr:
        out["warnings"].append(
            f"Weakest Bragg peak has SNR {min(bragg_snrs):.1f} "
            f"(< {min_bragg_snr:g}); the Bragg control is poorly determined.")

    for note in best["notes"]:
        out["warnings"].append(note)

    if not best_valid and cdw_model is None:
        out["status"] = "insufficient"
        out["message"] = (
            "No CDW superlattice model is supported by this map: "
            + "; ".join(best["reasons"][:3])
            + ".  Reporting no anisotropy rather than a number produced by "
              "apertures placed where no peak exists.  If you know the "
              "superlattice, pass cdw_model explicitly (and bragg_radius / "
              "bragg_angles_deg if the anchor is also wrong).")
        out["bragg_peaks"] = bragg_rows
        out["cdw_candidates_rejected"] = best["reasons"]
        out["alpha_bragg"] = float(anisotropy(B))
        return out

    if not best_valid:
        out["warnings"].append(
            "cdw_model was supplied explicitly but does not pass the "
            "geometry/intensity checks ("
            + "; ".join(best["reasons"][:3]) + ").  Intensities are reported "
            "as requested but alpha_CDW is not trustworthy.")

    out["cdw_model"] = {
        "name": best_name,
        "q_ratio_to_bragg": round(float(model_table[best_name][0]), 6),
        "rotation_deg": round(float(best_rot), 3),
        "selection": "user-specified" if cdw_model else "auto (max min-SNR)",
        "min_snr": (round(float(best_snr), 2)
                    if np.isfinite(best_snr) else None),
        "geometry_valid": bool(best_valid),
        "cdw_shell_independently_detected": bool(chosen["family_match"]),
        "enantiomorph_search": rotation_info.get(best_name),
    }
    runner = next((s for s in scored[1:] if s["name"] != best_name), None)
    if runner is not None:
        out["cdw_model"]["runner_up"] = {
            "name": runner["name"],
            "min_snr": (round(float(runner["min_snr"]), 2)
                        if np.isfinite(runner["min_snr"]) else None),
            "geometry_valid": bool(runner["valid"]),
            "family_match": bool(runner["family_match"]),
        }

    _rot = out["cdw_model"].get("rotation_deg")
    _unrotated = _rot is not None and abs(float(_rot)) < 1e-9
    _runner_unrotated = False
    if runner is not None:
        _rr = float(model_table[runner["name"]][1])
        _runner_unrotated = abs(_rr) < 1e-9
    out["cdw_model"]["identifiability"] = {
        "degenerate_family": bool(_unrotated and _runner_unrotated),
        "runner_up_geometry_valid": bool(runner["valid"]) if runner else None,
        "note": (
            "This model is unrotated, so it is separated from the other "
            "unrotated cells (2x2 / 3x3 / 4x4 ...) by RADIUS RATIO ALONE. "
            "With cdw_radius_rtol = 0.12 the acceptance windows around 1/3 "
            "and 1/4 are adjacent, so on a small field of view the "
            "assignment can move by one shell without any change in the "
            "data. Treat the cell label as provisional unless "
            "cdw_shell_independently_detected is true and the runner-up "
            "failed its geometry check."
            if (_unrotated and _runner_unrotated) else
            "The chosen cell is separated from its alternatives by rotation "
            "as well as radius, so the assignment is not radius-degenerate.")
    }
    if _unrotated and _runner_unrotated and runner is not None \
            and runner["valid"]:
        out["warnings"].append(
            f"Superlattice assignment {best_name!r} is radius-degenerate "
            f"with {runner['name']!r}: both are unrotated cells "
            "distinguished only by their radius ratio, and both passed the "
            "geometry check. The cell label should be treated as "
            "provisional.")

    C = best["intensities"]
    cdw_rows = best["rows"]

    if not all(np.isfinite(C)):
        out["status"] = "insufficient"
        out["message"] = "CDW photometry produced undefined intensities."
        out["bragg_peaks"], out["cdw_peaks"] = bragg_rows, cdw_rows
        return out

    n_resolved = sum(1 for r in cdw_rows
                     if r["snr"] is not None and r["snr"] >= min_cdw_snr)
    if n_resolved < 3:
        out["warnings"].append(
            f"Only {n_resolved}/3 CDW peaks reach SNR {min_cdw_snr:g}.  A "
            "large alpha here indicates unidirectional (1Q / stripe / "
            "nematic) order rather than an anisotropic 3Q state; 1Q order "
            "is not chiral.")

    weak_integrated = [r["index"] for r in cdw_rows
                       if r.get("snr_integrated") is not None
                       and r["snr_integrated"] < 3.0]
    if weak_integrated:
        out["warnings"].append(
            "Direction(s) " + ", ".join(str(i) for i in weak_integrated) +
            " clear the single-pixel SNR gate but their integrated aperture "
            "power is below 3 sigma.  The peak is speckle-dominated: the "
            "statistic used for the domain cut and the statistic alpha is "
            "built from disagree for this map.")

    C_legacy = [r.get("intensity_legacy") for r in cdw_rows]
    B_legacy = [r.get("intensity_legacy") for r in bragg_rows]

    non_positive = [i + 1 for i, c in enumerate(C) if not (c > 0)]
    alpha_measurable = not non_positive
    if non_positive:
        out["warnings"].append(
            "CDW aperture(s) " + ", ".join(str(i) for i in non_positive) +
            " integrate to <= 0 after background subtraction, so the peak is "
            "at or below the local noise floor and alpha is NOT MEASURABLE. "
            "It is reported as null rather than clamped; "
            "`alpha_cdw_lower_bound` carries the one-sided information.")

    alpha_cdw = anisotropy(C) if alpha_measurable else float("nan")
    alpha_bragg = anisotropy(B)
    alpha_corr = (float(alpha_cdw - alpha_bragg)
                  if np.isfinite(alpha_cdw) and np.isfinite(alpha_bragg)
                  else float("nan"))
    tipscaled = _tipscaled_excess(cdw_rows, bragg_rows, alpha_cdw)
    R = [float(c / b) if (np.isfinite(b) and b > 0) else float("nan")
         for c, b in zip(C, B)]
    alpha_norm = anisotropy(R)

    alpha_lower_bound = None
    if not alpha_measurable:
        floors = []
        for c, r in zip(C, cdw_rows):
            s = r.get("intensity_sigma")
            floors.append(max(float(c), float(s) if s else 0.0))
        if all(f > 0 for f in floors):
            lb = anisotropy(floors)
            alpha_lower_bound = round(float(lb), 4) if np.isfinite(lb) else None

    sig = [r.get("intensity_sigma") for r in cdw_rows]
    bgs = [r.get("background") for r in cdw_rows]
    naps = [r.get("n_aperture") for r in cdw_rows]
    if not all(isinstance(n, (int, float)) and n > 0 for n in naps):
        naps = [float(np.pi * (r_ap / df) ** 2)] * 3
    unc = (_alpha_uncertainty(C, sig, bgs, naps, n_draw=int(alpha_mc_draws))
           if int(alpha_mc_draws) > 0 else
           {"alpha_ci68": None, "alpha_ci95": None,
            "alpha_bias_equal_null": None,
            "p_value_equal_intensity": None, "n_draw": 0})

    alpha_cdw_legacy = float("nan")
    legacy_clamp_applied = False
    if all(x is not None and np.isfinite(x) for x in C_legacy):
        cl = list(C_legacy)
        if min(cl) <= 0:
            legacy_clamp_applied = True
            floor = max(1e-12, 1e-6 * max(cl))
            cl = [max(c, floor) for c in cl]
        alpha_cdw_legacy = anisotropy(cl)
    alpha_bragg_legacy = float("nan")
    if all(x is not None and np.isfinite(x) for x in B_legacy):
        bl = list(B_legacy)
        if min(bl) <= 0:
            floor = max(1e-12, 1e-6 * max(bl))
            bl = [max(b, floor) for b in bl]
        alpha_bragg_legacy = anisotropy(bl)

    cdw_angles = [r["angle_deg"] for r in cdw_rows]
    bragg_angles = [r["angle_deg"] for r in bragg_rows]

    out.update({
        "status": "ok",
        "estimator_version": ESTIMATOR_VERSION,
        "aperture_overlap": bool(aperture_overlap),
        "annulus_reaches_neighbour": bool(
            out["method"].get("annulus_reaches_neighbour") or False),
        "C": [float(c) for c in C],
        "B": [float(b) for b in B],
        "R": R,
        "alpha_cdw": (float(alpha_cdw) if np.isfinite(alpha_cdw) else None),
        "alpha_cdw_lower_bound": alpha_lower_bound,
        "alpha_measurable": bool(alpha_measurable),
        "alpha_bragg": (float(alpha_bragg) if np.isfinite(alpha_bragg)
                        else None),
        "alpha_corrected": (alpha_corr if np.isfinite(alpha_corr) else None),
        "alpha_excess_over_bragg": (alpha_corr if np.isfinite(alpha_corr)
                                    else None),
        "tip_scaled_diagnostic": tipscaled,
        "alpha_cdw_tipcorrected": tipscaled["alpha_cdw_tipcorrected"],
        "alpha_excess_tipscaled": tipscaled["alpha_excess_tipscaled"],
        "excess_statistic_disagreement": None,
        "alpha_normalized": (float(alpha_norm) if np.isfinite(alpha_norm)
                             else None),
        "alpha_cdw_legacy": (float(alpha_cdw_legacy)
                             if np.isfinite(alpha_cdw_legacy) else None),
        "legacy_clamp_applied": bool(legacy_clamp_applied),
        "alpha_bragg_legacy": (float(alpha_bragg_legacy)
                               if np.isfinite(alpha_bragg_legacy) else None),
        "alpha_uncertainty": unc,
        "cdw_peaks": cdw_rows,
        "bragg_peaks": bragg_rows,
        "cdw_angles_deg": cdw_angles,
        "bragg_angles_deg": bragg_angles,
        "n_directions_resolved": n_resolved,
        "width_bins_cdw": [r.get("width_bins") for r in cdw_rows],
        "width_bins_bragg": [r.get("width_bins") for r in bragg_rows],
        "width_bins_cdw_corrected": [r.get("width_bins_corrected")
                                     for r in cdw_rows],
        "width_saturated_cdw": [bool(r.get("width_saturated"))
                                for r in cdw_rows],
        "hierarchy_sense_cdw": _hierarchy_sense(cdw_angles, C),
        "hierarchy_sense_normalized": _hierarchy_sense(cdw_angles, R),
        "conjugate_check": {
            "aperture_pixel_mismatch": (int(max(herm_res))
                                        if herm_res else None),
            "passed": (not herm_res) or max(herm_res) == 0,
            "is_identity": True,
            "note": "The +q/-q INTENSITY comparison this replaces was "
                    "identically zero on every possible input: |F(+q)| == "
                    "|F(-q)| holds exactly for any real image, so it could "
                    "not fail and was not a check, though it was reported "
                    "as one. What is reported now is an aperture indexing "
                    "self-test: the +q and -q apertures must contain the "
                    "same pixel count, which fails only when a peak lies "
                    "near the even-N Nyquist edge and its -q aperture is "
                    "clipped by the array boundary - a real, silent bias on "
                    "the mean of the two. Passing carries no evidence about "
                    "the data.",
            "not_tested": "Split-half reproducibility - whether the three "
                          "intensities agree between the first and second "
                          "halves of the scan - is the check that would "
                          "carry evidence about this measurement, and it is "
                          "not implemented. It would catch drift, a tip "
                          "change mid-scan and a domain boundary inside the "
                          "field of view, none of which any +q/-q statistic "
                          "can see.",
        },
    })

    out["hermitian_check"] = {
        "max_relative_residual": 0.0,
        "passed": True,
        "superseded_by": "conjugate_check",
        "note": "Retained for compatibility. This residual is zero by "
                "construction for any real input and is not a check; read "
                "`conjugate_check` instead.",
    }

    _wc = [w for w in (out.get("width_bins_cdw_corrected") or [])
           if isinstance(w, (int, float))]
    out["width_accuracy"] = {
        "max_width_bins": (round(float(max(_wc)), 3) if _wc else None),
        "tens_of_percent_regime": bool(_wc and max(_wc) > 5.0),
        "drives_aperture": True,
        "drives_domain_cut": True,
        "note": ("The corrected width is accurate to tens of percent above "
                 "~5 bins, and it sizes the aperture that produces alpha "
                 "AND gates the domain cut that admits alpha. Those two "
                 "uses are not independent, so a width error is not "
                 "averaged away by the cut. Breaking the coupling needs a "
                 "joint peak+background fit rather than aperture "
                 "photometry; that fit is not implemented."),
    }
    if out["width_accuracy"]["tens_of_percent_regime"]:
        out["warnings"].append(
            "At least one corrected peak width exceeds 5 bins "
            f"({out['width_accuracy']['max_width_bins']:.2f}), where the "
            "moment estimator is accurate to tens of percent rather than "
            "percent. That width both sizes the integration aperture and "
            "gates the calibrated domain cut, so its error enters the "
            "measurement and its admissibility check together.")

    _DELTA_MIN = 0.2
    _raw = out.get("alpha_excess_over_bragg")
    _tip = out.get("alpha_excess_tipscaled")
    if isinstance(_raw, float) and isinstance(_tip, float):
        _raw_pass = _raw > _DELTA_MIN
        _tip_pass = _tip > _DELTA_MIN
        out["excess_statistic_disagreement"] = bool(_raw_pass != _tip_pass)
        out["delta_criterion_raw"] = _raw_pass
        out["delta_criterion_tipscaled"] = _tip_pass
        out["delta_criterion_threshold"] = _DELTA_MIN
        if _raw_pass != _tip_pass:
            out["warnings"].append(
                "EXCESS-STATISTIC SENSITIVE: the delta criterion "
                f"(> {_DELTA_MIN:g}) is "
                f"{'met' if _raw_pass else 'not met'} by the raw excess "
                f"({_raw:.3f}) but "
                f"{'met' if _tip_pass else 'not met'} by the q^2-scaled "
                f"tip-removed excess ({_tip:.3f}). The scorecard thresholds "
                "the raw one, which over-subtracts the Bragg anisotropy by "
                "roughly 9x for a 3x3 cell; the tip-scaled value is the "
                "physically correct statistic but has no fitted threshold. "
                "This map's delta verdict is an artifact of which statistic "
                "is thresholded, not a measurement.")

    if out["conjugate_check"]["aperture_pixel_mismatch"]:
        out["warnings"].append(
            "The +q and -q apertures differ by "
            f"{out['conjugate_check']['aperture_pixel_mismatch']} pixels, so "
            "at least one peak lies close enough to the Nyquist edge that "
            "its conjugate aperture is clipped by the array boundary. The "
            "mean of the two intensities is biased low for that direction.")

    return out


def analyze_didv_anisotropy(image_path: str,
                            colormap: Optional[str] = None,
                            **kwargs: Any) -> Dict[str, Any]:
    channel = kwargs.pop("channel", None)
    direction = kwargs.pop("direction", "forward")
    bias = kwargs.pop("bias", None)
    try:
        field, load_info = load_didv_scalar(image_path, colormap=colormap,
                                            channel=channel,
                                            direction=direction, bias=bias)
    except Exception as exc:
        return {
            "status": "failed",
            "source_path": image_path,
            "warnings": [],
            "message": f"could not load field: {exc}",
        }

    res = analyze_didv_field(field, source_label=image_path,
                             scalar_field_info=load_info, **kwargs)

    info = load_info or {}
    res["data_class"] = info.get("data_class", "rendered")
    res["quantitative"] = bool(info.get("quantitative", False))
    res["input_channel"] = info.get("channel")
    res["input_bias_v"] = info.get("bias_v")
    return res


def format_anisotropy_report(res: Dict[str, Any]) -> str:
    L = ["=" * 72,
         "dI/dV PEAK ANISOTROPY  (background-subtracted integrated |FFT|^2)",
         "=" * 72]

    if res.get("status") != "ok":
        L += [f"Status: {res.get('status', '?').upper()}",
              f"Message: {res.get('message', '-')}"]
        for w in res.get("warnings", []):
            L.append(f"  ! {w}")
        L.append("=" * 72)
        return "\n".join(L)

    L += [f"Image         : {res['image']['Nx']} x {res['image']['Ny']} px",
          f"Aperture      : r = {res['method']['aperture_radius_bins']:g} "
          f"bins, identical for every peak"
          + ("  [mode " + str(res['method'].get('aperture_mode')) + "]"
             if res['method'].get('aperture_mode') else ""),
          f"Bragg anchor  : {res.get('bragg_source')}",
          f"CDW model     : {res['cdw_model']['name']} "
          f"(|q|/|q_B| = {res['cdw_model']['q_ratio_to_bragg']:.4f}, "
          f"rot = {res['cdw_model']['rotation_deg']:+.2f} deg)",
          "-" * 72,
          f"{'dir':>4} {'angle':>8} {'C_i':>14} {'B_i':>14} {'R_i=C/B':>10} "
          f"{'SNR_C':>7}"]

    for k in range(3):
        c, b, r = res["C"][k], res["B"][k], res["R"][k]
        snr = res["cdw_peaks"][k]["snr"]
        L.append(f"{k + 1:>4} {res['cdw_angles_deg'][k]:>7.1f}d "
                 f"{c:>14.5g} {b:>14.5g} {r:>10.4g} "
                 f"{(snr if snr is not None else float('nan')):>7.1f}")

    def _fmt(value: Any, spec: str = "{:.4f}") -> str:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return "not measurable"
        return spec.format(f) if np.isfinite(f) else "not measurable"

    unc = res.get("alpha_uncertainty") or {}
    ci = unc.get("alpha_ci68")
    ci_txt = (f"   68% CI [{ci[0]:.4f}, {ci[1]:.4f}]"
              if isinstance(ci, (list, tuple)) and len(ci) == 2 else "")
    p_eq = unc.get("p_value_equal_intensity")
    bias = unc.get("alpha_bias_equal_null")

    L += ["-" * 72,
          f"alpha_CDW              = {_fmt(res.get('alpha_cdw'))}{ci_txt}",
          f"alpha_Bragg            = {_fmt(res.get('alpha_bragg'))}",
          f"alpha_corrected        = {_fmt(res.get('alpha_corrected'), '{:+.4f}')}"
          "   (alpha_CDW - alpha_Bragg)",
          f"alpha_normalized       = {_fmt(res.get('alpha_normalized'))}"
          "   (from R_i = C_i / B_i)"]
    if res.get("alpha_cdw") is None and res.get("alpha_cdw_lower_bound"):
        L.append(f"alpha_CDW lower bound  = "
                 f"{res['alpha_cdw_lower_bound']:.4f}"
                 "   (a peak is at or below the noise floor)")
    if bias is not None:
        L.append(f"alpha under equal-I    = {bias:.4f}"
                 "   (median of the noise-only null for THIS map)")
    if p_eq is not None:
        L.append(f"p(equal intensities)   = {p_eq:.4g}"
                 "   (photometric noise only; no artifact model)")
    if res.get("alpha_cdw_legacy") is not None:
        L.append(f"alpha_CDW (v1 legacy)  = {res['alpha_cdw_legacy']:.4f}"
                 "   (median-subtracted; the alpha_crit=0.374 input)")
    L += ["-" * 72,
          f"CDW peaks resolved     : {res['n_directions_resolved']}/3",
          f"Intensity hierarchy    : {res['hierarchy_sense_cdw']} "
          f"(normalized: {res['hierarchy_sense_normalized']})  [DESCRIPTIVE]",
          "    A cyclic sense exists for any three unequal numbers, including",
          "    pure noise.  This is the raw sense only: no licensing against",
          "    the artifact screen, no stability test, no tip-quadrupole",
          "    control.  didv_handedness applies all three and is what turns",
          "    a sense into a handedness; do not quote this line as one.",
          f"Hermitian +q/-q check  : "
          f"{'ok' if res['conjugate_check']['passed'] else 'CLIPPED'}"
          " (identity, not evidence)"]

    sens = res.get("aperture_sensitivity") or {}
    if sens.get("status") == "measured":
        L += ["-" * 72,
              f"Aperture systematic    : alpha = "
              f"{sens.get('alpha_cdw_primary'):.4f} calibrated "
              f"({sens.get('aperture_bins_primary')} bins)  vs "
              f"{sens.get('alpha_cdw_capped'):.4f} separation-capped "
              f"({sens.get('aperture_bins_capped')} bins)",
              f"                         delta = "
              f"{sens.get('delta_alpha'):+.4f}  ->  "
              f"{sens.get('bias_direction')}",
              f"                         {sens.get('bias_mechanism')}"]
        if sens.get("verdict_robust") is not None:
            L.append("                         verdict "
                     + ("ROBUST" if sens["verdict_robust"] else "NOT ROBUST")
                     + " to the aperture procedure "
                     + f"({sens.get('verdict_primary')} -> "
                       f"{sens.get('verdict_capped')})")
    elif sens.get("status") == "unavailable":
        L += ["-" * 72,
              "Aperture systematic    : not measurable - "
              + str(sens.get("reason", ""))]
    return "\n".join(L)


def plot_anisotropy_diagnostic(image_path: str, res: Dict[str, Any],
                               save_figure: Optional[str] = None,
                               max_order: float = FFT_VIEW_MAX_ORDER,
                               view_margin: float = FFT_VIEW_MARGIN,
                               channel: Optional[str] = None,
                               direction: Optional[str] = None,
                               bias: Optional[float] = None):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    label_family = _peak_label_family()

    if res.get("status") != "ok":
        raise ValueError("diagnostic requires a successful analysis")

    _sf = res.get("scalar_field") or {}
    if channel is None:
        channel = _sf.get("channel")
    if direction is None:
        direction = _sf.get("direction")
    if direction not in ("forward", "backward"):
        direction = "forward"
    if bias is None and _sf.get("is_spectroscopic_grid"):
        bias = _sf.get("bias_v")
    field, _ = load_didv_scalar(
        image_path, colormap=_sf.get("colormap"),
        channel=channel, direction=direction, bias=bias)
    spec = power_spectrum(field)
    power = spec["power"]
    Ny, Nx = spec["shape"]
    cy, cx = spec["center"]

    r_ap = res["method"]["aperture_radius_cycles_per_px"]

    if not res.get("bragg_peaks"):
        raise ValueError("plot_anisotropy_diagnostic needs an identified "
                         "Bragg family to set the second-order view limit")
    q_bragg = float(max(r["q_cycles_per_px"] for r in res["bragg_peaks"]))
    r_show = float(max_order) * q_bragg
    r_view = r_show * float(view_margin)
    hx = min(cx - 1, int(np.ceil(r_view * Nx)))
    hy = min(cy - 1, int(np.ceil(r_view * Ny)))

    sub = np.log1p(power[cy - hy:cy + hy + 1, cx - hx:cx + hx + 1])
    gy, gx = np.mgrid[-hy:hy + 1, -hx:hx + 1]
    dd = np.hypot(gy, gx)
    rr = np.hypot(gx / float(Nx), gy / float(Ny))
    beyond_second_order = rr > r_view
    ref = sub[(dd > 3) & ~beyond_second_order]
    sub = np.where(beyond_second_order, np.nan, sub)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    _fny, _fnx = np.asarray(field).shape[:2]
    axes[0].imshow(field, cmap="gray", origin="upper",
                   extent=[0, _fnx, 0, _fny])
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)   [$y$ up, as displayed]")

    ax = axes[1]
    try:
        _cmap = plt.get_cmap("magma").copy()
    except AttributeError:
        import copy as _copy
        _cmap = _copy.copy(plt.get_cmap("magma"))
    _cmap.set_bad(_cmap(0.0))
    ax.imshow(sub[::-1, :], origin="lower", cmap=_cmap, aspect="equal",
              extent=[-hx / Nx, hx / Nx, -hy / Ny, hy / Ny],
              vmin=np.percentile(ref, 40), vmax=np.percentile(ref, 99.8))
    ax.set_xlim(-r_view, r_view)
    ax.set_ylim(-r_view, r_view)

    _rr_all = np.hypot((np.arange(Nx) - cx)[None, :] / float(Nx),
                       (np.arange(Ny) - cy)[:, None] / float(Ny))
    _anchor_pw = float(np.median([
        power[int(round(-r["q_cycles_per_px"] * np.sin(
                    np.radians(r["angle_deg"])) * Ny)) + cy,
              int(round(r["q_cycles_per_px"] * np.cos(
                    np.radians(r["angle_deg"])) * Nx)) + cx]
        for r in res["bragg_peaks"]]))
    _outside = (_rr_all > r_view) & (_rr_all < 0.5)
    if _outside.any() and _anchor_pw > 0:
        _n_stronger = int(np.count_nonzero(power[_outside] > _anchor_pw))
        if _n_stronger > 0:
            ax.text(0.5, 0.02,
                    f"{_n_stronger} bin(s) outside this view exceed the "
                    "anchor's own peak power \u2014 the Bragg assignment "
                    "may be too small, in which case the true order reached "
                    "here is higher than stated",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8, color="#ffb570", alpha=0.9, zorder=6,
                    wrap=True)

    marker_r = 0.085 * r_view
    for rows, color, tag in ((res["bragg_peaks"], "cyan", "B"),
                             (res["cdw_peaks"], "lime", "C")):
        for row in rows:
            th = np.radians(row["angle_deg"])
            q = row["q_cycles_per_px"]
            if q > r_show:
                continue
            for sgn in (+1, -1):
                u, v = sgn * q * np.cos(th), sgn * q * np.sin(th)
                ax.add_patch(plt.Circle((u, v), min(marker_r, r_ap),
                                        fill=False, ec=color, lw=1.6))
            ax.annotate(f"{tag}{row['index']}",
                        (q * np.cos(th), q * np.sin(th)),
                        color=color, fontsize=PEAK_LABEL_FONTSIZE,
                        fontweight="bold", family=label_family,
                        xytext=(7, 7), textcoords="offset points",
                        path_effects=[
                            path_effects.Stroke(linewidth=2.5,
                                                foreground="black"),
                            path_effects.Normal()])

    ax.set_xlabel("$f_x$ (cycles / pixel)")
    ax.set_ylabel("$f_y$ (cycles / pixel)   [$y$ up, as displayed]")
    try:
        fig.set_layout_engine("constrained")
    except AttributeError:
        fig.tight_layout()

    if save_figure:
        fig.savefig(save_figure, dpi=150)
    plt.close(fig)
    return fig


def overlap_risk(res: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    m = ((res or {}).get("method") or {}) if isinstance(res, dict) else {}
    overlap = bool((res or {}).get("aperture_overlap"))
    annulus = bool(m.get("annulus_reaches_neighbour"))
    sev = m.get("aperture_overlap_severity")
    return {
        "at_risk": bool(overlap or annulus),
        "aperture_overlap": overlap,
        "annulus_reaches_neighbour": annulus,
        "aperture_overlap_severity": sev,
        "min_peak_separation_bins": m.get("min_peak_separation_bins"),
        "separation_capped_target_bins": m.get(
            "separation_capped_target_bins"),
    }


def compare_aperture_variants(
        primary: Optional[Dict[str, Any]],
        capped: Optional[Dict[str, Any]],
        alpha_crit: Optional[float] = None,
        separation_fraction: float = SEPARATION_FRACTION_RECOMMENDED,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": "unavailable",
        "separation_fraction": float(separation_fraction),
        "alpha_crit": (float(alpha_crit) if alpha_crit is not None else None),
        "verdict_robust": None,
    }
    if not isinstance(primary, dict) or primary.get("status") != "ok":
        out["reason"] = "No primary measurement to compare against."
        return out

    a_p = primary.get("alpha_cdw")
    m_p = primary.get("method") or {}
    out["aperture_bins_primary"] = m_p.get("aperture_radius_bins")
    out["alpha_cdw_primary"] = a_p

    if not isinstance(capped, dict) or capped.get("status") != "ok":
        out["reason"] = (
            "The capped counterfactual did not yield a measurement"
            + (f" ({capped.get('message')})"
               if isinstance(capped, dict) and capped.get("message") else "")
            + ", which is itself informative: the aperture the cap allows is "
              "too small to resolve this map's peaks, so the overlap here is "
              "not removable by shrinking the aperture.")
        return out

    a_c = capped.get("alpha_cdw")
    m_c = capped.get("method") or {}
    out["aperture_bins_capped"] = m_c.get("aperture_radius_bins")
    out["alpha_cdw_capped"] = a_c
    if a_p is None or a_c is None:
        out["reason"] = "One of the two measurements has no alpha_cdw."
        return out

    a_p, a_c = float(a_p), float(a_c)
    delta = a_c - a_p
    out.update(status="measured",
               delta_alpha=round(delta, 5),
               alpha_band=[round(min(a_p, a_c), 5), round(max(a_p, a_c), 5)],
               alpha_ratio=(round(a_c / a_p, 4) if a_p > 0 else None))

    tol = 0.01
    if delta > tol:
        out["bias_direction"] = "deployed alpha biased LOW"
        out["bias_mechanism"] = ("shared power between overlapping apertures "
                                 "dominates on this map")
    elif delta < -tol:
        out["bias_direction"] = "deployed alpha biased HIGH"
        out["bias_mechanism"] = ("background over-estimation from a "
                                 "neighbouring peak inside the annulus "
                                 "dominates on this map")
    else:
        out["bias_direction"] = "negligible"
        out["bias_mechanism"] = ("the two competing mechanisms cancel to "
                                 "within 0.01 in alpha on this map")

    if alpha_crit is None:
        out["reason"] = ("The bias is measured, but no alpha_crit was "
                         "available, so verdict robustness could not be "
                         "evaluated.")
        return out

    crit = float(alpha_crit)
    v_p = "above" if a_p >= crit else "below"
    v_c = "above" if a_c >= crit else "below"
    robust = bool(v_p == v_c)
    out.update(verdict_primary=v_p, verdict_capped=v_c,
               verdict_robust=robust,
               band_spans_threshold=bool(min(a_p, a_c) < crit <= max(a_p, a_c)))
    out["reason"] = (
        f"alpha = {a_p:.3f} with the calibrated aperture "
        f"({out['aperture_bins_primary']} bins) and {a_c:.3f} with the "
        f"separation cap at {separation_fraction:g} of the nearest peak "
        f"separation ({out['aperture_bins_capped']} bins); the "
        f"{out['bias_direction']} by {abs(delta):.3f}. "
        + (f"Both fall on the same side of alpha_crit = {crit:.3f}, so the "
           f"'{v_p}' verdict does not depend on the aperture procedure."
           if robust else
           f"They fall on OPPOSITE sides of alpha_crit = {crit:.3f} "
           f"('{v_p}' capped to '{v_c}'), so this verdict is an artifact of "
           f"the aperture choice and cannot be treated as conservative in "
           f"either direction."))
    return out


def didv_anisotropy_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    import base64

    didv_b64 = state.get("input_didv_base64")
    if not didv_b64:
        state["didv_anisotropy"] = {
            "status": "unavailable",
            "message": "No dI/dV map was provided.",
            "screen": {"verdict": "no_measurement",
                       "headline": "No measurement",
                       "reason": "No dI/dV map was provided."},
        }
        return state

    meta = state.get("metadata") or {}
    ext = _sdio.normalise_extension(meta.get("didv_ext"))

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "didv" + ext)
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(didv_b64))

            res = analyze_didv_anisotropy(
                path,
                colormap=meta.get("didv_colormap"),
                cdw_model=meta.get("cdw_model"),
                channel=meta.get("didv_channel"),
                direction=meta.get("didv_direction", "forward"),
                bias=meta.get("didv_bias"),
                aperture_bins="auto",
            )

            if res.get("status") == "ok":
                fig_path = os.path.join(tmp, "didv_anisotropy.png")
                try:
                    plot_anisotropy_diagnostic(
                        path, res, save_figure=fig_path,
                        channel=meta.get("didv_channel"),
                        direction=meta.get("didv_direction", "forward"),
                        bias=meta.get("didv_bias"))
                    with open(fig_path, "rb") as fh:
                        state["didv_anisotropy_figure_base64"] = \
                            base64.b64encode(fh.read()).decode("utf-8")
                except Exception as exc:
                    res.setdefault("warnings", []).append(
                        f"diagnostic figure failed: {exc}")

            try:
                from didv_alpha_crit import screen_result
                res["screen"] = screen_result(res)
            except Exception as exc:
                res.setdefault("warnings", []).append(
                    f"alpha_crit screen unavailable: {exc}")

            try:
                risk = overlap_risk(res)
                res["aperture_overlap_risk"] = risk
                screen = res.get("screen") or {}
                if res.get("status") == "ok" and risk["at_risk"]:
                    capped = analyze_didv_anisotropy(
                        path,
                        colormap=meta.get("didv_colormap"),
                        cdw_model=meta.get("cdw_model"),
                        channel=meta.get("didv_channel"),
                        direction=meta.get("didv_direction", "forward"),
                        bias=meta.get("didv_bias"),
                        aperture_bins="auto",
                        separation_fraction=SEPARATION_FRACTION_RECOMMENDED,
                    )
                    sens = compare_aperture_variants(
                        res, capped, alpha_crit=screen.get("alpha_crit"))
                else:
                    sens = {
                        "status": "not_required",
                        "verdict_robust": True,
                        "separation_fraction":
                            SEPARATION_FRACTION_RECOMMENDED,
                        "reason": ("Apertures do not overlap and no "
                                   "neighbouring peak lies inside the "
                                   "background annulus, so neither "
                                   "aperture-geometry bias is active and "
                                   "the verdict cannot depend on the cap."),
                    }
                res["aperture_sensitivity"] = sens

                if isinstance(screen, dict) and screen:
                    screen["aperture_sensitivity"] = sens
                    screen["verdict_robust_to_aperture"] = sens.get(
                        "verdict_robust")
                    if sens.get("verdict_robust") is False:
                        screen["aperture_overlap"] = True
                        screen.setdefault("notes", []).append(
                            "APERTURE-DEPENDENT VERDICT. "
                            + str(sens.get("reason", "")))
                        res.setdefault("warnings", []).append(
                            "The screen verdict flips when the aperture "
                            "separation cap is applied, so it is a property "
                            "of the aperture procedure and not of the map; "
                            "treat it as undetermined in both directions.")
                    elif sens.get("status") == "measured":
                        screen.setdefault("notes", []).append(
                            "Aperture-overlap counterfactual: "
                            + str(sens.get("reason", "")))
            except Exception as exc:
                res.setdefault("warnings", []).append(
                    f"aperture-overlap counterfactual unavailable: {exc}")

            res["report"] = format_anisotropy_report(res)
            state["didv_anisotropy"] = res
    except Exception as exc:
        state["didv_anisotropy"] = {
            "status": "failed", "message": str(exc),
            "screen": {"verdict": "no_measurement",
                       "headline": "No measurement", "reason": str(exc)},
        }

    return state
