import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import (distance_transform_edt, gaussian_filter,
                           gaussian_filter1d)


import stm_data_io as _sdio
FFT_CMAP = LinearSegmentedColormap.from_list(
    "fft_white_to_red",
    ["#ffffff", "#fbe4e4", "#f0a5a5", "#dc5a5a", "#b81a1a", "#7f0000"],
)
DEFAULT_FFT_CROP_FRACTION = 0.35

def estimate_crop_fraction(magnitude, dc_radius_frac=0.02, margin=1.7,
                           rel_threshold=0.05, n_shells=256,
                           axis_exclude_bins=2, min_fraction=0.02,
                           fallback=DEFAULT_FFT_CROP_FRACTION):
    mag = np.asarray(magnitude, dtype=np.float64)
    Ny, Nx = mag.shape
    cy, cx = Ny // 2, Nx // 2

    yy, xx = np.mgrid[0:Ny, 0:Nx]
    rn = np.hypot((yy - cy) / max(Ny / 2.0, 1e-12),
                  (xx - cx) / max(Nx / 2.0, 1e-12))

    valid = np.isfinite(mag) & (rn > float(dc_radius_frac)) & (rn <= 1.0)

    if axis_exclude_bins:
        k = int(axis_exclude_bins)
        valid &= ~((np.abs(yy - cy) <= k) | (np.abs(xx - cx) <= k))

    if valid.sum() < 256:
        return float(fallback)

    vals = mag[valid]
    med = float(np.median(vals))
    sigma = 1.4826 * float(np.median(np.abs(vals - med)))
    floor = med + 3.0 * sigma if sigma > 0 else med

    excess = np.where(valid, mag - floor, 0.0)
    np.clip(excess, 0.0, None, out=excess)
    peak_excess = float(excess.max())
    if not np.isfinite(peak_excess) or peak_excess <= 0:
        return float(fallback)

    shell = np.minimum((rn * n_shells).astype(int), n_shells - 1)
    shell_max = np.zeros(n_shells)
    np.maximum.at(shell_max, shell[valid], excess[valid])

    occupied = np.nonzero(shell_max >= rel_threshold * peak_excess)[0]
    if occupied.size == 0:
        return float(fallback)

    r_content = float(occupied.max() + 1) / n_shells
    return float(np.clip(margin * r_content, min_fraction, 1.0))


def crop_fft_region(data, axis_x, axis_y, crop_fraction):
    data = np.asarray(data)
    Ny, Nx = data.shape
    cy, cx = Ny // 2, Nx // 2
    crop_fraction = float(np.clip(crop_fraction, 1e-3, 1.0))

    half_ix = max(1, int(round(crop_fraction * (Nx // 2))))
    half_iy = max(1, int(round(crop_fraction * (Ny // 2))))
    x0, x1 = max(0, cx - half_ix), min(Nx, cx + half_ix + 1)
    y0, y1 = max(0, cy - half_iy), min(Ny, cy + half_iy + 1)

    sub = data[y0:y1, x0:x1]
    ax_x = np.asarray(axis_x)[x0:x1]
    ax_y = np.asarray(axis_y)[y0:y1]

    dx = float(ax_x[1] - ax_x[0]) if ax_x.size > 1 else 1.0
    dy = float(ax_y[1] - ax_y[0]) if ax_y.size > 1 else 1.0
    extent = [float(ax_x[0]) - 0.5 * dx, float(ax_x[-1]) + 0.5 * dx,
              float(ax_y[0]) - 0.5 * dy, float(ax_y[-1]) + 0.5 * dy]

    return {
        "data": sub,
        "extent": extent,
        "axis_x": ax_x,
        "axis_y": ax_y,
        "slices": (y0, y1, x0, x1),
        "center": (cy - y0, cx - x0),
        "half_width_x": max(abs(extent[0]), abs(extent[1])),
        "half_width_y": max(abs(extent[2]), abs(extent[3])),
    }


def _display_contrast(log_mag, center=None, dc_exclude_bins=None,
                      n_mad=3.0, hi_pct=99.9):
    log_mag = np.asarray(log_mag, dtype=np.float64)
    Ny, Nx = log_mag.shape
    cy, cx = (Ny // 2, Nx // 2) if center is None else center

    finite = np.isfinite(log_mag)
    if not finite.any():
        return None, None

    if dc_exclude_bins is None:
        dc_exclude_bins = max(6.0, 0.04 * min(Ny, Nx))

    yy, xx = np.mgrid[0:Ny, 0:Nx]
    outside_dc = ((yy - cy) ** 2 + (xx - cx) ** 2) > float(dc_exclude_bins) ** 2
    ref = log_mag[finite & outside_dc]
    if ref.size < 64:
        ref = log_mag[finite]

    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med)))
    sigma = 1.4826 * mad

    vmax = float(np.percentile(ref, hi_pct))
    vmin = med + n_mad * sigma if sigma > 0 else float(np.percentile(ref, 60.0))

    if not (vmax > vmin):
        vmin = float(np.percentile(ref, 50.0))
        vmax = float(np.nanmax(log_mag[finite]))
    if not (vmax > vmin):
        vmin = float(np.nanmin(log_mag[finite]))
        vmax = vmin + 1e-9
    return float(vmin), float(vmax)


_CMAP_CANDIDATES = (
    "viridis", "plasma", "inferno", "magma", "cividis", "jet", "turbo",
    "hot", "afmhot", "gist_heat", "gray", "bone", "copper", "cubehelix",
    "coolwarm", "seismic", "rainbow", "nipy_spectral", "terrain",
    "Blues", "Greens", "Reds", "Oranges", "Purples", "BuPu", "PuBu",
    "BuGn", "GnBu", "YlGnBu", "YlOrRd", "YlOrBr", "RdYlBu", "RdBu",
)


def _cmap_lut(name, n):
    import matplotlib
    try:
        cmap = matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        from matplotlib import cm
        cmap = cm.get_cmap(name)
    return np.asarray(cmap(np.linspace(0.0, 1.0, n)))[:, :3] * 255.0


def _fit_colormap(rgb, n_sample=20000, lut_size=256):
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None, np.inf

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


def _invert_colormap(rgb, name, lut_size=1024):
    from scipy.spatial import cKDTree
    lut = _cmap_lut(name, lut_size)
    flat = rgb.reshape(-1, 3).astype(np.float64)
    _, idx = cKDTree(lut).query(flat, k=1)
    return (idx / (lut_size - 1.0)).reshape(rgb.shape[:2])


def load_scalar_field(path, colormap_resid_tol=20.0,
                      channel=None, direction="forward", prefer="topo",
                      bias=None):
    return _sdio.load_field(
        path, colormap=None, colormap_resid_tol=colormap_resid_tol,
        channel=channel, direction=direction, prefer=prefer, bias=bias,
        image_loader=_load_rendered_scalar_field)


def _load_rendered_scalar_field(path, colormap=None,
                                colormap_resid_tol=20.0):
    ext = os.path.splitext(path)[1].lower()
    image_exts = set(_sdio.IMAGE_EXTENSIONS)
    if ext not in image_exts:
        raise ValueError(
            f"Unsupported file extension '{ext}' for the rendered-image "
            f"loader. Image formats: {sorted(image_exts)}; native data "
            f"formats are handled by load_scalar_field.")

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
        lum = np.asarray(im.convert("F"), dtype=np.float64) / 255.0

    info = {"requested_colormap": colormap}

    if np.allclose(rgb[..., 0], rgb[..., 1]) and \
            np.allclose(rgb[..., 1], rgb[..., 2]):
        info.update(method="grayscale", colormap=None,
                    note="image is already grayscale")
        return lum, info

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
                     "Relative peak intensities may be distorted.")
    return lum, info


def sanitize(img):
    img = np.asarray(img, dtype=np.float64)
    finite = np.isfinite(img)

    if not finite.any():
        return np.zeros_like(img)

    if finite.all():
        return img.copy()

    nearest = distance_transform_edt(
        ~finite, return_distances=False, return_indices=True)
    return img[tuple(nearest)]


def _remove_plane(img):
    Ny, Nx = img.shape
    y, x = np.mgrid[0:Ny, 0:Nx]
    A = np.column_stack([np.ones(img.size), x.ravel(), y.ravel()])
    coef, *_ = np.linalg.lstsq(A, img.ravel(), rcond=None)
    return img - (A @ coef).reshape(Ny, Nx)


CDW_MAX_PERIODS = 5.0
BACKGROUND_GUARD_FACTOR = 4.0
BACKGROUND_MIN_SIGMA_FRAC = 1.0 / 6.0


def subtract_background(img, bragg_period_px=None,
                        cdw_max_periods=CDW_MAX_PERIODS,
                        guard_factor=BACKGROUND_GUARD_FACTOR,
                        min_sigma_frac=BACKGROUND_MIN_SIGMA_FRAC):
    img = sanitize(np.asarray(img, dtype=np.float64))
    Ny, Nx = img.shape
    y, x = np.mgrid[0:Ny, 0:Nx]
    xn = (x - Nx / 2.0) / max(Nx / 2.0, 1.0)
    yn = (y - Ny / 2.0) / max(Ny / 2.0, 1.0)

    A = np.column_stack([np.ones(img.size), xn.ravel(), yn.ravel(),
                         (xn * xn).ravel(), (xn * yn).ravel(),
                         (yn * yn).ravel()])
    coef, *_ = np.linalg.lstsq(A, img.ravel(), rcond=None)
    out = img - (A @ coef).reshape(Ny, Nx)

    sigma_floor = float(min_sigma_frac) * min(Ny, Nx)
    lam_cdw = None
    if bragg_period_px is not None and np.isfinite(bragg_period_px) \
            and bragg_period_px > 0:
        lam_cdw = float(cdw_max_periods) * float(bragg_period_px)
        sigma = max(float(guard_factor) * lam_cdw, sigma_floor)
        scale_source = "bragg_period"
    else:
        sigma = sigma_floor
        scale_source = "image_fraction"

    offsets = np.median(out, axis=1)
    smooth = gaussian_filter1d(offsets, sigma, mode="reflect")
    out = out - (offsets - smooth)[:, None]

    bg = gaussian_filter(out, sigma, mode="reflect")
    out = out - bg

    def preserved(lam):
        return float(1.0 - np.exp(-2.0 * np.pi ** 2 * sigma ** 2 / lam ** 2))

    info = {
        "method": "quadratic surface + row-jitter alignment + Gaussian "
                  "high-pass",
        "sigma_px": round(float(sigma), 2),
        "scale_source": scale_source,
        "bragg_period_px": (round(float(bragg_period_px), 3)
                            if bragg_period_px else None),
        "cdw_max_period_px": (round(lam_cdw, 2) if lam_cdw else None),
        "cdw_preserved_fraction": (round(preserved(lam_cdw), 6)
                                   if lam_cdw else None),
        "half_removal_wavelength_px": round(
            float(np.pi * sigma * np.sqrt(2.0 / np.log(2.0))), 1),
    }
    return out, info

def analyze_image_fft(image_path, Lx=None, Ly=None, save_figure=None,
                      channel=None, direction="forward", bias=None,
                      prefer="topo"):
    img, scalar_info = load_scalar_field(image_path, channel=channel,
                                         direction=direction,
                                         prefer=prefer, bias=bias)
    img = sanitize(img)

    Ny, Nx = img.shape

    calibration_source = "caller" if (Lx is not None and Ly is not None) \
        else None
    if Lx is None or Ly is None:
        hx, hy, unit = _sdio.physical_size(scalar_info)
        if hx and hy:
            Lx, Ly = hx, hy
            calibration_source = f"file header ({unit or 'unknown unit'})"
            print(f"Calibration taken from the file header: "
                  f"{hx:.6g} x {hy:.6g} {unit or ''}".rstrip())
    scalar_info = dict(scalar_info or {})
    scalar_info["calibration_source"] = calibration_source

    processed = _remove_plane(img)

    window = np.outer(np.hanning(Ny), np.hanning(Nx))

    wsum = window.sum()
    weighted_mean = float((processed * window).sum() / wsum) if wsum > 0 \
        else float(processed.mean())
    mean_subtracted = processed - weighted_mean
    windowed_image = mean_subtracted * window

    fft_complex = np.fft.fftshift(np.fft.fft2(windowed_image))
    magnitude = np.abs(fft_complex)          
    power = magnitude**2                     
    log_magnitude = np.log1p(magnitude)      

    calibrated = Lx is not None and Ly is not None

    if (Lx is None) != (Ly is None):
        raise ValueError("Provide both Lx and Ly for calibration, or neither.")

    print("=" * 64)
    print(f"FFT analysis of: {image_path}")
    print(f"Image dimensions: Nx = {Nx} px (width), Ny = {Ny} px (height)")
    print(f"Scalar field: {scalar_info.get('note', scalar_info.get('method'))}")
    print("Preprocessing: plane removed, window-weighted mean subtracted")

    if calibrated:
        dx = Lx / Nx
        dy = Ly / Ny
        fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))   
        fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
        qx = 2 * np.pi * fx                             
        qy = 2 * np.pi * fy

        dfx = 1.0 / Lx
        dfy = 1.0 / Ly
        f_nyq_x = 1.0 / (2.0 * dx)
        f_nyq_y = 1.0 / (2.0 * dy)

        print(f"Calibrated: Lx = {Lx}, Ly = {Ly} (physical units)")
        print(f"Real-space sampling: dx = {dx:.6g}, dy = {dy:.6g} (unit/px)")
        print(f"Reciprocal-space resolution: dfx = {dfx:.6g}, "
              f"dfy = {dfy:.6g} (cycles/unit)")
        print(f"  (in q: dqx = {2*np.pi*dfx:.6g}, dqy = {2*np.pi*dfy:.6g} rad/unit)")
        print(f"Nyquist limits: fx_max = {f_nyq_x:.6g}, fy_max = {f_nyq_y:.6g} "
              f"(cycles/unit)")
        print(f"  (in q: qx_max = {2*np.pi*f_nyq_x:.6g}, "
              f"qy_max = {2*np.pi*f_nyq_y:.6g} rad/unit)")
        if np.isclose(dx, dy):
            print("Sampling is EQUAL in x and y (isotropic pixels).")
        else:
            print("WARNING: sampling differs in x and y (anisotropic pixels); "
                  "reciprocal-space distances are direction-dependent.")
        freq_label_x = r"$q_x$ (rad / unit)"
        freq_label_y = r"$q_y$ (rad / unit)"
    else:
        dx = dy = None
        fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0))  
        fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=1.0))
        qx = qy = None

        print("UNCALIBRATED analysis: no physical field of view supplied.")
        print("Frequencies are in cycles/pixel. Do NOT interpret these axes")
        print("in physical units (e.g., screenshots have no physical scale).")
        print(f"Reciprocal-space resolution: {1.0/Nx:.6g} (x), "
              f"{1.0/Ny:.6g} (y) cycles/pixel")
        print("Nyquist limit: 0.5 cycles/pixel in both directions.")
        freq_label_x = r"$f_x$ (cycles / pixel) — UNCALIBRATED"
        freq_label_y = r"$f_y$ (cycles / pixel) — UNCALIBRATED"

    print("=" * 64)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("2D FFT diagnostic" + ("" if calibrated else " (UNCALIBRATED)"),
                 fontsize=14)

    ax = axes[0, 0]
    im0 = ax.imshow(img, cmap="gray", origin="upper")
    ax.set_title("Original grayscale")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    fig.colorbar(im0, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im1 = ax.imshow(mean_subtracted, cmap="gray", origin="upper")
    ax.set_title("Mean-subtracted")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    fig.colorbar(im1, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    im2 = ax.imshow(window, cmap="viridis", origin="upper")
    ax.set_title("2D Hanning window")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    fig.colorbar(im2, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    im3 = ax.imshow(windowed_image, cmap="gray", origin="upper")
    ax.set_title("Windowed image")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    fig.colorbar(im3, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    axis_x = qx if calibrated else fx
    axis_y = qy if calibrated else fy
    frac = estimate_crop_fraction(magnitude)
    crop = crop_fft_region(log_magnitude, axis_x, axis_y, frac)
    vmin, vmax = _display_contrast(crop["data"], center=crop["center"])
    im4 = ax.imshow(crop["data"], cmap=FFT_CMAP, origin="lower",
                    extent=crop["extent"], aspect="equal",
                    vmin=vmin, vmax=vmax)
    ax.set_title("log(1 + |FFT|)  [visualization only]")
    ax.set_xlabel(freq_label_x)
    ax.set_ylabel(freq_label_y)
    fig.colorbar(im4, ax=ax, fraction=0.046)

    axes[1, 2].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_figure:
        fig.savefig(save_figure, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"Diagnostic figure saved to: {save_figure}")
    plt.close(fig)

    return {
        "image": img,
        "mean_subtracted": mean_subtracted,
        "window": window,
        "windowed_image": windowed_image,
        "fft_complex": fft_complex,
        "magnitude": magnitude,      
        "power": power,              
        "log_magnitude": log_magnitude,  
        "fx": fx,
        "fy": fy,
        "qx": qx,
        "qy": qy,
        "dx": dx,
        "dy": dy,
        "calibrated": calibrated,
        "preprocessing": {"plane_removed": True,
                          "line_flattened": False,
                          "weighted_mean": weighted_mean},
        "scalar_field": scalar_info,
        "data_class": scalar_info.get("data_class", "rendered"),
        "quantitative": bool(scalar_info.get("quantitative", False)),
        "calibration_source": scalar_info.get("calibration_source"),
        "source_path": image_path,
        "source_ext": os.path.splitext(image_path)[1].lower(),
    }


def plot_peak_diagnostics(results, detection, save_figure=None):
    log_mag = results["log_magnitude"]
    calibrated = results["calibrated"]
    fx_ax, fy_ax = results["fx"], results["fy"]
    if calibrated:
        axis_x, axis_y = results["qx"], results["qy"]
        xs = lambda p: p["qx"]
        ys = lambda p: p["qy"]
        xlabel, ylabel = r"$q_x$ (rad / unit)", r"$q_y$ (rad / unit)"
    else:
        axis_x, axis_y = fx_ax, fy_ax
        xs = lambda p: p["fx"]
        ys = lambda p: p["fy"]
        xlabel = r"$f_x$ (cycles / pixel) — UNCALIBRATED"
        ylabel = r"$f_y$ (cycles / pixel) — UNCALIBRATED"

    frac = estimate_crop_fraction(results["magnitude"])
    half = max(abs(float(np.min(axis_x))), abs(float(np.max(axis_x))),
               abs(float(np.min(axis_y))), abs(float(np.max(axis_y))))
    marked = [max(abs(xs(p)), abs(ys(p)))
              for p in (detection["accepted"] + detection["rejected"])
              if xs(p) is not None and ys(p) is not None]
    if marked and half > 0:
        frac = max(frac, min(1.0, 1.15 * max(marked) / half))
    crop = crop_fft_region(log_mag, axis_x, axis_y, frac)
    extent = crop["extent"]

    fig, ax = plt.subplots(figsize=(9, 8))
    vmin, vmax = _display_contrast(crop["data"], center=crop["center"])
    im = ax.imshow(crop["data"], cmap=FFT_CMAP, origin="lower",
                   extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046,
                 label="log(1 + |FFT|)  [visualization only]")

    acc = detection["accepted"]
    drawn = set()
    for i, p in enumerate(acc):
        j = p["pair_index"]
        if j is not None and (j, i) not in drawn:
            q = acc[j]
            ax.plot([xs(p), xs(q)], [ys(p), ys(q)],
                    color="cyan", lw=0.8, alpha=0.7, zorder=2)
            drawn.add((i, j))
    if acc:
        ax.scatter([xs(p) for p in acc], [ys(p) for p in acc],
                   s=120, facecolors="none", edgecolors="lime",
                   linewidths=1.6, zorder=3, label=f"accepted ({len(acc)})")
    rej = detection["rejected"]
    if rej:
        ax.scatter([xs(p) for p in rej], [ys(p) for p in rej],
                   s=70, marker="x", color="red", linewidths=1.2,
                   zorder=3, label=f"rejected ({len(rej)})")
        for p in rej:
            ax.annotate(p["reject_reason"], (xs(p), ys(p)),
                        color="red", fontsize=7,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Reciprocal-space peak detection"
                 + ("" if calibrated else " (UNCALIBRATED)"))
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if save_figure:
        fig.savefig(save_figure, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"Peak-diagnostic figure saved to: {save_figure}")
    plt.close(fig)
    return fig


def _auto_crop_fraction(identification, axis_x, axis_y, margin=1.6,
                        default=DEFAULT_FFT_CROP_FRACTION):
    radii = [p["f_radius"] for p in
             (identification.get("bragg") or []) + (identification.get("cdw") or [])
             if p.get("f_radius")]
    half = max(abs(float(np.min(axis_x))), abs(float(np.max(axis_x))),
               abs(float(np.min(axis_y))), abs(float(np.max(axis_y))))
    if not radii or half <= 0:
        return default
    return float(np.clip(margin * max(radii) / half, 0.05, 1.0))


def plot_bragg_cdw_map(results, identification, save_figure=None):
    log_mag = results["log_magnitude"]
    calibrated = results["calibrated"]
    identification = identification or {}

    if calibrated:
        axis_x, axis_y = results["qx"], results["qy"]
        xlabel = r"$q_x$ (rad / unit)"
        ylabel = r"$q_y$ (rad / unit)"
    else:
        axis_x, axis_y = results["fx"], results["fy"]
        xlabel = r"$f_x$ (cycles / pixel)"
        ylabel = r"$f_y$ (cycles / pixel)"

    crop_fraction = _auto_crop_fraction(
        identification, results["fx"], results["fy"],
        default=estimate_crop_fraction(results["magnitude"]))
    crop = crop_fft_region(log_mag, axis_x, axis_y, float(crop_fraction))
    vmin, vmax = _display_contrast(crop["data"], center=crop["center"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")
    ax_img, ax = axes[0], axes[1]
    period_px = None
    fam = identification.get("bragg_family") or {}
    f_r = fam.get("radius")
    if f_r:
        f_px = float(f_r) * float(results["dx"]) if (
            calibrated and results.get("dx")) else float(f_r)
        if f_px > 0:
            period_px = 1.0 / f_px
    field, bg_info = subtract_background(results["image"],
                                         bragg_period_px=period_px)
    results.setdefault("display_background", bg_info)

    finite = field[np.isfinite(field)]
    if finite.size:
        lo, hi = np.percentile(finite, [1.0, 99.0])
        if not (hi > lo):
            lo, hi = float(finite.min()), float(finite.max()) + 1e-12
    else:
        lo, hi = 0.0, 1.0
    ax_img.imshow(field, cmap="gray", origin="upper", vmin=lo, vmax=hi)
    ax_img.set_xlabel("x (px)")
    ax_img.set_ylabel("y (px)")
    ax_img.set_title("Background-subtracted topograph")

    im = ax.imshow(crop["data"], cmap=FFT_CMAP, origin="lower",
                   extent=crop["extent"], aspect="equal",
                   vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, label="log(1 + |FFT|)")

    ax.set_xlabel(xlabel + ("" if calibrated else "  -- UNCALIBRATED"))
    ax.set_ylabel(ylabel + ("" if calibrated else "  -- UNCALIBRATED"))
    ax.set_title("Reciprocal space: FFT modulus"
                 + ("" if calibrated else " (UNCALIBRATED)"))

    if save_figure:
        fig.savefig(save_figure, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"Bragg/CDW figure saved to: {save_figure}")
    plt.close(fig)
    return fig
