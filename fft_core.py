import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.ndimage import distance_transform_edt

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FFT_CMAP = LinearSegmentedColormap.from_list(
    "fft_white_to_red",
    ["#ffffff", "#fbe4e4", "#f0a5a5", "#dc5a5a", "#b81a1a", "#7f0000"],
)
DEFAULT_FFT_CROP_FRACTION = 0.35

def _central_crop(shape, axis_x, axis_y, crop_fraction):
    Ny, Nx = shape
    cy, cx = Ny // 2, Nx // 2  

    crop_fraction = float(np.clip(crop_fraction, 1e-3, 1.0))

    hx = crop_fraction * max(abs(float(axis_x.min())), abs(float(axis_x.max())))
    hy = crop_fraction * max(abs(float(axis_y.min())), abs(float(axis_y.max())))
    xlim = (-hx, hx)
    ylim = (-hy, hy)

    half_ix = max(1, int(round(crop_fraction * (Nx // 2))))
    half_iy = max(1, int(round(crop_fraction * (Ny // 2))))
    y0, y1 = max(0, cy - half_iy), min(Ny, cy + half_iy + 1)
    x0, x1 = max(0, cx - half_ix), min(Nx, cx + half_ix + 1)
    return xlim, ylim, (y0, y1, x0, x1)


def _fft_contrast(log_mag, window=None, dc_exclude_bins=3,
                  lo_pct=60.0, hi_pct=99.5):
    Ny, Nx = log_mag.shape
    cy, cx = Ny // 2, Nx // 2
    if window is None:
        y0, y1, x0, x1 = 0, Ny, 0, Nx
    else:
        y0, y1, x0, x1 = window

    sub = log_mag[y0:y1, x0:x1]
    finite = np.isfinite(sub)
    if not finite.any():
        return None, None
    sub_f = sub[finite]

    vmin = float(np.percentile(sub_f, lo_pct))

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dc = ((yy - cy) ** 2 + (xx - cx) ** 2) <= dc_exclude_bins ** 2
    non_dc = sub[(~dc) & finite]
    ref = non_dc if non_dc.size >= 16 else sub_f
    vmax = float(np.percentile(ref, hi_pct))

    if not (vmax > vmin):
        vmax = float(np.nanmax(sub_f))
    if not (vmax > vmin):
        vmin = float(np.nanmin(sub_f))
        vmax = vmin + 1e-9
    return vmin, vmax

def load_image_grayscale(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    with Image.open(path) as im:
        img = np.asarray(im.convert("F"), dtype=np.float64)

    if img.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array, got shape {img.shape}")

    return img


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


def _flatten_lines(img):
    return img - np.median(img, axis=1, keepdims=True)

def analyze_image_fft(image_path, Lx=None, Ly=None, show=True,
                      save_figure=None, remove_plane=True,
                      line_flatten=False,
                      fft_crop_fraction=DEFAULT_FFT_CROP_FRACTION):
    img = load_image_grayscale(image_path)
    img = sanitize(img)

    Ny, Nx = img.shape  

    processed = img
    if remove_plane:
        processed = _remove_plane(processed)
    if line_flatten:
        processed = _flatten_lines(processed)

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
    print("Preprocessing: "
          + ("plane removed" if remove_plane else "no plane removal")
          + (", line-flattened" if line_flatten else "")
          + ", window-weighted mean subtracted")

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
        extent = [qx.min(), qx.max(), qy.min(), qy.max()]
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
        extent = [fx.min(), fx.max(), fy.min(), fy.max()]

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
    xlim, ylim, crop_win = _central_crop(
        log_magnitude.shape, axis_x, axis_y, fft_crop_fraction)
    vmin, vmax = _fft_contrast(log_magnitude, window=crop_win)
    im4 = ax.imshow(log_magnitude, cmap=FFT_CMAP, origin="lower",
                    extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("log(1 + |FFT|)  [visualization only]")
    ax.set_xlabel(freq_label_x)
    ax.set_ylabel(freq_label_y)
    fig.colorbar(im4, ax=ax, fraction=0.046)

    axes[1, 2].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_figure:
        fig.savefig(save_figure, dpi=150)
        print(f"Diagnostic figure saved to: {save_figure}")
    if show:
        plt.show()
    else:
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
        "preprocessing": {"plane_removed": bool(remove_plane),
                          "line_flattened": bool(line_flatten),
                          "weighted_mean": weighted_mean},
        "source_path": image_path,
        "source_ext": os.path.splitext(image_path)[1].lower(),
    }


def plot_peak_diagnostics(results, detection, show=True, save_figure=None):
    log_mag = results["log_magnitude"]
    calibrated = results["calibrated"]
    fx_ax, fy_ax = results["fx"], results["fy"]
    if calibrated:
        extent = [results["qx"].min(), results["qx"].max(),
                  results["qy"].min(), results["qy"].max()]
        xs = lambda p: p["qx"]
        ys = lambda p: p["qy"]
        xlabel, ylabel = r"$q_x$ (rad / unit)", r"$q_y$ (rad / unit)"
    else:
        extent = [fx_ax.min(), fx_ax.max(), fy_ax.min(), fy_ax.max()]
        xs = lambda p: p["fx"]
        ys = lambda p: p["fy"]
        xlabel = r"$f_x$ (cycles / pixel) — UNCALIBRATED"
        ylabel = r"$f_y$ (cycles / pixel) — UNCALIBRATED"

    fig, ax = plt.subplots(figsize=(9, 8))
    vmin, vmax = _fft_contrast(log_mag, window=None)
    im = ax.imshow(log_mag, cmap=FFT_CMAP, origin="lower",
                   extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
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
        fig.savefig(save_figure, dpi=150)
        print(f"Peak-diagnostic figure saved to: {save_figure}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig