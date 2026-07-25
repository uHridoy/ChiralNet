import argparse

import numpy as np
from scipy.ndimage import maximum_filter

from fft_core import (
    DEFAULT_FFT_CROP_FRACTION,
    analyze_image_fft,
    plot_peak_diagnostics,
)
def make_dc_exclusion_mask(shape, dc_radius_bins=6, dc_radius_frac=None,
                           min_radius_bins=3):
    Ny, Nx = shape
    cy, cx = Ny // 2, Nx // 2  
    if dc_radius_frac is not None:
        ry = max(min_radius_bins, int(round(dc_radius_frac * Ny)))
        rx = max(min_radius_bins, int(round(dc_radius_frac * Nx)))
    else:
        ry = rx = max(min_radius_bins, int(dc_radius_bins))
    y, x = np.mgrid[0:Ny, 0:Nx]
    inside_dc = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0
    return ~inside_dc, (ry, rx)

def _find_local_maxima(data, neighborhood=3):
    footprint_max = maximum_filter(data, size=neighborhood, mode="nearest")
    return (data == footprint_max) & (data > 0)


def _radial_threshold_map(data, valid_mask, snr_threshold, n_radial_bins=32):
    Ny, Nx = data.shape
    cy, cx = Ny // 2, Nx // 2
    y, x = np.mgrid[0:Ny, 0:Nx]
    r = np.hypot((y - cy) / (Ny / 2.0), (x - cx) / (Nx / 2.0))
    r_norm = r / max(r.max(), 1e-12)
    r_idx = np.minimum((r_norm * n_radial_bins).astype(int),
                       n_radial_bins - 1)

    thr = np.full(n_radial_bins, np.nan)
    for b in range(n_radial_bins):
        vals = data[(r_idx == b) & valid_mask]
        if vals.size >= 16:
            med = float(np.median(vals))
            sig = 1.4826 * float(np.median(np.abs(vals - med)))
            thr[b] = med + snr_threshold * max(sig, 1e-300)
    last = np.nanmax(thr) if np.isfinite(thr).any() else np.inf
    for b in range(n_radial_bins):
        if np.isfinite(thr[b]):
            last = thr[b]
        else:
            thr[b] = last
    return thr[r_idx]


def _annulus_background(data, iy, ix, r_in, r_out, valid_mask=None):
    Ny, Nx = data.shape
    y0, y1 = max(0, iy - r_out), min(Ny, iy + r_out + 1)
    x0, x1 = max(0, ix - r_out), min(Nx, ix + r_out + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr2 = (yy - iy) ** 2 + (xx - ix) ** 2
    sel = (rr2 > r_in**2) & (rr2 <= r_out**2)
    if valid_mask is not None:
        sel &= valid_mask[y0:y1, x0:x1]
    ann = data[y0:y1, x0:x1][sel]
    if ann.size < 8:  
        return np.nan, np.nan
    med = float(np.median(ann))
    mad = float(np.median(np.abs(ann - med)))
    clipped = ann[ann <= med + 3.0 * 1.4826 * mad]
    if clipped.size >= 8:
        med = float(np.median(clipped))
        mad = float(np.median(np.abs(clipped - med)))
    sigma = 1.4826 * mad  
    return med, sigma


def _peak_width(data, iy, ix, background, r_max=16):
    Ny, Nx = data.shape
    apex = data[iy, ix] - background
    if not np.isfinite(apex) or apex <= 0:
        return np.nan, np.nan, 8
    half = apex / 2.0
    rays = ((0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1))
    half_widths, saturated = [], []
    for dy_, dx_ in rays:
        step = np.hypot(dy_, dx_)  
        hw, sat = float(r_max) * step, True
        for r in range(1, r_max + 1):
            y, x = iy + dy_ * r, ix + dx_ * r
            if not (0 <= y < Ny and 0 <= x < Nx):
                hw, sat = r * step, False  
                break
            if data[y, x] - background < half:
                prev = data[iy + dy_ * (r - 1), ix + dx_ * (r - 1)] \
                    - background
                cur = data[y, x] - background
                frac = (prev - half) / (prev - cur) if prev != cur else 0.5
                hw, sat = ((r - 1) + frac) * step, False
                break
        half_widths.append(hw)
        saturated.append(sat)

    hw_arr = np.asarray(half_widths)
    sat_arr = np.asarray(saturated)
    n_sat = int(sat_arr.sum())
    usable = hw_arr[~sat_arr] if (~sat_arr).any() else hw_arr
    fwhm = 2.0 * float(np.mean(usable))
    axis_w = [half_widths[0] + half_widths[1],
              half_widths[2] + half_widths[3],
              half_widths[4] + half_widths[5],
              half_widths[6] + half_widths[7]]
    anisotropy = float(max(axis_w) / max(min(axis_w), 1e-12))
    return fwhm, anisotropy, n_sat


def _subbin_offset(data, iy, ix):
    Ny, Nx = data.shape

    def off(vm, v0, vp):
        denom = vm - 2.0 * v0 + vp
        if denom >= 0 or not np.isfinite(denom):
            return 0.0
        return float(np.clip(0.5 * (vm - vp) / denom, -0.5, 0.5))

    dx = off(data[iy, ix - 1], data[iy, ix], data[iy, ix + 1]) \
        if 1 <= ix < Nx - 1 else 0.0
    dy = off(data[iy - 1, ix], data[iy, ix], data[iy + 1, ix]) \
        if 1 <= iy < Ny - 1 else 0.0
    return dy, dx


def _split_half_spectra(mean_subtracted):
    Ny, Nx = mean_subtracted.shape
    h = Ny // 2
    spectra = []
    for half in (mean_subtracted[:h, :], mean_subtracted[h:2 * h, :]):
        hy, hx = half.shape
        w = np.outer(np.hanning(hy), np.hanning(hx))
        wsum = w.sum()
        wmean = (half * w).sum() / wsum if wsum > 0 else half.mean()
        spectra.append(np.abs(np.fft.fftshift(
            np.fft.fft2((half - wmean) * w))))
    return spectra


def _split_consistency(half_spectra, u, v, snr_floor=3.0, search=2):
    snrs = []
    for F in half_spectra:
        hy, hx = F.shape
        ty = hy // 2 + int(round(v * hy))
        tx = hx // 2 + int(round(u * hx))
        pad = search + 6
        if not (pad <= ty < hy - pad and pad <= tx < hx - pad):
            return np.nan
        core = F[ty - search:ty + search + 1, tx - search:tx + search + 1]
        peak_val = float(core.max())
        box = F[ty - pad:ty + pad + 1, tx - pad:tx + pad + 1]
        ring = box.copy()
        ring[pad - search:pad + search + 1,
             pad - search:pad + search + 1] = np.nan
        ring_vals = ring[np.isfinite(ring)]
        if ring_vals.size < 12:
            return np.nan
        med = float(np.median(ring_vals))
        sig = 1.4826 * float(np.median(np.abs(ring_vals - med)))
        snrs.append((peak_val - med) / sig if sig > 0 else 0.0)

    s1, s2 = snrs
    if min(s1, s2) < snr_floor:
        return 0.0
    presence = min(1.0, min(s1, s2) / (2.0 * snr_floor))
    balance = min(s1, s2) / max(s1, s2)
    return float(presence * balance)


def _streak_score(data, iy, ix, background, half_len=6):
    Ny, Nx = data.shape

    def line_mean(dy_, dx_):
        vals = []
        for r in range(1, half_len + 1):
            for s in (+1, -1):
                y, x = iy + s * dy_ * r, ix + s * dx_ * r
                if 0 <= y < Ny and 0 <= x < Nx:
                    vals.append(data[y, x] - background)
        return float(np.mean(vals)) if vals else 0.0

    h = line_mean(0, 1)       
    v = line_mean(1, 0)       
    d = 0.5 * (line_mean(1, 1) + line_mean(1, -1))  
    d = max(d, 1e-12)
    ratio_h, ratio_v = h / d, v / d
    if ratio_h >= ratio_v:
        return ratio_h, "h"
    return ratio_v, "v"


def detect_peaks(results,
                 use_power=False,
                 dc_radius_bins=6,
                 dc_radius_frac=None,
                 neighborhood=3,
                 snr_threshold=5.0,
                 border_margin_frac=0.02,
                 streak_ratio_threshold=3.0,
                 streak_axis_tol_bins=2,
                 elongation_threshold=4.0,
                 pair_tol_bins=2.0,
                 max_candidates=200):
    mag = results["power"] if use_power else results["magnitude"]
    data = np.asarray(mag, dtype=np.float64)
    Ny, Nx = data.shape
    cy, cx = Ny // 2, Nx // 2
    fx_ax, fy_ax = results["fx"], results["fy"]
    dfx = float(fx_ax[1] - fx_ax[0]) if Nx > 1 else 1.0
    dfy = float(fy_ax[1] - fy_ax[0]) if Ny > 1 else 1.0
    calibrated = results["calibrated"]

    dc_mask, (ry_dc, rx_dc) = make_dc_exclusion_mask(
        (Ny, Nx), dc_radius_bins=dc_radius_bins,
        dc_radius_frac=dc_radius_frac)

    thr_map = _radial_threshold_map(data, dc_mask, snr_threshold)
    is_max = _find_local_maxima(data, neighborhood=neighborhood)
    cand_iy, cand_ix = np.nonzero(is_max & dc_mask & (data > thr_map))

    if cand_iy.size > max_candidates:
        order = np.argsort(data[cand_iy, cand_ix])[::-1][:max_candidates]
        cand_iy, cand_ix = cand_iy[order], cand_ix[order]

    r_in = max(2, neighborhood // 2 + 1)
    r_out = r_in + max(6, neighborhood + 2)

    border_y = max(2, int(round(border_margin_frac * Ny)))
    border_x = max(2, int(round(border_margin_frac * Nx)))

    half_spectra = _split_half_spectra(results["mean_subtracted"]) \
        if results.get("mean_subtracted") is not None else None

    accepted, rejected = [], []

    for iy, ix in zip(cand_iy.tolist(), cand_ix.tolist()):
        peak = {
            "iy": iy, "ix": ix,
            "fx": float(fx_ax[ix]), "fy": float(fy_ax[iy]),
            "qx": float(results["qx"][ix]) if calibrated else None,
            "qy": float(results["qy"][iy]) if calibrated else None,
            "intensity": float(data[iy, ix]),
            "pair_index": None,
            "pairing_score": 0.0,
            "consistency_score": np.nan,
            "anisotropy": np.nan,
        }

        if (iy < border_y or iy >= Ny - border_y or
                ix < border_x or ix >= Nx - border_x):
            peak.update(background=np.nan, prominence=np.nan,
                        local_sigma=np.nan, snr=np.nan, width_bins=np.nan,
                        reject_reason="border")
            rejected.append(peak)
            continue

        bg, sig = _annulus_background(data, iy, ix, r_in, r_out,
                                      valid_mask=dc_mask)
        if not (np.isfinite(sig) and sig > 0):
            bg, sig = _annulus_background(data, iy, ix, r_in, r_out + 8,
                                          valid_mask=dc_mask)
        if not (np.isfinite(bg) and np.isfinite(sig) and sig > 0):
            peak.update(background=bg, prominence=np.nan, local_sigma=sig,
                        snr=np.nan, width_bins=np.nan,
                        reject_reason="background_undefined")
            rejected.append(peak)
            continue

        width, aniso, n_sat = _peak_width(data, iy, ix, bg)
        if np.isfinite(width) and width / 2.0 + 1 > r_in:
            r_in2 = int(width / 2.0) + 2
            bg2, sig2 = _annulus_background(data, iy, ix, r_in2,
                                            r_in2 + 8, valid_mask=dc_mask)
            if np.isfinite(bg2) and np.isfinite(sig2) and sig2 > 0:
                bg, sig = bg2, sig2
                width, aniso, n_sat = _peak_width(data, iy, ix, bg)

        prom = peak["intensity"] - bg
        snr = prom / sig
        peak.update(background=bg, prominence=prom, local_sigma=sig,
                    snr=snr, width_bins=width, anisotropy=aniso)

        if not np.isfinite(prom) or prom <= 0 or snr < snr_threshold:
            peak["reject_reason"] = "low_prominence"
            rejected.append(peak)
            continue

        if np.isfinite(aniso) and aniso > elongation_threshold:
            peak["reject_reason"] = "elongated"
            rejected.append(peak)
            continue

        on_h_axis = abs(iy - cy) <= streak_axis_tol_bins
        on_v_axis = abs(ix - cx) <= streak_axis_tol_bins
        if on_h_axis or on_v_axis:
            score, axis = _streak_score(data, iy, ix, bg)
            aligned = (axis == "h" and on_h_axis) or \
                      (axis == "v" and on_v_axis)
            if aligned and score > streak_ratio_threshold:
                peak["reject_reason"] = f"streak_{axis}"
                rejected.append(peak)
                continue

        dyo, dxo = _subbin_offset(data, iy, ix)
        peak["fx"] = float(fx_ax[ix] + dxo * dfx)
        peak["fy"] = float(fy_ax[iy] + dyo * dfy)
        if calibrated:
            peak["qx"] = float(2 * np.pi * peak["fx"])
            peak["qy"] = float(2 * np.pi * peak["fy"])

        if half_spectra is not None:
            u = (ix + dxo - cx) / Nx
            v = (iy + dyo - cy) / Ny
            peak["consistency_score"] = _split_consistency(
                half_spectra, u, v)

        accepted.append(peak)

    for i, p in enumerate(accepted):
        if p["pair_index"] is not None:
            continue
        ty, tx = 2 * cy - p["iy"], 2 * cx - p["ix"]
        if not (0 <= ty < Ny and 0 <= tx < Nx):
            continue  
        best_j, best_d = None, None
        for j, q in enumerate(accepted):
            if j == i:
                continue
            d = np.hypot(q["iy"] - ty, q["ix"] - tx)
            if d <= pair_tol_bins and (best_d is None or d < best_d):
                best_j, best_d = j, d
        if best_j is not None:
            q = accepted[best_j]
            pos_score = 1.0 - best_d / pair_tol_bins            
            lo, hi = sorted([p["intensity"], q["intensity"]])
            int_score = lo / hi if hi > 0 else 0.0              
            score = float(pos_score * int_score)
            p["pair_index"], q["pair_index"] = best_j, i
            p["pairing_score"] = q["pairing_score"] = score

    return {
        "accepted": accepted,
        "rejected": rejected,
        "dc_mask": dc_mask,
        "dc_radii_bins": (ry_dc, rx_dc),
        "data_used": "power" if use_power else "magnitude",
        "threshold": float(np.median(thr_map)),
    }

def report_peaks(detection, calibrated):
    acc, rej = detection["accepted"], detection["rejected"]
    unit = "1/unit" if calibrated else "cyc/px"
    print("-" * 100)
    print(f"Peak detection on linear {detection['data_used']}; "
          f"adaptive floor = {detection['threshold']:.6g}")
    print(f"Accepted: {len(acc)}   Rejected: {len(rej)}")
    print("-" * 100)
    hdr = (f"{'#':>3} {'iy':>5} {'ix':>5} "
           f"{'fx ('+unit+')':>14} {'fy ('+unit+')':>14} "
           f"{'intensity':>12} {'prominence':>12} {'SNR':>8} "
           f"{'width':>7} {'consis':>7} {'pair':>5} {'score':>6}")
    print(hdr)
    for k, p in enumerate(acc):
        pair = p["pair_index"] if p["pair_index"] is not None else "-"
        cons = p.get("consistency_score", float("nan"))
        cons_txt = f"{cons:7.2f}" if cons == cons else "    n/a"
        print(f"{k:>3} {p['iy']:>5} {p['ix']:>5} "
              f"{p['fx']:>14.5g} {p['fy']:>14.5g} "
              f"{p['intensity']:>12.5g} {p['prominence']:>12.5g} "
              f"{p['snr']:>8.2f} {p['width_bins']:>7.2f} "
              f"{cons_txt} "
              f"{str(pair):>5} {p['pairing_score']:>6.3f}")
    if calibrated and acc:
        print("(qx, qy available per peak: q = 2*pi*f, rad/unit)")
    if rej:
        print("Rejected peaks (reason):")
        for p in rej:
            print(f"    (iy={p['iy']}, ix={p['ix']}) -> {p['reject_reason']}")
    print("-" * 100)

def _pair_list(accepted):
    pairs, seen = [], set()
    for i, p in enumerate(accepted):
        j = p["pair_index"]
        if j is None or (j, i) in seen:
            continue
        seen.add((i, j))
        pairs.append((i, j, p, accepted[j]))
    return pairs


def _jpeg_blockiness(magnitude):
    Ny, Nx = magnitude.shape
    cy, cx = Ny // 2, Nx // 2
    comb, ref = [], []
    for k in (1, 2, 3):
        for s in (+1, -1):
            col = cx + s * int(round(k * Nx / 8.0))
            if 3 <= col < Nx - 3:
                comb.append(float(np.median(magnitude[:, col])))
                ref.append(float(np.median(magnitude[:, col + 3 * s])))
            row = cy + s * int(round(k * Ny / 8.0))
            if 3 <= row < Ny - 3:
                comb.append(float(np.median(magnitude[row, :])))
                ref.append(float(np.median(magnitude[row + 3 * s, :])))
    if not comb:
        return 1.0
    return float(np.median(comb) / max(np.median(ref), 1e-300))

def assess_cdw_evidence(results, detection,
                        min_pairing_score=0.5,
                        min_snr=8.0,
                        min_consistency=0.4,
                        min_radius_frac=0.0,
                        central_margin_bins=3,
                        axis_tol_deg=3.0,
                        max_width_bins=12.0,
                        elongation_suspect=2.5,
                        radius_family_rtol=0.10,
                        angle_tol_deg=10.0,
                        jpeg_freq_tol=0.012,
                        jpeg_blockiness_threshold=1.5):
    acc = detection["accepted"]
    rej = detection["rejected"]
    Ny, Nx = results["magnitude"].shape
    cy, cx = Ny // 2, Nx // 2
    calibrated = results["calibrated"]
    is_jpeg_ext = results.get("source_ext") in (".jpg", ".jpeg")
    blockiness = _jpeg_blockiness(results["magnitude"])
    blocky = blockiness > jpeg_blockiness_threshold
    dc_ry, dc_rx = detection.get("dc_radii_bins", (6, 6))
    central_bins = max(dc_ry, dc_rx) + central_margin_bins

    warnings = []
    n_streak = sum(1 for p in rej
                   if str(p["reject_reason"]).startswith("streak"))
    n_elong = sum(1 for p in rej if p["reject_reason"] == "elongated")
    n_border = sum(1 for p in rej if p["reject_reason"] == "border")
    n_nobg = sum(1 for p in rej
                 if p["reject_reason"] == "background_undefined")
    if n_streak:
        warnings.append(f"{n_streak} streak-like peak(s) rejected along the "
                        "fx=0/fy=0 axes (possible scan-line or edge artifacts).")
    if n_elong:
        warnings.append(f"{n_elong} elongated peak(s) rejected (possible "
                        "tilted-streak / drift artifacts).")
    if n_border:
        warnings.append(f"{n_border} border peak(s) rejected (possible "
                        "windowing/edge artifacts).")
    if n_nobg:
        warnings.append(f"{n_nobg} peak(s) rejected with undefined local "
                        "background.")
    if blocky:
        warnings.append(
            f"Spectrum shows 8x8 block-compression combing (ratio "
            f"{blockiness:.2f}): peaks on the k/8 grid may be JPEG "
            "artifacts" + ("" if is_jpeg_ext
                           else " (file is not .jpg -- possibly a "
                                "re-saved JPEG)") + ".")
    elif is_jpeg_ext:
        warnings.append("Source is JPEG but no significant block combing "
                        "was measured; peaks are not demoted for "
                        "compression.")

    pair_records = []
    for i, j, p, q in _pair_list(acc):
        u = (p["ix"] - cx) / Nx
        v = (p["iy"] - cy) / Ny
        r_frac = float(np.hypot(u, v) / 0.5)       
        f_r = float(np.hypot(p["fx"], p["fy"]))
        q_r = float(np.hypot(p["qx"], p["qy"])) if calibrated else None
        angle = float(np.degrees(np.arctan2(p["fy"], p["fx"])) % 180.0)

        snr_pair = float(min(p["snr"], q["snr"]))
        width_pair = float(np.nanmax([p["width_bins"], q["width_bins"]]))
        score = float(p["pairing_score"])
        cons_vals = [c for c in (p.get("consistency_score"),
                                 q.get("consistency_score"))
                     if c is not None and c == c]  
        cons_pair = float(min(cons_vals)) if cons_vals else float("nan")
        aniso_pair = float(np.nanmax([p.get("anisotropy", np.nan),
                                      q.get("anisotropy", np.nan)]))

        axis_aligned = (min(angle, 180.0 - angle) <= axis_tol_deg or
                        abs(angle - 90.0) <= axis_tol_deg)
        jpeg_suspect = False
        if blocky:
            def on_comb(w):
                k = round(w / 0.125)
                return abs(w - 0.125 * k) <= jpeg_freq_tol, k
            on_u, ku = on_comb(abs(u))
            on_v, kv = on_comb(abs(v))
            jpeg_suspect = on_u and on_v and (ku >= 1 or kv >= 1)

        failures = []
        if score < min_pairing_score:
            failures.append("low_pairing_score")
        if snr_pair < min_snr:
            failures.append("low_snr")
        r_bins = float(np.hypot(p["iy"] - cy, p["ix"] - cx))
        if r_bins < central_bins or r_frac < min_radius_frac:
            failures.append("central_region")
        if not np.isfinite(width_pair) or width_pair > max_width_bins:
            failures.append("too_broad")
        if cons_pair == cons_pair and cons_pair < min_consistency:
            failures.append("split_inconsistent")

        axis_suspect = axis_aligned and (
            (np.isfinite(aniso_pair) and aniso_pair > elongation_suspect)
            or (cons_pair == cons_pair and cons_pair < min_consistency)
            or (cons_pair != cons_pair))  

        clean = (not failures) and (not axis_suspect) and (not jpeg_suspect)
        status = ("clean" if clean else
                  "suspect" if not failures else "not_counted")

        pair_records.append({
            "peak_indices": (i, j),
            "f_radius": f_r,
            "q_radius": q_r,
            "angle_deg": angle,
            "radius_frac_nyquist": r_frac,
            "min_snr": snr_pair,
            "max_width_bins": width_pair,
            "pairing_score": score,
            "min_consistency": cons_pair,
            "max_anisotropy": aniso_pair,
            "axis_aligned": axis_aligned,
            "jpeg_suspect": jpeg_suspect,
            "failures": failures,
            "status": status,
        })

        if axis_aligned and clean:
            warnings.append(
                f"Pair at |f|={f_r:.4g}, {angle:.1f} deg lies on a "
                "principal axis but is isotropic and split-consistent; "
                "counted as clean -- verify it is not scan-synchronous.")
        if axis_suspect and not failures:
            warnings.append(
                f"Pair at |f|={f_r:.4g}, {angle:.1f} deg lies on a "
                "principal axis with corroborating artifact signs: "
                "could be a scan-line/raster artifact.")
        if jpeg_suspect and not failures:
            warnings.append(
                f"Pair at |f|={f_r:.4g} matches the JPEG DCT comb in a "
                "blocky spectrum: possible compression artifact.")

    clean_pairs = [r for r in pair_records if r["status"] == "clean"]
    suspect_pairs = [r for r in pair_records if r["status"] == "suspect"]

    q_org = None
    if clean_pairs:
        ref = max(clean_pairs, key=lambda r: r["min_snr"])
        fam = [r for r in clean_pairs
               if abs(r["f_radius"] - ref["f_radius"])
               <= radius_family_rtol * ref["f_radius"]]
        angles = sorted(r["angle_deg"] for r in fam)
        n = len(fam)

        def _sep_ok(target):
            seps = [(angles[k + 1] - angles[k]) for k in range(n - 1)]
            return all(abs(s - target) <= angle_tol_deg for s in seps)

        if n == 1:
            q_org = "possible 1Q"
        elif n == 2 and _sep_ok(90.0):
            q_org = "possible 2Q"
        elif n == 3 and _sep_ok(60.0):
            q_org = "possible 3Q"
        else:
            q_org = "unclear"

    if clean_pairs:
        evidence = "strong"
    elif suspect_pairs:
        evidence = "weak"
    else:
        evidence = "insufficient"

    unit = "cycles/unit" if calibrated else "cycles/pixel"
    lines = []
    lines.append(f"{len(pair_records)} symmetric (+q,-q) pair(s) accepted by "
                 f"peak detection; {len(clean_pairs)} clean, "
                 f"{len(suspect_pairs)} suspect, "
                 f"{len(pair_records) - len(clean_pairs) - len(suspect_pairs)}"
                 " not counted as evidence.")
    for r in pair_records:
        extra = (f", |q|={r['q_radius']:.5g} rad/unit"
                 if r["q_radius"] is not None else "")
        tags = []
        if r["axis_aligned"]:
            tags.append("axis-aligned")
        if r["jpeg_suspect"]:
            tags.append("JPEG-comb")
        tags += r["failures"]
        cons_txt = (f"{r['min_consistency']:.2f}"
                    if r["min_consistency"] == r["min_consistency"]
                    else "n/a")
        lines.append(
            f"  Pair {r['peak_indices']}: |f|={r['f_radius']:.5g} {unit}"
            f"{extra}, angle={r['angle_deg']:.1f} deg, "
            f"min SNR={r['min_snr']:.1f}, width={r['max_width_bins']:.2f} "
            f"bins, split-consistency={cons_txt} "
            f"[{r['status']}{': ' + ', '.join(tags) if tags else ''}]")
    if evidence == "strong":
        lines.append(
            f"Evidence level STRONG: {len(clean_pairs)} clean symmetric "
            f"pair(s) with SNR >= {min_snr:g}, split-half consistency >= "
            f"{min_consistency:g} (the peak appears independently in the "
            f"FFTs of both image halves), radius >= {central_bins:g} "
            f"frequency bins from DC, width <= {max_width_bins:g} bins, "
            "not elongated, and not on a measured JPEG comb. "
            + (f"Angular arrangement suggests {q_org}." if q_org else ""))
    elif evidence == "weak":
        lines.append(
            "Evidence level WEAK: symmetric pairs exist but every pair "
            "carries artifact flags (axis-aligned with corroborating "
            "signs, and/or on a measured JPEG comb), so artifacts cannot "
            "be excluded numerically.")
    else:
        lines.append(
            "Evidence level INSUFFICIENT: no reliable symmetric (+q,-q) "
            "pair outside the central region. Broad central intensity, "
            "isolated peaks, streaks, elongated features, border peaks, "
            "and split-inconsistent peaks are not counted.")
    lines.append(
        "Note: +q/-q pairing itself is guaranteed by the Hermitian "
        "symmetry of a real image's FFT and is used here for geometry "
        "only; reliability rests on the split-half consistency check.")
    lines.append(
        "Caveat: sharp symmetric FFT peak pairs are NECESSARY but NOT "
        "SUFFICIENT for a CDW. Atomic-lattice, structural, and moire "
        "periodicities produce identical signatures; this is "
        "CDW-compatible FFT evidence, not a CDW classification.")
    if not calibrated:
        lines.append(
            "Analysis is UNCALIBRATED (cycles/pixel); peak radii cannot be "
            "compared to physical lattice or CDW wavevectors.")

    return {
        "n_pairs_total": len(pair_records),
        "n_pairs_clean": len(clean_pairs),
        "n_pairs_suspect": len(suspect_pairs),
        "pairs": pair_records,
        "q_organization": q_org,
        "warnings": warnings,
        "evidence_level": evidence,
        "explanation": "\n".join(lines),
    }

def report_cdw_evidence(assessment):
    print("=" * 72)
    print("CDW-COMPATIBLE FFT EVIDENCE (not a definitive CDW classification)")
    print("=" * 72)
    print(f"Accepted symmetric pairs: {assessment['n_pairs_total']} "
          f"(clean: {assessment['n_pairs_clean']}, "
          f"suspect: {assessment['n_pairs_suspect']})")
    print(f"Possible Q-organization: {assessment['q_organization']}")
    print(f"Evidence level: {assessment['evidence_level'].upper()}")
    if assessment["warnings"]:
        print("Artifact warnings:")
        for w in assessment["warnings"]:
            print(f"  - {w}")
    print("-" * 72)
    print(assessment["explanation"])
    print("=" * 72)

def main():
    parser = argparse.ArgumentParser(
        description="Windowed 2D FFT analysis of an image "
                    "(PNG/JPG/JPEG/TIFF)."
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--Lx", type=float, default=None,
                        help="Physical field of view along x (width), "
                             "in your chosen unit (e.g., nm)")
    parser.add_argument("--Ly", type=float, default=None,
                        help="Physical field of view along y (height)")
    parser.add_argument("--save-figure", default=None,
                        help="Path to save the diagnostic figure (optional)")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display the figure interactively")
    parser.add_argument("--fft-crop-fraction", type=float,
                        default=DEFAULT_FFT_CROP_FRACTION,
                        help="DISPLAY-ONLY: fraction of the reciprocal-space "
                             "half-extent shown around the zero-frequency "
                             "point in the FFT panel (default %(default)s). "
                             "Keeps the central and first-order Bragg peaks; "
                             "raise it to reveal higher-order peaks.")
    parser.add_argument("--detect-peaks", action="store_true",
                        help="NEW: run conservative reciprocal-space "
                             "peak detection after the FFT")
    parser.add_argument("--use-power", action="store_true",
                        help="NEW: detect on power (|FFT|^2) instead of "
                             "linear magnitude")
    parser.add_argument("--snr", type=float, default=5.0,
                        help="NEW: SNR threshold for prominence (default 5)")
    parser.add_argument("--save-peaks-figure", default=None,
                        help="NEW: path to save the peak-diagnostic figure")
    parser.add_argument("--assess-cdw", action="store_true",
                        help="NEW: assess CDW-compatible FFT evidence "
                             "(implies --detect-peaks; NOT a definitive "
                             "CDW classification)")
    args = parser.parse_args()

    results = analyze_image_fft(
        args.image,
        Lx=args.Lx,
        Ly=args.Ly,
        show=not args.no_show,
        save_figure=args.save_figure,
        fft_crop_fraction=args.fft_crop_fraction,
    )

    if args.detect_peaks or args.assess_cdw:
        detection = detect_peaks(results,
                                 use_power=args.use_power,
                                 snr_threshold=args.snr)
        report_peaks(detection, results["calibrated"])
        plot_peak_diagnostics(results, detection,
                              show=not args.no_show,
                              save_figure=args.save_peaks_figure)

        if args.assess_cdw:
            assessment = assess_cdw_evidence(results, detection)
            report_cdw_evidence(assessment)

if __name__ == "__main__":
    main()