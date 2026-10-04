import numpy as np
from scipy.ndimage import maximum_filter

SUPERLATTICE_MODELS = {
    "2x2": (1.0 / 2.0, 0.0),
    "3x3": (1.0 / 3.0, 0.0),
    "4x4": (1.0 / 4.0, 0.0),
    "sqrt3xsqrt3": (1.0 / np.sqrt(3.0), 30.0),
    "sqrt13xsqrt13": (1.0 / np.sqrt(13.0), 13.898),
}
def make_dc_exclusion_mask(shape, dc_radius_bins=6, min_radius_bins=3):
    Ny, Nx = shape
    cy, cx = Ny // 2, Nx // 2  
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
                 dc_radius_bins=6,
                 neighborhood=3,
                 snr_threshold=5.0,
                 border_margin_frac=0.02,
                 streak_ratio_threshold=3.0,
                 streak_axis_tol_bins=2,
                 elongation_threshold=4.0,
                 pair_tol_bins=2.0,
                 max_candidates=200):
    data = np.asarray(results["magnitude"], dtype=np.float64)
    Ny, Nx = data.shape
    cy, cx = Ny // 2, Nx // 2
    fx_ax, fy_ax = results["fx"], results["fy"]
    dfx = float(fx_ax[1] - fx_ax[0]) if Nx > 1 else 1.0
    dfy = float(fy_ax[1] - fy_ax[0]) if Ny > 1 else 1.0
    calibrated = results["calibrated"]

    dc_mask, (ry_dc, rx_dc) = make_dc_exclusion_mask(
        (Ny, Nx), dc_radius_bins=dc_radius_bins)

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
        "data_used": "magnitude",
        "threshold": float(np.median(thr_map)),
    }

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
    _axis_vetoed = n_streak + n_elong
    n_border = sum(1 for p in rej if p["reject_reason"] == "border")
    n_nobg = sum(1 for p in rej
                 if p["reject_reason"] == "background_undefined")
    if n_streak:
        warnings.append(f"{n_streak} streak-like peak(s) rejected along the "
                        "fx=0/fy=0 axes (possible scan-line or edge artifacts).")
        warnings.append(
            "PATH DISAGREEMENT: those peaks are vetoed here but are RETAINED "
            "by the lattice-identification path (`_lattice_candidates`), "
            "which does not apply the streak/elongation vetoes so that a "
            "genuine Bragg direction lying along the scan axis is not "
            "discarded. This evidence level is therefore computed on a "
            "stricter peak set than the one the superlattice assignment "
            "uses, and the two can disagree for a reason that depends on "
            "the scan direction rather than on the sample. Read this level "
            "as a lower bound when peaks were vetoed on the axes.")
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
        "axis_vetoed_peaks": int(_axis_vetoed),
        "evidence_level_is_lower_bound": bool(_axis_vetoed),
        "evidence_level_path": (
            "detect_peaks (streak/elongation vetoes APPLIED); the lattice "
            "identification path does not apply them"),
        "explanation": "\n".join(lines),
    }

def _fold_angle(deg):
    return float(deg) % 180.0


def _angular_gap(a, b):
    d = abs(_fold_angle(a) - _fold_angle(b)) % 180.0
    return min(d, 180.0 - d)


def _canonicalize(row):
    th = np.radians(row["angle_deg"])
    r = row["f_radius"]
    row["fx_measured"], row["fy_measured"] = row["fx"], row["fy"]
    row["fx"] = float(r * np.cos(th))
    row["fy"] = float(r * np.sin(th))
    return row


def _axis_units(results):
    fx_ax = np.asarray(results["fx"], dtype=np.float64)
    fy_ax = np.asarray(results["fy"], dtype=np.float64)
    Ny, Nx = results["magnitude"].shape
    dfx = float(fx_ax[1] - fx_ax[0]) if Nx > 1 else 1.0
    dfy = float(fy_ax[1] - fy_ax[0]) if Ny > 1 else 1.0
    return fx_ax, fy_ax, dfx, dfy, (Ny // 2, Nx // 2), (Ny, Nx)


def _bin_of(u, v, dfx, dfy, center):
    cy, cx = center
    return int(round(v / dfy)) + cy, int(round(u / dfx)) + cx


def _lattice_candidates(results, dc_mask, snr_threshold=4.0, neighborhood=5,
                        border_margin_frac=0.02, max_candidates=400):
    data = np.asarray(results["magnitude"], dtype=np.float64)
    fx_ax, fy_ax, dfx, dfy, center, (Ny, Nx) = _axis_units(results)
    cy, cx = center

    thr_map = _radial_threshold_map(data, dc_mask, snr_threshold)
    is_max = _find_local_maxima(data, neighborhood=neighborhood)
    cand_iy, cand_ix = np.nonzero(is_max & dc_mask & (data > thr_map))

    if cand_iy.size > max_candidates * 4:
        order = np.argsort(data[cand_iy, cand_ix])[::-1][:max_candidates * 4]
        cand_iy, cand_ix = cand_iy[order], cand_ix[order]

    border_y = max(2, int(round(border_margin_frac * Ny)))
    border_x = max(2, int(round(border_margin_frac * Nx)))
    r_in = max(2, neighborhood // 2 + 1)
    r_out = r_in + max(6, neighborhood + 2)

    cands = []
    for iy, ix in zip(cand_iy.tolist(), cand_ix.tolist()):
        if (iy < border_y or iy >= Ny - border_y
                or ix < border_x or ix >= Nx - border_x):
            continue
        bg, sig = _annulus_background(data, iy, ix, r_in, r_out,
                                      valid_mask=dc_mask)
        if not (np.isfinite(bg) and np.isfinite(sig) and sig > 0):
            continue
        prom = float(data[iy, ix]) - bg
        snr = prom / sig
        if not np.isfinite(snr) or prom <= 0 or snr < snr_threshold:
            continue

        dyo, dxo = _subbin_offset(data, iy, ix)
        u = float(fx_ax[ix] + dxo * dfx)
        v = float(fy_ax[iy] + dyo * dfy)
        cands.append({
            "iy": int(iy), "ix": int(ix),
            "fx": u, "fy": v,
            "f_radius": float(np.hypot(u, v)),
            "angle_deg": _fold_angle(np.degrees(np.arctan2(v, u))),
            "intensity": float(data[iy, ix]),
            "prominence": float(prom),
            "background": float(bg),
            "local_sigma": float(sig),
            "snr": float(snr),
        })

    cands.sort(key=lambda p: -p["prominence"])
    return cands[:max_candidates]


def find_hex_families(cands, radius_rtol=0.10, angle_tol_deg=12.0,
                      max_directions=8):
    from itertools import combinations

    families = []
    for seed in cands:
        if seed["f_radius"] <= 0:
            continue
        shell = [p for p in cands
                 if abs(p["f_radius"] - seed["f_radius"])
                 <= radius_rtol * seed["f_radius"]]
        if len(shell) < 4:
            continue

        dirs = {}
        for p in sorted(shell, key=lambda z: -z["prominence"]):
            key = None
            for k in dirs:
                if _angular_gap(p["angle_deg"], k) <= angle_tol_deg:
                    key = k
                    break
            dirs.setdefault(p["angle_deg"] if key is None else key,
                            []).append(p)

        paired = {k: max(v, key=lambda z: z["prominence"])
                  for k, v in dirs.items() if len(v) >= 2}
        if len(paired) < 3:
            continue

        keys = sorted(paired, key=lambda k: -paired[k]["prominence"])
        keys = keys[:max_directions]

        best = None
        for combo in combinations(keys, 3):
            angs = sorted(combo)
            seps = [angs[1] - angs[0], angs[2] - angs[1],
                    180.0 - (angs[2] - angs[0])]
            if not all(abs(s - 60.0) <= angle_tol_deg for s in seps):
                continue
            peaks = [paired[a] for a in angs]

            rs = [p["f_radius"] for p in peaks]
            if (max(rs) - min(rs)) > radius_rtol * float(np.mean(rs)):
                continue

            min_prom = float(min(p["prominence"] for p in peaks))
            if best is not None and min_prom <= best["min_prominence"]:
                continue
            best = {
                "radius": float(np.mean([p["f_radius"] for p in peaks])),
                "angles_deg": [float(a) for a in angs],
                "peaks": peaks,
                "min_prominence": min_prom,
                "min_snr": float(min(p["snr"] for p in peaks)),
                "hex_deviation_deg": float(np.mean([abs(s - 60.0)
                                                    for s in seps])),
                "radius_spread": float(np.std([p["f_radius"] for p in peaks])
                                       / max(np.mean([p["f_radius"]
                                                      for p in peaks]), 1e-12)),
            }

        if best is None:
            continue
        if not any(abs(f["radius"] - best["radius"])
                   <= radius_rtol * best["radius"] for f in families):
            families.append(best)

    families.sort(key=lambda f: f["radius"])
    return families


def _is_harmonic_of(inner, outer, ratio_tol=0.06, angle_tol_deg=12.0,
                    max_order=6):
    if inner["radius"] <= 0:
        return False
    order = outer["radius"] / inner["radius"]
    n = int(round(order))
    if n < 2 or n > max_order:
        return False
    if abs(order - n) > ratio_tol * n:
        return False
    for a in outer["angles_deg"]:
        if not any(_angular_gap(a, b) <= angle_tol_deg
                   for b in inner["angles_deg"]):
            return False
    return True


def _select_bragg_family(families, ref_prominence, min_snr=5.0,
                         min_prominence_frac=0.02, min_radius=0.0):
    if not families:
        return None

    gated = []
    for fam in families:
        if fam["radius"] < min_radius:
            continue
        if fam["min_prominence"] < min_prominence_frac * float(ref_prominence):
            continue
        if fam["min_snr"] < min_snr:
            continue
        gated.append(fam)
    if not gated:
        return None

    fundamentals, harmonics = [], []
    for fam in gated:
        parent = next((g for g in gated
                       if g is not fam
                       and g["min_snr"] > fam["min_snr"]
                       and _is_harmonic_of(g, fam)), None)
        (harmonics if parent is not None else fundamentals).append(
            (fam, parent))

    pool = [f for f, _ in fundamentals] or [f for f, _ in harmonics]
    if not pool:
        return None

    chosen = dict(max(pool, key=lambda f: f["min_snr"]))
    chosen["selection"] = "strongest non-harmonic family"
    chosen["harmonics_removed"] = [
        {"radius": f["radius"],
         "order": round(f["radius"] / max(p["radius"], 1e-12), 2),
         "prominence_frac": float(f["min_prominence"]
                                  / max(ref_prominence, 1e-300))}
        for f, p in harmonics]
    chosen["prominence_frac"] = float(chosen["min_prominence"]
                                      / max(ref_prominence, 1e-300))
    return chosen


def _measure_at(data, dc_mask, u0, v0, dfx, dfy, center, shape,
                search_bins=3, r_in=3, r_out=11):
    Ny, Nx = shape
    cy, cx = center
    iy0, ix0 = _bin_of(u0, v0, dfx, dfy, center)
    s = int(search_bins)
    y0, y1 = max(0, iy0 - s), min(Ny, iy0 + s + 1)
    x0, x1 = max(0, ix0 - s), min(Nx, ix0 + s + 1)
    if y1 <= y0 or x1 <= x0:
        return None

    sub = data[y0:y1, x0:x1]
    dy, dx = np.unravel_index(int(np.argmax(sub)), sub.shape)
    iy, ix = int(y0 + dy), int(x0 + dx)
    if not (1 <= iy < Ny - 1 and 1 <= ix < Nx - 1):
        return None

    dyo, dxo = _subbin_offset(data, iy, ix)
    u = float((ix - cx + dxo) * dfx)
    v = float((iy - cy + dyo) * dfy)

    bg, sig = _annulus_background(data, iy, ix, r_in, r_out,
                                  valid_mask=dc_mask)
    if not (np.isfinite(bg) and np.isfinite(sig) and sig > 0):
        bg, sig = _annulus_background(data, iy, ix, r_in, r_out + 8,
                                      valid_mask=dc_mask)
    if not (np.isfinite(bg) and np.isfinite(sig) and sig > 0):
        return None

    prom = float(data[iy, ix]) - bg
    shift_bins = float(np.hypot((u - u0) / dfx, (v - v0) / dfy))
    return {
        "iy": iy, "ix": ix, "fx": u, "fy": v,
        "f_radius": float(np.hypot(u, v)),
        "angle_deg": _fold_angle(np.degrees(np.arctan2(v, u))),
        "intensity": float(data[iy, ix]),
        "background": float(bg), "local_sigma": float(sig),
        "prominence": prom, "snr": float(prom / sig),
        "shift_bins": shift_bins,
        "predicted_fx": float(u0), "predicted_fy": float(v0),
        "predicted_f_radius": float(np.hypot(u0, v0)),
    }


def _identify_cdw(results, dc_mask, bragg_family, ref_prominence,
                  dc_radius_f, min_cdw_snr=5.0,
                  min_cdw_prominence_frac=5e-3, refine_tol_bins=2.0,
                  cdw_radius_rtol=0.12,
                  search_bins=3, families=(),
                  radius_rtol=0.10):
    data = np.asarray(results["magnitude"], dtype=np.float64)
    _, _, dfx, dfy, center, shape = _axis_units(results)

    table = dict(SUPERLATTICE_MODELS)
    floor = min_cdw_prominence_frac * float(ref_prominence)
    bragg_dirs = [(p["f_radius"], p["angle_deg"])
                  for p in bragg_family["peaks"]]

    results_by_model, rejections = [], {}
    for name, (ratio, rot) in table.items():
        rows, reasons = [], []
        for k, (r_b, ang) in enumerate(bragg_dirs, start=1):
            r_c = r_b * ratio
            th = np.radians(ang + rot)
            if r_c <= dc_radius_f:
                reasons.append(f"direction {k}: predicted |q| is inside the "
                               f"DC exclusion zone")
                continue
            u0, v0 = r_c * np.cos(th), r_c * np.sin(th)

            r_c_bins = float(np.hypot(r_c * np.cos(th) / dfx,
                                      r_c * np.sin(th) / dfy))
            max_shift_bins = max(float(refine_tol_bins),
                                 cdw_radius_rtol * r_c_bins)

            m = _measure_at(data, dc_mask, u0, v0, dfx, dfy, center, shape,
                            search_bins=max(search_bins,
                                            int(np.ceil(max_shift_bins))))
            if m is None:
                reasons.append(f"direction {k}: no measurable maximum")
                continue
            if m["shift_bins"] > max_shift_bins:
                reasons.append(
                    f"direction {k}: nearest maximum is "
                    f"{m['shift_bins']:.1f} bins from the predicted position "
                    f"(tolerance {max_shift_bins:.1f})")
                continue
            if m["snr"] < min_cdw_snr:
                reasons.append(f"direction {k}: SNR {m['snr']:.1f} below "
                               f"{min_cdw_snr:g}")
                continue
            if m["prominence"] < floor:
                reasons.append(
                    f"direction {k}: peak holds only "
                    f"{m['prominence'] / max(ref_prominence, 1e-300):.2e} of "
                    f"the strongest prominence (gate "
                    f"{min_cdw_prominence_frac:g})")
                continue
            m["index"] = k
            rows.append(m)

        if len(rows) < 3:
            rejections[name] = reasons
            continue
        angs = [r["angle_deg"] for r in rows]
        if min(_angular_gap(angs[i], angs[j])
               for i in range(3) for j in range(i + 1, 3)) < 15.0:
            rejections[name] = reasons + ["directions collapsed onto each "
                                          "other"]
            continue
        r_pred = float(np.mean([r["f_radius"] for r in rows]))
        family_match = any(
            abs(f["radius"] - r_pred) <= radius_rtol * max(r_pred, 1e-12)
            for f in families)
        results_by_model.append({
            "name": name,
            "q_ratio_to_bragg": float(ratio),
            "rotation_deg": float(rot),
            "rows": rows,
            "family_match": bool(family_match),
            "min_snr": float(min(r["snr"] for r in rows)),
            "mean_shift_bins": float(np.mean([r["shift_bins"] for r in rows])),
        })

    if not results_by_model:
        return None, rejections

    best = min(results_by_model,
               key=lambda m: (not m["family_match"],
                              round(m["mean_shift_bins"], 1),
                              -m["min_snr"]))
    runner_up = [m["name"] for m in results_by_model if m is not best]
    best = dict(best)
    best["alternatives"] = runner_up
    return best, rejections


def identify_lattice_peaks(results, dc_radius_bins=6, candidate_snr=4.0,
                           neighborhood=5, radius_rtol=0.10,
                           angle_tol_deg=12.0, min_bragg_snr=5.0,
                           min_bragg_prominence_frac=0.02,
                           min_cdw_snr=5.0,
                           min_cdw_prominence_frac=5e-3,
                           refine_tol_bins=2.0, search_bins=3):
    fx_ax, fy_ax, dfx, dfy, center, shape = _axis_units(results)
    calibrated = bool(results["calibrated"])

    dc_mask, (ry_dc, rx_dc) = make_dc_exclusion_mask(
        shape, dc_radius_bins=dc_radius_bins)
    dc_radius_f = float(max(ry_dc * abs(dfy), rx_dc * abs(dfx)))

    out = {
        "bragg": [], "cdw": [], "model": None, "status": "no_bragg",
        "bragg_assignment": None,
        "messages": [], "cdw_message": None, "calibrated": calibrated,
        "families": [], "data_used": "magnitude",
    }

    cands = _lattice_candidates(results, dc_mask, snr_threshold=candidate_snr,
                                neighborhood=neighborhood)
    if len(cands) < 6:
        out["messages"].append(
            f"only {len(cands)} candidate maxima above the adaptive floor; "
            "a hexagonal family needs at least six (three directions, "
            "+q and -q).")
        return out

    ref_prominence = max(p["prominence"] for p in cands)
    families = find_hex_families(cands, radius_rtol=radius_rtol,
                                 angle_tol_deg=angle_tol_deg)
    out["families"] = [{"radius": f["radius"], "angles_deg": f["angles_deg"],
                        "min_snr": f["min_snr"],
                        "prominence_frac": f["min_prominence"]
                        / max(ref_prominence, 1e-300)} for f in families]

    if not families:
        out["messages"].append(
            "no radius shell contains three directions separated by "
            f"60 +/- {angle_tol_deg:g} degrees; Bragg peaks not identified.")
        return out

    fam = _select_bragg_family(
        families, ref_prominence, min_snr=min_bragg_snr,
        min_prominence_frac=min_bragg_prominence_frac,
        min_radius=1.5 * dc_radius_f)
    if fam is None:
        out["messages"].append(
            f"{len(families)} hexagonal family/families found, but none "
            f"clears both gates (SNR >= {min_bragg_snr:g} and prominence "
            f">= {min_bragg_prominence_frac:g} of the strongest peak); "
            "Bragg peaks not identified.")
        return out

    order = np.argsort([p["angle_deg"] for p in fam["peaks"]])
    bragg_rows = []
    for k, idx in enumerate(order, start=1):
        p = _canonicalize(dict(fam["peaks"][int(idx)]))
        p["index"] = k
        if calibrated:
            p["qx"] = 2.0 * np.pi * p["fx"]
            p["qy"] = 2.0 * np.pi * p["fy"]
            p["q_radius"] = 2.0 * np.pi * p["f_radius"]
        bragg_rows.append(p)
    out["bragg"] = bragg_rows
    out["status"] = "bragg_only"

    inner = [f for f in families
             if f["radius"] < fam["radius"] * (1.0 - radius_rtol)]
    out["bragg_assignment"] = "verified" if inner else "unverified"
    if not inner:
        out["messages"].append(
            "Bragg assignment UNVERIFIED: no independent lattice was found "
            "inside the anchor, so this single periodicity could be the "
            "atomic lattice, or a superlattice whose parent lattice is not "
            "resolved in this field of view. Image an atomically-resolved "
            "region, or pin the anchor with bragg_radius, to settle it.")
    out["bragg_family"] = {
        "radius": fam["radius"],
        "angles_deg": fam["angles_deg"],
        "hex_deviation_deg": fam["hex_deviation_deg"],
        "min_snr": fam["min_snr"],
        "prominence_frac": fam.get("prominence_frac"),
        "selection": fam.get("selection"),
        "harmonics_removed": fam.get("harmonics_removed") or [],
    }
    out["family_ratios"] = sorted(
        ({"radius": f["radius"],
          "x_bragg": round(f["radius"] / max(fam["radius"], 1e-12), 3),
          "prominence_frac": round(f["min_prominence"]
                                   / max(ref_prominence, 1e-300), 4)}
         for f in families), key=lambda d: d["x_bragg"])

    cdw, rejections = _identify_cdw(
        results, dc_mask, fam, ref_prominence, dc_radius_f,
        min_cdw_snr=min_cdw_snr,
        min_cdw_prominence_frac=min_cdw_prominence_frac,
        refine_tol_bins=refine_tol_bins, search_bins=search_bins,
        families=families, radius_rtol=radius_rtol)

    if cdw is None:
        tried = ", ".join(sorted(rejections)) or "none"
        out["cdw_message"] = ("no superlattice model confirmed "
                              f"(tried: {tried})")
        out["messages"].append(
            "CDW peaks NOT identified: no tested superlattice "
            f"({tried}) has a peak at all three predicted positions that "
            f"clears SNR >= {min_cdw_snr:g} and the prominence gate. "
            "Nothing is labelled C1-C3.")
        out["cdw_rejections"] = rejections
        return out

    cdw_rows = []
    for p in sorted(cdw["rows"], key=lambda r: r["angle_deg"]):
        p = _canonicalize(dict(p))
        if calibrated:
            p["qx"] = 2.0 * np.pi * p["fx"]
            p["qy"] = 2.0 * np.pi * p["fy"]
            p["q_radius"] = 2.0 * np.pi * p["f_radius"]
        cdw_rows.append(p)
    for k, p in enumerate(cdw_rows, start=1):
        p["index"] = k

    out["cdw"] = cdw_rows
    out["model"] = {
        "name": cdw["name"],
        "q_ratio_to_bragg": cdw["q_ratio_to_bragg"],
        "rotation_deg": cdw["rotation_deg"],
        "selection": "auto (family match, then position, then SNR)",
        "shell_independently_detected": cdw.get("family_match"),
        "mean_shift_bins": cdw["mean_shift_bins"],
        "alternatives": cdw["alternatives"],
    }
    out["status"] = "bragg_and_cdw"
    out["cdw_rejections"] = rejections
    return out
