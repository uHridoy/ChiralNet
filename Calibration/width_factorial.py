from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np

import didv_anisotropy as da
from forward_model import CDW_GEOMETRY, draw_random_spec, synthesize

WIDTHS = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5)
NOISE_WHITE = (0.03, 0.1, 0.3, 1.0, 3.0)
TIPS = (0.0, 0.3, 0.6)
APERTURES = (3.0, 5.0)
ALPHA_LEVELS = (0.0, 0.5)


def _one(spec, aperture: float, geom: int) -> Dict[str, Any]:
    field, _ = synthesize(spec)
    res = da.analyze_didv_field(field, cdw_model=spec.model,
                                aperture_bins=aperture,
                                bragg_radius=spec.q_bragg,
                                bragg_angles_deg=spec.bragg_angles_deg)

    row: Dict[str, Any] = {
        "geometry": geom, "status": res.get("status"),
        "imposed_width": spec.peak_width_bins,
        "noise_white": spec.noise_white,
        "tip": spec.tip_alpha_bragg,
        "aperture_bins": aperture,
        "alpha_true": spec.alpha_true_cdw,
        "model": spec.model, "Nx": spec.Nx, "Ny": spec.Ny,
        "q_bragg": spec.q_bragg,
    }
    if res.get("status") != "ok":
        return row

    snrs = [p["snr"] for p in res["cdw_peaks"] if p["snr"] is not None]
    w = [x for x in (res.get("width_bins_cdw") or []) if x]
    row.update({
        "alpha_cdw": res["alpha_cdw"],
        "alpha_bragg": res["alpha_bragg"],
        "min_snr": min(snrs) if len(snrs) == 3 else None,
        "measured_width": float(np.mean(w)) if len(w) == 3 else None,
        "n_resolved": res.get("n_directions_resolved"),
    })
    return row


def run(n: int, seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gi, child in enumerate(np.random.SeedSequence(seed).spawn(n)):
        rng = np.random.Generator(np.random.PCG64(child))
        base = int(child.entropy) % (2 ** 31)
        for width in WIDTHS:
            for nw in NOISE_WHITE:
                for tip in TIPS:
                    for a_true in ALPHA_LEVELS:
                        spec = draw_random_spec(
                            rng, alpha_true_cdw=a_true, square=True,
                            peak_width_bins=width, tip_alpha_bragg=tip,
                            noise_white=nw, seed=base)
                        for ap in APERTURES:
                            rows.append(_one(spec, ap, gi))
        if (gi + 1) % 5 == 0:
            print(f"  {gi + 1}/{n} geometries", flush=True)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    null = ok[ok.alpha_true == 0]
    sig = ok[ok.alpha_true == 0.5]

    L = ["=" * 84, "WIDTH x SNR x TIP FACTORIAL", "=" * 84,
         f"{len(df)} runs, {len(ok)} ok, {len(df) - len(ok)} failed",
         f"null n = {len(null)}, signal n = {len(sig)}", ""]
    if not len(null):
        return "\n".join(L + ["nothing to report"])

    def p95(s):
        s = s.dropna()
        return float(s.quantile(.95)) if len(s) else np.nan

    L.append("-- Sanity: does imposed width control measured width? -------")
    L.append(f"{'imposed':>9}{'measured p50':>14}{'min_snr p50':>13}")
    for w, g in null.groupby("imposed_width"):
        L.append(f"{w:>9.1f}{g.measured_width.median():>14.2f}"
                 f"{g.min_snr.median():>13.1f}")
    L.append("")

    a3 = null[null.aperture_bins == 3.0].copy()

    L.append("-- SNR overlap across widths (does post-stratification work?) -")
    L.append("   If these ranges do not overlap, width and SNR cannot be")
    L.append("   separated and the domain cut must gate on SNR alone.")
    L.append(f"{'width':>7}{'snr p5':>10}{'snr p50':>10}{'snr p95':>10}")
    for w, g in a3.groupby("imposed_width"):
        L.append(f"{w:>7.1f}{g.min_snr.quantile(.05):>10.1f}"
                 f"{g.min_snr.median():>10.1f}{g.min_snr.quantile(.95):>10.1f}")
    L.append("")

    L.append("-- Q1: width effect at FIXED ACHIEVED SNR and tip (aper 3) ---")
    L.append("   null p95 of alpha; read ACROSS a row for the width effect.")
    L.append("   Stratified on MEASURED min_snr, not on nominal noise, since")
    L.append("   width shifts SNR by itself.")
    bins = [0, 30, 100, 300, 1000, 1e12]
    labels = ["<30", "30-100", "100-300", "300-1k", ">1k"]
    a3["snr_bin"] = pd.cut(a3.min_snr, bins, labels=labels)
    hdr = f"{'SNR':>9}{'tip':>6}" + "".join(f"{w:>8.1f}" for w in WIDTHS)
    L.append(hdr)
    for sb in labels:
        for tip in TIPS:
            cell = a3[(a3.snr_bin == sb) & (a3.tip == tip)]
            if len(cell) < 20:
                continue
            row = f"{sb:>9}{tip:>6.1f}"
            for w in WIDTHS:
                s = cell[cell.imposed_width == w].alpha_cdw
                row += f"{p95(s):>8.3f}" if len(s) >= 8 else f"{'-':>8}"
            L.append(row)
    L.append("")

    L.append("-- Q2: is the high-width rise just aperture truncation? -----")
    L.append(f"{'width':>7}{'aper 3':>9}{'aper 5':>9}{'change':>9}")
    for w in WIDTHS:
        v = []
        for ap in APERTURES:
            v.append(p95(null[(null.imposed_width == w)
                              & (null.aperture_bins == ap)].alpha_cdw))
        L.append(f"{w:>7.1f}{v[0]:>9.3f}{v[1]:>9.3f}{v[1] - v[0]:>+9.3f}")
    L.append("")

    L.append("-- Marginal effects (each factor, others averaged over) -----")
    for fac, lab in (("imposed_width", "width"), ("noise_white", "noise"),
                     ("tip", "tip")):
        L.append(f"   by {lab}:")
        for v, g in null.groupby(fac):
            L.append(f"     {v:>6.1f}: null p95 {p95(g.alpha_cdw):>6.3f}"
                     f"   n={len(g)}")
    L.append("")

    L.append("-- Detection power at fixed 5% FPR, by width (aperture 3) ---")
    L.append(f"{'width':>7}{'a_crit':>9}{'power@0.5':>11}{'n_null':>8}")
    for w in WIDTHS:
        nn = a3[a3.imposed_width == w].alpha_cdw
        ss = sig[(sig.aperture_bins == 3.0)
                 & (sig.imposed_width == w)].alpha_cdw
        if len(nn) < 20 or not len(ss):
            continue
        crit = nn.quantile(.95)
        L.append(f"{w:>7.1f}{crit:>9.3f}{(ss > crit).mean():>11.1%}"
                 f"{len(nn):>8}")
    L.append("")
    L.append("Read: if the width row in Q1 is flat at every (noise, tip)")
    L.append("cell, width is a proxy and the domain cut should gate on SNR")
    L.append("alone.  If the high-width end collapses in Q2 when the")
    L.append("aperture widens, the fix is an adaptive aperture, not a")
    L.append("refusal - B1 already supports aperture_bins='auto'.")
    L.append("=" * 84)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = run(args.n, args.seed)

    import pandas as pd
    csv = os.path.join(args.out, "width_factorial.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)

    report = summarize(rows)
    print(report)
    with open(os.path.join(args.out,
                           "width_factorial_report.txt"), "w") as fh:
        fh.write(report + "\n")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
