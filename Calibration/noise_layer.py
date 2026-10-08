from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np

import didv_anisotropy as da
from forward_model import CDW_GEOMETRY, draw_random_spec, synthesize

NOISE_LEVELS = (0.0, 0.3, 1.0, 3.0, 10.0)
REALISTIC_WIDTHS = (2.0, 3.0, 4.0)
SIGNAL_LADDER = (0.2, 0.5)
EST = ["alpha_cdw", "alpha_corrected", "alpha_normalized",
       "alpha_q1corrected", "alpha_q2corrected"]


def _artanh(a: Optional[float]) -> float:
    if a is None or not np.isfinite(a):
        return float("nan")
    return float(np.arctanh(float(np.clip(a, 0.0, 1.0 - 1e-12))))


def _one(spec, arm: str, geom: int) -> Dict[str, Any]:
    field, _ = synthesize(spec)
    res = da.analyze_didv_field(field, cdw_model=spec.model)

    row: Dict[str, Any] = {
        "arm": arm, "geometry": geom, "status": res.get("status"),
        "model": spec.model, "alpha_true": spec.alpha_true_cdw,
        "noise_white": spec.noise_white, "noise_pink": spec.noise_pink,
        "noise_line": spec.noise_line,
        "tip": spec.tip_alpha_bragg,
        "peak_width_bins": spec.peak_width_bins,
        "theta0_deg": spec.theta0_deg,
        "Nx": spec.Nx, "Ny": spec.Ny, "q_bragg": spec.q_bragg,
        "aspect": spec.Nx / spec.Ny,
    }
    if res.get("status") != "ok":
        return row

    ratio = CDW_GEOMETRY[spec.model][0]
    a_c, a_b = res["alpha_cdw"], res["alpha_bragg"]
    lc, lb = _artanh(a_c), _artanh(a_b)
    row.update({
        "alpha_cdw": a_c, "alpha_bragg": a_b,
        "alpha_corrected": res["alpha_corrected"],
        "alpha_normalized": res["alpha_normalized"],
        "alpha_q1corrected": float(np.tanh(lc - ratio * lb)),
        "alpha_q2corrected": float(np.tanh(lc - ratio ** 2 * lb)),
        "cdw_over_bragg": (float(a_c / a_b) if a_b and a_b > 1e-9
                           else float("nan")),
        "q_ratio": ratio,
    })

    snr_c = [p["snr"] for p in res["cdw_peaks"] if p["snr"] is not None]
    snr_b = [p["snr"] for p in res["bragg_peaks"] if p["snr"] is not None]
    row["min_snr_cdw"] = min(snr_c) if len(snr_c) == 3 else None
    row["min_snr_bragg"] = min(snr_b) if len(snr_b) == 3 else None
    w = [x for x in (res.get("width_bins_cdw") or []) if x]
    wb = [x for x in (res.get("width_bins_bragg") or []) if x]
    row["width_cdw"] = float(np.mean(w)) if len(w) == 3 else None
    row["width_bragg"] = float(np.mean(wb)) if len(wb) == 3 else None
    row["n_resolved"] = res.get("n_directions_resolved")
    return row


def run(n: int, seed: int, arms: str = "01234") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def spawn(o):
        return np.random.SeedSequence(seed + o).spawn(n)

    single = (("0", "noise_white", "N0_white"),
              ("1", "noise_pink", "N1_pink"),
              ("2", "noise_line", "N2_line"))

    for flag, channel, arm in single:
        if flag not in arms:
            continue
        for i, ch in enumerate(spawn(int(flag))):
            rng = np.random.Generator(np.random.PCG64(ch))
            for lev in NOISE_LEVELS:
                sp = draw_random_spec(rng, alpha_true_cdw=0.0, square=True,
                                      peak_width_bins=0.0,
                                      seed=int(ch.entropy) % (2 ** 31),
                                      **{channel: lev})
                rows.append(_one(sp, arm, i))
            if (i + 1) % 20 == 0:
                print(f"  {arm} {i + 1}/{n}", flush=True)

    if "3" in arms or "4" in arms:
        alphas = ([0.0] if "3" in arms else []) + \
                 (list(SIGNAL_LADDER) if "4" in arms else [])
        for i, ch in enumerate(spawn(9)):
            rng = np.random.Generator(np.random.PCG64(ch))
            for a_true in alphas:
                for line in (0.0, 0.3, 1.0, 3.0):
                    w = float(rng.choice(REALISTIC_WIDTHS))
                    sp = draw_random_spec(
                        rng, alpha_true_cdw=a_true, square=True,
                        peak_width_bins=w,
                        tip_alpha_bragg=float(rng.choice([0.0, 0.3])),
                        noise_white=float(rng.uniform(0.1, 1.0)),
                        noise_pink=float(rng.uniform(0.1, 1.0)),
                        noise_line=line,
                        seed=int(ch.entropy) % (2 ** 31))
                    sp.drift_pixels = 0.02 * sp.Nx
                    rows.append(_one(sp, "N3_combined" if a_true == 0.0
                                     else "N4_signal", i))
            if (i + 1) % 20 == 0:
                print(f"  N3/N4 {i + 1}/{n}", flush=True)

    return rows


def summarize(rows: List[Dict[str, Any]]) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    L = ["=" * 88, "NOISE LAYER (white / isotropic 1-f / scan-correlated)",
         "=" * 88,
         f"{len(df)} runs, {len(ok)} ok, {len(df) - len(ok)} failed", ""]
    if not len(ok):
        return "\n".join(L + ["nothing to report"])

    def med(s):
        s = s.dropna()
        return float(np.median(s)) if len(s) else np.nan

    for arm, chan, label in (("N0_white", "noise_white", "white (flat)"),
                             ("N1_pink", "noise_pink", "isotropic 1/f"),
                             ("N2_line", "noise_line", "scan-correlated 1/f")):
        g0 = ok[ok.arm == arm]
        if not len(g0):
            continue
        L.append(f"-- {label} " + "-" * (62 - len(label)))
        L.append(f"{'level':>7}{'aCDW p50':>10}{'aCDW p95':>10}"
                 f"{'aBragg p50':>12}{'C/B ratio':>11}{'minSNR':>10}{'fail%':>7}")
        for lev, g in g0.groupby(chan):
            tot = ((df.arm == arm) & (df[chan] == lev)).sum()
            L.append(f"{lev:>7.1f}{med(g.alpha_cdw):>10.4f}"
                     f"{g.alpha_cdw.quantile(.95):>10.4f}"
                     f"{med(g.alpha_bragg):>12.4f}"
                     f"{med(g.cdw_over_bragg):>11.2f}"
                     f"{med(g.min_snr_cdw):>10.1f}"
                     f"{1 - len(g) / max(tot, 1):>7.1%}")
        L.append("")

    L.append("-- FINGERPRINT: alpha_CDW / alpha_Bragg by channel -------------")
    L.append("   tip attenuates high-q  -> ratio < 1")
    L.append("   1/f contaminates low-q -> ratio > 1")
    for arm, label in (("N0_white", "white"), ("N1_pink", "isotropic 1/f"),
                       ("N2_line", "scan-correlated 1/f")):
        g = ok[(ok.arm == arm) & ok.cdw_over_bragg.notna()]
        g = g[g.alpha_bragg > 0.01]
        if len(g):
            L.append(f"   {label:>22}: median {med(g.cdw_over_bragg):>6.2f}"
                     f"   IQR [{g.cdw_over_bragg.quantile(.25):.2f}, "
                     f"{g.cdw_over_bragg.quantile(.75):.2f}]")
    L.append("")
    L.append("   For reference, the experimental maps gave alpha_CDW/alpha_Bragg")
    L.append("   of 2.05 (CsV3Sb5), 0.82 (RbV3Sb5), 0.60 (TaS2), 0.21 (sl-NbSe2).")
    L.append("")

    n3 = ok[ok.arm == "N3_combined"]
    n4 = ok[ok.arm == "N4_signal"]
    if len(n3):
        L.append("-- N3: can the artifact budget reach the observed +0.27? ------")
        L.append(f"{'line':>7}{'aCDW p50':>10}{'aCDW p95':>10}"
                 f"{'aBragg p50':>12}{'width':>8}{'minSNR':>10}")
        for lev, g in n3.groupby("noise_line"):
            L.append(f"{lev:>7.1f}{med(g.alpha_cdw):>10.4f}"
                     f"{g.alpha_cdw.quantile(.95):>10.4f}"
                     f"{med(g.alpha_bragg):>12.4f}"
                     f"{med(g.width_cdw):>8.2f}{med(g.min_snr_cdw):>10.1f}")
        L.append("")

    if len(n3) and len(n4):
        L.append("-- N3/N4: detection at matched 5% false-positive rate ---------")
        L.append(f"{'estimator':>20}{'a_crit':>9}{'pow@0.2':>9}"
                 f"{'pow@0.5':>9}{'frac<0':>8}")
        for c in EST:
            crit = n3[c].quantile(.95)
            if not np.isfinite(crit):
                continue
            p2 = (n4[n4.alpha_true == 0.2][c] > crit).mean()
            p5 = (n4[n4.alpha_true == 0.5][c] > crit).mean()
            L.append(f"{c:>20}{crit:>+9.4f}{p2:>9.1%}{p5:>9.1%}"
                     f"{(n3[c] < 0).mean():>8.1%}")
        L.append("")

    L.append("-- alpha_crit covariates: is the null predictable from things")
    L.append("   B1 can actually measure? (Spearman vs alpha_cdw, N3 null) ---")
    if len(n3):
        for cov in ("min_snr_cdw", "min_snr_bragg", "width_cdw",
                    "alpha_bragg", "Nx"):
            s = n3[[cov, "alpha_cdw"]].dropna()
            if len(s) > 20:
                L.append(f"     {cov:>16}: rho = "
                         f"{s[cov].corr(s.alpha_cdw, method='spearman'):+.3f}")
    L.append("=" * 88)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--arms", default="01234")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = run(args.n, args.seed, args.arms)

    import pandas as pd
    csv = os.path.join(args.out, "noise_layer.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)

    report = summarize(rows)
    print(report)
    with open(os.path.join(args.out, "noise_layer_report.txt"), "w") as fh:
        fh.write(report + "\n")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
