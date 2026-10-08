from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np

import didv_anisotropy as da
from forward_model import CDW_GEOMETRY, draw_random_spec, synthesize

TIP_LADDER = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
REALISTIC_WIDTHS = (2.0, 3.0, 4.0)
SIGNAL_LADDER = (0.2, 0.5)
MODELS = ("2x2", "3x3")


def _safe_artanh(a: Optional[float]) -> float:
    if a is None or not np.isfinite(a):
        return float("nan")
    return float(np.arctanh(float(np.clip(a, 0.0, 1.0 - 1e-12))))


def estimators(res: Dict[str, Any], ratio: float) -> Dict[str, float]:
    a_c = res.get("alpha_cdw")
    a_b = res.get("alpha_bragg")
    out = {
        "alpha_cdw": a_c,
        "alpha_bragg": a_b,
        "alpha_corrected": res.get("alpha_corrected"),
        "alpha_normalized": res.get("alpha_normalized"),
    }
    lc, lb = _safe_artanh(a_c), _safe_artanh(a_b)
    out["alpha_q2corrected"] = (float(np.tanh(lc - ratio ** 2 * lb))
                                if np.isfinite(lc) and np.isfinite(lb)
                                else float("nan"))
    out["alpha_cdw_predicted"] = (float(np.tanh(ratio ** 2 * lb))
                                  if np.isfinite(lb) else float("nan"))
    return out


def _one(spec, arm: str) -> Optional[Dict[str, Any]]:
    field, truth = synthesize(spec)
    res = da.analyze_didv_field(
        field, cdw_model=spec.model, bragg_radius=spec.q_bragg,
        bragg_angles_deg=spec.bragg_angles_deg)
    if res.get("status") != "ok":
        return {"arm": arm, "status": res.get("status"),
                "tip_alpha_bragg": spec.tip_alpha_bragg,
                "model": spec.model,
                "peak_width_bins": spec.peak_width_bins,
                "alpha_true_cdw": spec.alpha_true_cdw}

    ratio = CDW_GEOMETRY[spec.model][0]
    row: Dict[str, Any] = {
        "arm": arm, "status": "ok",
        "model": spec.model, "q_ratio": ratio,
        "tip_alpha_bragg": spec.tip_alpha_bragg,
        "tip_angle_deg": spec.tip_angle_deg,
        "peak_width_bins": spec.peak_width_bins,
        "alpha_true_cdw": spec.alpha_true_cdw,
        "Ny": spec.Ny, "Nx": spec.Nx, "q_bragg": spec.q_bragg,
        "theta0_deg": spec.theta0_deg,
    }
    row.update(estimators(res, ratio))
    w = [x for x in (res.get("width_bins_cdw") or []) if x]
    row["width_cdw"] = float(np.mean(w)) if len(w) == 3 else None
    return row


def run(n: int, seed: int, arms: str = "012") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if "0" in arms:
        for i, child in enumerate(np.random.SeedSequence(seed).spawn(n)):
            rng = np.random.Generator(np.random.PCG64(child))
            for model in MODELS:
                for tip in TIP_LADDER:
                    spec = draw_random_spec(
                        rng, alpha_true_cdw=0.0, square=True,
                        models=(model,), peak_width_bins=0.0,
                        tip_alpha_bragg=tip,
                        seed=int(child.entropy) % (2 ** 31))
                    spec.model = model
                    r = _one(spec, "T0_perfect_order")
                    if r:
                        rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"  T0 {i + 1}/{n}", flush=True)

    if "1" in arms:
        for i, child in enumerate(np.random.SeedSequence(seed + 1).spawn(n)):
            rng = np.random.Generator(np.random.PCG64(child))
            for tip in TIP_LADDER:
                w = float(rng.choice(REALISTIC_WIDTHS))
                spec = draw_random_spec(
                    rng, alpha_true_cdw=0.0, square=True,
                    peak_width_bins=w, tip_alpha_bragg=tip,
                    seed=int(child.entropy) % (2 ** 31))
                r = _one(spec, "T1_realistic")
                if r:
                    rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"  T1 {i + 1}/{n}", flush=True)

    if "2" in arms:
        for i, child in enumerate(np.random.SeedSequence(seed + 2).spawn(n)):
            rng = np.random.Generator(np.random.PCG64(child))
            for a_true in SIGNAL_LADDER:
                for tip in (0.0, 0.3, 0.6, 0.9):
                    w = float(rng.choice(REALISTIC_WIDTHS))
                    spec = draw_random_spec(
                        rng, alpha_true_cdw=a_true, square=True,
                        peak_width_bins=w, tip_alpha_bragg=tip,
                        seed=int(child.entropy) % (2 ** 31))
                    r = _one(spec, "T2_signal")
                    if r:
                        rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"  T2 {i + 1}/{n}", flush=True)

    return rows


def summarize(rows: List[Dict[str, Any]]) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    L = ["=" * 82,
         "ELLIPTICAL TIP LAYER",
         "=" * 82,
         f"{len(df)} runs, {len(ok)} ok, {len(df) - len(ok)} failed", ""]
    if not len(ok):
        return "\n".join(L + ["nothing to report"])

    def med(s):
        return float(np.median(s.dropna())) if len(s.dropna()) else np.nan

    t0 = ok[ok.arm == "T0_perfect_order"]
    if len(t0):
        L.append("-- T0: perfect order, does the q^2 law hold? -----------------")
        L.append(f"{'model':>7}{'tip aB':>8}{'aBragg':>9}{'aCDW':>9}"
                 f"{'predicted':>11}{'resid':>9}")
        for model, gm in t0.groupby("model"):
            for tip, g in gm.groupby("tip_alpha_bragg"):
                L.append(f"{model:>7}{tip:>8.1f}{med(g.alpha_bragg):>9.4f}"
                         f"{med(g.alpha_cdw):>9.4f}"
                         f"{med(g.alpha_cdw_predicted):>11.4f}"
                         f"{med(g.alpha_cdw - g.alpha_cdw_predicted):>+9.4f}")
        L.append("")
        L.append("-- T0: estimator comparison (alpha_true = 0) -----------------")
        L.append(f"{'tip aB':>8}{'a_CDW':>9}{'a_corr':>9}{'a_norm':>9}"
                 f"{'a_q2':>9}")
        for tip, g in t0.groupby("tip_alpha_bragg"):
            L.append(f"{tip:>8.1f}{med(g.alpha_cdw):>9.4f}"
                     f"{med(g.alpha_corrected):>+9.4f}"
                     f"{med(g.alpha_normalized):>9.4f}"
                     f"{med(g.alpha_q2corrected):>+9.4f}")
        L.append("")

    t1 = ok[ok.arm == "T1_realistic"]
    if len(t1):
        L.append("-- T1: with realistic speckle (alpha_true = 0) ---------------")
        L.append(f"{'tip aB':>8}{'a_CDW p50':>11}{'a_CDW p95':>11}"
                 f"{'a_q2 p50':>10}{'a_q2 p95':>10}")
        for tip, g in t1.groupby("tip_alpha_bragg"):
            L.append(f"{tip:>8.1f}{med(g.alpha_cdw):>11.4f}"
                     f"{g.alpha_cdw.quantile(.95):>11.4f}"
                     f"{med(g.alpha_q2corrected):>10.4f}"
                     f"{g.alpha_q2corrected.quantile(.95):>10.4f}")
        L.append("")
        L.append("   null spread (p95, the quantity alpha_crit is built from):")
        for c in ("alpha_cdw", "alpha_corrected", "alpha_normalized",
                  "alpha_q2corrected"):
            L.append(f"     {c:>20}: {t1[c].quantile(.95):+.4f}")
        L.append("")

    t2 = ok[ok.arm == "T2_signal"]
    if len(t2):
        L.append("-- T2: is real signal preserved? -----------------------------")
        L.append(f"{'a_true':>8}{'tip aB':>8}{'a_CDW':>9}{'a_q2':>9}"
                 f"{'a_corr':>9}")
        for (at, tip), g in t2.groupby(["alpha_true_cdw", "tip_alpha_bragg"]):
            L.append(f"{at:>8.2f}{tip:>8.1f}{med(g.alpha_cdw):>9.4f}"
                     f"{med(g.alpha_q2corrected):>9.4f}"
                     f"{med(g.alpha_corrected):>+9.4f}")
        L.append("")

    L.append("Read: T0 residuals near zero confirm the q^2 mechanism.  In T1")
    L.append("the estimator with the SMALLEST p95 under alpha_true=0, while")
    L.append("still tracking alpha_true in T2, is the one B3 should report.")
    L.append("=" * 82)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--arms", default="012")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = run(args.n, args.seed, args.arms)

    import pandas as pd
    csv = os.path.join(args.out, "tip_layer.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)

    report = summarize(rows)
    print(report)
    with open(os.path.join(args.out, "tip_layer_report.txt"), "w") as fh:
        fh.write(report + "\n")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
