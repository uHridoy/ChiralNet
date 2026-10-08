from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np

import didv_anisotropy as da
from forward_model import CDW_GEOMETRY, draw_random_spec, synthesize

DRIFT_FRACS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
CREEP_FRACS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
CREEP_TAUS = (0.15, 0.5)
ASPECT_ERRORS = (0.0, 0.02, 0.05)
TIP_LEVELS = (0.0, 0.3, 0.6)
REALISTIC_WIDTHS = (2.0, 3.0, 4.0)
SIGNAL_LADDER = (0.2, 0.5)


def _artanh(a: Optional[float]) -> float:
    if a is None or not np.isfinite(a):
        return float("nan")
    return float(np.arctanh(float(np.clip(a, 0.0, 1.0 - 1e-12))))


def estimators(res: Dict[str, Any], ratio: float) -> Dict[str, float]:
    a_c, a_b = res.get("alpha_cdw"), res.get("alpha_bragg")
    out = {"alpha_cdw": a_c, "alpha_bragg": a_b,
           "alpha_corrected": res.get("alpha_corrected"),
           "alpha_normalized": res.get("alpha_normalized")}
    lc, lb = _artanh(a_c), _artanh(a_b)
    out["alpha_q2corrected"] = (float(np.tanh(lc - ratio ** 2 * lb))
                                if np.isfinite(lc) and np.isfinite(lb)
                                else np.nan)
    out["alpha_q1corrected"] = (float(np.tanh(lc - ratio * lb))
                                if np.isfinite(lc) and np.isfinite(lb)
                                else np.nan)
    return out


def _one(spec, arm: str) -> Dict[str, Any]:
    field, _ = synthesize(spec)
    res = da.analyze_didv_field(field, cdw_model=spec.model)

    row: Dict[str, Any] = {
        "arm": arm, "status": res.get("status"), "model": spec.model,
        "alpha_true_cdw": spec.alpha_true_cdw,
        "peak_width_bins": spec.peak_width_bins,
        "tip_alpha_bragg": spec.tip_alpha_bragg,
        "drift_frac": spec.drift_pixels / max(spec.Nx, 1),
        "creep_frac": spec.creep_pixels / max(spec.Nx, 1),
        "creep_tau": spec.creep_tau,
        "aspect_error": spec.aspect_error,
        "drift_pixels": spec.drift_pixels,
        "creep_pixels": spec.creep_pixels,
        "Nx": spec.Nx, "Ny": spec.Ny, "q_bragg": spec.q_bragg,
    }
    if res.get("status") != "ok":
        return row

    row.update(estimators(res, CDW_GEOMETRY[spec.model][0]))
    w = [x for x in (res.get("width_bins_cdw") or []) if x]
    row["width_cdw"] = float(np.mean(w)) if len(w) == 3 else None
    wb = [x for x in (res.get("width_bins_bragg") or []) if x]
    row["width_bragg"] = float(np.mean(wb)) if len(wb) == 3 else None
    row["model_recovered"] = res.get("cdw_model", {}).get("name")
    return row


def run(n: int, seed: int, arms: str = "01234") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def spawn(offset: int):
        return np.random.SeedSequence(seed + offset).spawn(n)

    if "0" in arms:
        for i, ch in enumerate(spawn(0)):
            rng = np.random.Generator(np.random.PCG64(ch))
            for dfrac in DRIFT_FRACS:
                for aerr in ASPECT_ERRORS:
                    sp = draw_random_spec(rng, alpha_true_cdw=0.0,
                                          square=True, peak_width_bins=0.0,
                                          aspect_error=aerr,
                                          seed=int(ch.entropy) % (2 ** 31))
                    sp.drift_pixels = dfrac * sp.Nx
                    rows.append(_one(sp, "C0_affine"))
            if (i + 1) % 10 == 0:
                print(f"  C0 {i + 1}/{n}", flush=True)

    if "1" in arms:
        for i, ch in enumerate(spawn(1)):
            rng = np.random.Generator(np.random.PCG64(ch))
            for cfrac in CREEP_FRACS:
                for tau in CREEP_TAUS:
                    sp = draw_random_spec(rng, alpha_true_cdw=0.0,
                                          square=True, peak_width_bins=0.0,
                                          creep_tau=tau,
                                          seed=int(ch.entropy) % (2 ** 31))
                    sp.creep_pixels = cfrac * sp.Nx
                    rows.append(_one(sp, "C1_creep"))
            if (i + 1) % 10 == 0:
                print(f"  C1 {i + 1}/{n}", flush=True)

    if "2" in arms:
        for i, ch in enumerate(spawn(2)):
            rng = np.random.Generator(np.random.PCG64(ch))
            for cfrac in CREEP_FRACS:
                w = float(rng.choice(REALISTIC_WIDTHS))
                sp = draw_random_spec(rng, alpha_true_cdw=0.0, square=True,
                                      peak_width_bins=w,
                                      creep_tau=float(rng.choice(CREEP_TAUS)),
                                      seed=int(ch.entropy) % (2 ** 31))
                sp.creep_pixels = cfrac * sp.Nx
                sp.drift_pixels = float(rng.choice(DRIFT_FRACS)) * sp.Nx
                rows.append(_one(sp, "C2_realistic"))
            if (i + 1) % 10 == 0:
                print(f"  C2 {i + 1}/{n}", flush=True)

    if "3" in arms:
        for i, ch in enumerate(spawn(3)):
            rng = np.random.Generator(np.random.PCG64(ch))
            for tip in TIP_LEVELS:
                for cfrac in (0.0, 0.05, 0.10, 0.20):
                    w = float(rng.choice(REALISTIC_WIDTHS))
                    sp = draw_random_spec(
                        rng, alpha_true_cdw=0.0, square=True,
                        peak_width_bins=w, tip_alpha_bragg=tip,
                        creep_tau=float(rng.choice(CREEP_TAUS)),
                        seed=int(ch.entropy) % (2 ** 31))
                    sp.creep_pixels = cfrac * sp.Nx
                    sp.drift_pixels = 0.02 * sp.Nx
                    rows.append(_one(sp, "C3_combined"))
            if (i + 1) % 10 == 0:
                print(f"  C3 {i + 1}/{n}", flush=True)

    if "4" in arms:
        for i, ch in enumerate(spawn(4)):
            rng = np.random.Generator(np.random.PCG64(ch))
            for a_true in SIGNAL_LADDER:
                for tip in TIP_LEVELS:
                    for cfrac in (0.0, 0.10):
                        w = float(rng.choice(REALISTIC_WIDTHS))
                        sp = draw_random_spec(
                            rng, alpha_true_cdw=a_true, square=True,
                            peak_width_bins=w, tip_alpha_bragg=tip,
                            creep_tau=float(rng.choice(CREEP_TAUS)),
                            seed=int(ch.entropy) % (2 ** 31))
                        sp.creep_pixels = cfrac * sp.Nx
                        sp.drift_pixels = 0.02 * sp.Nx
                        rows.append(_one(sp, "C4_signal"))
            if (i + 1) % 10 == 0:
                print(f"  C4 {i + 1}/{n}", flush=True)

    return rows


EST = ["alpha_cdw", "alpha_corrected", "alpha_normalized",
       "alpha_q1corrected", "alpha_q2corrected"]


def summarize(rows: List[Dict[str, Any]]) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    L = ["=" * 84, "DRIFT / CREEP / ASPECT LAYER", "=" * 84,
         f"{len(df)} runs, {len(ok)} ok, {len(df) - len(ok)} failed",
         "(anchor auto-detected; only the superlattice model is pinned)", ""]
    if not len(ok):
        return "\n".join(L + ["nothing to report"])

    def med(s):
        s = s.dropna()
        return float(np.median(s)) if len(s) else np.nan

    c0 = ok[ok.arm == "C0_affine"]
    if len(c0):
        L.append("-- C0: are affine distortions benign? ----------------------")
        L.append(f"{'drift':>8}{'aspect':>8}{'aCDW p50':>10}{'aCDW p95':>10}"
                 f"{'aBragg p95':>12}{'fail%':>7}")
        for (dfr, ae), g in c0.groupby(["drift_frac", "aspect_error"]):
            tot = ((df.arm == "C0_affine") & (df.drift_frac == dfr)
                   & (df.aspect_error == ae)).sum()
            L.append(f"{dfr:>8.2f}{ae:>8.2f}{med(g.alpha_cdw):>10.4f}"
                     f"{g.alpha_cdw.quantile(.95):>10.4f}"
                     f"{g.alpha_bragg.quantile(.95):>12.4f}"
                     f"{1 - len(g) / max(tot, 1):>7.1%}")
        L.append("")

    c1 = ok[ok.arm == "C1_creep"]
    if len(c1):
        L.append("-- C1: creep alone (perfect order) -------------------------")
        L.append(f"{'creep':>8}{'tau':>6}{'aCDW p50':>10}{'aCDW p95':>10}"
                 f"{'aBragg p50':>12}{'width':>8}")
        for (cfr, tau), g in c1.groupby(["creep_frac", "creep_tau"]):
            L.append(f"{cfr:>8.2f}{tau:>6.2f}{med(g.alpha_cdw):>10.4f}"
                     f"{g.alpha_cdw.quantile(.95):>10.4f}"
                     f"{med(g.alpha_bragg):>12.4f}{med(g.width_cdw):>8.2f}")
        L.append("")

    c2 = ok[ok.arm == "C2_realistic"]
    if len(c2):
        L.append("-- C2: creep with realistic speckle ------------------------")
        L.append(f"{'creep':>8}{'aCDW p50':>10}{'aCDW p95':>10}"
                 f"{'q2 p50':>9}{'q2 p95':>9}{'width':>8}")
        for cfr, g in c2.groupby("creep_frac"):
            L.append(f"{cfr:>8.2f}{med(g.alpha_cdw):>10.4f}"
                     f"{g.alpha_cdw.quantile(.95):>10.4f}"
                     f"{med(g.alpha_q2corrected):>9.4f}"
                     f"{g.alpha_q2corrected.quantile(.95):>9.4f}"
                     f"{med(g.width_cdw):>8.2f}")
        L.append("")

    c3 = ok[ok.arm == "C3_combined"]
    c4 = ok[ok.arm == "C4_signal"]
    if len(c3):
        L.append("-- C3/C4: DETECTION at matched 5% false-positive rate ------")
        L.append("   alpha_crit = p95 of the C3 null; power measured on C4")
        L.append(f"{'estimator':>20}{'a_crit':>9}{'pow@0.2':>9}{'pow@0.5':>9}")
        for c in EST:
            crit = c3[c].quantile(.95)
            if not np.isfinite(crit):
                continue
            p2 = ((c4[c4.alpha_true_cdw == 0.2][c] > crit).mean()
                  if len(c4) else np.nan)
            p5 = ((c4[c4.alpha_true_cdw == 0.5][c] > crit).mean()
                  if len(c4) else np.nan)
            L.append(f"{c:>20}{crit:>+9.4f}{p2:>9.1%}{p5:>9.1%}")
        L.append("")
        L.append("   null median and sign (a low p95 from a negative-biased")
        L.append("   estimator is not a good threshold - see the tip layer):")
        for c in EST:
            s = c3[c].dropna()
            if len(s):
                L.append(f"     {c:>20}: median {s.median():+.4f}"
                         f"  frac<0 {(s < 0).mean():>6.1%}")
        L.append("")
        L.append("   null p95 vs creep strength (does alpha_crit have to")
        L.append("   depend on a quantity you cannot measure?):")
        for c in ("alpha_cdw", "alpha_q1corrected", "alpha_q2corrected"):
            lo = c3[c3.creep_frac <= 0.05][c].quantile(.95)
            hi = c3[c3.creep_frac >= 0.10][c].quantile(.95)
            if np.isfinite(lo) and np.isfinite(hi) and lo != 0:
                L.append(f"     {c:>20}: weak {lo:+.4f}  strong {hi:+.4f}"
                         f"  ratio {hi / lo:.2f}")
    L.append("=" * 84)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--arms", default="01234")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = run(args.n, args.seed, args.arms)

    import pandas as pd
    csv = os.path.join(args.out, "drift_creep.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)

    report = summarize(rows)
    print(report)
    with open(os.path.join(args.out, "drift_creep_report.txt"), "w") as fh:
        fh.write(report + "\n")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
