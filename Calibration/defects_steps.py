from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np

import didv_anisotropy as da
from forward_model import CDW_GEOMETRY, draw_random_spec, synthesize

DEFECT_DENSITIES = (0.0, 0.5, 2.0, 8.0, 20.0)
DEFECT_AMPLITUDE = 0.5
STEP_HEIGHTS = (0.0, 0.3, 1.0, 3.0)
STEP_COUNTS = (1, 3)
REL_ANGLES = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0)
REALISTIC_WIDTHS = (2.0, 3.0, 4.0)
ALPHA_LEVELS = (0.0, 0.5)
EST = ["alpha_cdw", "alpha_corrected", "alpha_q2corrected"]


def _one(spec, arm: str, geom: int, extra: Optional[Dict] = None
         ) -> Dict[str, Any]:
    field, _ = synthesize(spec)
    res = da.analyze_didv_field(field, cdw_model=spec.model,
                                bragg_radius=spec.q_bragg,
                                bragg_angles_deg=spec.bragg_angles_deg)
    row: Dict[str, Any] = {
        "arm": arm, "geometry": geom, "status": res.get("status"),
        "alpha_true": spec.alpha_true_cdw, "model": spec.model,
        "defect_density": spec.defect_density,
        "n_steps": spec.n_steps, "step_height": spec.step_height,
        "step_angle_deg": spec.step_angle_deg,
        "step_registry_px": spec.step_registry_px,
        "peak_width_bins": spec.peak_width_bins,
        "theta0_deg": spec.theta0_deg,
        "Nx": spec.Nx, "q_bragg": spec.q_bragg,
    }
    if extra:
        row.update(extra)
    if res.get("status") != "ok":
        return row

    ratio = CDW_GEOMETRY[spec.model][0]
    a_c, a_b = res["alpha_cdw"], res["alpha_bragg"]
    lc = np.arctanh(np.clip(a_c, 0, 1 - 1e-12))
    lb = np.arctanh(np.clip(a_b, 0, 1 - 1e-12))
    snrs = [p["snr"] for p in res["cdw_peaks"] if p["snr"] is not None]
    w = [x for x in (res.get("width_bins_cdw") or []) if x]
    row.update({
        "alpha_cdw": a_c, "alpha_bragg": a_b,
        "alpha_corrected": res["alpha_corrected"],
        "alpha_q2corrected": float(np.tanh(lc - ratio ** 2 * lb)),
        "cdw_over_bragg": (a_c / a_b if a_b > 1e-9 else np.nan),
        "min_snr": min(snrs) if len(snrs) == 3 else None,
        "width_cdw": float(np.mean(w)) if len(w) == 3 else None,
    })
    return row


def run(n: int, seed: int, arms: str = "0123") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def spawn(o):
        return np.random.SeedSequence(seed + o).spawn(n)

    if "0" in arms:
        for i, ch in enumerate(spawn(0)):
            rng = np.random.Generator(np.random.PCG64(ch))
            for dens in DEFECT_DENSITIES:
                sp = draw_random_spec(
                    rng, alpha_true_cdw=0.0, square=True,
                    peak_width_bins=float(rng.choice(REALISTIC_WIDTHS)),
                    defect_density=dens, defect_amplitude=DEFECT_AMPLITUDE,
                    seed=int(ch.entropy) % (2 ** 31))
                rows.append(_one(sp, "S0_defects", i))
            if (i + 1) % 10 == 0:
                print(f"  S0 {i + 1}/{n}", flush=True)

    if "1" in arms:
        for i, ch in enumerate(spawn(1)):
            rng = np.random.Generator(np.random.PCG64(ch))
            for h in STEP_HEIGHTS:
                for ns in STEP_COUNTS:
                    sp = draw_random_spec(
                        rng, alpha_true_cdw=0.0, square=True,
                        peak_width_bins=float(rng.choice(REALISTIC_WIDTHS)),
                        n_steps=ns, step_height=h, step_registry_px=2.0,
                        seed=int(ch.entropy) % (2 ** 31))
                    rows.append(_one(sp, "S1_steps", i))
            if (i + 1) % 10 == 0:
                print(f"  S1 {i + 1}/{n}", flush=True)

    if "2" in arms:
        for i, ch in enumerate(spawn(2)):
            rng = np.random.Generator(np.random.PCG64(ch))
            for rel in REL_ANGLES:
                sp = draw_random_spec(
                    rng, alpha_true_cdw=0.0, square=True,
                    peak_width_bins=float(rng.choice(REALISTIC_WIDTHS)),
                    n_steps=2, step_height=1.0, step_registry_px=2.0,
                    seed=int(ch.entropy) % (2 ** 31))
                sp.step_angle_deg = float((sp.theta0_deg + rel) % 180.0)
                rows.append(_one(sp, "S2_angle", i, {"rel_angle": rel}))
            if (i + 1) % 10 == 0:
                print(f"  S2 {i + 1}/{n}", flush=True)

    if "3" in arms:
        for i, ch in enumerate(spawn(3)):
            rng = np.random.Generator(np.random.PCG64(ch))
            for a_true in ALPHA_LEVELS:
                for h in (0.0, 1.0, 3.0):
                    sp = draw_random_spec(
                        rng, alpha_true_cdw=a_true, square=True,
                        peak_width_bins=float(rng.choice(REALISTIC_WIDTHS)),
                        defect_density=float(rng.uniform(0.5, 8.0)),
                        defect_amplitude=DEFECT_AMPLITUDE,
                        n_steps=int(rng.integers(1, 4)), step_height=h,
                        step_registry_px=float(rng.uniform(0.0, 3.0)),
                        noise_white=float(rng.uniform(0.1, 1.0)),
                        seed=int(ch.entropy) % (2 ** 31))
                    rows.append(_one(sp, "S3_combined" if a_true == 0
                                     else "S3_signal", i))
            if (i + 1) % 10 == 0:
                print(f"  S3 {i + 1}/{n}", flush=True)

    return rows


def summarize(rows: List[Dict[str, Any]]) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    L = ["=" * 84, "DEFECTS AND STEP EDGES", "=" * 84,
         f"{len(df)} runs, {len(ok)} ok, {len(df) - len(ok)} failed", ""]
    if not len(ok):
        return "\n".join(L + ["nothing to report"])

    def q(s, p):
        s = s.dropna()
        return float(s.quantile(p)) if len(s) else np.nan

    s0 = ok[ok.arm == "S0_defects"]
    if len(s0):
        L.append("-- S0: point defects (expected BENIGN: isotropic) ----------")
        L.append(f"{'density':>9}{'aCDW p50':>10}{'aCDW p95':>10}"
                 f"{'aBragg p50':>12}{'minSNR':>10}")
        for d, g in s0.groupby("defect_density"):
            L.append(f"{d:>9.1f}{q(g.alpha_cdw, .5):>10.3f}"
                     f"{q(g.alpha_cdw, .95):>10.3f}"
                     f"{q(g.alpha_bragg, .5):>12.3f}"
                     f"{q(g.min_snr, .5):>10.1f}")
        L.append("   -> if alpha is flat while minSNR falls, defects act")
        L.append("      only through SNR and need no separate treatment.")
        L.append("")

    s1 = ok[ok.arm == "S1_steps"]
    if len(s1):
        L.append("-- S1: step edges (expected DIRECTIONAL: biasing) ----------")
        L.append(f"{'height':>8}{'steps':>7}{'aCDW p50':>10}{'aCDW p95':>10}"
                 f"{'aBragg p50':>12}{'C/B':>7}")
        for (h, ns), g in s1.groupby(["step_height", "n_steps"]):
            L.append(f"{h:>8.1f}{int(ns):>7}{q(g.alpha_cdw, .5):>10.3f}"
                     f"{q(g.alpha_cdw, .95):>10.3f}"
                     f"{q(g.alpha_bragg, .5):>12.3f}"
                     f"{q(g.cdw_over_bragg, .5):>7.2f}")
        L.append("")

    s2 = ok[ok.arm == "S2_angle"]
    if len(s2):
        L.append("-- S2: step angle relative to the lattice ------------------")
        L.append("   The geometric test.  A streak perpendicular to the step")
        L.append("   should hit some Q directions and miss others, so alpha")
        L.append("   must depend on this angle if the mechanism is real.")
        L.append(f"{'rel angle':>11}{'aCDW p50':>10}{'aCDW p95':>10}"
                 f"{'aBragg p50':>12}")
        for a, g in s2.groupby("rel_angle"):
            L.append(f"{a:>11.0f}{q(g.alpha_cdw, .5):>10.3f}"
                     f"{q(g.alpha_cdw, .95):>10.3f}"
                     f"{q(g.alpha_bragg, .5):>12.3f}")
        spread = (s2.groupby("rel_angle").alpha_cdw.median().max()
                  - s2.groupby("rel_angle").alpha_cdw.median().min())
        L.append(f"   spread across angles: {spread:.3f}")
        L.append("   -> a large spread means step ORIENTATION must be")
        L.append("      recorded relative to Q, which no analysis does now.")
        L.append("")

    n3 = ok[ok.arm == "S3_combined"]
    s3 = ok[ok.arm == "S3_signal"]
    if len(n3) and len(s3):
        L.append("-- S3: detection at matched 5% false-positive rate ---------")
        L.append(f"{'estimator':>20}{'a_crit':>9}{'power@0.5':>11}")
        for c in EST:
            crit = q(n3[c], .95)
            L.append(f"{c:>20}{crit:>+9.3f}"
                     f"{(s3[c] > crit).mean():>11.1%}")
        L.append("")
        L.append("   compare with the noise layer, where the same rule gave")
        L.append("   alpha_crit = 0.379 and power 83.6% inside the domain.")
    L.append("=" * 84)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--arms", default="0123")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = run(args.n, args.seed, args.arms)

    import pandas as pd
    csv = os.path.join(args.out, "defects_steps.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)

    report = summarize(rows)
    print(report)
    with open(os.path.join(args.out,
                           "defects_steps_report.txt"), "w") as fh:
        fh.write(report + "\n")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
