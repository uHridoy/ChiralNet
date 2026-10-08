from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import didv_anisotropy as da
from forward_model import CDW_GEOMETRY, draw_random_spec, synthesize

TIP_LEVELS = (0.0, 0.3, 0.6, 0.9)
ALPHA_LEVELS = (0.0, 0.2, 0.5)
CREEP_LEVELS = (0.0, 0.10)
ANCHOR_MODES = ("pinned", "auto")
REALISTIC_WIDTHS = (2.0, 3.0, 4.0)
DRIFT_FRAC = 0.02

EST = ["alpha_cdw", "alpha_corrected", "alpha_normalized",
       "alpha_q1corrected", "alpha_q2corrected"]


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
    good = np.isfinite(lc) and np.isfinite(lb)
    out["alpha_q1corrected"] = float(np.tanh(lc - ratio * lb)) if good else np.nan
    out["alpha_q2corrected"] = (float(np.tanh(lc - ratio ** 2 * lb))
                                if good else np.nan)
    return out


def _one(spec, anchor_mode: str, geom: int) -> Dict[str, Any]:
    field, _ = synthesize(spec)
    kwargs: Dict[str, Any] = {"cdw_model": spec.model}
    if anchor_mode == "pinned":
        kwargs.update(bragg_radius=spec.q_bragg,
                      bragg_angles_deg=spec.bragg_angles_deg)
    res = da.analyze_didv_field(field, **kwargs)

    row: Dict[str, Any] = {
        "geometry": geom, "anchor": anchor_mode, "status": res.get("status"),
        "model": spec.model, "alpha_true": spec.alpha_true_cdw,
        "tip": spec.tip_alpha_bragg, "creep_frac": spec.creep_pixels / spec.Nx,
        "peak_width_bins": spec.peak_width_bins,
        "Nx": spec.Nx, "q_bragg": spec.q_bragg,
    }
    if res.get("status") != "ok":
        return row
    row.update(estimators(res, CDW_GEOMETRY[spec.model][0]))
    w = [x for x in (res.get("width_bins_cdw") or []) if x]
    row["width_cdw"] = float(np.mean(w)) if len(w) == 3 else None
    return row


def run(n: int, seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gi, child in enumerate(np.random.SeedSequence(seed).spawn(n)):
        rng = np.random.Generator(np.random.PCG64(child))
        width = float(rng.choice(REALISTIC_WIDTHS))
        base_seed = int(child.entropy) % (2 ** 31)

        for creep in CREEP_LEVELS:
            for tip in TIP_LEVELS:
                for a_true in ALPHA_LEVELS:
                    spec = draw_random_spec(
                        rng, alpha_true_cdw=a_true, square=True,
                        peak_width_bins=width, tip_alpha_bragg=tip,
                        seed=base_seed)
                    spec.creep_pixels = creep * spec.Nx
                    spec.drift_pixels = DRIFT_FRAC * spec.Nx
                    for mode in ANCHOR_MODES:
                        rows.append(_one(spec, mode, gi))
        if (gi + 1) % 10 == 0:
            print(f"  {gi + 1}/{n} geometries", flush=True)
    return rows


def _power(null: "pd.DataFrame", sig: "pd.DataFrame", col: str,
           q: float = 0.95) -> float:
    crit = null[col].quantile(q)
    s = sig[col].dropna()
    return float((s > crit).mean()) if len(s) and np.isfinite(crit) else np.nan


def _bootstrap_diff(null, sig, a: str, b: str, n_boot: int = 400,
                    rng=None) -> Sequence[float]:
    rng = rng or np.random.default_rng(0)
    geoms = np.union1d(null.geometry.unique(), sig.geometry.unique())
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(geoms, size=len(geoms), replace=True)
        n_b = null[null.geometry.isin(pick)]
        s_b = sig[sig.geometry.isin(pick)]
        if not len(n_b) or not len(s_b):
            continue
        diffs.append(_power(n_b, s_b, a) - _power(n_b, s_b, b))
    if not diffs:
        return (np.nan, np.nan)
    return (float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def summarize(rows: List[Dict[str, Any]]) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    L = ["=" * 86, "ESTIMATOR SHOWDOWN (2x2: anchoring x creep)", "=" * 86,
         f"{len(df)} runs, {len(ok)} ok, {len(df) - len(ok)} failed", ""]
    if not len(ok):
        return "\n".join(L + ["nothing to report"])

    L.append("-- failure rate by cell -----------------------------------")
    for (mode, creep), g in df.groupby(["anchor", "creep_frac"]):
        L.append(f"   anchor={mode:>6} creep={creep:.2f}: "
                 f"{(g.status != 'ok').mean():.1%}")
    L.append("")

    rng = np.random.default_rng(7)
    for mode in ANCHOR_MODES:
        for creep in CREEP_LEVELS:
            cell = ok[(ok.anchor == mode) & (np.isclose(ok.creep_frac, creep))]
            null = cell[cell.alpha_true == 0.0]
            if not len(null):
                continue
            L.append(f"-- anchor={mode}, creep={creep:.2f}  "
                     f"(null n={len(null)}) " + "-" * 20)
            L.append(f"{'estimator':>20}{'a_crit':>9}{'pow@0.2':>9}"
                     f"{'pow@0.5':>9}{'frac<0':>8}")
            for c in EST:
                s2 = cell[cell.alpha_true == 0.2]
                s5 = cell[cell.alpha_true == 0.5]
                crit = null[c].quantile(.95)
                L.append(f"{c:>20}{crit:>+9.4f}"
                         f"{_power(null, s2, c):>9.1%}"
                         f"{_power(null, s5, c):>9.1%}"
                         f"{(null[c] < 0).mean():>8.1%}")
            lo, hi = _bootstrap_diff(null, cell[cell.alpha_true == 0.5],
                                     "alpha_q2corrected", "alpha_cdw",
                                     rng=rng)
            d = (_power(null, cell[cell.alpha_true == 0.5],
                        "alpha_q2corrected")
                 - _power(null, cell[cell.alpha_true == 0.5], "alpha_cdw"))
            verdict = ("q2 better" if lo > 0 else
                       "cdw better" if hi < 0 else "NOT DISTINGUISHABLE")
            L.append(f"   q2 - cdw power@0.5 = {d:+.1%}  "
                     f"95% CI [{lo:+.1%}, {hi:+.1%}]  -> {verdict}")
            L.append("")

    L.append("-- Does anchoring change the answer? (paired, same fields) --")
    for creep in CREEP_LEVELS:
        for c in ("alpha_cdw", "alpha_q2corrected"):
            vals = {}
            for mode in ANCHOR_MODES:
                cell = ok[(ok.anchor == mode)
                          & (np.isclose(ok.creep_frac, creep))]
                vals[mode] = _power(cell[cell.alpha_true == 0.0],
                                    cell[cell.alpha_true == 0.5], c)
            L.append(f"   creep={creep:.2f} {c:>20}: "
                     f"pinned {vals['pinned']:.1%}  auto {vals['auto']:.1%}")
    L.append("")
    L.append("Read: if the CI on q2-cdw straddles zero in the auto cells,")
    L.append("neither estimator is demonstrably better and B3 should report")
    L.append("the simpler one.  Auto-anchoring is the condition that counts,")
    L.append("since that is what a published figure gets.")
    L.append("=" * 86)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = run(args.n, args.seed)

    import pandas as pd
    csv = os.path.join(args.out, "estimator_showdown.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)

    report = summarize(rows)
    print(report)
    with open(os.path.join(args.out,
                           "estimator_showdown_report.txt"), "w") as fh:
        fh.write(report + "\n")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
