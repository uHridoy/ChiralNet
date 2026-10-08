from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

import didv_anisotropy as da
from forward_model import (
    SyntheticSpec, draw_random_spec, subbin_offsets, synthesize,
)

APERTURES = (2.0, 3.0, 4.0, 5.0)
ALPHA_LADDER = (0.02, 0.05, 0.10, 0.20, 0.40, 0.80)
PEAK_WIDTHS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _row_from_result(spec: SyntheticSpec, truth: Dict[str, Any],
                     res: Dict[str, Any], arm: str,
                     aperture_bins: float, pinned: bool) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "arm": arm,
        "pinned": pinned,
        "aperture_bins": aperture_bins,
        "status": res.get("status"),
        "message": res.get("message"),
        "n_warnings": len(res.get("warnings", [])),
    }
    row.update({k: v for k, v in truth.items()
                if k not in ("bragg_angles_deg", "cdw_angles_deg",
                             "I_bragg_true", "I_cdw_true")})
    row.update(subbin_offsets(spec))
    row["aspect"] = spec.Nx / spec.Ny
    row["peak_width_bins"] = spec.peak_width_bins
    row["q_bragg_bins"] = spec.q_bragg * np.sqrt(spec.Nx * spec.Ny)
    row["q_cdw_bins"] = spec.q_cdw * np.sqrt(spec.Nx * spec.Ny)

    if res.get("status") == "ok":
        row.update({
            "alpha_cdw": res["alpha_cdw"],
            "alpha_bragg": res["alpha_bragg"],
            "alpha_corrected": res["alpha_corrected"],
            "alpha_normalized": res["alpha_normalized"],
            "n_resolved": res.get("n_directions_resolved"),
            "model_recovered": res.get("cdw_model", {}).get("name"),
            "hermitian_ok": res.get("hermitian_check", {}).get("passed"),
        })
        snr_c = [r["snr"] for r in res["cdw_peaks"] if r["snr"] is not None]
        snr_b = [r["snr"] for r in res["bragg_peaks"] if r["snr"] is not None]
        row["min_snr_cdw"] = min(snr_c) if len(snr_c) == 3 else None
        row["min_snr_bragg"] = min(snr_b) if len(snr_b) == 3 else None
        for tag, key in (("cdw", "alpha_cdw"), ("bragg", "alpha_bragg")):
            a = float(np.clip(res[key], 0.0, 1.0 - 1e-15))
            row[f"logspread_{tag}"] = float(np.arctanh(a))
        ratio = truth["q_cdw"] / truth["q_bragg"]
        row["alpha_q2corrected"] = float(np.tanh(
            row["logspread_cdw"] - ratio ** 2 * row["logspread_bragg"]))
    else:
        for k in ("alpha_cdw", "alpha_bragg", "alpha_corrected",
                  "alpha_normalized", "alpha_q2corrected", "n_resolved",
                  "model_recovered", "hermitian_ok", "min_snr_cdw",
                  "min_snr_bragg", "logspread_cdw", "logspread_bragg"):
            row[k] = None
    return row


def _run_one(spec: SyntheticSpec, arm: str, aperture_bins: float,
             pinned: bool) -> Dict[str, Any]:
    field, truth = synthesize(spec)
    kwargs: Dict[str, Any] = {"aperture_bins": aperture_bins}
    if pinned:
        kwargs.update(cdw_model=spec.model,
                      bragg_radius=spec.q_bragg,
                      bragg_angles_deg=spec.bragg_angles_deg)
    res = da.analyze_didv_field(field, source_label=f"synthetic:{arm}",
                                **kwargs)
    return _row_from_result(spec, truth, res, arm, aperture_bins, pinned)


def run_study(n_geometries: int = 400, seed: int = 20260819,
              arms: str = "ABCD") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = np.random.SeedSequence(seed)

    if "A" in arms:
        t0 = time.time()
        for i, child in enumerate(root.spawn(n_geometries)):
            rng = np.random.Generator(np.random.PCG64(child))
            spec = draw_random_spec(rng, alpha_true_cdw=0.0,
                                    alpha_true_bragg=0.0, square=True,
                                    seed=int(child.entropy) % (2 ** 31))
            for ap in APERTURES:
                rows.append(_run_one(spec, "A_pinned_square", ap, True))
            if (i + 1) % 50 == 0:
                print(f"  arm A {i + 1}/{n_geometries} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    if "B" in arms:
        t0 = time.time()
        n_b = max(1, n_geometries // 2)
        for i, child in enumerate(np.random.SeedSequence(seed + 1).spawn(n_b)):
            rng = np.random.Generator(np.random.PCG64(child))
            spec = draw_random_spec(rng, alpha_true_cdw=0.0, square=True,
                                    seed=int(child.entropy) % (2 ** 31))
            rows.append(_run_one(spec, "B_auto_square", 3.0, False))
            if (i + 1) % 50 == 0:
                print(f"  arm B {i + 1}/{n_b} ({time.time() - t0:.0f}s)",
                      flush=True)

    if "C" in arms:
        n_c = max(1, n_geometries // 2)
        for child in np.random.SeedSequence(seed + 2).spawn(n_c):
            rng = np.random.Generator(np.random.PCG64(child))
            spec = draw_random_spec(rng, alpha_true_cdw=0.0, square=False,
                                    seed=int(child.entropy) % (2 ** 31))
            rows.append(_run_one(spec, "C_pinned_aspect", 3.0, True))
        print("  arm C done", flush=True)

    if "E" in arms:
        n_e = max(1, n_geometries // 2)
        for w in PEAK_WIDTHS:
            for child in np.random.SeedSequence(
                    seed + 40 + int(w * 10)).spawn(n_e):
                rng = np.random.Generator(np.random.PCG64(child))
                spec = draw_random_spec(rng, alpha_true_cdw=0.0, square=True,
                                        peak_width_bins=w,
                                        seed=int(child.entropy) % (2 ** 31))
                for ap in (2.0, 3.0, 5.0):
                    rows.append(_run_one(spec, "E_width", ap, True))
        print("  arm E done", flush=True)

    if "D" in arms:
        n_d = max(1, n_geometries // 4)
        for a_true in ALPHA_LADDER:
            for child in np.random.SeedSequence(
                    seed + 3 + int(a_true * 1000)).spawn(n_d):
                rng = np.random.Generator(np.random.PCG64(child))
                spec = draw_random_spec(rng, alpha_true_cdw=a_true,
                                        square=True,
                                        seed=int(child.entropy) % (2 ** 31))
                rows.append(_run_one(spec, "D_recovery", 3.0, True))
        print("  arm D done", flush=True)

    return rows


def summarize(rows: List[Dict[str, Any]]) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    L: List[str] = []

    def q(series, p):
        s = series.dropna()
        return float(np.percentile(s, p)) if len(s) else float("nan")

    L.append("=" * 74)
    L.append("NOISELESS SCALLOPING FLOOR")
    L.append("=" * 74)
    L.append(f"{len(df)} runs; status counts: "
             + ", ".join(f"{k}={v}" for k, v in
                         df['status'].value_counts().items()))
    L.append("")

    null = df[(df["alpha_true_cdw"] == 0) & (df["status"] == "ok")]

    L.append("-- Arm A: pure photometry floor (anchor + model pinned) -------")
    L.append(f"{'aperture':>9} {'n':>5} {'median':>10} {'p95':>10} "
             f"{'p99':>10} {'max':>10}")
    a = null[null["arm"] == "A_pinned_square"]
    for ap in sorted(a["aperture_bins"].unique()):
        s = a[a["aperture_bins"] == ap]["alpha_cdw"]
        L.append(f"{ap:>9.1f} {len(s):>5} {q(s, 50):>10.2e} {q(s, 95):>10.2e} "
                 f"{q(s, 99):>10.2e} {s.max():>10.2e}")
    L.append("")

    L.append("-- Floor by estimator (arm A, aperture 3) ---------------------")
    a3 = a[a["aperture_bins"] == 3.0]
    L.append(f"{'estimator':>22} {'median':>10} {'p95':>10} {'p99':>10}")
    for key in ("alpha_cdw", "alpha_bragg", "alpha_corrected",
                "alpha_normalized", "alpha_q2corrected"):
        s = a3[key].abs()
        L.append(f"{key:>22} {q(s, 50):>10.2e} {q(s, 95):>10.2e} "
                 f"{q(s, 99):>10.2e}")
    L.append("")

    for arm, label in (("B_auto_square", "Arm B: automatic identification"),
                       ("C_pinned_aspect", "Arm C: non-square maps")):
        s = null[null["arm"] == arm]
        if not len(s):
            continue
        tot = (df["arm"] == arm).sum()
        L.append(f"-- {label} " + "-" * max(0, 60 - len(label)))
        L.append(f"   ok {len(s)}/{tot}; alpha_cdw median {q(s['alpha_cdw'], 50):.2e}"
                 f"  p95 {q(s['alpha_cdw'], 95):.2e}"
                 f"  p99 {q(s['alpha_cdw'], 99):.2e}")
        if arm == "B_auto_square" and "model_recovered" in s:
            ok = (s["model_recovered"] == s["model"]).mean()
            L.append(f"   superlattice identified correctly: {ok:.1%}")
        L.append("")

    d = df[(df["arm"] == "D_recovery") & (df["status"] == "ok")]
    if len(d):
        L.append("-- Arm D: recovery of imposed anisotropy ----------------------")
        L.append(f"{'alpha_true':>11} {'n':>5} {'median':>10} {'bias':>11} "
                 f"{'p5':>10} {'p95':>10}")
        for at in sorted(d["alpha_true_cdw"].unique()):
            s = d[d["alpha_true_cdw"] == at]["alpha_cdw"]
            L.append(f"{at:>11.2f} {len(s):>5} {q(s, 50):>10.4f} "
                     f"{q(s, 50) - at:>+11.2e} {q(s, 5):>10.4f} "
                     f"{q(s, 95):>10.4f}")
        L.append("")

    e = df[(df["arm"] == "E_width") & (df["status"] == "ok")]
    if len(e):
        L.append("-- Arm E: floor vs intrinsic peak width (alpha_true=0) --------")
        L.append(f"{'width':>7} {'aper':>6} {'n':>5} {'median':>10} "
                 f"{'p95':>10} {'p99':>10}")
        for w in sorted(e["peak_width_bins"].unique()):
            for apx in sorted(e["aperture_bins"].unique()):
                s_ = e[(e["peak_width_bins"] == w)
                       & (e["aperture_bins"] == apx)]["alpha_cdw"]
                if len(s_):
                    L.append(f"{w:>7.1f} {apx:>6.1f} {len(s_):>5} "
                             f"{q(s_, 50):>10.2e} {q(s_, 95):>10.2e} "
                             f"{q(s_, 99):>10.2e}")
        L.append("")

    L.append("-- Drivers of the floor (arm A, Spearman vs alpha_cdw) --------")
    for cov in ("subbin_cdw_spread", "subbin_cdw_mean", "q_cdw_bins",
                "aperture_bins", "cdw_power_ratio"):
        sub = a[[cov, "alpha_cdw"]].dropna()
        if len(sub) > 10:
            rho = sub[cov].corr(sub["alpha_cdw"], method="spearman")
            L.append(f"   {cov:>22}  rho = {rho:+.3f}")
    L.append("=" * 74)
    return "\n".join(L)


def make_figure(rows: List[Dict[str, Any]], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows)
    null = df[(df["alpha_true_cdw"] == 0) & (df["status"] == "ok")]
    a = null[null["arm"] == "A_pinned_square"]
    a3 = a[a["aperture_bins"] == 3.0]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    vals = a3["alpha_cdw"].dropna()
    vals = vals[vals > 0]
    if len(vals):
        ax[0].hist(np.log10(vals), bins=40, color="#6F5AA7",
                   edgecolor="white")
        for p, c in ((95, "#b81a1a"), (99, "#000000")):
            ax[0].axvline(np.log10(np.percentile(vals, p)), color=c, ls="--",
                          lw=1.3, label=f"p{p}")
        ax[0].legend(fontsize=8)
    ax[0].set_xlabel(r"$\log_{10}\ \hat{\alpha}_{CDW}$")
    ax[0].set_ylabel("count")
    ax[0].set_title(r"Null floor, $\alpha_{true}=0$ (aperture 3 bins)")

    for ap in sorted(a["aperture_bins"].unique()):
        s = a[a["aperture_bins"] == ap]["alpha_cdw"].dropna()
        s = s[s > 0]
        if len(s):
            ax[1].scatter([ap] * len(s), s, s=5, alpha=0.25, color="#6F5AA7")
            ax[1].scatter([ap], [np.percentile(s, 95)], marker="_", s=600,
                          color="#b81a1a", zorder=5)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("aperture radius (bins)")
    ax[1].set_ylabel(r"$\hat{\alpha}_{CDW}$")
    ax[1].set_title("Floor vs aperture (red = p95)")

    sub = a3[["subbin_cdw_spread", "alpha_cdw"]].dropna()
    sub = sub[sub["alpha_cdw"] > 0]
    if len(sub):
        ax[2].scatter(sub["subbin_cdw_spread"], sub["alpha_cdw"], s=6,
                      alpha=0.35, color="#6F5AA7")
        ax[2].set_yscale("log")
    ax[2].set_xlabel("spread of sub-bin offsets across the three Q")
    ax[2].set_ylabel(r"$\hat{\alpha}_{CDW}$")
    ax[2].set_title("Scalloping mechanism")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-geometries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--arms", default="ABCDE")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rows = run_study(args.n_geometries, args.seed, args.arms)
    print(f"{len(rows)} runs in {time.time() - t0:.0f}s", flush=True)

    import pandas as pd
    csv = os.path.join(args.out, "scalloping_runs.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)

    report = summarize(rows)
    print(report)
    with open(os.path.join(args.out, "scalloping_report.txt"), "w") as fh:
        fh.write(report + "\n")

    fig = os.path.join(args.out, "scalloping_floor.png")
    make_figure(rows, fig)

    with open(os.path.join(args.out, "scalloping_meta.json"), "w") as fh:
        json.dump({"n_geometries": args.n_geometries, "seed": args.seed,
                   "arms": args.arms, "apertures": list(APERTURES),
                   "alpha_ladder": list(ALPHA_LADDER),
                   "peak_widths": list(PEAK_WIDTHS),
                   "n_runs": len(rows)}, fh, indent=2)
    print(f"\nwrote {csv}, {fig}")


if __name__ == "__main__":
    main()
