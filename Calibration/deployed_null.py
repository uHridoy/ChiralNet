import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from didv_anisotropy import analyze_didv_field
from didv_alpha_crit import screen_result, ACTIVE_CALIBRATION

TAU = 0.95


def run_one(seed, alpha_true, N=384):
    from null_corpus import make_null_map
    rng = np.random.default_rng(seed)
    f = make_null_map(seed, N=N, alpha_true=alpha_true, rng=rng)
    res = analyze_didv_field(f, aperture_bins="auto", cdw_model=None)
    if res.get("status") != "ok":
        return {"ok": False, "reason": "no_measurement"}
    sc = screen_result(res)
    return {
        "ok": True,
        "verdict": sc.get("verdict"),
        "in_domain": sc.get("verdict") in ("above", "below"),
        "alpha_cdw": res.get("alpha_cdw"),
        "alpha_bragg": res.get("alpha_bragg"),
        "delta": (None if res.get("alpha_cdw") is None
                  or res.get("alpha_bragg") is None
                  else res["alpha_cdw"] - res["alpha_bragg"]),
        "model": (res.get("cdw_model") or {}).get("name"),
        "aperture_bins": (res.get("method") or {}).get("aperture_radius_bins"),
        "width_corr": res.get("width_bins_cdw_corrected"),
        "min_snr": res.get("min_snr_cdw"),
    }


def sweep(n, alpha_true, seed0, label, N=384, budget_s=None):
    rows, t0 = [], time.time()
    for i in range(n):
        if budget_s and time.time() - t0 > budget_s:
            print(f"   [{label}] budget reached after {i} runs", flush=True)
            break
        try:
            rows.append(run_one(seed0 + i, alpha_true, N=N))
        except Exception as exc:
            rows.append({"ok": False, "reason": f"error: {exc}"})
        if (i + 1) % 25 == 0:
            done = [r for r in rows if r.get("in_domain")]
            print(f"   [{label}] {i + 1} runs, {len(done)} in domain, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return rows


def summarise(rows):
    n = len(rows)
    meas = [r for r in rows if r.get("ok")]
    dom = [r for r in meas if r.get("in_domain")
           and r.get("alpha_cdw") is not None]
    a = np.array([r["alpha_cdw"] for r in dom], dtype=float)
    d = np.array([r["delta"] for r in dom if r.get("delta") is not None],
                 dtype=float)
    return {
        "n_runs": n,
        "n_measured": len(meas),
        "n_in_domain": len(dom),
        "measurement_failure_rate": round(1 - len(meas) / max(n, 1), 4),
        "domain_pass_rate": round(len(dom) / max(n, 1), 4),
        "alpha": a, "delta": d,
    }


if __name__ == "__main__":
    n_null = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_pow = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
    out_path = sys.argv[4] if len(sys.argv) > 4 else "null_result.json"

    print(f"active rule: {ACTIVE_CALIBRATION['name']}  "
          f"alpha_crit = {ACTIVE_CALIBRATION['alpha_crit']}")
    print(f"NULL arm: {n_null} achiral maps through the deployed path")
    null_rows = sweep(n_null, 0.0, 10_000, "null", budget_s=budget * 0.62)
    print(f"POWER arm: {n_pow} maps at alpha_true = 0.5")
    pow_rows = sweep(n_pow, 0.5, 90_000, "power", budget_s=budget * 0.38)

    json.dump({"null": null_rows, "power": pow_rows}, open(out_path, "w"))
    print(f"\nwrote {out_path}")
