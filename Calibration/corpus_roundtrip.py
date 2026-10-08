from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

import didv_anisotropy as da
from forward_model import draw_random_spec, synthesize

COLORMAPS = ("viridis", "inferno", "Blues", "jet")

ENCODINGS = (("png", None), ("jpeg", 95), ("jpeg", 75), ("jpeg", 50))

SCALES = (1.0, 0.5, 2.0)

PEAK_WIDTHS = (2.0, 3.0, 4.0)


def render(field: np.ndarray, cmap_name: str) -> "Image.Image":
    from PIL import Image
    import matplotlib
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except (AttributeError, KeyError):
        from matplotlib import cm
        cmap = cm.get_cmap(cmap_name)

    lo, hi = float(np.nanmin(field)), float(np.nanmax(field))
    norm = (field - lo) / max(hi - lo, 1e-12)
    rgb = (cmap(norm)[:, :, :3] * 255.0).round().astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def round_trip(field: np.ndarray, cmap_name: str, fmt: str,
               quality: Optional[int], scale: float,
               tmpdir: str) -> str:
    from PIL import Image
    img = render(field, cmap_name)

    if scale != 1.0:
        w, h = img.size
        small = img.resize((max(8, int(w * scale)), max(8, int(h * scale))),
                           Image.BICUBIC)
        img = small.resize((w, h), Image.BICUBIC) if scale < 1.0 else small

    ext = ".png" if fmt == "png" else ".jpg"
    path = os.path.join(tmpdir, f"rt_{cmap_name}_{fmt}_{quality}_{scale}{ext}")
    if fmt == "png":
        img.save(path)
    else:
        img.save(path, quality=int(quality), subsampling=0)
    return path


def run(n_geometries: int, seed: int, alpha_true: float,
        widths=PEAK_WIDTHS) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = np.random.SeedSequence(seed)

    for gi, child in enumerate(root.spawn(n_geometries)):
        rng = np.random.Generator(np.random.PCG64(child))
        width = float(rng.choice(widths))
        spec = draw_random_spec(rng, alpha_true_cdw=alpha_true, square=True,
                                peak_width_bins=width,
                                seed=int(child.entropy) % (2 ** 31))
        field, _ = synthesize(spec)

        pinned = dict(cdw_model=spec.model, bragg_radius=spec.q_bragg,
                      bragg_angles_deg=spec.bragg_angles_deg)

        ref = da.analyze_didv_field(field, **pinned)
        if ref.get("status") != "ok":
            continue
        ref_a, ref_b = ref["alpha_cdw"], ref["alpha_bragg"]

        with tempfile.TemporaryDirectory() as tmp:
            for cmap_name in COLORMAPS:
                for fmt, qual in ENCODINGS:
                    for scale in SCALES:
                        path = round_trip(field, cmap_name, fmt, qual,
                                          scale, tmp)
                        q_scale = 1.0 / scale if scale > 1.0 else 1.0
                        pinned_rt = dict(
                            cdw_model=spec.model,
                            bragg_radius=spec.q_bragg * q_scale,
                            bragg_angles_deg=spec.bragg_angles_deg)
                        res = da.analyze_didv_anisotropy(path, **pinned_rt)
                        row = {
                            "geometry": gi,
                            "alpha_true": alpha_true,
                            "peak_width_bins": width,
                            "model": spec.model,
                            "colormap": cmap_name,
                            "format": fmt,
                            "quality": qual,
                            "scale": scale,
                            "status": res.get("status"),
                            "alpha_ref": ref_a,
                            "alpha_bragg_ref": ref_b,
                        }
                        if res.get("status") == "ok":
                            row.update({
                                "alpha_cdw": res["alpha_cdw"],
                                "alpha_bragg": res["alpha_bragg"],
                                "delta_cdw": res["alpha_cdw"] - ref_a,
                                "delta_bragg": res["alpha_bragg"] - ref_b,
                                "cmap_method":
                                    res.get("scalar_field", {}).get("method"),
                                "cmap_found":
                                    res.get("scalar_field", {}).get("colormap"),
                            })
                            row["cmap_correct"] = (
                                row["cmap_found"] == cmap_name)
                        rows.append(row)
        print(f"  geometry {gi + 1}/{n_geometries}", flush=True)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    L = ["=" * 78,
         "CORPUS ROUND-TRIP: alpha manufactured by figure encoding",
         "=" * 78,
         f"{len(df)} round trips, {len(ok)} measured "
         f"({len(df) - len(ok)} failed to analyse)", ""]

    if not len(ok):
        return "\n".join(L + ["no successful round trips"])

    L.append("-- Colormap recovery ---------------------------------------")
    for cm, g in ok.groupby("colormap"):
        L.append(f"   {cm:>10}: identified correctly {g.cmap_correct.mean():.0%}"
                 f"   |delta alpha_CDW| median "
                 f"{g.delta_cdw.abs().median():.3f}")
    L.append("")

    L.append("-- Encoding ------------------------------------------------")
    for (fmt, q), g in ok.groupby(["format", "quality"], dropna=False):
        L.append(f"   {fmt:>5} q={str(q):>4}: |delta alpha_CDW| median "
                 f"{g.delta_cdw.abs().median():.3f}  p95 "
                 f"{g.delta_cdw.abs().quantile(.95):.3f}")
    L.append("")

    L.append("-- Rescaling -----------------------------------------------")
    for sc, g in ok.groupby("scale"):
        L.append(f"   scale {sc:>4}: |delta alpha_CDW| median "
                 f"{g.delta_cdw.abs().median():.3f}  p95 "
                 f"{g.delta_cdw.abs().quantile(.95):.3f}")
    L.append("")

    L.append("-- Worst realistic corpus case (jet + JPEG75) --------------")
    bad = ok[(ok.colormap == "jet") & (ok.format == "jpeg")
             & (ok.quality == 75)]
    if len(bad):
        L.append(f"   n={len(bad)}  |delta alpha_CDW| median "
                 f"{bad.delta_cdw.abs().median():.3f}  "
                 f"p95 {bad.delta_cdw.abs().quantile(.95):.3f}")
    L.append("")
    L.append("-- Overall -------------------------------------------------")
    L.append(f"   |delta alpha_CDW|  median {ok.delta_cdw.abs().median():.3f}"
             f"   p95 {ok.delta_cdw.abs().quantile(.95):.3f}")
    L.append(f"   |delta alpha_Bragg| median "
             f"{ok.delta_bragg.abs().median():.3f}"
             f"   p95 {ok.delta_bragg.abs().quantile(.95):.3f}")
    L.append("")
    L.append("Compare against the +0.27 median excess of measured "
             "alpha_Bragg over")
    L.append("the finite-domain speckle floor on the experimental maps.  If "
             "the round")
    L.append("trip alone reaches that scale, the excess is largely a "
             "publication")
    L.append("artifact rather than an instrumental one.")
    L.append("=" * 78)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60,
                    help="number of geometries (each -> 48 round trips)")
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--alpha-true", type=float, default=0.0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = run(args.n, args.seed, args.alpha_true)

    import pandas as pd
    csv = os.path.join(args.out, "corpus_roundtrip.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)

    report = summarize(rows)
    print(report)
    with open(os.path.join(args.out, "corpus_roundtrip_report.txt"),
              "w") as fh:
        fh.write(report + "\n")
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
