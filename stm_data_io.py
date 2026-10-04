from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

NATIVE_EXTENSIONS = {".sxm", ".3ds"}

SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | NATIVE_EXTENSIONS

DATA_CLASS_NATIVE = "native"
DATA_CLASS_RENDERED = "rendered"

_DIDV_HINTS = ("lix", "li x", "li_x", "li demod", "lock-in", "lockin", "didv",
               "di/dv", "dif", "conductance", "demod", "lia")
_TOPO_HINTS = ("z (m)", "z(m)", "topo", "height", "zsensor", "z_sensor", "z")


def normalise_extension(ext: Optional[str], default: str = ".png") -> str:
    ext = (ext or default).lower().strip()
    if ext and not ext.startswith("."):
        ext = "." + ext
    if ext == ".tif":
        ext = ".tiff"
    if ext not in SUPPORTED_EXTENSIONS:
        return default
    return ext


def is_native_extension(ext_or_path: str) -> bool:
    s = str(ext_or_path).lower()
    ext = s if s.startswith(".") and "/" not in s else os.path.splitext(s)[1]
    if ext == ".tif":
        ext = ".tiff"
    return ext in NATIVE_EXTENSIONS


def _blank_info(path: str, fmt: str) -> Dict[str, Any]:
    return {
        "data_class": DATA_CLASS_NATIVE,
        "quantitative": True,
        "source_format": fmt,
        "source_path": os.path.basename(str(path)),
        "channel": None,
        "direction": None,
        "unit": None,
        "lx": None, "ly": None, "length_unit": None,
        "bias_v": None, "current_a": None,
        "notes": [],
        "orientation_note": "row 0 at top (display orientation)",
    }


def _finish(field: np.ndarray, info: Dict[str, Any]) -> Tuple[np.ndarray,
                                                              Dict[str, Any]]:
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D field, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("field is empty")
    n_bad = int(np.count_nonzero(~np.isfinite(arr)))
    if n_bad:
        info["n_nonfinite"] = n_bad
        info["notes"].append(
            f"{n_bad} non-finite samples ({100.0 * n_bad / arr.size:.2f}% of "
            "the frame) - unfinished scan lines or dropped points. They are "
            "filled from the nearest finite neighbour by the caller's "
            "sanitiser, not here.")
    bits = [f"native {info['source_format']}"]
    if info.get("channel"):
        bits.append(f"channel {info['channel']!r}")
    if info.get("direction"):
        bits.append(str(info["direction"]))
    if info.get("unit"):
        bits.append(f"values in {info['unit']}")
    if info.get("lx") and info.get("ly"):
        bits.append(f"scan {info['lx']:.4g} x {info['ly']:.4g} "
                    f"{info.get('length_unit') or ''}".strip())
    info["method"] = "native_read"
    info["note"] = ", ".join(bits)
    info.setdefault("colormap", None)
    info.setdefault("colormap_residual", None)
    return arr, info


def _pick_channel(names: List[str], want: Optional[str],
                  prefer: str = "auto") -> int:
    if not names:
        raise ValueError("file declares no channels")
    low = [n.lower() for n in names]
    if want:
        w = str(want).lower().strip()
        for i, n in enumerate(low):
            if w == n:
                return i
        for i, n in enumerate(low):
            if w in n:
                return i
        raise ValueError(
            f"channel {want!r} not found; available: {names}")
    hints = (_DIDV_HINTS if prefer == "didv"
             else _TOPO_HINTS if prefer == "topo" else ())
    for h in hints:
        for i, n in enumerate(low):
            if h in n:
                return i
    return 0


def _si_scale(unit: str) -> Tuple[float, str]:
    if unit == "m":
        return 1e9, "nm"
    return 1.0, unit


_SXM_END = b":SCANIT_END:"


def _sxm_header(raw: bytes) -> Tuple[Dict[str, List[str]], int]:
    end = raw.find(_SXM_END)
    if end < 0:
        raise ValueError("not a Nanonis .sxm file (no :SCANIT_END: marker)")
    text = raw[:end].decode("latin-1", errors="replace")

    entries: Dict[str, List[str]] = {}
    key = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(":") and s.endswith(":") and len(s) > 2:
            key = s[1:-1]
            entries[key] = []
        elif key is not None:
            entries[key].append(line.rstrip("\r\n"))

    marker = raw.find(b"\x1a\x04", end)
    if marker < 0:
        raise ValueError("Nanonis .sxm header is not terminated by \\x1a\\x04")
    return entries, marker + 2


def _sxm_data_info(lines: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    header: Optional[List[str]] = None
    for line in lines:
        cells = [c.strip() for c in line.split("\t") if c.strip() != ""]
        if not cells:
            continue
        if header is None:
            header = [c.lower() for c in cells]
            continue
        row = {header[i]: cells[i] for i in range(min(len(header),
                                                     len(cells)))}
        if row.get("name"):
            rows.append(row)
    return rows


def _read_sxm(path: str, channel: Optional[str], direction: str,
              prefer: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with open(path, "rb") as fh:
        raw = fh.read()

    entries, offset = _sxm_header(raw)
    info = _blank_info(path, "Nanonis .sxm")

    def first(key: str) -> str:
        vals = entries.get(key) or []
        return vals[0].strip() if vals else ""

    px = first("SCAN_PIXELS").split()
    if len(px) < 2:
        raise ValueError(".sxm header carries no SCAN_PIXELS")
    nx, ny = int(float(px[0])), int(float(px[1]))

    rng = first("SCAN_RANGE").split()
    if len(rng) >= 2:
        scale, unit = _si_scale("m")
        info["lx"] = float(rng[0]) * scale
        info["ly"] = float(rng[1]) * scale
        info["length_unit"] = unit

    scan_dir = (first("SCAN_DIR") or "down").lower()
    ang = first("SCAN_ANGLE")
    try:
        info["scan_angle_deg"] = float(ang) if ang else 0.0
    except ValueError:
        info["scan_angle_deg"] = 0.0
    try:
        info["bias_v"] = float(first("BIAS"))
    except ValueError:
        pass

    rows = _sxm_data_info(entries.get("DATA_INFO") or [])
    if not rows:
        raise ValueError(".sxm header carries no DATA_INFO channel table")

    names = [r.get("name", f"ch{i}") for i, r in enumerate(rows)]
    info["available_channels"] = names
    idx = _pick_channel(names, channel, prefer=prefer)
    row = rows[idx]

    frame = nx * ny
    planes_before = 0
    for r in rows[:idx]:
        planes_before += 2 if r.get("direction", "").lower() == "both" else 1
    both = row.get("direction", "").lower() == "both"
    want_bwd = str(direction).lower().startswith("b") and both
    plane = planes_before + (1 if want_bwd else 0)

    need = (plane + 1) * frame * 4
    if len(raw) - offset < need:
        raise ValueError(
            f".sxm binary block is short: need {need} bytes for plane "
            f"{plane}, have {len(raw) - offset}")

    flat = np.frombuffer(raw, dtype=">f4", count=frame,
                         offset=offset + plane * frame * 4)
    field = flat.reshape(ny, nx).astype(np.float64)

    if want_bwd:
        field = field[:, ::-1]

    if scan_dir.startswith("up"):
        field = field[::-1, :]
        info["orientation_note"] = (
            "scan direction 'up': frame flipped vertically so row 0 is the "
            "top, matching the display frame the angle convention assumes")

    try:
        cal = float(row.get("calibration", "1"))
        off = float(row.get("offset", "0"))
        if cal != 1.0 or off != 0.0:
            field = field * cal + off
            info["notes"].append(
                f"applied header calibration {cal:g} and offset {off:g}")
    except (TypeError, ValueError):
        pass

    info["channel"] = names[idx]
    info["direction"] = "backward" if want_bwd else "forward"
    info["unit"] = row.get("unit")
    info["header"] = {k: (v[0].strip() if len(v) == 1 else v)
                      for k, v in entries.items()
                      if k in ("SCAN_TIME", "SCAN_OFFSET", "REC_DATE",
                               "REC_TIME", "SCANIT_TYPE", "COMMENT",
                               "Z-CONTROLLER")}
    if both and not want_bwd:
        info["notes"].append(
            "This channel also has a backward frame. Comparing the two is "
            "the cheapest available drift and tip-stability check; pass "
            "direction='backward' to load it.")
    return _finish(field, info)


GRID_ORIENTATION = "physical"

_3DS_TOPO_PARAMS = ("Z (m)", "Scan:Z (m)")


def _3ds_param(block: np.ndarray, params: List[str], name: str
               ) -> Optional[np.ndarray]:
    low = [p.strip().lower() for p in params]
    try:
        j = low.index(name.lower())
    except ValueError:
        return None
    if j >= block.shape[1]:
        return None
    return block[:, j].astype(np.float64)


def _read_3ds(path: str, channel: Optional[str], bias: Optional[float],
              prefer: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with open(path, "rb") as fh:
        raw = fh.read()

    end = raw.find(b":HEADER_END:")
    if end < 0:
        raise ValueError("not a Nanonis .3ds file (no :HEADER_END: marker)")
    text = raw[:end].decode("latin-1", errors="replace")
    offset = raw.find(b"\n", end) + 1

    hdr: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            hdr[k.strip()] = v.strip().strip('"')

    info = _blank_info(path, "Nanonis .3ds")

    dim = re.findall(r"\d+", hdr.get("Grid dim", ""))
    if len(dim) < 2:
        raise ValueError(".3ds header carries no 'Grid dim'")
    nx, ny = int(dim[0]), int(dim[1])

    gs = [float(t) for t in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?",
                                       hdr.get("Grid settings", ""))]
    if len(gs) >= 4:
        scale, unit = _si_scale("m")
        info["lx"] = gs[2] * scale
        info["ly"] = gs[3] * scale
        info["length_unit"] = unit
    grid_angle = gs[4] if len(gs) >= 5 else 0.0
    info["scan_angle_deg"] = float(grid_angle)

    n_pts = int(float(hdr.get("Points", "0") or 0))
    channels = [c.strip() for c in (hdr.get("Channels", "") or "").split(";")
                if c.strip()]
    if not channels or n_pts <= 0:
        raise ValueError(".3ds header carries no Channels/Points")

    n_par = int(float(hdr.get("# Parameters (4 byte)", "0") or 0))
    fixed = [p for p in (hdr.get("Fixed parameters", "") or "").split(";")
             if p.strip()]
    exp = [p for p in (hdr.get("Experiment parameters", "") or "").split(";")
           if p.strip()]
    if n_par <= 0:
        n_par = len(fixed) + len(exp)
    params = [p.strip() for p in fixed + exp]

    per_point = n_par + len(channels) * n_pts
    total = nx * ny * per_point
    avail = (len(raw) - offset) // 4
    if avail < total:
        complete = avail // per_point
        if complete < nx:
            raise ValueError(
                f".3ds data block holds only {complete} complete grid points")
        info["notes"].append(
            f"Grid is incomplete: {complete} of {nx * ny} points were "
            "written. The unmeasured tail is filled with NaN.")
        total = complete * per_point
    else:
        complete = nx * ny

    block = np.frombuffer(raw, dtype=">f4", count=total,
                          offset=offset).reshape(complete, per_point)

    topo_params = []
    for name in _3DS_TOPO_PARAMS:
        col = _3ds_param(block, params, name)
        if col is not None and np.isfinite(col).any():
            topo_params.append(name)
    selectable = list(channels) + topo_params
    info["available_channels"] = selectable
    info["topography_channels"] = list(topo_params)
    info["spectroscopic_channels"] = list(channels)

    idx = _pick_channel(selectable, channel,
                        prefer=("didv" if prefer == "auto" else prefer))
    chosen = selectable[idx]
    is_topo = idx >= len(channels)

    sweep = None
    if n_par >= 2:
        starts = block[:, 0].astype(np.float64)
        ends = block[:, 1].astype(np.float64)
        sweep = np.linspace(float(np.nanmedian(starts)),
                            float(np.nanmedian(ends)), n_pts)

    sp_bias = _3ds_param(block, params, "Scan:Bias (V)")
    sp_bias = (float(np.nanmedian(sp_bias))
               if sp_bias is not None and np.isfinite(sp_bias).any() else None)
    sp_cur = _3ds_param(block, params, "Scan:Current (A)")
    sp_cur = (float(np.nanmedian(sp_cur))
              if sp_cur is not None and np.isfinite(sp_cur).any() else None)
    info["setpoint_bias_v"] = sp_bias
    info["setpoint_current_a"] = sp_cur
    info["current_a"] = sp_cur

    if sweep is not None:
        info["sweep_points"] = int(n_pts)
        info["sweep_range_v"] = [float(sweep[0]), float(sweep[-1])]
        info["sweep_values_v"] = [float(v) for v in sweep]

    flip_v = flip_h = False
    if GRID_ORIENTATION == "physical":
        x_col = _3ds_param(block, params, "X (m)")
        y_col = _3ds_param(block, params, "Y (m)")
        if len(gs) >= 2 and x_col is not None and y_col is not None:
            cx0, cy0 = gs[0], gs[1]
            th = np.radians(grid_angle)
            dx0, dy0 = x_col[0] - cx0, y_col[0] - cy0
            u0 = dx0 * np.cos(th) + dy0 * np.sin(th)
            v0 = -dx0 * np.sin(th) + dy0 * np.cos(th)
            flip_v = bool(v0 < 0)
            flip_h = bool(u0 > 0)
        else:
            flip_v = True
    info["orientation_note"] = (
        ("grid acquired bottom-up: flipped vertically so row 0 is the top "
         "(largest Y), X right / Y up as in the sample"
         if flip_v else "grid acquired top-down: row 0 is already the top")
        + ("; columns acquired right-to-left: flipped horizontally"
           if flip_h else "")
        + (f"; frame is the grid frame, rotated {grid_angle:g} deg"
           if grid_angle else "")
        if GRID_ORIENTATION == "physical" else
        "orientation 'stored': row 0 is the first acquired grid line; for a "
        "bottom-up grid this is the VERTICAL MIRROR of the sample")
    info["grid_orientation"] = GRID_ORIENTATION

    if is_topo:
        flat = _3ds_param(block, params, chosen)
        info["bias_v"] = sp_bias
        info["is_spectroscopic_grid"] = False
        info["direction"] = "grid"
        info["unit"] = "m"
        info["notes"].append(
            f"Topograph of the grid: the per-point tip height {chosen!r} "
            "recorded under constant-current feedback"
            + (f" at the setpoint {sp_bias * 1e3:+.4g} mV" if sp_bias is not None
               else "")
            + (f", {sp_cur * 1e12:.4g} pA" if sp_cur is not None else "")
            + ".")
    else:
        if bias is not None and sweep is not None:
            k = int(np.argmin(np.abs(sweep - float(bias))))
        elif sweep is not None and sp_bias is not None and (
                min(sweep[0], sweep[-1]) - 1e-9 <= sp_bias
                <= max(sweep[0], sweep[-1]) + 1e-9):
            k = int(np.argmin(np.abs(sweep - sp_bias)))
        else:
            k = n_pts // 2
        if sweep is not None:
            info["bias_v"] = float(sweep[k])
            info["bias_index"] = int(k)
            if bias is not None:
                info["notes"].append(
                    f"Requested bias {float(bias):+.4g} V; nearest swept point "
                    f"is {sweep[k]:+.4g} V (index {k} of {n_pts}).")
            elif sp_bias is not None and abs(sweep[k] - sp_bias) < 1e-6:
                info["notes"].append(
                    f"No bias requested; took the slice at the setpoint bias "
                    f"{sweep[k]:+.4g} V, the energy the grid's topograph was "
                    "recorded at. Pass bias=... to choose another slice.")
            else:
                info["notes"].append(
                    f"No bias requested; took the mid-sweep slice at "
                    f"{sweep[k]:+.4g} V. Pass bias=... to choose the slice.")
        col = n_par + idx * n_pts + k
        flat = block[:, col].astype(np.float64)
        info["direction"] = "grid"
        m = re.search(r"\(([^)]*)\)\s*$", chosen)
        info["unit"] = m.group(1) if m else None
        info["is_spectroscopic_grid"] = True
        info["notes"].append(
            "Constant-bias slice through a grid of spectra, not a rendered "
            "figure. Every value is the recorded signal.")

    field = np.full(nx * ny, np.nan)
    field[:complete] = flat
    field = field.reshape(ny, nx)
    if flip_v:
        field = field[::-1, :]
    if flip_h:
        field = field[:, ::-1]

    info["channel"] = chosen
    return _finish(field, info)


_NATIVE_DISPATCH = {
    ".sxm": lambda p, ch, di, pr, bi: _read_sxm(p, ch, di, pr),
    ".3ds": lambda p, ch, di, pr, bi: _read_3ds(p, ch, bi, pr),
}


def load_native_field(path: str, channel: Optional[str] = None,
                      direction: str = "forward", prefer: str = "auto",
                      bias: Optional[float] = None
                      ) -> Tuple[np.ndarray, Dict[str, Any]]:
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".tif":
        ext = ".tiff"
    if ext in _NATIVE_DISPATCH:
        return _NATIVE_DISPATCH[ext](path, channel, direction, prefer, bias)
    raise ValueError(
        f"{ext!r} is not a native data format. Supported: "
        f"{sorted(NATIVE_EXTENSIONS)}")


def normalise_native_scale(field: np.ndarray,
                           info: Optional[Dict[str, Any]] = None
                           ) -> Tuple[np.ndarray, Dict[str, Any]]:
    info = dict(info or {})
    arr = np.asarray(field, dtype=np.float64)
    finite = np.isfinite(arr)
    scale = None
    if finite.sum() >= 16:
        ny, nx = arr.shape
        y, x = np.mgrid[0:ny, 0:nx]
        yn = (y - ny / 2.0) / max(ny / 2.0, 1.0)
        xn = (x - nx / 2.0) / max(nx / 2.0, 1.0)
        A = np.column_stack([np.ones(int(finite.sum())), xn[finite],
                             yn[finite], xn[finite] ** 2,
                             (xn * yn)[finite], yn[finite] ** 2])
        v = arr[finite]
        try:
            coef, *_ = np.linalg.lstsq(A, v, rcond=None)
            resid = v - A @ coef
        except np.linalg.LinAlgError:
            resid = v - np.median(v)
        lo, hi = np.percentile(resid, [0.5, 99.5])
        spread = float(hi - lo)
        if np.isfinite(spread) and spread > 0:
            scale = spread
    if scale is None:
        finite_vals = arr[finite]
        m = float(np.max(np.abs(finite_vals))) if finite_vals.size else 0.0
        scale = m if (np.isfinite(m) and m > 0) else 1.0

    info["value_scale"] = scale
    info["physical_unit"] = info.get("unit")
    if info.get("note"):
        info["note"] += f", scaled by 1/{scale:.4g}"
    if info.get("unit"):
        info["unit"] = f"{info['unit']} / {scale:.4g}"
    info.setdefault("notes", []).append(
        f"Values divided by {scale:.4g} {info.get('physical_unit') or ''} "
        "(one positive constant: ratios, phases and angles are unchanged) so "
        "the field is on the same numerical scale as a rendered image. "
        "Multiply by info['value_scale'] for physical units.".replace("  ", " "))
    return arr / scale, info


def physical_size(info: Optional[Dict[str, Any]]
                  ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    info = info or {}
    lx, ly = info.get("lx"), info.get("ly")
    if lx and ly and np.isfinite(lx) and np.isfinite(ly) and lx > 0 and ly > 0:
        return float(lx), float(ly), info.get("length_unit")
    return None, None, None


def load_field(path: str, colormap: Optional[str] = None,
               colormap_resid_tol: float = 20.0,
               channel: Optional[str] = None,
               direction: str = "forward",
               prefer: str = "auto",
               bias: Optional[float] = None,
               image_loader=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".tif":
        ext = ".tiff"

    if ext in NATIVE_EXTENSIONS:
        field, info = load_native_field(path, channel=channel,
                                        direction=direction, prefer=prefer,
                                        bias=bias)
        field, info = normalise_native_scale(field, info)
        info["data_class"] = DATA_CLASS_NATIVE
        info["quantitative"] = True
        info["requested_colormap"] = None
        return field, info

    if ext not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension {ext!r}. Supported: "
            f"{sorted(SUPPORTED_EXTENSIONS)}")

    if image_loader is None:
        raise ValueError(
            "a rendered image was supplied but no image_loader was given")
    field, info = image_loader(path, colormap, colormap_resid_tol)
    info = dict(info or {})
    info["data_class"] = DATA_CLASS_RENDERED
    info["quantitative"] = False
    return np.asarray(field, dtype=np.float64), info


def render_preview_png(field: np.ndarray, info: Optional[Dict[str, Any]] = None,
                       cmap: str = "magma", max_px: int = 1024) -> bytes:
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(field, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        lo, hi = np.percentile(finite, [1.0, 99.0])
        if not (hi > lo):
            lo, hi = float(finite.min()), float(finite.max()) + 1e-12
    else:
        lo, hi = 0.0, 1.0

    h, w = arr.shape
    s = min(1.0, max_px / float(max(h, w)))
    fig = plt.figure(figsize=(w * s / 100.0, h * s / 100.0), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(arr, cmap=cmap, origin="upper", vmin=lo, vmax=hi,
              interpolation="nearest")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, pad_inches=0)
    plt.close(fig)
    return buf.getvalue()
