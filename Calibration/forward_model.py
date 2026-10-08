from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CDW_GEOMETRY",
    "SyntheticSpec",
    "intensities_from_alpha",
    "synthesize",
    "draw_random_spec",
]

CDW_GEOMETRY: Dict[str, Tuple[float, float]] = {
    "2x2": (1.0 / 2.0, 0.0),
    "3x3": (1.0 / 3.0, 0.0),
    "4x4": (1.0 / 4.0, 0.0),
    "sqrt3xsqrt3": (1.0 / np.sqrt(3.0), 30.0),
    "sqrt13xsqrt13": (1.0 / np.sqrt(13.0), 13.898),
}

OBSERVED_Q_BRAGG = (0.089, 0.155)
OBSERVED_SIZES = (226, 300, 394, 460, 512, 616, 861, 1024)


def intensities_from_alpha(alpha: float, middle: float = 0.5,
                           sense: int = +1,
                           strongest_index: int = 0) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0 - 1e-12))
    i_max = 1.0
    i_min = (1.0 - alpha) / (1.0 + alpha)
    i_mid = i_min + float(np.clip(middle, 0.0, 1.0)) * (i_max - i_min)

    ordered = np.array([i_max, i_mid, i_min], dtype=np.float64)
    if sense < 0:
        ordered = ordered[::-1].copy()
        ordered = np.roll(ordered, 1)
    return np.roll(ordered, int(strongest_index) % 3)


@dataclass
class SyntheticSpec:
    Ny: int
    Nx: int
    q_bragg: float
    theta0_deg: float
    model: str = "2x2"
    alpha_true_cdw: float = 0.0
    alpha_true_bragg: float = 0.0
    cdw_power_ratio: float = 1.0
    middle_cdw: float = 0.5
    middle_bragg: float = 0.5
    sense_cdw: int = +1
    strongest_index_cdw: int = 0
    phases_bragg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    phases_cdw: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    peak_width_bins: float = 0.0
    tip_alpha_bragg: float = 0.0
    tip_angle_deg: float = 0.0
    tip_blur: float = 0.0
    drift_pixels: float = 0.0
    drift_angle_deg: float = 0.0
    creep_pixels: float = 0.0
    creep_tau: float = 0.15
    creep_angle_deg: float = 0.0
    aspect_error: float = 0.0
    noise_white: float = 0.0
    noise_pink: float = 0.0
    noise_line: float = 0.0
    noise_beta: float = 1.0
    defect_density: float = 0.0
    defect_amplitude: float = 0.0
    defect_radius_px: float = 3.0
    n_steps: int = 0
    step_height: float = 0.0
    step_angle_deg: float = 0.0
    step_registry_px: float = 0.0
    seed: Optional[int] = None
    meta: Dict[str, Any] = _dc_field(default_factory=dict)

    @property
    def bragg_angles_deg(self) -> np.ndarray:
        return (self.theta0_deg + np.array([0.0, 60.0, 120.0])) % 180.0

    @property
    def cdw_angles_deg(self) -> np.ndarray:
        rot = CDW_GEOMETRY[self.model][1]
        return (self.theta0_deg + rot + np.array([0.0, 60.0, 120.0])) % 180.0

    @property
    def q_cdw(self) -> float:
        return self.q_bragg * CDW_GEOMETRY[self.model][0]


def _add_wave(out: np.ndarray, x: np.ndarray, y: np.ndarray,
              amplitude: float, q: float, angle_deg: float,
              phase: float) -> None:
    th = np.radians(angle_deg)
    out += amplitude * np.cos(
        2.0 * np.pi * q * (x * np.cos(th) + y * np.sin(th)) + phase)


def _add_blob(spectrum: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray,
              intensity: float, q: float, angle_deg: float, phase: float,
              sigma: float, rng: np.random.Generator) -> None:
    th = np.radians(angle_deg)
    u0, v0 = q * np.cos(th), q * np.sin(th)
    Ny, Nx = spectrum.shape

    half_x = max(2, int(np.ceil(6.0 * sigma * Nx)))
    half_y = max(2, int(np.ceil(6.0 * sigma * Ny)))

    cx = int(round(u0 * Nx)) + Nx // 2
    cy = int(round(v0 * Ny)) + Ny // 2
    x0, x1 = max(0, cx - half_x), min(Nx, cx + half_x + 1)
    y0, y1 = max(0, cy - half_y), min(Ny, cy + half_y + 1)
    if x1 <= x0 or y1 <= y0:
        return

    du = u_axis[x0:x1][None, :] - u0
    dv = v_axis[y0:y1][:, None] - v0
    g = np.exp(-((du ** 2 + dv ** 2) / (2.0 * sigma ** 2)))
    norm = np.sqrt(intensity / max(float((g ** 2).sum()), 1e-300))

    ph = phase + rng.uniform(0.0, 2.0 * np.pi, size=g.shape)
    spectrum[y0:y1, x0:x1] += norm * g * np.exp(1j * ph)


def _step_offsets(spec: "SyntheticSpec", x: np.ndarray, y: np.ndarray
                  ) -> np.ndarray:
    if spec.n_steps <= 0:
        return np.zeros_like(x)
    th = np.radians(spec.step_angle_deg)
    nx, ny = -np.sin(th), np.cos(th)
    proj = nx * x + ny * y
    lo, hi = float(proj.min()), float(proj.max())

    out = np.zeros_like(x)
    for k in range(int(spec.n_steps)):
        frac = (k + 1.0) / (spec.n_steps + 1.0)
        out += np.sign(proj - (lo + frac * (hi - lo)))
    return out


def _add_defects_and_steps(field: np.ndarray, spec: "SyntheticSpec",
                           rng: np.random.Generator) -> np.ndarray:
    if not (spec.defect_density > 0 or (spec.n_steps > 0
                                        and spec.step_height > 0)):
        return field

    Ny, Nx = field.shape
    ref = float(field.std()) or 1.0
    out = field.copy()

    if spec.defect_density > 0 and spec.defect_amplitude > 0:
        n_def = int(round(spec.defect_density * (Nx * Ny) / 1000.0))
        if n_def > 0:
            sig = max(float(spec.defect_radius_px), 0.5)
            half = max(2, int(np.ceil(3.0 * sig)))
            cx = rng.integers(0, Nx, n_def)
            cy = rng.integers(0, Ny, n_def)
            sign = rng.choice([-1.0, 1.0], n_def)
            amp = spec.defect_amplitude * ref * rng.uniform(0.5, 1.5, n_def)
            dy, dx = np.mgrid[-half:half + 1, -half:half + 1]
            kernel = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sig ** 2))
            for i in range(n_def):
                y0, y1 = max(0, cy[i] - half), min(Ny, cy[i] + half + 1)
                x0, x1 = max(0, cx[i] - half), min(Nx, cx[i] + half + 1)
                ky0, kx0 = y0 - (cy[i] - half), x0 - (cx[i] - half)
                out[y0:y1, x0:x1] += (sign[i] * amp[i]
                                      * kernel[ky0:ky0 + (y1 - y0),
                                               kx0:kx0 + (x1 - x0)])

    if spec.n_steps > 0 and spec.step_height > 0:
        y, x = np.mgrid[0:Ny, 0:Nx].astype(np.float64)
        out += spec.step_height * ref * _step_offsets(spec, x, y)

    return out


def _warp_coords(spec: "SyntheticSpec", x: np.ndarray, y: np.ndarray):
    Ny, Nx = spec.Ny, spec.Nx
    t = (y * Nx + x) / float(Ny * Nx)

    xw, yw = x.astype(np.float64), y.astype(np.float64)

    if spec.aspect_error:
        xw = xw * (1.0 + spec.aspect_error)

    if spec.drift_pixels:
        th = np.radians(spec.drift_angle_deg)
        xw = xw + spec.drift_pixels * t * np.cos(th)
        yw = yw + spec.drift_pixels * t * np.sin(th)

    if spec.n_steps > 0 and spec.step_registry_px:
        th_s = np.radians(spec.step_angle_deg)
        shift = 0.5 * spec.step_registry_px * _step_offsets(spec, x, y)
        xw = xw + shift * (-np.sin(th_s))
        yw = yw + shift * np.cos(th_s)

    if spec.creep_pixels:
        th = np.radians(spec.creep_angle_deg)
        tau = max(float(spec.creep_tau), 1e-3)
        relax = 1.0 - np.exp(-t / tau)
        xw = xw + spec.creep_pixels * relax * np.cos(th)
        yw = yw + spec.creep_pixels * relax * np.sin(th)

    return xw, yw


def _has_warp(spec: "SyntheticSpec") -> bool:
    return bool(spec.drift_pixels or spec.creep_pixels or spec.aspect_error
                or (spec.n_steps > 0 and spec.step_registry_px))


def _pink_field(Ny: int, Nx: int, beta: float,
                rng: np.random.Generator) -> np.ndarray:
    fy = np.fft.fftfreq(Ny)[:, None]
    fx = np.fft.fftfreq(Nx)[None, :]
    f = np.hypot(fy, fx)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(Ny, Nx))
    out = np.real(np.fft.ifft2(amp * np.exp(1j * phase)))
    s = out.std()
    return out / s if s > 0 else out


def _line_noise(Ny: int, Nx: int, beta: float, rng: np.random.Generator,
                streak_width: float = 0.02) -> np.ndarray:
    fy = np.fft.fftfreq(Ny)[:, None]
    fx = np.fft.fftfreq(Nx)[None, :]

    ay = np.abs(fy).copy()
    ay[ay == 0] = 1.0
    amp = ay ** (-beta / 2.0)
    amp = amp * np.exp(-0.5 * (fx / max(streak_width, 1e-6)) ** 2)
    amp[0, 0] = 0.0

    phase = rng.uniform(0.0, 2.0 * np.pi, size=(Ny, Nx))
    out = np.real(np.fft.ifft2(amp * np.exp(1j * phase)))
    s = out.std()
    return out / s if s > 0 else out


def _add_noise(field: np.ndarray, spec: "SyntheticSpec",
               rng: np.random.Generator) -> np.ndarray:
    if not (spec.noise_white or spec.noise_pink or spec.noise_line):
        return field

    ref = float(field.std())
    if ref <= 0:
        ref = 1.0
    Ny, Nx = field.shape
    out = field.copy()

    if spec.noise_white:
        out += spec.noise_white * ref * rng.standard_normal((Ny, Nx))
    if spec.noise_pink:
        out += spec.noise_pink * ref * _pink_field(Ny, Nx, spec.noise_beta,
                                                   rng)
    if spec.noise_line:
        out += spec.noise_line * ref * _line_noise(Ny, Nx, spec.noise_beta,
                                                   rng)
    return out


def _apply_tip(field: np.ndarray, spec: "SyntheticSpec") -> np.ndarray:
    Ny, Nx = field.shape
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=1.0))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=1.0))
    u = fx[None, :]
    v = fy[:, None]

    s2 = (u ** 2 + v ** 2) / max(spec.q_bragg ** 2, 1e-30)
    theta = np.arctan2(v, u)

    b = np.arctanh(np.clip(spec.tip_alpha_bragg, 0.0, 0.999)) / 1.5
    a = max(spec.tip_blur, 0.0)

    ln_t = -(a + b * np.cos(2.0 * (theta - np.radians(spec.tip_angle_deg)))) * s2
    ln_t = np.clip(ln_t, -60.0, 60.0)

    spec_f = np.fft.fftshift(np.fft.fft2(field)) * np.exp(ln_t)
    return np.real(np.fft.ifft2(np.fft.ifftshift(spec_f)))


def synthesize(spec: SyntheticSpec,
               rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, Dict[str, Any]]:
    if rng is None:
        rng = np.random.default_rng(spec.seed if spec.seed is not None else 0)
    y, x = np.mgrid[0:spec.Ny, 0:spec.Nx].astype(np.float64)

    i_bragg = intensities_from_alpha(spec.alpha_true_bragg,
                                     middle=spec.middle_bragg)
    i_cdw = intensities_from_alpha(spec.alpha_true_cdw,
                                   middle=spec.middle_cdw,
                                   sense=spec.sense_cdw,
                                   strongest_index=spec.strongest_index_cdw)

    i_bragg = i_bragg / i_bragg.mean()
    i_cdw = spec.cdw_power_ratio * i_cdw / i_cdw.mean()

    xs, ys = (_warp_coords(spec, x, y) if _has_warp(spec) else (x, y))

    if spec.peak_width_bins <= 0.0:
        field = np.zeros((spec.Ny, spec.Nx), dtype=np.float64)
        for k in range(3):
            _add_wave(field, xs, ys, np.sqrt(i_bragg[k]), spec.q_bragg,
                      spec.bragg_angles_deg[k], spec.phases_bragg[k])
        for k in range(3):
            _add_wave(field, xs, ys, np.sqrt(i_cdw[k]), spec.q_cdw,
                      spec.cdw_angles_deg[k], spec.phases_cdw[k])
    else:
        df = 1.0 / np.sqrt(float(spec.Nx) * float(spec.Ny))
        sigma = spec.peak_width_bins * df
        cy, cx = spec.Ny // 2, spec.Nx // 2
        u_axis = (np.arange(spec.Nx) - cx) / float(spec.Nx)
        v_axis = (np.arange(spec.Ny) - cy) / float(spec.Ny)

        spectrum = np.zeros((spec.Ny, spec.Nx), dtype=np.complex128)
        for k in range(3):
            _add_blob(spectrum, u_axis, v_axis, i_bragg[k], spec.q_bragg,
                      spec.bragg_angles_deg[k], spec.phases_bragg[k], sigma, rng)
        for k in range(3):
            _add_blob(spectrum, u_axis, v_axis, i_cdw[k], spec.q_cdw,
                      spec.cdw_angles_deg[k], spec.phases_cdw[k], sigma, rng)
        field = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))
        if _has_warp(spec):
            from scipy.ndimage import map_coordinates
            field = map_coordinates(field, [ys, xs], order=3,
                                    mode="reflect")

    if spec.tip_alpha_bragg > 0.0 or spec.tip_blur > 0.0:
        field = _apply_tip(field, spec)

    field = _add_defects_and_steps(field, spec, rng)
    field = _add_noise(field, spec, rng)

    truth = {
        "peak_width_bins": float(spec.peak_width_bins),
        "tip_alpha_bragg": float(spec.tip_alpha_bragg),
        "tip_angle_deg": float(spec.tip_angle_deg),
        "tip_blur": float(spec.tip_blur),
        "drift_pixels": float(spec.drift_pixels),
        "drift_angle_deg": float(spec.drift_angle_deg),
        "creep_pixels": float(spec.creep_pixels),
        "creep_tau": float(spec.creep_tau),
        "creep_angle_deg": float(spec.creep_angle_deg),
        "aspect_error": float(spec.aspect_error),
        "noise_white": float(spec.noise_white),
        "noise_pink": float(spec.noise_pink),
        "noise_line": float(spec.noise_line),
        "noise_beta": float(spec.noise_beta),
        "defect_density": float(spec.defect_density),
        "defect_amplitude": float(spec.defect_amplitude),
        "n_steps": int(spec.n_steps),
        "step_height": float(spec.step_height),
        "step_angle_deg": float(spec.step_angle_deg),
        "step_registry_px": float(spec.step_registry_px),
        "alpha_true_cdw": float(spec.alpha_true_cdw),
        "alpha_true_bragg": float(spec.alpha_true_bragg),
        "model": spec.model,
        "q_bragg": float(spec.q_bragg),
        "q_cdw": float(spec.q_cdw),
        "theta0_deg": float(spec.theta0_deg),
        "bragg_angles_deg": spec.bragg_angles_deg.tolist(),
        "cdw_angles_deg": spec.cdw_angles_deg.tolist(),
        "I_bragg_true": i_bragg.tolist(),
        "I_cdw_true": i_cdw.tolist(),
        "Ny": spec.Ny, "Nx": spec.Nx,
        "cdw_power_ratio": float(spec.cdw_power_ratio),
        "seed": spec.seed,
    }
    return field, truth


def subbin_offsets(spec: SyntheticSpec) -> Dict[str, float]:
    def offsets(q: float, angles: np.ndarray) -> np.ndarray:
        d = []
        for ang in angles:
            th = np.radians(ang)
            bx = q * np.cos(th) * spec.Nx
            by = q * np.sin(th) * spec.Ny
            fx = abs(bx - round(bx))
            fy = abs(by - round(by))
            d.append(float(np.hypot(fx, fy)))
        return np.asarray(d)

    ob = offsets(spec.q_bragg, spec.bragg_angles_deg)
    oc = offsets(spec.q_cdw, spec.cdw_angles_deg)
    return {
        "subbin_bragg_mean": float(ob.mean()),
        "subbin_bragg_spread": float(ob.max() - ob.min()),
        "subbin_cdw_mean": float(oc.mean()),
        "subbin_cdw_spread": float(oc.max() - oc.min()),
    }


def draw_random_spec(rng: np.random.Generator,
                     alpha_true_cdw: float = 0.0,
                     alpha_true_bragg: float = 0.0,
                     square: bool = True,
                     models: Sequence[str] = ("2x2", "3x3"),
                     peak_width_bins: float = 0.0,
                     tip_alpha_bragg: float = 0.0,
                     tip_angle_deg: Optional[float] = None,
                     tip_blur: float = 0.0,
                     drift_pixels: float = 0.0,
                     creep_pixels: float = 0.0,
                     creep_tau: float = 0.15,
                     aspect_error: float = 0.0,
                     noise_white: float = 0.0,
                     noise_pink: float = 0.0,
                     noise_line: float = 0.0,
                     noise_beta: float = 1.0,
                     defect_density: float = 0.0,
                     defect_amplitude: float = 0.0,
                     n_steps: int = 0,
                     step_height: float = 0.0,
                     step_angle_deg: Optional[float] = None,
                     step_registry_px: float = 0.0,
                     seed: Optional[int] = None) -> SyntheticSpec:
    n = int(rng.choice(OBSERVED_SIZES))
    if square:
        Ny = Nx = n
    else:
        aspect = float(rng.uniform(0.95, 2.20))
        Ny, Nx = n, int(round(n * aspect))

    return SyntheticSpec(
        Ny=Ny, Nx=Nx,
        q_bragg=float(rng.uniform(*OBSERVED_Q_BRAGG)),
        theta0_deg=float(rng.uniform(0.0, 60.0)),
        model=str(rng.choice(list(models))),
        alpha_true_cdw=alpha_true_cdw,
        alpha_true_bragg=alpha_true_bragg,
        cdw_power_ratio=float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))),
        middle_cdw=float(rng.uniform(0.0, 1.0)),
        middle_bragg=float(rng.uniform(0.0, 1.0)),
        sense_cdw=int(rng.choice([-1, +1])),
        strongest_index_cdw=int(rng.integers(0, 3)),
        phases_bragg=tuple(rng.uniform(0, 2 * np.pi, 3)),
        phases_cdw=tuple(rng.uniform(0, 2 * np.pi, 3)),
        peak_width_bins=float(peak_width_bins),
        tip_alpha_bragg=float(tip_alpha_bragg),
        tip_angle_deg=(float(rng.uniform(0.0, 180.0))
                       if tip_angle_deg is None else float(tip_angle_deg)),
        tip_blur=float(tip_blur),
        drift_pixels=float(drift_pixels),
        drift_angle_deg=float(rng.uniform(0.0, 360.0)),
        creep_pixels=float(creep_pixels),
        creep_tau=float(creep_tau),
        creep_angle_deg=float(rng.uniform(0.0, 360.0)),
        aspect_error=float(aspect_error),
        noise_white=float(noise_white),
        noise_pink=float(noise_pink),
        noise_line=float(noise_line),
        noise_beta=float(noise_beta),
        defect_density=float(defect_density),
        defect_amplitude=float(defect_amplitude),
        n_steps=int(n_steps),
        step_height=float(step_height),
        step_angle_deg=(float(rng.uniform(0.0, 180.0))
                        if step_angle_deg is None else float(step_angle_deg)),
        step_registry_px=float(step_registry_px),
        seed=seed,
    )
