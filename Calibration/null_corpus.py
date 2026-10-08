import numpy as np
from scipy.ndimage import gaussian_filter

CDW_DENOM = 3.0


def _smooth_noise(rng, N, sigma):
    z = gaussian_filter(rng.standard_normal((N, N)), sigma, mode="wrap")
    s = float(z.std())
    return z / (s if s > 1e-12 else 1.0)


def make_null_map(seed, N=384, a_px=9.0, alpha_true=0.0, rng=None,
                  layers=None):
    rng = rng or np.random.default_rng(seed)
    L = layers if layers is not None else {}

    def opt(key, default):
        return L.get(key, default)

    a = float(alpha_true)
    I3 = (1.0 - a) / (1.0 + a) if a > 0 else 1.0
    I = np.array([1.0, np.sqrt(max(I3, 1e-9)), max(I3, 1e-9)])
    amps = np.sqrt(I)
    amps = amps / amps.max()
    amps = amps[rng.permutation(3)]

    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    x -= N / 2.0
    y -= N / 2.0
    shear = rng.uniform(*opt("shear", (0.0, 0.16)))
    stretch = rng.uniform(*opt("stretch", (0.0, 0.07)))
    creep = rng.uniform(*opt("creep", (0.0, 0.09)))
    xs = (1.0 + stretch) * x + shear * y
    ys = y.copy()
    if creep > 0:
        tau = N / 3.0
        xs = xs + creep * N * np.exp(-(y + N / 2.0) / tau)

    theta0 = rng.uniform(0, 60.0)
    fb = 1.0 / a_px
    field = np.zeros((N, N))
    for k in range(3):
        th = np.radians(theta0 + 60.0 * k)
        field += 0.6 * np.cos(
            2 * np.pi * fb * (np.cos(th) * xs + np.sin(th) * ys))

    xi = rng.uniform(*opt("xi", (12.0, 90.0)))
    phase_amp = rng.uniform(*opt("phase_amp", (0.3, 1.7)))
    for k in range(3):
        th = np.radians(theta0 + 60.0 * k)
        q = fb / CDW_DENOM
        ph = phase_amp * _smooth_noise(rng, N, xi)
        field += amps[k] * np.cos(
            2 * np.pi * q * (np.cos(th) * xs + np.sin(th) * ys) + ph)

    n_def = rng.integers(*opt("n_defects", (0, 45)))
    for _ in range(int(n_def)):
        cx, cy = rng.uniform(-N / 2, N / 2, 2)
        w = rng.uniform(2.0, 6.0)
        amp = rng.uniform(-1.5, 1.5)
        field += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w ** 2))

    n_step = rng.integers(*opt("n_steps", (0, 3)))
    for _ in range(int(n_step)):
        ang = rng.uniform(0, np.pi)
        off = rng.uniform(-N / 3, N / 3)
        d = np.cos(ang) * x + np.sin(ang) * y - off
        field += rng.uniform(0.3, 1.2) * np.tanh(d / rng.uniform(1.0, 4.0))

    tip = rng.uniform(*opt("tip", (0.0, 2.2)))
    if tip > 0:
        tth = rng.uniform(0, np.pi)
        F = np.fft.fftshift(np.fft.fft2(field))
        fy = np.fft.fftshift(np.fft.fftfreq(N))[:, None]
        fx = np.fft.fftshift(np.fft.fftfreq(N))[None, :]
        qq = np.hypot(fx, fy) / max(fb, 1e-9)
        tt = np.arctan2(fy, fx)
        env = np.exp(-tip * 0.25 * qq ** 2 * np.cos(2 * (tt - tth)))
        field = np.real(np.fft.ifft2(np.fft.ifftshift(F * env)))

    white = rng.uniform(*opt("white", (0.02, 0.55)))
    pink = rng.uniform(*opt("pink", (0.0, 0.55)))
    scan = rng.uniform(*opt("scan", (0.0, 0.45)))
    field = field + white * rng.standard_normal((N, N))
    if pink > 0:
        field = field + pink * _smooth_noise(rng, N, rng.uniform(10, 40))
    if scan > 0:
        line = gaussian_filter(rng.standard_normal(N), rng.uniform(2, 12))
        line /= max(float(line.std()), 1e-12)
        field = field + scan * line[:, None]
    return field
