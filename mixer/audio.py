"""Entrées/sorties audio et mesures (LUFS, true peak, tempo, spectre)."""
from math import gcd

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import lfilter, resample_poly


def db2g(db):
    return 10.0 ** (db / 20.0)


def load(path, sr=None):
    """Charge un fichier en float32 (2, n). Rééchantillonne si `sr` est donné."""
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.T
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    x = np.ascontiguousarray(x[:2])
    if sr and sr != file_sr:
        x = resample(x, file_sr, sr)
        file_sr = sr
    return x, file_sr


def resample(x, sr_in, sr_out):
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g, axis=1).astype(np.float32)


def save(path, x, sr, subtype="PCM_24"):
    sf.write(str(path), np.ascontiguousarray(x.T), sr, subtype=subtype)


def lufs(x, sr):
    """Loudness intégrée (LUFS). -inf si silence."""
    if x.shape[1] < sr // 2:
        return float("-inf")
    return float(pyln.Meter(sr).integrated_loudness(x.T.astype(np.float64)))


def true_peak_db(x):
    up = resample_poly(x, 4, 1, axis=1)
    return float(20 * np.log10(max(np.max(np.abs(up)), 1e-9)))


def peak_db(x):
    return float(20 * np.log10(max(np.max(np.abs(x)), 1e-9)))


def estimate_bpm(x, sr, lo=70, hi=180):
    """Tempo approximatif par autocorrélation de l'enveloppe d'attaques."""
    mono = x.mean(axis=0)
    rate = 200  # trames de 5 ms
    hop = sr // rate
    n = len(mono) // hop
    if n < 800:
        return None
    energy = np.sqrt(np.mean(mono[: n * hop].reshape(n, hop) ** 2, axis=1))
    onset = np.maximum(np.diff(np.log(energy + 1e-6)), 0)
    onset -= onset.mean()
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    a, b = int(60 * rate / hi), int(60 * rate / lo) + 1
    if b + 1 >= len(ac):
        return None
    lag = a + int(np.argmax(ac[a:b]))
    # interpolation parabolique autour du pic
    y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = y0 - 2 * y1 + y2
    lag = lag + (0.5 * (y0 - y2) / denom if denom else 0.0)
    bpm = 60.0 * rate / lag
    # privilégier la zone 80–160 BPM (double/moitié)
    while bpm < 80:
        bpm *= 2
    while bpm > 160:
        bpm /= 2
    return round(float(bpm), 1)


BANDS = [("sub", 20, 60), ("bas", 60, 250), ("bas-médium", 250, 2000),
         ("haut-médium", 2000, 6000), ("aigus", 6000, 12000), ("air", 12000, 20000)]


def band_profile(x, sr):
    """Énergie relative par bande (dB, relatif au total)."""
    mono = x.mean(axis=0)
    n = min(len(mono), sr * 60)
    start = max(0, (len(mono) - n) // 2)
    seg = mono[start:start + n]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    total = spec.sum() + 1e-12
    return {name: round(float(10 * np.log10(spec[(freqs >= a) & (freqs < b)].sum() / total + 1e-12)), 1)
            for name, a, b in BANDS}


def envelope(x, sr, ms):
    """Enveloppe RMS lissée (mono), constante de temps `ms`."""
    p = np.mean(x ** 2, axis=0) if x.ndim == 2 else x ** 2
    a = np.exp(-1.0 / (sr * ms / 1000.0))
    e = lfilter([1 - a], [1, -a], p.astype(np.float64))
    return np.sqrt(np.maximum(e, 1e-12)).astype(np.float32)
