"""Traitements : chaque fonction prend un signal (2, n) float32 et renvoie (2, n)."""
import numpy as np
from pedalboard import (Compressor, Delay, HighpassFilter, HighShelfFilter, LowpassFilter,
                        LowShelfFilter, Pedalboard, PeakFilter, PitchShift, Reverb)
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

from .audio import db2g, envelope


def run(board, x, sr):
    return board(x, sr).astype(np.float32)


def eq(x, sr, filters):
    filters = [f for f in filters if f is not None]
    return run(Pedalboard(filters), x, sr) if filters else x


def peak(freq, gain_db, q=1.0):
    return PeakFilter(cutoff_frequency_hz=freq, gain_db=gain_db, q=q) if abs(gain_db) > 0.05 else None


def low_shelf(freq, gain_db):
    return LowShelfFilter(cutoff_frequency_hz=freq, gain_db=gain_db) if abs(gain_db) > 0.05 else None


def high_shelf(freq, gain_db):
    return HighShelfFilter(cutoff_frequency_hz=freq, gain_db=gain_db) if abs(gain_db) > 0.05 else None


def deess(x, sr, amount, freq=5500.0):
    """De-esser à bande séparée : atténue la partie > freq quand elle dépasse
    son niveau habituel (sifflantes « s », « ch »)."""
    if amount <= 0:
        return x
    hi = run(Pedalboard([HighpassFilter(freq), HighpassFilter(freq)]), x, sr)
    e_hi = envelope(hi, sr, 3.0)
    active = envelope(x, sr, 20.0)
    gate = active > np.percentile(active, 40)
    if not gate.any():
        return x
    thr = np.percentile(e_hi[gate], 70)
    over = np.maximum(e_hi / (thr + 1e-9), 1.0)
    floor = db2g(-14.0 * amount)
    g = np.clip(over ** (-0.9 * amount), floor, 1.0)
    g = uniform_filter1d(g, size=max(1, int(sr * 0.004)))
    return x - hi * (1.0 - g)[None, :]


def compress(x, sr, amount, base_thr=-16.0, attack=5.0, release=90.0):
    if amount <= 0:
        return x
    c = Compressor(threshold_db=base_thr - 10.0 * amount, ratio=1.5 + 4.5 * amount,
                   attack_ms=attack, release_ms=release)
    return run(Pedalboard([c]), x, sr)


def saturate(x, amount, drive_max=5.0):
    """Saturation douce type bande/préampli, en parallèle, au même niveau RMS."""
    if amount <= 0:
        return x
    drive = 1.0 + drive_max * amount
    wet = np.tanh(drive * x + 0.06 * amount * x * x)  # légère asymétrie = harmoniques paires
    wet -= wet.mean(axis=1, keepdims=True)
    rms_x = np.sqrt(np.mean(x ** 2)) + 1e-9
    rms_w = np.sqrt(np.mean(wet ** 2)) + 1e-9
    mix = min(1.0, 0.25 + 0.75 * amount)
    return ((1 - mix) * x + mix * wet * (rms_x / rms_w)).astype(np.float32)


def double(x, sr, amount):
    """Doublage discret : deux copies légèrement désaccordées et décalées, gauche/droite."""
    if amount <= 0:
        return x
    mono = x.mean(axis=0, keepdims=True)
    pre = Pedalboard([HighpassFilter(150), LowpassFilter(8000)])
    left = run(Pedalboard([PitchShift(semitones=0.09), Delay(delay_seconds=0.017, feedback=0.0, mix=1.0)]),
               run(pre, mono, sr), sr)
    right = run(Pedalboard([PitchShift(semitones=-0.09), Delay(delay_seconds=0.026, feedback=0.0, mix=1.0)]),
                run(pre, mono, sr), sr)
    n = x.shape[1]
    g = 0.45 * amount
    return x + g * np.vstack([left[0, :n], right[0, :n]])


def reverb_send(x, sr, amount, room=0.35, damping=0.6):
    if amount <= 0:
        return x
    send = run(Pedalboard([HighpassFilter(250), LowpassFilter(7500)]), x, sr)
    wet = run(Pedalboard([Reverb(room_size=room, damping=damping, wet_level=1.0, dry_level=0.0, width=1.0)]),
              send, sr)
    return x + 0.5 * amount * wet


def delay_send(x, sr, amount, bpm):
    if amount <= 0:
        return x
    t = 60.0 / (bpm or 95.0) / 2  # croche
    send = run(Pedalboard([HighpassFilter(400), LowpassFilter(5000)]), x, sr)
    wet = run(Pedalboard([Delay(delay_seconds=t, feedback=0.28, mix=1.0)]), send, sr)
    # ping-pong simple : décaler un canal
    wet[1] = np.roll(wet[1], int(t * sr / 2))
    return x + 0.45 * amount * wet


def transient(x, sr, amount):
    """Accentue les attaques (punch) : gain = (enveloppe rapide / lente)^amount."""
    if amount <= 0:
        return x
    fast = envelope(x, sr, 1.0)
    slow = envelope(x, sr, 40.0)
    g = np.clip((fast / (slow + 1e-9)) ** (0.9 * amount), 0.6, 3.0)
    g = uniform_filter1d(g, size=max(1, int(sr * 0.001)))
    return x * g[None, :]


def parallel_comp(x, sr, amount):
    if amount <= 0:
        return x
    crushed = run(Pedalboard([Compressor(threshold_db=-32, ratio=8, attack_ms=1.0, release_ms=120)]), x, sr)
    return x + 0.6 * amount * crushed


def width(x, sr, w, mono_below=0.0):
    """Largeur stéréo par mid/side ; `mono_below` (Hz) met les basses en mono."""
    if abs(w - 1.0) < 1e-3 and mono_below <= 0:
        return x
    mid = (x[0] + x[1]) / 2
    side = (x[0] - x[1]) / 2 * w
    if mono_below > 0:
        side = run(Pedalboard([HighpassFilter(mono_below), HighpassFilter(mono_below)]), side[None], sr)[0]
    return np.vstack([mid + side, mid - side]).astype(np.float32)


def duck(x, sr, key_env, depth_db):
    """Creuse le médium (1–5 kHz) de `x` quand la voix (key_env 0..1) est présente."""
    if depth_db <= 0:
        return x
    band = run(Pedalboard([HighpassFilter(1000), LowpassFilter(5000)]), x, sr)
    g = 1.0 - (1.0 - db2g(-depth_db)) * key_env
    return x - band * (1.0 - g)[None, :]


def limit(x, sr, ceiling_db, window_ms=12.0):
    """Limiteur à anticipation (gain lié sur les deux canaux), sensible aux crêtes
    inter-échantillons (enveloppe mesurée suréchantillonnée ×4)."""
    ceil = db2g(ceiling_db)
    n = x.shape[1]
    up = np.max(np.abs(resample_poly(x, 4, 1, axis=1)), axis=0)[: n * 4]
    amp = np.maximum(np.abs(x).max(axis=0), np.pad(up, (0, n * 4 - len(up))).reshape(n, 4).max(axis=1))
    need = np.minimum(1.0, ceil / np.maximum(amp, 1e-9))
    w = max(3, int(sr * window_ms / 1000))
    g = minimum_filter1d(need, size=2 * w + 1)
    g = uniform_filter1d(g, size=w + 1)
    return (x * g[None, :]).astype(np.float32)
