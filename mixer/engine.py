"""Moteur de mix : traitement par rôle, bus master, mise au volume des versions finales."""
import copy
import json
from pathlib import Path

import numpy as np

from . import dsp
from .audio import db2g, envelope, lufs, true_peak_db

PRESETS = Path(__file__).parent / "presets"
WORK_LUFS = -20.0  # chaque piste est ramenée à ce niveau avant traitement (seuils stables)


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_preset(name):
    default = json.loads((PRESETS / "default.json").read_text())
    if name in (None, "", "default"):
        return default
    path = PRESETS / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"Préset inconnu : {name} (dispo : {', '.join(list_presets())})")
    return deep_merge(default, json.loads(path.read_text()))


def list_presets():
    return sorted(p.stem for p in PRESETS.glob("*.json"))


def to_level(x, sr, target):
    level = lufs(x, sr)
    if not np.isfinite(level):
        return x, None
    return x * db2g(target - level), level


# --- traitements par rôle -------------------------------------------------

def process_vocals(x, sr, p, bpm):
    x = dsp.eq(x, sr, [dsp.HighpassFilter(90),
                       dsp.peak(250, p["warmth"], 0.8),
                       dsp.peak(450, -p["mud"], 1.2),
                       dsp.peak(3200, -2.5 * p["harsh"], 1.4),
                       dsp.peak(7500, -4.5 * p["harsh"], 1.6),
                       dsp.high_shelf(12000, p["air"])])
    x = dsp.deess(x, sr, p["deess"])
    x = dsp.compress(x, sr, p["compression"])
    x = dsp.saturate(x, p["saturation"])
    x = dsp.double(x, sr, p["double"])
    x = dsp.width(x, sr, p.get("width", 1.0))
    x = dsp.delay_send(x, sr, p["delay"], bpm)
    x = dsp.reverb_send(x, sr, p["reverb"], room=0.3)
    return x


def process_drums(x, sr, p, bpm):
    x = dsp.eq(x, sr, [dsp.HighpassFilter(30), dsp.low_shelf(90, p["low"]), dsp.high_shelf(7000, p["snap"])])
    x = dsp.transient(x, sr, p["punch"])
    x = dsp.parallel_comp(x, sr, p["punch"])
    x = dsp.saturate(x, p["saturation"], drive_max=3.0)
    return x


def process_bass(x, sr, p, bpm):
    x = dsp.eq(x, sr, [dsp.HighpassFilter(28), dsp.low_shelf(80, p["low"])])
    x = dsp.compress(x, sr, p["compression"], attack=15.0, release=120.0)
    x = dsp.saturate(x, p["saturation"], drive_max=4.0)
    if p.get("mono", True):
        x = dsp.width(x, sr, 0.0)
    return x


def process_music(x, sr, p, bpm, vocal_env=None):
    x = dsp.eq(x, sr, [dsp.HighpassFilter(35),
                       dsp.peak(350, -p["mud"], 1.0),
                       dsp.peak(3500, -2.0 * p["harsh"], 1.2)])
    if vocal_env is not None:
        x = dsp.duck(x, sr, vocal_env, p["duck"])
    x = dsp.width(x, sr, p.get("width", 1.0), mono_below=120)
    return x


PROCESSORS = {"vocals": process_vocals, "backing": process_vocals,
              "drums": process_drums, "bass": process_bass, "music": process_music}


def vocal_key(stems, sr):
    """Enveloppe 0..1 de présence de la voix principale (pour le ducking)."""
    voc = [s["audio"] for s in stems if s["role"] == "vocals"]
    if not voc:
        return None
    e = envelope(sum(voc), sr, 60.0)
    hi = np.percentile(e, 85) + 1e-9
    return np.clip(e / hi, 0.0, 1.0)


def render(stems, sr, s, segment=None):
    """Mix des pistes → bus pré-master (avant master et mise au volume).

    `segment` = (début_s, durée_s) pour une pré-écoute rapide.
    Renvoie (mix, infos par piste).
    """
    if segment:
        a = int(segment[0] * sr)
        b = a + int(segment[1] * sr)
        stems = [{**st, "audio": st["audio"][:, a:b]} for st in stems]
    bpm = s.get("bpm") or 95.0
    key = vocal_key(stems, sr)

    orig_levels = {i: lufs(st["audio"], sr) for i, st in enumerate(stems)}
    voc = [orig_levels[i] for i, st in enumerate(stems) if st["role"] == "vocals" and np.isfinite(orig_levels[i])]
    finite = [v for v in orig_levels.values() if np.isfinite(v)]
    anchor = max(voc) if voc else (max(finite) if finite else WORK_LUFS)

    out, info = None, []
    for i, st in enumerate(stems):
        role = st["role"]
        if st["file"] in s.get("mute", []):
            continue
        x, level = to_level(st["audio"], sr, WORK_LUFS)
        if level is None:
            continue
        params = s[role]
        if role == "music":
            y = process_music(x, sr, params, bpm, key)
        else:
            y = PROCESSORS[role](x, sr, params, bpm)
        processed = lufs(y, sr)
        # niveau voulu : mélange entre l'équilibre d'origine et la cible du préset
        target = (1 - s["balance"]) * level + s["balance"] * (anchor + s["targets"][role])
        target += s["levels"].get(role, 0.0) + s["stem_levels"].get(st["file"], 0.0)
        y = y * db2g(target - processed)
        out = y if out is None else _add(out, y)
        info.append({"file": st["file"], "name": st["name"], "role": role,
                     "lufs_origine": round(level, 1), "lufs_mix": round(target, 1)})
    return out, info


def _add(a, b):
    n = max(a.shape[1], b.shape[1])
    a = np.pad(a, ((0, 0), (0, n - a.shape[1])))
    b = np.pad(b, ((0, 0), (0, n - b.shape[1])))
    return a + b


def master(x, sr, m):
    """Bus master : EQ, compression de colle, chaleur, largeur, basses mono."""
    x, _ = to_level(x, sr, -16.0)
    x = dsp.eq(x, sr, [dsp.low_shelf(100, m["low"]), dsp.high_shelf(10000, m["high"])])
    if m["glue"] > 0:
        x = dsp.run(dsp.Pedalboard([dsp.Compressor(threshold_db=-10 - 8 * m["glue"], ratio=1.5 + 1.5 * m["glue"],
                                                   attack_ms=30, release_ms=150)]), x, sr)
    x = dsp.saturate(x, m["warmth"], drive_max=2.5)
    x = dsp.width(x, sr, m["width"], mono_below=m["mono_bass"])
    return x


def finalize(x, sr, target_lufs, ceiling_db):
    """Ramène à `target_lufs` sous un plafond true peak `ceiling_db`."""
    level = lufs(x, sr)
    y = x * db2g(target_lufs - level)
    ceil = ceiling_db - 0.3
    for _ in range(12):
        out = dsp.limit(y, sr, ceil)
        level = lufs(out, sr)
        tp = true_peak_db(out)
        if tp > ceiling_db:
            ceil -= tp - ceiling_db + 0.05
            continue
        if abs(target_lufs - level) < 0.15:
            break
        y = y * db2g(target_lufs - level)
    return out, {"lufs": round(level, 1), "true_peak": round(tp, 1)}
