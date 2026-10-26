"""Chaîne complète d'un morceau : réglages → mix → master → fichiers de sortie."""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import engine
from .audio import band_profile, db2g, estimate_bpm, load, lufs, resample, save, true_peak_db
from .stems import load_stems

ROOT = Path(__file__).resolve().parent.parent
SONGS = ROOT / "songs"


def song_dir(slug):
    return SONGS / slug


def read_overrides(sdir):
    path = Path(sdir) / "settings.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_overrides(sdir, overrides):
    (Path(sdir) / "settings.json").write_text(json.dumps(overrides, ensure_ascii=False, indent=1))


def effective_settings(sdir, overrides=None):
    over = read_overrides(sdir) if overrides is None else overrides
    return engine.deep_merge(engine.load_preset(over.get("preset")), over)


def ensure_bpm(sdir, stems, sr):
    """Tempo détecté une fois pour toutes (stocké dans settings.json)."""
    over = read_overrides(sdir)
    if not over.get("bpm"):
        drums = [s["audio"] for s in stems if s["role"] == "drums"]
        src = sum(drums) if drums else sum(s["audio"] for s in stems)
        over["bpm"] = estimate_bpm(src, sr) or 95.0
        write_overrides(sdir, over)
    return over["bpm"]


def match_reference(x, sr, ref_path):
    """Égalisation et dynamique calées sur un morceau de référence (Matchering)."""
    import matchering as mg
    with tempfile.TemporaryDirectory() as tmp:
        tgt, out = Path(tmp) / "target.wav", Path(tmp) / "matched.wav"
        save(tgt, x, sr, subtype="FLOAT")
        mg.log(lambda *_: None)
        mg.process(target=str(tgt), reference=str(ref_path),
                   results=[mg.Result(str(out), subtype="FLOAT", use_limiter=False, normalize=False)])
        y, ysr = load(out)
    if ysr != sr:
        y = resample(y, ysr, sr)
    return y


def bus(stems, sr, s, segment=None):
    """Mix pré-master (+ calage sur la référence si fournie)."""
    pre, info = engine.render(stems, sr, s, segment)
    ref = s.get("reference")
    if ref and Path(ref).exists():
        pre = match_reference(pre, sr, ref)
    return pre, info


def output_version(pre, sr, s, name):
    cfg = s["outputs"][name]
    m = engine.deep_merge(s["master"], cfg.get("master", {}))
    x = engine.master(pre, sr, m)
    return engine.finalize(x, sr, cfg["lufs"], cfg["ceiling"])


def ab_file(orig, mix, sr, bpm, bars=4, fade_ms=25):
    """Alterne original / mix toutes les `bars` mesures (4/4), au même volume."""
    n = min(orig.shape[1], mix.shape[1])
    orig, mix = orig[:, :n], mix[:, :n]
    seg = int(sr * 60.0 / (bpm or 95.0) * 4 * bars)
    sel = ((np.arange(n) // seg) % 2).astype(np.float32)  # 0 = original, 1 = mix
    f = max(1, int(sr * fade_ms / 1000))
    sel = np.convolve(sel, np.ones(f) / f, mode="same")
    return orig * (1 - sel) + mix * sel


def export(slug, clip_dir=None, log=print):
    sdir = song_dir(slug)
    stems, sr = load_stems(sdir)
    ensure_bpm(sdir, stems, sr)
    s = effective_settings(sdir)
    out = sdir / "out"
    out.mkdir(exist_ok=True)

    log(f"  mix de {len(stems)} pistes (préset {s.get('preset') or 'default'}, {s['bpm']} BPM)…")
    pre, info = bus(stems, sr, s)
    results = {}
    for name in s["outputs"]:
        x, meas = output_version(pre, sr, s, name)
        path = out / f"mix_{name}.wav"
        save(path, x, sr, subtype="PCM_24")
        results[name] = {"file": str(path), **meas}
        log(f"  → {path.name} : {meas['lufs']} LUFS, true peak {meas['true_peak']} dBTP")

    # original au même volume que le mix DistroKid, pour comparer à l'oreille
    orig = load(sdir / "original.wav")[0]
    mix_main = load(out / "mix_distrokid.wav")[0]
    l_orig, l_mix = lufs(orig, sr), lufs(mix_main, sr)
    common = min(l_orig, l_mix)
    orig_m = orig * db2g(common - l_orig)
    mix_m = mix_main * db2g(common - l_mix)
    save(out / "original_meme_volume.wav", orig_m, sr)
    save(out / "ab_original_puis_mix.wav", ab_file(orig_m, mix_m, sr, s["bpm"]), sr)

    report = {
        "preset": s.get("preset") or "default", "bpm": s["bpm"],
        "original": {"lufs": round(l_orig, 1), "true_peak": round(true_peak_db(orig), 1),
                     "bandes": band_profile(orig, sr)},
        "mix": {"bandes": band_profile(mix_main, sr), **results},
        "pistes": info,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    if clip_dir:
        copy_to_clip(Path(clip_dir), out / "mix_distrokid.wav", log)
    return report


def copy_to_clip(clip_dir, mix_path, log=print):
    """Copie le mix dans un projet create_video_music et le désigne comme audio du clip."""
    proj = clip_dir / "project.json"
    (clip_dir / "audio").mkdir(exist_ok=True)
    data = json.loads(proj.read_text()) if proj.exists() else {}
    stem = Path(data.get("audio_original") or data.get("audio") or "audio/mix.wav").stem
    dest = clip_dir / "audio" / f"{stem} (mix).wav"
    shutil.copy2(mix_path, dest)
    if proj.exists():
        data.setdefault("audio_original", data.get("audio"))
        data["audio"] = str(dest.relative_to(clip_dir))
        proj.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    log(f"  → copié dans {dest}" + (" (project.json mis à jour)" if proj.exists() else ""))
