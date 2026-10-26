"""Interface web locale : curseurs + pré-écoute A/B.

Usage : .venv/bin/python server.py [--port 8765]   puis ouvrir http://127.0.0.1:8765
"""
import argparse
import copy
import json
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from mixer import engine, pipeline
from mixer.audio import db2g, lufs, save
from mixer.stems import load_stems

ROOT = Path(__file__).resolve().parent
app = FastAPI()
_cache = {}          # slug → (stems, sr)
_lock = threading.Lock()
_jobs = {}           # slug → état de l'export


def stems_of(slug):
    with _lock:
        if slug not in _cache:
            sdir = pipeline.song_dir(slug)
            if not (sdir / "stems.json").exists():
                raise HTTPException(404, f"morceau inconnu : {slug}")
            _cache[slug] = load_stems(sdir)
        return _cache[slug]


def diff(full, base):
    """Ne garde que ce qui diffère du préset (ce qu'on écrit dans settings.json)."""
    out = {}
    for k, v in full.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            d = diff(v, base[k])
            if d:
                out[k] = d
        elif base.get(k) != v:
            out[k] = v
    return out


def to_overrides(settings):
    s = copy.deepcopy(settings)
    preset = s.get("preset")
    base = engine.load_preset(preset)
    over = diff(s, base)
    over["preset"] = preset
    for k in ("bpm", "reference", "clip_dir"):
        if s.get(k) is not None:
            over[k] = s[k]
    return over


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text()


@app.get("/api/songs")
def songs():
    root = pipeline.SONGS
    return sorted(p.name for p in root.iterdir() if (p / "stems.json").exists()) if root.exists() else []


@app.get("/api/presets")
def presets():
    return {name: engine.load_preset(name) for name in engine.list_presets()}


@app.get("/api/song/{slug}")
def song(slug):
    stems, sr = stems_of(slug)
    sdir = pipeline.song_dir(slug)
    pipeline.ensure_bpm(sdir, stems, sr)
    over = pipeline.read_overrides(sdir)
    s = pipeline.effective_settings(sdir, over)
    s["preset"] = over.get("preset") or "default"
    s["clip_dir"] = over.get("clip_dir")
    out = sdir / "out"
    return {
        "slug": slug, "settings": s, "duration": stems[0]["audio"].shape[1] / sr,
        "stems": [{"file": st["file"], "name": st["name"], "role": st["role"]} for st in stems],
        "outputs": sorted(p.name for p in out.glob("*.wav")) if out.exists() else [],
        "report": json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else None,
    }


class PreviewReq(BaseModel):
    settings: dict
    start: float = 30.0
    duration: float = 25.0
    version: str = "distrokid"


@app.post("/api/song/{slug}/preview")
def preview(slug, req: PreviewReq):
    stems, sr = stems_of(slug)
    sdir = pipeline.song_dir(slug)
    s = engine.deep_merge(engine.load_preset(req.settings.get("preset")), req.settings)
    t = time.time()
    seg = (max(0.0, req.start), req.duration)
    pre, info = pipeline.bus(stems, sr, s, seg)
    mix, meas = pipeline.output_version(pre, sr, s, req.version)
    a = int(seg[0] * sr)
    orig = pipeline.load(sdir / "original.wav")[0][:, a:a + mix.shape[1]]
    # même volume pour une comparaison honnête
    l_o, l_m = lufs(orig, sr), lufs(mix, sr)
    common = min(l_o, l_m)
    pdir = sdir / "preview"
    pdir.mkdir(exist_ok=True)
    for old in pdir.glob("*.wav"):
        if time.time() - old.stat().st_mtime > 120:
            old.unlink()
    tag = uuid.uuid4().hex[:8]
    save(pdir / f"mix_{tag}.wav", mix * db2g(common - l_m), sr, subtype="PCM_16")
    save(pdir / f"orig_{tag}.wav", orig * db2g(common - l_o), sr, subtype="PCM_16")
    return {"mix": f"/preview/{slug}/mix_{tag}.wav", "orig": f"/preview/{slug}/orig_{tag}.wav",
            "measures": meas, "lufs_original": round(l_o, 1), "pistes": info,
            "seconds": round(time.time() - t, 1)}


@app.get("/preview/{slug}/{name}")
def preview_file(slug, name):
    path = pipeline.song_dir(slug) / "preview" / Path(name).name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.get("/out/{slug}/{name}")
def out_file(slug, name):
    path = pipeline.song_dir(slug) / "out" / Path(name).name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


class SaveReq(BaseModel):
    settings: dict


@app.post("/api/song/{slug}/save")
def save_settings(slug, req: SaveReq):
    over = to_overrides(req.settings)
    pipeline.write_overrides(pipeline.song_dir(slug), over)
    return {"ok": True, "overrides": over}


@app.post("/api/song/{slug}/export")
def export(slug, req: SaveReq):
    save_settings(slug, req)
    if _jobs.get(slug, {}).get("state") == "running":
        return _jobs[slug]
    _jobs[slug] = {"state": "running", "log": []}

    def job():
        try:
            over = pipeline.read_overrides(pipeline.song_dir(slug))
            pipeline.export(slug, clip_dir=over.get("clip_dir"), log=_jobs[slug]["log"].append)
            _jobs[slug]["state"] = "done"
        except Exception as e:  # remonté à l'interface
            _jobs[slug].update(state="error", error=str(e))

    threading.Thread(target=job, daemon=True).start()
    return _jobs[slug]


@app.get("/api/song/{slug}/export")
def export_status(slug):
    return _jobs.get(slug, {"state": "idle", "log": []})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"Interface : http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
