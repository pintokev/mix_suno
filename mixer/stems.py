"""Préparation des pistes d'un morceau : stems Suno (dossier ou zip) ou séparation Demucs.

Résultat dans songs/<slug>/ :
  original.wav   le morceau tel quel (fourni, ou somme des stems)
  stems/*.wav    une piste par fichier, à la fréquence de l'original
  stems.json     [{"file", "name", "role"}] ; role ∈ vocals, backing, drums, bass, music
La somme des stems reproduit l'original : rien n'est perdu à la séparation.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path

import numpy as np

from .audio import load, save

AUDIO_EXT = {".wav", ".flac", ".mp3", ".aiff", ".aif", ".m4a", ".ogg"}

# mot-clé (dans le nom du fichier) → rôle ; l'ordre compte (« backing vocals » avant « vocals »)
ROLE_KEYWORDS = [
    (("backing", "choir", "chœur", "choeur", "harmony", "bgv"), "backing"),
    (("vocal", "voice", "voix", "lead", "acapella", "a cappella"), "vocals"),
    (("instrumental", "instru", "karaoke", "accompaniment"), "instrumental"),
    (("drum", "percussion", "perc", "batterie", "kick", "snare"), "drums"),
    (("bass", "basse", "808", "sub"), "bass"),
]


def slugify(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "morceau"


def guess_role(filename):
    low = filename.lower()
    for words, role in ROLE_KEYWORDS:
        if any(w in low for w in words):
            return role
    return "music"


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def demucs_separate(path, workdir):
    """Sépare `path` en vocals/drums/bass/other. Renvoie {nom: Path}."""
    gpu = cuda_available()
    model = "htdemucs_ft" if gpu else "htdemucs"
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-d", "cuda" if gpu else "cpu",
           "--float32", "-o", str(workdir), str(path)]
    if not gpu:
        cmd[3:3] = ["-j", "4"]
    print(f"  séparation Demucs ({model}, {'GPU' if gpu else 'CPU'})…", flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    out = next((Path(workdir) / model).iterdir())
    return {p.stem: p for p in out.glob("*.wav")}


def collect_inputs(source, tmp):
    """Liste les fichiers audio d'une source (fichier, dossier ou zip)."""
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            z.extractall(tmp)
        source = Path(tmp)
    if source.is_dir():
        files = sorted(p for p in source.rglob("*") if p.suffix.lower() in AUDIO_EXT
                       and not p.name.startswith("."))
        if not files:
            raise SystemExit(f"Aucun fichier audio dans {source}")
        return files
    if source.suffix.lower() in AUDIO_EXT:
        return [source]
    raise SystemExit(f"Source non reconnue : {source}")


def prepare(source, song_dir):
    song_dir = Path(song_dir)
    stems_dir = song_dir / "stems"
    if stems_dir.exists():
        shutil.rmtree(stems_dir)
    stems_dir.mkdir(parents=True)
    entries, tracks = [], []

    with tempfile.TemporaryDirectory() as tmp:
        files = collect_inputs(source, Path(tmp) / "in")
        # Un seul fichier = morceau mixé ; sinon ce sont des stems. Un fichier « mix/full »
        # parmi les stems sert d'original.
        full = None
        if len(files) == 1:
            full, files = files[0], []
        else:
            for f in files:
                if re.search(r"\b(full|mix|master|original)\b", f.stem.lower()):
                    full = f
            files = [f for f in files if f != full]

        sr = load(full or files[0])[1]

        def add(name, role, audio):
            fname = f"{len(entries):02d}_{slugify(name)}.wav"
            save(stems_dir / fname, audio, sr, subtype="FLOAT")
            entries.append({"file": fname, "name": name, "role": role})
            tracks.append(audio)

        def add_separated(path, keep_vocals):
            parts = demucs_separate(path, Path(tmp) / "sep")
            roles = {"vocals": "vocals", "drums": "drums", "bass": "bass", "other": "music"}
            for key in ("vocals", "drums", "bass", "other"):
                audio = load(parts[key], sr)[0]
                role = roles[key] if (key != "vocals" or keep_vocals) else "music"
                label = {"vocals": "Voix", "drums": "Batterie", "bass": "Basse", "other": "Instruments"}[key]
                add(label if keep_vocals or key != "vocals" else "Résidu voix", role, audio)

        if not files:
            add_separated(full, keep_vocals=True)
        else:
            for f in files:
                role = guess_role(f.stem)
                if role == "instrumental":
                    add_separated(f, keep_vocals=False)
                else:
                    add(f.stem, role, load(f, sr)[0])

        n = max(t.shape[1] for t in tracks)
        if full:
            orig = load(full, sr)[0]
            n = max(n, orig.shape[1])
        else:
            orig = None
        # même longueur partout
        for e, t in zip(entries, tracks):
            if t.shape[1] < n:
                save(stems_dir / e["file"], np.pad(t, ((0, 0), (0, n - t.shape[1]))), sr, subtype="FLOAT")
        if orig is None:
            orig = sum(np.pad(t, ((0, 0), (0, n - t.shape[1]))) for t in tracks)
        else:
            # Demucs ne reconstitue pas l'original à l'échantillon près : le résidu
            # (souvent des aigus fins, de la reverb) rejoint les instruments.
            orig = np.pad(orig, ((0, 0), (0, n - orig.shape[1])))
            total = sum(np.pad(t, ((0, 0), (0, n - t.shape[1]))) for t in tracks)
            music = [i for i, e in enumerate(entries) if e["role"] == "music"]
            if music:
                i = music[-1]
                fixed = np.pad(tracks[i], ((0, 0), (0, n - tracks[i].shape[1]))) + (orig - total)
                save(stems_dir / entries[i]["file"], fixed, sr, subtype="FLOAT")
        save(song_dir / "original.wav", np.pad(orig, ((0, 0), (0, n - orig.shape[1]))), sr, subtype="FLOAT")

    (song_dir / "stems.json").write_text(json.dumps({"sr": sr, "stems": entries}, ensure_ascii=False, indent=1))
    return entries


def load_stems(song_dir):
    song_dir = Path(song_dir)
    meta = json.loads((song_dir / "stems.json").read_text())
    stems = []
    for e in meta["stems"]:
        audio, sr = load(song_dir / "stems" / e["file"])
        stems.append({**e, "audio": audio})
    return stems, meta["sr"]
