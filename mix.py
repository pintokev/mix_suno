"""Remixe un morceau Suno.

Usage :
  .venv/bin/python mix.py <source> [--name slug] [--preset boombap-reggae] [--ref ref.wav] [--clip dossier]
  .venv/bin/python mix.py <slug> [--set vocals.reverb=0.2 ...]      # relance sur un morceau déjà importé

<source> : un WAV mixé (séparé par Demucs), un dossier ou un zip de stems Suno.
Tout est rangé dans songs/<slug>/ ; les sorties dans songs/<slug>/out/ :
  mix_distrokid.wav (24-bit, -14 LUFS, -1 dBTP), mix_tiktok.wav (plus fort),
  original_meme_volume.wav, ab_original_puis_mix.wav, report.json
--set modifie un réglage (enregistré dans songs/<slug>/settings.json), ex. levels.vocals=2
"""
import argparse
import json
import sys
import time
from pathlib import Path

from mixer import pipeline
from mixer.engine import list_presets
from mixer.stems import prepare, slugify


def parse_value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def apply_set(over, assignment):
    path, _, value = assignment.partition("=")
    keys = path.split(".")
    node = over
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = parse_value(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="fichier/dossier/zip à importer, ou slug d'un morceau déjà importé")
    ap.add_argument("--name", help="slug du morceau (défaut : nom du fichier)")
    ap.add_argument("--preset", help=f"préset : {', '.join(list_presets())}")
    ap.add_argument("--ref", help="morceau de référence pour le mastering (Matchering) ; 'none' pour l'enlever")
    ap.add_argument("--set", action="append", default=[], metavar="clé=valeur")
    ap.add_argument("--clip", help="dossier create_video_music où copier le mix")
    ap.add_argument("--reimport", action="store_true", help="refaire l'import / la séparation")
    args = ap.parse_args()

    src = Path(args.source).expanduser()
    if src.exists():
        slug = args.name or slugify(src.stem)
        sdir = pipeline.song_dir(slug)
        if args.reimport or not (sdir / "stems.json").exists():
            sdir.mkdir(parents=True, exist_ok=True)
            print(f"Import de {src.name} → songs/{slug}/")
            t = time.time()
            entries = prepare(src, sdir)
            print(f"  {len(entries)} pistes : " + ", ".join(f"{e['name']} ({e['role']})" for e in entries)
                  + f" [{time.time() - t:.0f} s]")
    else:
        slug = args.source
        sdir = pipeline.song_dir(slug)
        if not (sdir / "stems.json").exists():
            sys.exit(f"Ni fichier ni morceau importé : {args.source}")

    over = pipeline.read_overrides(sdir)
    if args.preset:
        over["preset"] = args.preset
    if args.ref:
        over["reference"] = None if args.ref == "none" else str(Path(args.ref).expanduser().resolve())
    if args.clip:
        over["clip_dir"] = str(Path(args.clip).expanduser().resolve())
    for a in args.set:
        apply_set(over, a)
    pipeline.write_overrides(sdir, over)

    t = time.time()
    pipeline.export(slug, clip_dir=over.get("clip_dir"))
    print(f"Terminé en {time.time() - t:.0f} s → {sdir / 'out'}")


if __name__ == "__main__":
    main()
