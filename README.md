# mix_suno

Remix / mastering des morceaux Suno : séparation des pistes (Demucs ou stems Suno), traitement
par piste (voix anti-« IA », punch, équilibre), master DistroKid (−14 LUFS, −1 dBTP) + version
TikTok, extrait A/B, interface web à curseurs.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
.venv/bin/python mix.py <wav|zip|dossier> --name <slug> --preset boombap-reggae   # mix complet
.venv/bin/python server.py                                                        # http://127.0.0.1:8765
.venv/bin/python suno_fetch.py "<titre|url>" [--confirm] [--stems]                # WAV depuis Suno
```

- `mixer/stems.py` import et séparation · `mixer/engine.py` traitements par rôle, master, volume final
- `mixer/dsp.py` effets (de-esser, saturation, doublage, transitoires, limiteur true peak)
- `mixer/presets/*.json` présets de genre (par-dessus `default.json`)
- `songs/<slug>/` un dossier par morceau (non versionné)
