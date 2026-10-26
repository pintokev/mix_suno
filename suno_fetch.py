"""Récupère le WAV d'un morceau Suno via le navigateur piloté de create_suno (CDP :9222).

Usage :
  .venv/bin/python suno_fetch.py <url|id|titre>              # prépare, affiche le quota, ne télécharge PAS
  .venv/bin/python suno_fetch.py <url|id|titre> --confirm    # clique « Unlock & Download » (consomme le quota)
  .venv/bin/python suno_fetch.py <url|id|titre> --stems      # ouvre « Stems & MIDI › Manage » et décrit les options

Un titre est cherché parmi les morceaux affichés sur l'onglet Suno ouvert (le plus récent d'abord).
Le fichier arrive dans songs/<slug>/source/ ; lancer ensuite mix.py dessus.
Sans --confirm, rien n'est consommé : la fenêtre est refermée.
"""
import argparse
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from mixer.pipeline import SONGS
from mixer.stems import slugify

CDP = "http://127.0.0.1:9222"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def find_song(browser, query):
    """Renvoie (url, titre) à partir d'une URL, d'un id ou d'un titre."""
    m = UUID.search(query)
    if m:
        return f"https://suno.com/song/{m.group(0)}", None
    for ctx in browser.contexts:
        for pg in ctx.pages:
            if "suno.com" not in pg.url:
                continue
            links = pg.evaluate("""() => [...document.querySelectorAll('a[href*="/song/"]')]
                .map(a => ({h: a.getAttribute('href'), t: (a.innerText || '').trim()}))""")
            for link in links:
                if link["t"] and slugify(link["t"]) == slugify(query):
                    return "https://suno.com" + link["h"].split("?")[0], link["t"]
    sys.exit(f"Morceau « {query} » introuvable dans les onglets Suno ouverts : donne son URL.")


def open_download_dialog(page):
    page.get_by_role("button", name="More options").first.click()
    page.wait_for_timeout(800)
    page.get_by_role("menuitem", name="Download").first.click()
    page.wait_for_timeout(1500)
    dialog = page.get_by_role("dialog")
    if not dialog.count():
        sys.exit("Fenêtre de téléchargement introuvable (l'interface Suno a peut-être changé).")
    return dialog.first


def quota(dialog):
    text = dialog.inner_text()
    m = re.search(r"Downloads\s*\n?\s*(\d+)", text)
    r = re.search(r"Refreshes\s*\D*?([\d/]{5,})", text)
    return (m.group(1) if m else "?"), (r.group(1) if r else "?")


def close(page):
    for _ in range(2):
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song")
    ap.add_argument("--confirm", action="store_true", help="télécharger pour de vrai (consomme le quota)")
    ap.add_argument("--stems", action="store_true", help="explorer l'option Stems & MIDI (sans rien valider)")
    ap.add_argument("--name", help="slug du morceau (défaut : titre Suno)")
    args = ap.parse_args()

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(CDP)
        except Exception:
            sys.exit("Navigateur Suno non lancé : cd ~/Agents/claude/create_suno && .venv/bin/python launch.py")
        url, title = find_song(browser, args.song)
        ctx = browser.contexts[0]
        page = ctx.new_page()
        try:
            try:
                page.goto(url, wait_until="commit")
            except Exception:
                pass  # Suno interrompt parfois la navigation initiale (routeur client)
            page.wait_for_timeout(6000)
            title = title or page.evaluate(
                "document.querySelector('meta[property=\"og:title\"]')?.content") or page.title().split(" by ")[0]
            slug = args.name or slugify(title)
            dialog = open_download_dialog(page)
            left, refresh = quota(dialog)
            print(f"Morceau : {title} ({url})")
            print(f"Quota de téléchargements restant : {left} (renouvelé le {refresh})")

            if args.stems:
                dialog.get_by_text("Manage").first.click()
                page.wait_for_timeout(2000)
                print("Options « Stems & MIDI » affichées (rien n'a été validé) :")
                for line in page.get_by_role("dialog").last.inner_text().splitlines():
                    if line.strip():
                        print("  " + line.strip())
                shot = SONGS / slug / "suno_stems.png"
                shot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(shot))
                print(f"Capture : {shot}")
                close(page)
                return

            dialog.get_by_text("WAV", exact=True).first.click()
            page.wait_for_timeout(500)
            button = dialog.get_by_role("button", name=re.compile(r"Download", re.I)).last
            label = button.inner_text().strip()
            if not args.confirm:
                print(f"Prêt : WAV sélectionné, bouton « {label} ».")
                print("Rien n'a été téléchargé. Relance avec --confirm pour télécharger "
                      + ("(consomme 1 téléchargement du quota)." if "Unlock" in label else "."))
                close(page)
                sys.exit(2)

            dest = SONGS / slug / "source"
            dest.mkdir(parents=True, exist_ok=True)
            before = set(dest.iterdir())
            cdp = browser.new_browser_cdp_session()
            cdp.send("Browser.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(dest)})
            button.click()
            print("Téléchargement…", flush=True)
            deadline = time.time() + 180
            while time.time() < deadline:
                new = [f for f in set(dest.iterdir()) - before if not f.name.endswith(".crdownload")]
                if new and not any(f.name.endswith(".crdownload") for f in dest.iterdir()):
                    print(f"OK : {new[0]}")
                    print(f"Mix : .venv/bin/python mix.py \"{new[0]}\" --name {slug}")
                    return
                time.sleep(1)
            sys.exit("Pas de fichier reçu après 3 min (Suno prépare peut-être encore le WAV : réessaie).")
        finally:
            page.close()


if __name__ == "__main__":
    main()
