# ==========================================================
#  fetch_human_abstracts_epmc.py
#  Télécharge 30 abstracts humains réels (Europe PMC)
#  Adam Malila — Projet Détection IA / Data Science
# ==========================================================

import os
import csv
import requests
from pathlib import Path

# -----------------------------
# 🔧 Paramètres de base
# -----------------------------
BASE = Path("/Users/adamdavidmalila/Library/Mobile Documents/com~apple~CloudDocs/projet-detec-ia")
OUT_DIR = BASE / "data" / "humain" / "diabete"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_GLOBAL = OUT_DIR / "diabetes_humain_abstracts.txt"
META_CSV = OUT_DIR / "diabetes_humain_meta.csv"

API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
MAX_ABS = 30  # nombre d'abstracts à récupérer

# 🔍 Requête Europe PMC :
# On filtre sur le mot "diabetes", période < 2020, abstracts disponibles
# On retire OPEN_ACCESS:Y pour élargir la recherche
QUERY = '(TITLE:"diabetes" OR ABSTRACT:"diabetes") AND PUB_YEAR:[1990 TO 2019] AND HAS_ABSTRACT:Y'

# -----------------------------
# 🧠 Fonctions utilitaires
# -----------------------------
def normalize_text(s):
    """Nettoie les espaces et caractères parasites."""
    return " ".join((s or "").split())

def fetch_epmc_articles(max_abs=MAX_ABS):
    """Récupère les articles depuis Europe PMC"""
    print("🔎 Europe PMC — recherche d’articles humains (avant 2020)…")
    params = {
        "query": QUERY,
        "format": "json",
        "pageSize": 100,   # On prend large puis on filtre
        "sort": "cited desc"
    }
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = []
    all_results = data.get("resultList", {}).get("result", [])

    print(f"   ~{len(all_results)} résultats initiaux")

    for rec in all_results:
        abstract = rec.get("abstractText")
        if not abstract:
            continue

        pub_type = (rec.get("pubType") or "").lower()
        if "preprint" in pub_type:
            continue  # exclut preprints

        title = normalize_text(rec.get("title", ""))
        journal = normalize_text(rec.get("journalTitle", ""))
        year = rec.get("pubYear", "")
        doi = rec.get("doi", "")
        pmid = rec.get("pmid", "")
        pmcid = rec.get("pmcid", "")
        src = rec.get("source", "")
        url = f"https://europepmc.org/abstract/{src.lower()}/{pmid}" if pmid else ""

        abstract = normalize_text(abstract)

        results.append({
            "title": title,
            "journal": journal,
            "year": year,
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "source": src,
            "url": url,
            "abstract": abstract
        })

        if len(results) >= max_abs:
            break

    return results


# -----------------------------
# 💾 Sauvegarde fichiers
# -----------------------------
def save_results(results):
    if not results:
        print("❌ Aucun abstract valide trouvé.")
        return

    # Fichier texte global
    with open(TXT_GLOBAL, "w", encoding="utf-8") as fw:
        for i, r in enumerate(results, start=1):
            fw.write(f"AbstractH_{i:02d}\n")
            fw.write(f"Title: {r['title']}\n")
            fw.write(f"Journal: {r['journal']} ({r['year']})\n")
            if r['doi']:
                fw.write(f"DOI: {r['doi']}\n")
            if r['url']:
                fw.write(f"URL: {r['url']}\n")
            fw.write("\n")
            fw.write(r["abstract"].strip() + "\n\n")

    # Fichier CSV de métadonnées
    with open(META_CSV, "w", encoding="utf-8", newline="") as fcsv:
        w = csv.writer(fcsv, delimiter=";")
        w.writerow(["idx", "title", "journal", "year", "doi", "pmid", "pmcid", "source", "url", "words"])
        for i, r in enumerate(results, start=1):
            w.writerow([
                i,
                r["title"],
                r["journal"],
                r["year"],
                r["doi"],
                r["pmid"],
                r["pmcid"],
                r["source"],
                r["url"],
                len(r["abstract"].split())
            ])

    print(f"\n✅ Fichier global créé : {TXT_GLOBAL}")
    print(f"✅ Métadonnées enregistrées : {META_CSV}")
    print("ℹ️ Tous les abstracts proviennent d’articles scientifiques publiés avant 2020.")


# -----------------------------
# 🚀 Main
# -----------------------------
def main():
    results = fetch_epmc_articles()
    save_results(results)
    if len(results) < 5:
        print("⚠️ Peu d’articles trouvés. Essaie d’élargir la période ou d’enlever certains filtres.")
    else:
        print(f"📄 {len(results)} abstracts humains sauvegardés avec succès !")


if __name__ == "__main__":
    main()