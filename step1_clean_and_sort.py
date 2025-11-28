# -*- coding: utf-8 -*-
"""
STEP 1 — CLEAN & SORT (version unifiée corrigée)
-----------------------------------------------

But :
- Lire TOUS les .txt dans data/raw_texts/
- Détecter le label (= humain ou IA) à partir du nom de fichier
- Nettoyer le texte (espaces, en-têtes, boilerplate, IMRaD, etc.)
- Compter les mots avant/après nettoyage
- Filtrer :
    * trop court  : < MIN_WORDS  (skipped_short)
    * trop long   : > MAX_WORDS  (skipped_long)
- Détecter les doublons (exactement même texte nettoyé)
- Sauvegarder :
    * data_clean/human/*.txt
    * data_clean/ai/*.txt
    * data_clean/unknown/*.txt (si on ne sait pas)
    * data_clean/duplicates/*.txt (doublons)
- Produire un CSV : outputs_step1/clean_dataset_overview.csv
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# -----------------------------
# Réglages généraux
# -----------------------------
MIN_WORDS = 200     # longueur minimale voulue
MAX_WORDS = 320    # longueur maximale voulue

ROOT = Path(__file__).resolve().parents[1]  # .../projet-detec-ia
RAW_DIR = ROOT / "data" / "raw_texts"
CLEAN_ROOT = ROOT / "data_clean"
OUT_DIR = ROOT / "outputs_step1"

CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Regex / nettoyage
# -----------------------------

# Lignes IMRaD seules (Introduction / Methods / Results / Discussion)
IMRAD_LINE_RE = re.compile(r"(?mi)^(?:\s*(introduction|methods|results|discussion)\s*)$")

# En-têtes classiques dans tes fichiers humains
HEADER_LINES_RE = re.compile(
    r"(?mi)^(abstracth_\d+|title:.*|authors:.*|journal:.*|doi:.*|keywords:.*)$"
)

# Boilerplate générique à supprimer
BOILER_PLATE_TERMS = [
    # sections / meta
    "introduction", "methods", "results", "discussion",
    "background", "objective", "objectives", "aim", "aims",
    "conclusion", "conclusions", "materials and methods",
    "method", "this study", "the present study", "this review",
    "we reviewed",
    # bibliographie / meta
    "doi", "title", "keywords", "keywords diabetes", "diabetes human",
    "journal", "study", "studies", "article", "manuscript",
    "research", "scientific", "disease", "diseases",
    # IA génériques
    "remains", "accurate", "adherence", "this analysis", "our study",
    "findings suggest", "authors"
]

WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


# -----------------------------
# Helpers
# -----------------------------
def detect_label_from_filename(path: Path) -> str:
    """
    Détecte grossièrement le label à partir du nom de fichier.
    - contient 'human' ou 'humain' -> 'human'
    - contient 'ai' ou 'ia'        -> 'ai'
    - sinon                        -> 'unknown'
    """
    name = path.name.lower()
    if "human" in name or "humain" in name:
        return "human"
    # on tolère plusieurs formes pour l'IA
    if re.search(r"\bai\b", name) or "ia_" in name or "ia-" in name or name.startswith("ia"):
        return "ai"
    return "unknown"


def normalize_ws(text: str) -> str:
    """
    Normalise les fins de lignes et compresse les lignes vides.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out = []
    blank = 0
    for ln in lines:
        if ln.strip() == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln.rstrip())
    t = "\n".join(out).strip()
    # espaces multiples -> simple
    t = re.sub(r"[ \t]+", " ", t)
    return t


def strip_headers_and_imrad(text: str) -> str:
    """
    Supprime :
    - lignes d'en-têtes type "AbstractH_01", "Title:", "Journal:", "DOI:", "Keywords:"
    - lignes IMRaD isolées
    - boilerplate générique
    """
    # suppression des headers sur lignes entières
    lines = text.split("\n")
    kept_lines = []
    for ln in lines:
        if HEADER_LINES_RE.match(ln.strip()):
            continue
        kept_lines.append(ln)
    t = "\n".join(kept_lines)

    # supprimer lignes IMRaD seules
    t = IMRAD_LINE_RE.sub(" ", t)

    # supprimer quelques boilerplates "vides"
    for term in BOILER_PLATE_TERMS:
        t = re.sub(rf"(?i)\b{re.escape(term)}\b", " ", t)

    # compactage final
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def clean_text(raw: str) -> str:
    """
    Pipeline de nettoyage complet :
    - normalisation espaces
    - suppression headers + IMRaD + boilerplate
    """
    t = normalize_ws(raw)
    t = strip_headers_and_imrad(t)
    # on s'assure encore une fois qu'il n'y a pas de gros trous
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def text_hash(text: str) -> str:
    """
    Hash SHA1 du texte nettoyé pour détecter les doublons exacts.
    """
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------
# Main
# -----------------------------
def main():
    print(f"🧭 Racine projet : {ROOT}")
    print(f"📂 Lecture des fichiers bruts dans : {RAW_DIR}")

    if not RAW_DIR.exists():
        raise SystemExit(f"❌ Dossier introuvable : {RAW_DIR}")

    all_txt = sorted(RAW_DIR.glob("*.txt"))
    if not all_txt:
        raise SystemExit("❌ Aucun fichier .txt trouvé dans data/raw_texts/. "
                         "Place tes textes humains + IA dans ce dossier.")

    rows = []
    hash_to_first = {}  # hash -> index première apparition
    duplicates_info = []

    for idx, p in enumerate(all_txt, start=1):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        raw_norm = normalize_ws(raw)
        raw_words = len(WORD_RE.findall(raw_norm))

        label = detect_label_from_filename(p)

        # Nettoyage complet
        cleaned = clean_text(raw_norm)
        clean_words = len(WORD_RE.findall(cleaned))

        # Déterminer statut par longueur
        if clean_words < MIN_WORDS:
            status = "skipped_short"
            duplicate_of = ""
        elif clean_words > MAX_WORDS:
            status = "skipped_long"
            duplicate_of = ""
        else:
            # candidat "kept" ou "duplicate"
            h = text_hash(cleaned)
            if h in hash_to_first:
                status = "duplicate"
                duplicate_of = rows[hash_to_first[h]]["filename_raw"]
            else:
                status = "kept"
                duplicate_of = ""
                hash_to_first[h] = len(rows)  # index dans rows

        row = {
            "filename_raw": p.name,
            "path_raw": str(p),
            "label_detected": label,
            "raw_words": raw_words,
            "clean_words": clean_words,
            "status": status,
            "duplicate_of": duplicate_of,
            "clean_text": cleaned,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # -------------------------
    # Stats globales
    # -------------------------
    print("\n=== STATISTIQUES BRUTES ===")
    if not df.empty:
        by_label = (df
                    .groupby("label_detected")
                    .agg(
                        raw_words_count=("raw_words", "count"),
                        raw_words_mean=("raw_words", "mean"),
                        raw_words_min=("raw_words", "min"),
                        raw_words_max=("raw_words", "max"),
                        clean_words_mean=("clean_words", "mean"),
                        clean_words_min=("clean_words", "min"),
                        clean_words_max=("clean_words", "max"),
                    ))
        print("Par label détecté :")
        print(by_label)

        by_status = df["status"].value_counts()
        print("\nPar status :")
        print(by_status)

    # -------------------------
    # Création des dossiers de sortie (CORRIGÉ : 'ai' et pas 'ia')
    # -------------------------
    for sub in ["human", "ai", "unknown", "duplicates"]:
        out_dir = CLEAN_ROOT / sub
        out_dir.mkdir(parents=True, exist_ok=True)
    print("\n📦 Dossiers de sortie assurés :")
    print(f"  - {CLEAN_ROOT / 'human'}")
    print(f"  - {CLEAN_ROOT / 'ai'}")
    print(f"  - {CLEAN_ROOT / 'unknown'}")
    print(f"  - {CLEAN_ROOT / 'duplicates'}")

    # -------------------------
    # Sauvegarde des textes nettoyés
    # -------------------------
    counters = defaultdict(int)
    clean_paths = []

    for _, row in df.iterrows():
        status = row["status"]
        label = row["label_detected"]
        cleaned = row["clean_text"]
        fname_raw = row["filename_raw"]

        if status == "kept":
            # dossier selon label
            if label not in {"human", "ai"}:
                out_dir = CLEAN_ROOT / "unknown"
                label_for_name = "unknown"
            else:
                out_dir = CLEAN_ROOT / label  # 'human' ou 'ai'
                label_for_name = label

            counters[label_for_name] += 1
            new_name = f"{label_for_name}_{counters[label_for_name]:03d}.txt"
            out_path = out_dir / new_name
            out_path.write_text(cleaned + "\n", encoding="utf-8")
            clean_paths.append((fname_raw, str(out_path)))

        elif status == "duplicate":
            # on sauvegarde aussi les doublons dans un dossier à part
            out_dir = CLEAN_ROOT / "duplicates"
            out_path = out_dir / fname_raw
            out_path.write_text(cleaned + "\n", encoding="utf-8")

    # Ajouter chemin nettoyé dans le DataFrame
    path_map = dict(clean_paths)
    df["path_clean"] = df["filename_raw"].map(path_map).fillna("")

    # -------------------------
    # Sauvegarde CSV + JSON
    # -------------------------
    csv_path = OUT_DIR / "clean_dataset_overview.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📄 CSV récapitulatif : {csv_path}")

    summary = {
        "counts_by_label": df["label_detected"].value_counts().to_dict(),
        "counts_by_status": df["status"].value_counts().to_dict(),
        "kept_by_label": df[df["status"] == "kept"]["label_detected"].value_counts().to_dict(),
        "paths": {
            "raw_dir": str(RAW_DIR),
            "clean_root": str(CLEAN_ROOT),
            "out_dir": str(OUT_DIR),
        },
        "params": {
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
        },
    }
    (OUT_DIR / "clean_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # -------------------------
    # Affichage des doublons
    # -------------------------
    df_dup = df[df["status"] == "duplicate"]
    if not df_dup.empty:
        print("\n=== DOUBLONS (status='duplicate') ===")
        print(f"{len(df_dup)} doublons trouvés. Exemples :")
        print(df_dup[["filename_raw", "duplicate_of"]].head(10))
    else:
        print("\n✅ Aucun doublon détecté (après nettoyage).")

    # -------------------------
    # Résumé final console
    # -------------------------
    print("\n=== RÉSUMÉ FINAL ===")
    for k, v in summary["kept_by_label"].items():
        print(f"- {k} kept : {v}")
    print("Terminé ✅")


if __name__ == "__main__":
    main()