# -*- coding: utf-8 -*-
"""
STEP 2 — Split 80/20 + TF-IDF (Spyder-ready, dataset déjà nettoyé par step1)

Objectif :
    - Lire les textes propres dans data_clean/human et data_clean/ai
    - Construire un dataset équilibré : 100 humains + 100 IA
    - Split en train/test avec stratification (80% / 20%)
    - Transformer les textes en vecteurs TF-IDF (fit sur le train uniquement)
    - Sauvegarder toutes les matrices + métadonnées pour l'étape 3 (modèles)

Sorties dans outputs_step2/ :
    - X_train_tfidf.npz         (matrice sparse TF-IDF train)
    - X_test_tfidf.npz          (matrice sparse TF-IDF test)
    - y_train.npy, y_test.npy   (labels)
    - feature_names.json        (vocabulaire TF-IDF)
    - train_meta.csv            (path, label, word_count pour le train)
    - test_meta.csv             (idem pour le test)
    - summary_step2.json        (récap structuré)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# CHEMINS GLOBAUX
# ----------------------------------------------------------------------

# Racine du projet : .../projet-detec-ia
ROOT = Path(__file__).resolve().parents[1]

# Dossiers des textes NETTOYÉS (résultat de step1_clean_and_sort.py)
DATA_CLEAN_HUM = ROOT / "data_clean" / "human"
DATA_CLEAN_AI = ROOT / "data_clean" / "ai"

# Dossier de sortie de l'étape 2
OUT_DIR = ROOT / "outputs_step2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres du dataset équilibré et du split
TARGET_PER_CLASS = 100     # 100 IA + 100 humains
TEST_SIZE = 0.20           # 80/20
RANDOM_SEED = 42           # reproductible

# ----------------------------------------------------------------------
# Petits helpers
# ----------------------------------------------------------------------

WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def normalize_ws(text: str) -> str:
    """
    Normalise légèrement les espaces :
        - remplace \r\n par \n
        - compresse les espaces multiples en un seul
        - strip global
    (step1 a déjà fait le gros du nettoyage, ici c'est du polish.)
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # remplace les séquences d'espaces / tab par un espace
    text = re.sub(r"[ \t]+", " ", text)
    # retire espaces superflus en début/fin
    return text.strip()

def read_clean_texts(dirpath: Path, label: str) -> pd.DataFrame:
    """
    Lit tous les .txt d'un dossier nettoyé et retourne un DataFrame :
        columns = [path, label, text, words]
    """
    rows = []
    for p in sorted(dirpath.glob("*.txt")):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        txt = normalize_ws(raw)
        wc = len(WORD_RE.findall(txt))
        rows.append({
            "path": str(p),
            "label": label,
            "text": txt,
            "words": wc,
        })
    return pd.DataFrame(rows)

def sample_balanced_100(df: pd.DataFrame, target_per_class: int = 100,
                        seed: int = 42) -> pd.DataFrame:
    """
    Prend le DataFrame complet (human + ai) et :
        - vérifie qu'il y a au moins target_per_class par label
        - échantillonne aléatoirement exactement target_per_class
          'human' + target_per_class 'ai'
        - mélange le tout.
    """
    dfs = []
    for label in ["human", "ai"]:
        sub = df[df["label"] == label]
        n = len(sub)
        if n < target_per_class:
            raise SystemExit(
                f"❌ Pas assez de textes pour '{label}': {n} < {target_per_class}. "
                f"Ajoute des textes ou baisse TARGET_PER_CLASS."
            )
        dfs.append(sub.sample(n=target_per_class, random_state=seed))
    df_bal = pd.concat(dfs, ignore_index=True)
    # mélange global pour ne pas garder "human" puis "ai"
    df_bal = df_bal.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_bal

def show_top_means_per_class(vec: TfidfVectorizer, X, y, k: int = 15):
    """
    Calcule, pour chaque classe, la moyenne TF-IDF par terme
    et retourne un dict {label: [(term, mean_tfidf), ...]} pour les k top termes.
    """
    vocab = np.array(vec.get_feature_names_out())
    classes = np.unique(y)
    report = {}
    for c in classes:
        mask = (y == c)
        if not np.any(mask):
            continue
        Xc = X[mask]
        # moyenne sur toutes les lignes de la classe
        mean_vec = np.asarray(Xc.mean(axis=0)).ravel()
        top_idx = np.argsort(mean_vec)[::-1][:k]
        report[c] = list(zip(vocab[top_idx].tolist(),
                             [float(m) for m in mean_vec[top_idx]]))
    return report

def print_top_terms(report: dict):
    """
    Affiche joliment les top termes moyens par classe dans la console.
    """
    for label, items in report.items():
        print(f"\n🔎 Top termes moyens ({label} — train):")
        for term, val in items:
            print(f"  - {term: <30} {val:0.4f}")

# ----------------------------------------------------------------------
# Paramètres TF-IDF
# ----------------------------------------------------------------------

VEC_KW = dict(
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",  # évite tokens ridicules
    ngram_range=(1, 2),          # unigrams + bigrams
    min_df=2,                    # terme doit apparaître dans ≥ 2 docs
    max_df=0.90,                 # ignoré si dans > 90% des docs
    max_features=30000,          # plafond large
    sublinear_tf=True,           # log-scaling sur tf
    norm="l2",
    dtype=np.float32,
    stop_words="english",        # enlève is, on, that, the, in, of, etc.
)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print(f"🧭 Racine projet : {ROOT}")
    print(f"📂 Lecture des textes nettoyés dans :\n"
          f"   - {DATA_CLEAN_HUM}\n   - {DATA_CLEAN_AI}\n")

    if not DATA_CLEAN_HUM.exists() or not DATA_CLEAN_AI.exists():
        raise SystemExit("❌ Dossiers data_clean/human ou data_clean/ai introuvables. "
                         "Assure-toi que step1_clean_and_sort.py a bien tourné.")

    # 1) Lecture des fichiers nettoyés
    df_h = read_clean_texts(DATA_CLEAN_HUM, "human")
    df_a = read_clean_texts(DATA_CLEAN_AI, "ai")

    print("=== STATISTIQUES APRÈS STEP1 (DATA_CLEAN) ===")
    print(df_h.groupby("label")["words"].agg(
        count="count", mean="mean", min="min", max="max"
    ))
    print(df_a.groupby("label")["words"].agg(
        count="count", mean="mean", min="min", max="max"
    ))

    # Concat human + ai
    df_all = pd.concat([df_h, df_a], ignore_index=True)

    print("\n📊 Répartition globale (avant équilibrage 100/100) :")
    print(df_all["label"].value_counts())

    # 2) Dataset équilibré : 100 humains + 100 IA
    print(f"\n⚖️ Construction d'un dataset équilibré : {TARGET_PER_CLASS} human + {TARGET_PER_CLASS} ai")
    df_bal = sample_balanced_100(df_all,
                                 target_per_class=TARGET_PER_CLASS,
                                 seed=RANDOM_SEED)

    print("\n📊 Répartition après équilibrage (doit être 100/100) :")
    print(df_bal["label"].value_counts())

    print("\n📈 Stats mots (dataset équilibré) :")
    print(df_bal.groupby("label")["words"].agg(
        count="count", mean="mean", min="min", max="max"
    ))

    # 3) Split train/test 80/20, stratifié
    print(f"\n🔀 Split train/test avec stratification (test_size={TEST_SIZE:.2f}) ...")
    X_text = df_bal["text"].tolist()
    y = df_bal["label"].to_numpy()
    meta = df_bal[["path", "label", "words"]].copy()

    X_train_text, X_test_text, y_train, y_test, meta_train, meta_test = train_test_split(
        X_text,
        y,
        meta,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    print("\n📊 Répartition après split :")
    print(f"  - Train : {len(y_train)} textes "
          f"(human={np.sum(y_train=='human')}, ai={np.sum(y_train=='ai')})")
    print(f"  - Test  : {len(y_test)} textes "
          f"(human={np.sum(y_test=='human')}, ai={np.sum(y_test=='ai')})")

    # 4) Vectorisation TF-IDF (fit sur TRAIN uniquement)
    print("\n🧮 Vectorisation TF-IDF sur le TRAIN (fit) puis transformation du TEST ...")
    vec = TfidfVectorizer(**VEC_KW)
    X_train = vec.fit_transform(X_train_text)
    X_test = vec.transform(X_test_text)

    feature_names = vec.get_feature_names_out()
    vocab_size = len(feature_names)

    print(f"✅ TF-IDF terminé.")
    print(f"   - X_train shape : {X_train.shape}")
    print(f"   - X_test  shape : {X_test.shape}")
    print(f"   - Taille du vocabulaire : {vocab_size} termes")

    # 5) Top termes moyens par classe sur le TRAIN
    print("\n📌 Analyse qualitative : top termes moyens par classe (TRAIN seulement)")
    top_means = show_top_means_per_class(vec, X_train, y_train, k=15)
    print_top_terms(top_means)

    # 6) Sauvegardes (matrices + labels + méta + résumé)
    print(f"\n💾 Sauvegarde des matrices et métadonnées dans : {OUT_DIR}")

    # Matrices sparse TF-IDF
    sparse.save_npz(OUT_DIR / "X_train_tfidf.npz", X_train)
    sparse.save_npz(OUT_DIR / "X_test_tfidf.npz", X_test)

    # Labels
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "y_test.npy", y_test)

    # Vocabulaire
    with open(OUT_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names.tolist(), f, ensure_ascii=False, indent=2)

    # Meta (utile pour tracer plus tard, retrouver les textes originaux, etc.)
    meta_train.to_csv(OUT_DIR / "train_meta.csv", index=False)
    meta_test.to_csv(OUT_DIR / "test_meta.csv", index=False)

    # Résumé structuré pour le rapport
    summary = {
        "paths": {
            "root": str(ROOT),
            "data_clean_human": str(DATA_CLEAN_HUM),
            "data_clean_ai": str(DATA_CLEAN_AI),
            "output_dir": str(OUT_DIR),
        },
        "settings": {
            "target_per_class": int(TARGET_PER_CLASS),
            "test_size": float(TEST_SIZE),
            "random_seed": int(RANDOM_SEED),
        },
        "counts": {
            "clean_total": int(len(df_all)),
            "clean_per_label": {
                "human": int(len(df_h)),
                "ai": int(len(df_a)),
            },
            "balanced_per_label": {
                "human": int(np.sum(df_bal["label"] == "human")),
                "ai": int(np.sum(df_bal["label"] == "ai")),
            },
            "train": int(len(y_train)),
            "test": int(len(y_test)),
        },
        "tfidf": {
            "vocab_size": int(vocab_size),
            "ngram_range": list(VEC_KW["ngram_range"]),
            "min_df": int(VEC_KW["min_df"]),
            "max_df": float(VEC_KW["max_df"]),
            "max_features": int(VEC_KW["max_features"]),
            "stop_words": "english",
            "token_pattern": VEC_KW["token_pattern"],
            "norm": VEC_KW["norm"],
            "sublinear_tf": bool(VEC_KW["sublinear_tf"]),
        },
        "top_terms": {
            label: [
                {"term": t, "mean_tfidf": float(v)} for t, v in items
            ]
            for label, items in top_means.items()
        },
    }

    (OUT_DIR / "summary_step2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== RÉSUMÉ ÉTAPE 2 (TF-IDF + SPLIT) ===")
    print(f"- Dataset équilibré : {TARGET_PER_CLASS} human + {TARGET_PER_CLASS} ai")
    print(f"- Split : {len(y_train)} train / {len(y_test)} test (80/20)")
    print(f"- Vocabulaire TF-IDF : {vocab_size} termes")
    print(f"- Dossier de sortie : {OUT_DIR}")
    print("Étape 2 terminée ✅ — prête pour l'étape 3 (modèles ML).")

if __name__ == "__main__":
    main()