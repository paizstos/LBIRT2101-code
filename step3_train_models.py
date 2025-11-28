#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:43:54 2025

@author: adamdavidmalila
"""

"""
STEP 3 — Entraînement et sélection de modèles (train uniquement)

Ce script :
  - charge X_train, y_train depuis outputs_step2
  - définit plusieurs modèles de classification texte (IA vs humain)
  - effectue une validation croisée (StratifiedKFold, k=5)
  - compare les modèles (accuracy + F1_macro)
  - entraîne le MEILLEUR modèle sur tout le train
  - sauvegarde :
      * best_model.joblib
      * models_cv_results.json
      * models_cv_results.csv

À lancer dans Spyder :
%runfile '.../code/step3_train_models.py' --wdir
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score

import joblib

# ----------------------------------------------------------------------
# CHEMINS GLOBAUX
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]          # .../projet-detec-ia
OUT_STEP2 = ROOT / "outputs_step2"                  # là où step2 a écrit X/y
OUT_STEP3 = ROOT / "outputs_step3"                  # nouveaux résultats
OUT_STEP3.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# FONCTIONS UTILITAIRES
# ----------------------------------------------------------------------

def load_step2_train():
    """
    Charge X_train, y_train et quelques métadonnées de l'étape 2.
    On suppose que step2_tfidf_split.py a créé ces fichiers :
      - X_train_tfidf.npz
      - y_train.npy
      - train_meta.csv (optionnel pour info)
    """
    X_train_path = OUT_STEP2 / "X_train_tfidf.npz"
    y_train_path = OUT_STEP2 / "y_train.npy"
    meta_path    = OUT_STEP2 / "train_meta.csv"

    if not X_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError(
            f"Impossible de trouver X_train / y_train dans {OUT_STEP2}.\n"
            "Assure-toi d'avoir exécuté step2_tfidf_split.py avant."
        )

    print(f"📂 Chargement X_train depuis : {X_train_path}")
    print(f"📂 Chargement y_train depuis : {y_train_path}")

    X_train = sparse.load_npz(X_train_path)
    y_train = np.load(y_train_path, allow_pickle=True)

    meta = None
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        print(f"📄 train_meta.csv trouvé ({len(meta)} lignes)")

    print(f"   → X_train shape : {X_train.shape}")
    print(f"   → y_train shape : {y_train.shape}")
    return X_train, y_train, meta


def get_models():
    """
    Définit plusieurs modèles classiques pour texte :
      - LogisticRegression (baseline très solide)
      - LinearSVC (SVM linéaire)
      - MultinomialNB (Naive Bayes)
      - RandomForestClassifier (pour comparer avec un modèle non-linéaire)
    """
    models = {
        "logreg": LogisticRegression(
            max_iter=2000,
            solver="liblinear",   # robuste pour petit dataset, binaire
            class_weight="balanced"
        ),
        "svm_linear": LinearSVC(
            C=1.0,
            class_weight="balanced"
        ),
        "nb_multinomial": MultinomialNB(),
        "rf_ensemble": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced_subsample"
        )
    }
    return models


def evaluate_models_cv(X, y, models, n_splits=5):
    """
    Effectue une validation croisée (StratifiedKFold) pour chaque modèle.

    Retourne :
      - results_df : DataFrame avec mean/std accuracy/F1 pour chaque modèle
      - raw_results : dict avec les scores détaillés par fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    raw_results = {}

    print("\n🔁 Validation croisée (k-fold) en cours...")
    for name, model in models.items():
        print(f"\n➡️ Modèle : {name}")
        # On demande accuracy + f1_macro sur chaque fold
        cv_scores = cross_validate(
            model,
            X,
            y,
            cv=skf,
            scoring=["accuracy", "f1_macro"],
            return_train_score=False,
            n_jobs=-1
        )

        acc_mean = float(cv_scores["test_accuracy"].mean())
        acc_std  = float(cv_scores["test_accuracy"].std())
        f1_mean  = float(cv_scores["test_f1_macro"].mean())
        f1_std   = float(cv_scores["test_f1_macro"].std())

        print(f"   - accuracy (moy ± std) : {acc_mean:.3f} ± {acc_std:.3f}")
        print(f"   - F1_macro (moy ± std) : {f1_mean:.3f} ± {f1_std:.3f}")

        rows.append({
            "model": name,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std
        })

        raw_results[name] = {
            "test_accuracy": list(map(float, cv_scores["test_accuracy"])),
            "test_f1_macro": list(map(float, cv_scores["test_f1_macro"]))
        }

    results_df = pd.DataFrame(rows).sort_values(by="acc_mean", ascending=False)
    return results_df, raw_results


def choose_best_model(results_df):
    """
    Choisit le meilleur modèle selon :
      - accuracy moyenne
      - F1_macro (en critère secondaire si égalité)
    """
    # On trie d'abord par acc_mean, puis par f1_mean
    sorted_df = results_df.sort_values(
        by=["acc_mean", "f1_mean"],
        ascending=False
    ).reset_index(drop=True)

    best_row = sorted_df.iloc[0]
    best_name = best_row["model"]
    print("\n🏆 Meilleur modèle d'après la CV :")
    print(sorted_df.head())
    print(f"\n   → Sélectionné : {best_name}")
    return best_name, sorted_df


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print(f"🧭 Racine projet : {ROOT}")
    print("=== STEP 3 — Entraînement + sélection de modèles ===")

    # 1) Charger les données de train
    X_train, y_train, meta = load_step2_train()

    # 2) Définir les modèles
    models = get_models()

    # 3) Validation croisée
    results_df, raw_cv = evaluate_models_cv(X_train, y_train, models, n_splits=5)

    # 4) Choisir le meilleur modèle
    best_name, sorted_df = choose_best_model(results_df)
    best_model = models[best_name]

    # 5) Entraîner le meilleur modèle sur TOUT le train
    print(f"\n🚂 Entraînement final du modèle '{best_name}' sur tout le TRAIN...")
    best_model.fit(X_train, y_train)

    # 6) Évaluer la perf sur le train (juste pour voir s’il overfit déjà)
    y_train_pred = best_model.predict(X_train)
    acc_train = float(accuracy_score(y_train, y_train_pred))
    f1_train = float(f1_score(y_train, y_train_pred, average="macro"))

    print(f"\n📌 Perf sur TRAIN complet ({best_name}) :")
    print(f"   - accuracy_train : {acc_train:.3f}")
    print(f"   - F1_macro_train : {f1_train:.3f}")

    # 7) Sauvegarder le modèle et les résultats
    model_path = OUT_STEP3 / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\n💾 Modèle sauvegardé dans : {model_path}")

    # Résultats détaillés
    results_csv_path = OUT_STEP3 / "models_cv_results.csv"
    sorted_df.to_csv(results_csv_path, index=False)

    results_json_path = OUT_STEP3 / "models_cv_results.json"
    results_payload = {
        "cv_results": sorted_df.to_dict(orient="records"),
        "raw_folds": raw_cv,
        "best_model_name": best_name,
        "train_performance": {
            "accuracy_train": acc_train,
            "f1_macro_train": f1_train
        }
    }
    results_json_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    print(f"📄 Résultats CV (triés) sauvegardés dans : {results_csv_path}")
    print(f"📄 Résumé JSON sauvegardé dans      : {results_json_path}")

    print("\n✅ STEP 3 (train + sélection) terminée.")
    print("   → Tu peux maintenant lancer step3_evaluate_best.py pour évaluer sur le TEST.")


if __name__ == "__main__":
    main()