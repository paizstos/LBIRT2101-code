#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:51:02 2025

@author: adamdavidmalila
"""

# -*- coding: utf-8 -*-
"""
STEP 3 — Contrôle qualité des résultats et recommandations

Ce script :
  - lit metrics_step3.json (généré par step3_evaluate_best.py)
  - lit y_train.npy / y_test.npy pour connaître la taille des datasets
  - applique quelques règles simples :
      * test_accuracy < 0.70 → performance jugée faible
      * 0.70 <= test_accuracy < 0.80 → performance moyenne
      * test_accuracy >= 0.80 → performance bonne
      * |accuracy_train - accuracy_test| > 0.15 → suspicion d'overfitting
  - imprime un diagnostic en français + recommandations :
      * ajouter des textes IA/humains
      * renforcer la régularisation
      * éventuellement augmenter le vocabulaire ou retravailler le nettoyage texte
"""

from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_STEP2 = ROOT / "outputs_step2"
OUT_STEP3 = ROOT / "outputs_step3"


def main():
    print(f"🧭 Racine projet : {ROOT}")
    print("=== STEP 3 — Contrôle qualité ===")

    metrics_path = OUT_STEP3 / "metrics_step3.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics_step3.json introuvable dans {OUT_STEP3}.\n"
            "Lance d'abord step3_evaluate_best.py."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_model_name = metrics.get("best_model_name", "UNKNOWN")
    train_scores = metrics.get("train_scores", {})
    test_scores = metrics.get("test_scores", {})

    acc_train = train_scores.get("accuracy", None)
    acc_test  = test_scores.get("accuracy", None)

    # Taille des datasets
    y_train = np.load(OUT_STEP2 / "y_train.npy", allow_pickle=True)
    y_test  = np.load(OUT_STEP2 / "y_test.npy", allow_pickle=True)

    n_train = len(y_train)
    n_test  = len(y_test)

    print(f"\n📦 Modèle évalué : {best_model_name}")
    print(f"   - Taille TRAIN : {n_train}")
    print(f"   - Taille TEST  : {n_test}")

    print("\n📊 Rappel des scores :")
    print(f"   - accuracy_train : {acc_train:.3f}")
    print(f"   - accuracy_test  : {acc_test:.3f}")

    gap = abs(acc_train - acc_test)
    print(f"   - écart train/test : {gap:.3f}")

    # Diagnostic selon accuracy test
    print("\n🩺 Diagnostic qualitatif :")
    if acc_test < 0.70:
        print("❌ Performance TEST faible (< 0.70).")
        print("   → Le modèle distingue mal IA vs humain sur des données jamais vues.")
        print("   → Recommandations :")
        print("      - augmenter le nombre de textes IA ET humains (ex : passer à 150–200 par classe),")
        print("      - vérifier le nettoyage (en-têtes, boilerplate, doublons),")
        print("      - éventuellement tester d'autres représentations (n-grammes plus larges, min_df plus bas).")
    elif acc_test < 0.80:
        print("⚠️ Performance TEST moyenne (entre 0.70 et 0.80).")
        print("   → Le modèle capte une partie des patterns, mais les frontières sont encore floues.")
        print("   → Recommandations :")
        print("      - si possible, ajouter quelques dizaines de textes IA/humains pour enrichir le signal,")
        print("      - affiner l'architecture (C du SVM, régularisation de la logistic, etc.),")
        print("      - vérifier que les textes IA ne sont pas trop ‘proches’ des textes humains (même style, même longueur).")
    else:
        print("✅ Performance TEST bonne (≥ 0.80).")
        print("   → Le modèle sépare correctement IA vs humain sur ce dataset.")
        print("   → Tu peux considérer ce niveau comme satisfaisant pour un projet académique.")
        print("   → Tu peux maintenant te concentrer sur l'interprétation et les visualisations (UMAP, cooccurrence, etc.).")

    # Diagnostic sur l’overfitting
    print("\n🔎 Analyse du risque d'overfitting :")
    if gap > 0.15:
        print("⚠️ Gros écart entre TRAIN et TEST (> 0.15) → suspicion d'overfitting.")
        print("   → Le modèle apprend trop les spécificités du TRAIN et généralise mal.")
        print("   → Recommandations :")
        print("      - ajouter davantage de données (surtout dans la classe la moins variée),")
        print("      - renforcer la régularisation (ex : C plus petit pour SVM/logreg),")
        print("      - vérifier qu'il n'y a pas de textes quasi identiques entre train/test.")
    else:
        print("✅ Pas d’overfitting massif apparent (écart train/test raisonnable).")

    print("\n📌 Résumé :")
    print("   - Ce script ne modifie pas les données, il t’aide à décider si le dataset est suffisant.")
    print("   - Si la performance est jugée limite, privilégie l’ajout de nouveaux abstracts IA/humains,")
    print("     en gardant la symétrie entre les deux classes (même ordre de grandeur).")

    print("\n✅ Contrôle qualité terminé.")


if __name__ == "__main__":
    main()