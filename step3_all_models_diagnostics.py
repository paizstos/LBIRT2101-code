#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 15:04:19 2025

@author: adamdavidmalila
"""

# -*- coding: utf-8 -*-
from __future__ import annotations
"""
STEP 3 — Diagnostics complets sur TOUS les modèles
--------------------------------------------------
Ce script :
  - recharge X_train, X_test, y_train, y_test depuis outputs_step2/
  - redéfinit les 4 modèles ML utilisés à l'étape 3 :
        * logreg        : LogisticRegression
        * svm_linear    : LinearSVC
        * nb_multinomial: MultinomialNB
        * rf_ensemble   : RandomForestClassifier
  - entraîne chaque modèle sur le TRAIN
  - évalue chaque modèle sur le TEST
  - pour CHAQUE modèle, génère :
        * métriques (accuracy, precision, recall, f1_macro)
        * matrice de confusion (CSV + PNG)
        * courbe ROC (PNG) avec AUC
        * classification_report (TXT)
  - enregistre un tableau récapitulatif dans outputs_step3/all_models_metrics.csv
  - trace un barplot d'accuracy par modèle.

Ce script ne modifie pas les données : il sert uniquement à l'INTERPRÉTATION
et à la comparaison des modèles sur le même split TRAIN/TEST.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

# ----------------------------------------------------------------------
# CHEMINS GLOBAUX
# ----------------------------------------------------------------------

# ROOT = dossier racine du projet, un cran au-dessus de "code/"
ROOT = Path(__file__).resolve().parents[1]

# Dossiers d'entrées / sorties
OUT_STEP2 = ROOT / "outputs_step2"       # là où se trouvent X_train_tfidf / X_test_tfidf / y_train / y_test
OUT_STEP3 = ROOT / "outputs_step3"       # là où on a déjà best_model, confusion_matrix, etc.
OUT_FIG   = OUT_STEP3 / "all_models"     # nouveau sous-dossier pour les figures de CE script

# On s'assure que les dossiers existent
OUT_STEP3.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# FONCTIONS UTILITAIRES
# ----------------------------------------------------------------------

def load_train_test():
    """
    Charge X_train, y_train, X_test, y_test depuis outputs_step2.

    On met allow_pickle=True pour éviter l'erreur liée au dtype=object,
    puis on convertit explicitement les labels en np.array de str.
    """
    print(f"🧭 Racine projet : {ROOT}")
    print("=== STEP 3 — Diagnostics multi-modèles ===")

    # Matrices TF-IDF (sparse CSR)
    X_train = sparse.load_npz(OUT_STEP2 / "X_train_tfidf.npz")
    X_test  = sparse.load_npz(OUT_STEP2 / "X_test_tfidf.npz")

    # Labels (np.array de str)
    y_train = np.load(OUT_STEP2 / "y_train.npy", allow_pickle=True)
    y_test  = np.load(OUT_STEP2 / "y_test.npy", allow_pickle=True)
    y_train = np.array(y_train, dtype=str)
    y_test  = np.array(y_test, dtype=str)

    print(f"   → X_train shape : {X_train.shape}, y_train shape : {y_train.shape}")
    print(f"   → X_test  shape : {X_test.shape},  y_test  shape : {y_test.shape}")
    print(f"   → Labels uniques train : {np.unique(y_train)}")
    print(f"   → Labels uniques test  : {np.unique(y_test)}\n")

    return X_train, y_train, X_test, y_test


def get_models():
    """
    Définit les 4 modèles utilisés à l'étape 3.

    On reprend exactement les mêmes hyperparamètres que dans step3_train_models.py
    pour être cohérent.
    """
    models = {
        # Régression logistique (linéaire, L2) — baseline forte pour texte
        "logreg": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            penalty="l2",
            n_jobs=-1
        ),

        # SVM linéaire (LinearSVC) — très efficace sur TF-IDF
        "svm_linear": LinearSVC(
            C=1.0
        ),

        # Naive Bayes multinomial — très classique pour BOW / TF-IDF
        "nb_multinomial": MultinomialNB(),

        # Random Forest — modèle d'ensemble non linéaire
        "rf_ensemble": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        ),
    }
    return models


def get_scores_for_roc(model, name, X_test, y_test, positive_label="ai"):
    """
    Calcule un vecteur de scores continus pour la ROC.

    - Si le modèle possède predict_proba, on utilise la proba de la classe positive.
    - Sinon, si le modèle possède decision_function, on utilise cette sortie.
    - Sinon, on renvoie None (on ne pourra pas tracer la ROC pour ce modèle).

    On renvoie :
        y_true_bin : np.array binaire (0/1) pour y_test
        scores     : np.array de scores continus ou None
    """
    # Classe positive = "ai" (cohérent avec le projet)
    y_true_bin = (y_test == positive_label).astype(int)

    scores = None
    try:
        if hasattr(model, "predict_proba"):
            # Probas par classe
            probas = model.predict_proba(X_test)
            # Index de la classe positive dans model.classes_
            classes = list(model.classes_)
            pos_idx = classes.index(positive_label)
            scores = probas[:, pos_idx]
        elif hasattr(model, "decision_function"):
            # Sortie marge / score (SVM linéaire)
            scores = model.decision_function(X_test)
        else:
            scores = None
    except Exception as e:
        print(f"   ⚠️ Impossible de récupérer des scores continus pour ROC ({name}) : {e}")
        scores = None

    return y_true_bin, scores


def plot_confusion_matrix(cm, labels, title, out_path):
    """
    Trace et sauvegarde une matrice de confusion cm (2x2) au format PNG.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax)

    # Noms des axes
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Affichage des valeurs dans chaque case
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="yellow" if cm[i, j] > cm.max() * 0.5 else "white",
                fontsize=12,
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true_bin, scores, model_name, out_path):
    """
    Trace la courbe ROC et sauvegarde en PNG.

    y_true_bin : labels 0/1 (1 = classe positive 'ai')
    scores     : scores continus (probas ou decision_function)
    """
    # ROC classique
    fpr, tpr, thresholds = roc_curve(y_true_bin, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Hasard (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve — {model_name}")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return roc_auc


# ----------------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ----------------------------------------------------------------------

def main():
    # 1) Charger les données TRAIN / TEST
    X_train, y_train, X_test, y_test = load_train_test()

    # 2) Récupérer les modèles
    models = get_models()

    # On définit un ordre fixe pour les labels (pour la matrice de confusion)
    label_order = ["human", "ai"]

    # Liste pour stocker les métriques de chaque modèle
    all_results = []

    print("🔁 Entraînement + évaluation de TOUS les modèles sur le TRAIN/TEST...\n")

    for name, model in models.items():
        print(f"➡️ Modèle : {name}")

        # --------------------------------------------------------------
        # 2.1 Entraînement sur tout le TRAIN
        # --------------------------------------------------------------
        model.fit(X_train, y_train)

        # --------------------------------------------------------------
        # 2.2 Prédictions sur le TEST
        # --------------------------------------------------------------
        y_pred = model.predict(X_test)

        # --------------------------------------------------------------
        # 2.3 Métriques globales
        # --------------------------------------------------------------
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"   - accuracy        : {acc:.3f}")
        print(f"   - precision_macro : {prec:.3f}")
        print(f"   - recall_macro    : {rec:.3f}")
        print(f"   - f1_macro        : {f1:.3f}")

        # --------------------------------------------------------------
        # 2.4 Matrice de confusion (2x2)
        # --------------------------------------------------------------
        cm = confusion_matrix(y_test, y_pred, labels=label_order)

        # Sauvegarde CSV
        cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
        cm_csv_path = OUT_FIG / f"confusion_matrix_{name}.csv"
        cm_df.to_csv(cm_csv_path)
        print(f"   - Matrice de confusion CSV : {cm_csv_path}")

        # Sauvegarde figure PNG
        cm_png_path = OUT_FIG / f"confusion_matrix_{name}.png"
        plot_confusion_matrix(
            cm,
            labels=label_order,
            title=f"Confusion matrix — {name}",
            out_path=cm_png_path,
        )
        print(f"   - Matrice de confusion PNG : {cm_png_path}")

        # --------------------------------------------------------------
        # 2.5 Classification report détaillé (TXT)
        # --------------------------------------------------------------
        report = classification_report(
            y_test,
            y_pred,
            labels=label_order,
            target_names=label_order,
            digits=3,
        )
        report_path = OUT_FIG / f"classification_report_{name}.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n\n")
            f.write(report)
        print(f"   - Classification report    : {report_path}")

        # --------------------------------------------------------------
        # 2.6 Courbe ROC + AUC (si possible)
        # --------------------------------------------------------------
        y_true_bin, scores = get_scores_for_roc(model, name, X_test, y_test, positive_label="ai")
        roc_auc = None
        if scores is not None:
            roc_png_path = OUT_FIG / f"roc_curve_{name}.png"
            roc_auc = plot_roc_curve(y_true_bin, scores, name, roc_png_path)
            print(f"   - ROC curve PNG            : {roc_png_path} (AUC = {roc_auc:.3f})")
        else:
            print("   ⚠️ ROC non tracée (pas de scores continus disponibles).")

        print("")  # ligne vide pour lisibilité

        # On stocke les résultats dans une liste pour le tableau final
        all_results.append({
            "model": name,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
            "roc_auc": roc_auc,
        })

    # ------------------------------------------------------------------
    # 3) Tableau récapitulatif de tous les modèles
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    metrics_csv_path = OUT_FIG / "all_models_metrics.csv"
    results_df.to_csv(metrics_csv_path, index=False)
    print("📊 Tableau récapitulatif des modèles :")
    print(results_df)
    print(f"\n💾 Sauvegardé dans : {metrics_csv_path}")

    # Sauvegarde aussi en JSON pour réutilisation éventuelle
    metrics_json_path = OUT_FIG / "all_models_metrics.json"
    metrics_json_path.write_text(
        json.dumps(results_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    print(f"💾 Version JSON        : {metrics_json_path}")

    # ------------------------------------------------------------------
    # 4) Barplot simple de l'accuracy par modèle
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(results_df["model"], results_df["accuracy"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (TEST)")
    ax.set_title("Comparaison des modèles — accuracy sur le TEST")

    # valeurs numériques au-dessus des barres
    for i, v in enumerate(results_df["accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    barplot_path = OUT_FIG / "accuracy_barplot_all_models.png"
    fig.savefig(barplot_path, dpi=150)
    plt.close(fig)
    print(f"🖼 Barplot accuracy     : {barplot_path}")

    print("\n✅ Diagnostics multi-modèles terminés.")
    print(f"   → Toutes les figures et rapports sont dans : {OUT_FIG}")


# Point d'entrée du script
if __name__ == "__main__":
    main()