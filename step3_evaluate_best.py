#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:50:52 2025

@author: adamdavidmalila
"""


"""
STEP 3 — Évaluation du meilleur modèle sur TRAIN / TEST

Ce script :
  - recharge le best_model.joblib choisi à l'étape précédente
  - charge X_train, y_train, X_test, y_test
  - calcule des métriques complètes sur TEST (et TRAIN pour comparer)
  - génère des fichiers :
      * classification_report.txt
      * confusion_matrix.csv
      * confusion_matrix.png
      * roc_curve.png (si possible)
      * metrics_step3.json
      * metrics_step3.csv

À lancer après step3_train_models.py.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import joblib

# ----------------------------------------------------------------------
# CHEMINS
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUT_STEP2 = ROOT / "outputs_step2"
OUT_STEP3 = ROOT / "outputs_step3"
OUT_STEP3.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# UTILITAIRES
# ----------------------------------------------------------------------

def load_data_train_test():
    """
    Charge X_train, y_train, X_test, y_test depuis outputs_step2.
    On autorise allow_pickle=True car les fichiers .npy peuvent contenir
    un dtype=object (par ex. chaînes 'human'/'ai' encodées par Pandas).
    """
    X_train = sparse.load_npz(OUT_STEP2 / "X_train_tfidf.npz")
    X_test  = sparse.load_npz(OUT_STEP2 / "X_test_tfidf.npz")

    y_train = np.load(OUT_STEP2 / "y_train.npy", allow_pickle=True)
    y_test  = np.load(OUT_STEP2 / "y_test.npy",  allow_pickle=True)

    # Optionnel : on force en array de str pour éviter les surprises
    y_train = np.array(y_train, dtype=str)
    y_test  = np.array(y_test, dtype=str)

    print(f"   → X_train shape : {X_train.shape}, y_train shape : {y_train.shape}")
    print(f"   → X_test  shape : {X_test.shape},  y_test  shape : {y_test.shape}")
    print(f"   → Labels uniques train : {np.unique(y_train)}")
    print(f"   → Labels uniques test  : {np.unique(y_test)}")

    return X_train, y_train, X_test, y_test


def load_best_model():
    """Recharge best_model.joblib + le nom du modèle depuis models_cv_results.json."""
    model_path = OUT_STEP3 / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"best_model.joblib introuvable dans {OUT_STEP3}.\n"
            "Lance d'abord step3_train_models.py."
        )
    model = joblib.load(model_path)

    json_path = OUT_STEP3 / "models_cv_results.json"
    best_name = "UNKNOWN"
    if json_path.exists():
        info = json.loads(json_path.read_text(encoding="utf-8"))
        best_name = info.get("best_model_name", "UNKNOWN")

    print(f"📦 Modèle chargé : {best_name} ({model.__class__.__name__})")
    return model, best_name


def get_scores(y_true, y_pred):
    """Retourne un dict avec les métriques principales (accuracy / precision / recall / f1_macro)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
    }


def try_get_scores_proba_or_decision(model, X):
    """
    Pour la courbe ROC :
    - si le modèle a predict_proba, on utilise proba[:, 1]
    - sinon, si decision_function, on utilise ce score
    - sinon, on renvoie None et on ne trace pas de ROC.
    """
    scores = None
    method_used = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # on suppose que la classe positive est 'ai' → on cherche l'index
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if "ai" in classes:
                idx_ai = classes.index("ai")
            else:
                # par défaut classe 1
                idx_ai = 1 if len(classes) > 1 else 0
        else:
            idx_ai = 1 if proba.shape[1] > 1 else 0

        scores = proba[:, idx_ai]
        method_used = "predict_proba"

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        method_used = "decision_function"

    return scores, method_used


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print(f"🧭 Racine projet : {ROOT}")
    print("=== STEP 3 — Évaluation du meilleur modèle ===")

    # 1) Charger données + modèle
    X_train, y_train, X_test, y_test = load_data_train_test()
    model, best_name = load_best_model()

    # 2) Prédictions TRAIN et TEST
    print("\n🔮 Prédictions en cours...")
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # 3) Scores global TRAIN / TEST
    train_scores = get_scores(y_train, y_train_pred)
    test_scores  = get_scores(y_test, y_test_pred)

    print("\n📊 Scores sur TRAIN :")
    for k, v in train_scores.items():
        print(f"   - {k} : {v:.3f}")

    print("\n📊 Scores sur TEST :")
    for k, v in test_scores.items():
        print(f"   - {k} : {v:.3f}")

    # 4) Classification report (sur TEST)
    report_txt = classification_report(y_test, y_test_pred, digits=3)
    report_path = OUT_STEP3 / "classification_report.txt"
    report_path.write_text(report_txt, encoding="utf-8")

    print(f"\n📄 Classification report (TEST) enregistré dans : {report_path}")
    print(report_txt)

    # 5) Matrice de confusion (TEST) + heatmap
    cm = confusion_matrix(y_test, y_test_pred, labels=["human", "ai"])
    cm_df = pd.DataFrame(cm, index=["human", "ai"], columns=["human", "ai"])
    cm_csv_path = OUT_STEP3 / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    print(f"\n📄 Confusion matrix (csv) enregistrée dans : {cm_csv_path}")
    print("   Matrice de confusion :")
    print(cm_df)

    cm_fig_path = OUT_STEP3 / "confusion_matrix.png"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["human", "ai"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title(f"Confusion matrix — {best_name}")
    plt.tight_layout()
    fig.savefig(cm_fig_path, dpi=150)
    plt.close(fig)
    print(f"🖼 Confusion matrix (png) enregistrée dans : {cm_fig_path}")

    # 6) ROC + AUC (si possible)
    roc_info = None
    scores_test, method_used = try_get_scores_proba_or_decision(model, X_test)
    roc_fig_path = OUT_STEP3 / "roc_curve.png"

    if scores_test is not None:
        # On doit binariser y_test : 1 pour 'ai', 0 pour 'human'
        y_test_bin = np.array([1 if lbl == "ai" else 0 for lbl in y_test])
        fpr, tpr, thr = roc_curve(y_test_bin, scores_test)
        roc_auc = float(auc(fpr, tpr))
        roc_info = {
            "fpr": list(map(float, fpr)),
            "tpr": list(map(float, tpr)),
            "thresholds": list(map(float, thr)),
            "auc": roc_auc,
            "method": method_used
        }

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC curve — {best_name}")
        ax.legend(loc="lower right")
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(roc_fig_path, dpi=150)
        plt.close(fig)
        print(f"🖼 ROC curve enregistrée dans : {roc_fig_path}")
    else:
        print("\n⚠️ Modèle sans predict_proba / decision_function : ROC non disponible.")

    # 7) Sauvegarde des métriques globales
    metrics = {
        "best_model_name": best_name,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "roc_info": roc_info
    }

    metrics_json_path = OUT_STEP3 / "metrics_step3.json"
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics_csv_path = OUT_STEP3 / "metrics_step3.csv"
    # on met les scores train/test dans un petit tableau
    metrics_rows = []
    for split_name, scores in [("train", train_scores), ("test", test_scores)]:
        row = {"split": split_name}
        row.update(scores)
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(metrics_csv_path, index=False)

    print(f"\n📄 Métriques globales enregistrées dans : {metrics_json_path}")
    print(f"📄 Résumé CSV enregistré dans        : {metrics_csv_path}")

    print("\n✅ Évaluation terminée.")
    print("   → Tu peux maintenant lancer step3_quality_check.py pour un diagnostic rapide.")


if __name__ == "__main__":
    main()