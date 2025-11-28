# -*- coding: utf-8 -*-
"""
STEP 3 — Visualisations avancées (IA vs humains)

Génère et sauvegarde dans outputs_step3 :
    - top_words_ai.png / top_words_human.png        (poids SVM linéaire)
    - tsne_2d.png                                   (projection t-SNE 2D)
    - umap_2d.png                                   (projection UMAP 2D)
    - hist_lengths.png                              (histogrammes nbre de mots)
    - tfidf_mean_boxplot.png                        (distribution TF-IDF moyen)
    - wordcloud_ai.png / wordcloud_human.png        (nuages de mots)
    - learning_curve_svm_linear.png                 (courbe d’apprentissage)
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.manifold import TSNE
from sklearn.model_selection import learning_curve
import umap
import joblib

# ------------------------------------------------------------------
# CHEMINS
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT2 = ROOT / "outputs_step2"
OUT3 = ROOT / "outputs_step3"
OUT3.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# CHARGEMENT DES DONNÉES & DU MODÈLE
# ------------------------------------------------------------------
print("📂 Chargement données TF-IDF & métadonnées…")
X_train = sparse.load_npz(OUT2 / "X_train_tfidf.npz")
X_test  = sparse.load_npz(OUT2 / "X_test_tfidf.npz")
y_train = np.load(OUT2 / "y_train.npy", allow_pickle=True)
y_test  = np.load(OUT2 / "y_test.npy",  allow_pickle=True)

with open(OUT2 / "feature_names.json", "r", encoding="utf-8") as f:
    feature_names = np.array(json.load(f))

train_meta = pd.read_csv(OUT2 / "train_meta.csv")
test_meta  = pd.read_csv(OUT2 / "test_meta.csv")
df_meta = pd.concat([train_meta, test_meta], ignore_index=True)

# colonne longueur de texte : on accepte soit "n_words" soit "words"
if "n_words" in df_meta.columns:
    len_col = "n_words"
elif "words" in df_meta.columns:
    len_col = "words"
else:
    # si vraiment rien, on met NaN
    df_meta["n_words"] = np.nan
    len_col = "n_words"

print("📦 Chargement du meilleur modèle (supposé SVM linéaire)…")
model = joblib.load(OUT3 / "best_model.joblib")

# ------------------------------------------------------------------
# (A) IMPORTANCE DES FEATURES (SVM LINÉAIRE)
# ------------------------------------------------------------------
print("📊 Graphique : importance des features (SVM)…")

# On suppose un modèle linéaire binaire avec attribut coef_
coef = model.coef_.ravel()          # poids pour chaque feature
idx_sorted = np.argsort(coef)

# top pour IA (poids positifs forts)
top_k = 20
top_ai_idx = idx_sorted[-top_k:]
top_hum_idx = idx_sorted[:top_k]

def plot_top_words(indices, title, filename):
    words = feature_names[indices]
    weights = coef[indices]
    order = np.argsort(np.abs(weights))
    words = words[order]
    weights = weights[order]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(words)), weights)
    plt.yticks(range(len(words)), words)
    plt.title(title)
    plt.xlabel("Poids du modèle (coef)")
    plt.tight_layout()
    out_path = OUT3 / filename
    plt.savefig(out_path, dpi=150)
    print(f"   → enregistré : {out_path}")
    plt.close()

plot_top_words(top_ai_idx, "Top mots prédictifs → classe AI", "top_words_ai.png")
plot_top_words(top_hum_idx, "Top mots prédictifs → classe human", "top_words_human.png")

# ------------------------------------------------------------------
# (B) t-SNE & UMAP 2D
# ------------------------------------------------------------------
print("🌀 t-SNE 2D & UMAP 2D…")

X_all = sparse.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

# t-SNE
tsne = TSNE(n_components=2, random_state=42, init="random", learning_rate="auto")
X_tsne = tsne.fit_transform(X_all.toarray())

plt.figure(figsize=(8, 6))
for lab, col in [("ai", "orange"), ("human", "royalblue")]:
    mask = (y_all == lab)
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=25, c=col, label=lab, alpha=0.8)
plt.legend()
plt.title("t-SNE 2D — Séparation IA vs Human")
plt.tight_layout()
out_tsne = OUT3 / "tsne_2d.png"
plt.savefig(out_tsne, dpi=150)
print(f"   → enregistré : {out_tsne}")
plt.close()

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_all)

plt.figure(figsize=(8, 6))
for lab, col in [("ai", "orange"), ("human", "royalblue")]:
    mask = (y_all == lab)
    plt.scatter(X_umap[mask, 0], X_umap[mask, 1], s=25, c=col, label=lab, alpha=0.8)
plt.legend()
plt.title("UMAP 2D — Séparation IA vs Human")
plt.tight_layout()
out_umap = OUT3 / "umap_2d.png"
plt.savefig(out_umap, dpi=150)
print(f"   → enregistré : {out_umap}")
plt.close()

# ------------------------------------------------------------------
# (C) HISTOGRAMMES DES LONGUEURS DE TEXTE
# ------------------------------------------------------------------
print("📏 Histogrammes des longueurs de textes…")

plt.figure(figsize=(8, 5))
for lab, col in [("ai", "orange"), ("human", "royalblue")]:
    sub = df_meta[df_meta["label"] == lab]
    if len(sub) == 0:
        continue
    plt.hist(sub[len_col], bins=15, alpha=0.6, label=lab, edgecolor="black")
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre d’abstracts")
plt.title("Distribution de la longueur des abstracts (AI vs Human)")
plt.legend()
plt.tight_layout()
out_hist = OUT3 / "hist_lengths.png"
plt.savefig(out_hist, dpi=150)
print(f"   → enregistré : {out_hist}")
plt.close()

# ------------------------------------------------------------------
# (D) DISTRIBUTION DU TF-IDF MOYEN PAR CLASSE
# ------------------------------------------------------------------
print("📈 Distribution du TF-IDF moyen par classe…")

# moyenne des coefficients TF-IDF par document
doc_means = np.asarray(X_all.mean(axis=1)).ravel()
df_tfidf = pd.DataFrame({"label": y_all, "tfidf_mean": doc_means})

plt.figure(figsize=(8, 5))
for lab, col in [("ai", "orange"), ("human", "royalblue")]:
    sub = df_tfidf[df_tfidf["label"] == lab]["tfidf_mean"]
    plt.hist(sub, bins=15, alpha=0.6, label=lab, edgecolor="black")
plt.xlabel("TF-IDF moyen par document")
plt.ylabel("Nombre d’abstracts")
plt.title("Distribution du TF-IDF moyen (AI vs Human)")
plt.legend()
plt.tight_layout()
out_tfidf = OUT3 / "tfidf_mean_hist.png"
plt.savefig(out_tfidf, dpi=150)
print(f"   → enregistré : {out_tfidf}")
plt.close()

# ------------------------------------------------------------------
# (E) WORDCLOUDS IA vs HUMAN (si df_balanced dispo)
# ------------------------------------------------------------------
df_bal_path = OUT2 / "df_balanced.csv"
if df_bal_path.exists():
    print("☁️ Wordclouds IA vs Human…")
    df_bal = pd.read_csv(df_bal_path)
    texts_ai = " ".join(df_bal[df_bal["label"] == "ai"]["text"].astype(str).tolist())
    texts_hu = " ".join(df_bal[df_bal["label"] == "human"]["text"].astype(str).tolist())

    wc_ai = WordCloud(width=1200, height=800, background_color="white").generate(texts_ai)
    plt.figure(figsize=(8, 6))
    plt.imshow(wc_ai, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud — IA")
    out_wc_ai = OUT3 / "wordcloud_ai.png"
    plt.savefig(out_wc_ai, dpi=150)
    print(f"   → enregistré : {out_wc_ai}")
    plt.close()

    wc_hu = WordCloud(width=1200, height=800, background_color="white").generate(texts_hu)
    plt.figure(figsize=(8, 6))
    plt.imshow(wc_hu, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud — Human")
    out_wc_hu = OUT3 / "wordcloud_human.png"
    plt.savefig(out_wc_hu, dpi=150)
    print(f"   → enregistré : {out_wc_hu}")
    plt.close()
else:
    print("⚠️ Pas de df_balanced.csv → wordclouds ignorés.")

# ------------------------------------------------------------------
# (F) LEARNING CURVE DU SVM LINÉAIRE
# ------------------------------------------------------------------
print("📉 Courbe d’apprentissage (learning curve) du SVM linéaire…")

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_train,
    y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="accuracy",
    n_jobs=-1,
    shuffle=True,
    random_state=42
)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_std    = val_scores.std(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, marker="o", label="Accuracy TRAIN")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)

plt.plot(train_sizes, val_mean, marker="s", label="Accuracy CV (5-fold)")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

plt.xlabel("Nombre d’exemples d’entraînement")
plt.ylabel("Accuracy")
plt.title("Learning curve — SVM linéaire")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_lc = OUT3 / "learning_curve_svm_linear.png"
plt.savefig(out_lc, dpi=150)
print(f"   → enregistré : {out_lc}")
plt.close()

print("✅ Visualisations STEP 3 terminées.")