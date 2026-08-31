"""Regenera las figuras del informe (report/figures/*.png) a partir del pipeline
de lab5_avances.ipynb y lab5_clasificacion_sentimiento.ipynb, para que el
informe en LaTeX tenga imagenes reproducibles sin depender de exportar
manualmente cada celda de los notebooks.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.lab5_pipeline import cargar_datos, entrenar_modelo_base, limpiar_texto  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / nombre, dpi=150, bbox_inches="tight")
    plt.close()
    print("Guardado:", nombre)


df = cargar_datos(str(REPO_ROOT / "data" / "train.csv"))
resultado = entrenar_modelo_base(df)
tfidf = resultado["tfidf"]
modelo = resultado["modelo"]
X_train = resultado["X_train"]
X_test = resultado["X_test"]
y_train = resultado["y_train"]
y_test = resultado["y_test"]
X_train_tfidf = resultado["X_train_tfidf"]
X_test_tfidf = resultado["X_test_tfidf"]
y_pred = modelo.predict(X_test_tfidf)

# 1. Valores faltantes por columna
porcentajes_faltantes = (df.isna().mean() * 100).round(2)
porcentajes_faltantes.sort_values(ascending=False).plot(
    kind="bar",
    figsize=(8, 4),
    color=["#c44e52", "#dd8452", "#55a868", "#4c72b0", "#8172b3"],
    title="Porcentaje de valores faltantes por columna",
)
plt.ylabel("Porcentaje (%)")
plt.xlabel("Columna")
plt.xticks(rotation=0)
guardar("missing_values.png")

# 2. Distribucion de la variable objetivo
distribucion_target = df["target"].value_counts().sort_index()
ax = distribucion_target.plot(
    kind="bar",
    figsize=(6, 4),
    color=["#4c72b0", "#dd8452"],
    title="Distribucion de la variable objetivo",
)
ax.set_xticklabels(["No desastre (0)", "Desastre (1)"], rotation=0)
plt.ylabel("Cantidad de tweets")
plt.xlabel("Clase")
guardar("target_distribution.png")

# 3. Caracteristicas del texto original por clase (boxplots)
df["n_caracteres"] = df["text"].str.len()
df["n_palabras_original"] = df["text"].str.split().str.len()
df["n_hashtags"] = df["text"].str.count(r"#\w+")
df["n_menciones"] = df["text"].str.count(r"@\w+")
df["n_urls"] = df["text"].str.count(r"http\S+|www\.\S+")
df["n_numeros"] = df["text"].str.count(r"\b\d+\b")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
variables = [
    ("n_caracteres", "Cantidad de caracteres"),
    ("n_palabras_original", "Cantidad de palabras"),
    ("n_hashtags", "Cantidad de hashtags"),
    ("n_menciones", "Cantidad de menciones"),
    ("n_urls", "Cantidad de URLs"),
    ("n_numeros", "Cantidad de numeros"),
]
for ax, (columna, titulo) in zip(axes.ravel(), variables):
    df.boxplot(column=columna, by="target", ax=ax, grid=False)
    ax.set_title(titulo)
    ax.set_xlabel("target")
    ax.set_ylabel(columna)
plt.suptitle("Comparacion de caracteristicas del texto original por clase", y=1.02)
guardar("text_features_boxplots.png")


def obtener_ngramas_frecuentes(textos, ngram_range=(1, 1), top_n=20, min_df=1):
    textos = pd.Series(textos).fillna("").astype(str)
    textos = textos[textos.str.strip() != ""]
    vectorizador = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    matriz = vectorizador.fit_transform(textos)
    frecuencias = np.asarray(matriz.sum(axis=0)).ravel()
    terminos = vectorizador.get_feature_names_out()
    tabla = pd.DataFrame({"termino": terminos, "frecuencia": frecuencias})
    return tabla.sort_values(["frecuencia", "termino"], ascending=[False, True]).head(top_n)


def graficar_top_terminos(tabla, titulo, color, ax):
    tabla_ordenada = tabla.sort_values("frecuencia")
    ax.barh(tabla_ordenada["termino"], tabla_ordenada["frecuencia"], color=color)
    ax.set_title(titulo)
    ax.set_xlabel("Frecuencia")
    ax.set_ylabel("Termino")


uni_no_desastre = obtener_ngramas_frecuentes(df.loc[df["target"] == 0, "text_clean"], (1, 1), 20)
uni_desastre = obtener_ngramas_frecuentes(df.loc[df["target"] == 1, "text_clean"], (1, 1), 20)

# 4. Unigramas mas frecuentes por clase
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
graficar_top_terminos(uni_no_desastre, "Unigramas mas frecuentes - No desastre", "#4c72b0", axes[0])
graficar_top_terminos(uni_desastre, "Unigramas mas frecuentes - Desastre", "#dd8452", axes[1])
guardar("unigramas_por_clase.png")

bigramas_no_desastre = obtener_ngramas_frecuentes(df.loc[df["target"] == 0, "text_clean"], (2, 2), 20)
bigramas_desastre = obtener_ngramas_frecuentes(df.loc[df["target"] == 1, "text_clean"], (2, 2), 20)

# 5. Bigramas mas frecuentes por clase
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
graficar_top_terminos(bigramas_no_desastre, "Bigramas mas frecuentes - No desastre", "#55a868", axes[0])
graficar_top_terminos(bigramas_desastre, "Bigramas mas frecuentes - Desastre", "#c44e52", axes[1])
guardar("bigramas_por_clase.png")

# 6. Nube de palabras por clase (requerida explicitamente por el enunciado)
texto_no_desastre = " ".join(df.loc[df["target"] == 0, "text_clean"])
texto_desastre = " ".join(df.loc[df["target"] == 1, "text_clean"])
wc_no = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(texto_no_desastre)
wc_si = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(texto_desastre)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(wc_no, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title("Nube de palabras - No desastre")
axes[1].imshow(wc_si, interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title("Nube de palabras - Desastre")
guardar("wordcloud.png")

# 7. Matrices de confusion: Regresion Logistica y Naive Bayes
modelo_nb = MultinomialNB()
modelo_nb.fit(X_train_tfidf, y_train)
y_pred_nb = modelo_nb.predict(X_test_tfidf)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["No desastre", "Desastre"], cmap="Blues", ax=axes[0], colorbar=False
)
axes[0].set_title("Regresion Logistica")
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_nb, display_labels=["No desastre", "Desastre"], cmap="Greens", ax=axes[1], colorbar=False
)
axes[1].set_title("Naive Bayes Multinomial")
guardar("matrices_confusion.png")

# 8. Log-odds de n-gramas (probabilidad por clase)
unigramas_globales = obtener_ngramas_frecuentes(df["text_clean"], (1, 1), 20)
bigramas_globales = obtener_ngramas_frecuentes(df["text_clean"], (2, 2), 20)
terminos_interes = sorted(set(unigramas_globales["termino"]).union(set(bigramas_globales["termino"])))

vectorizador_presencia = CountVectorizer(vocabulary=terminos_interes, ngram_range=(1, 2), binary=True)
matriz_presencia = vectorizador_presencia.fit_transform(df["text_clean"])
terminos_presencia = vectorizador_presencia.get_feature_names_out()
df_presencia = pd.DataFrame(matriz_presencia.toarray(), columns=terminos_presencia, index=df.index)

epsilon = 1e-4
filas_probabilidad = []
for termino in terminos_presencia:
    contiene = df_presencia[termino].astype(bool)
    p_desastre = contiene[df["target"] == 1].mean()
    p_no_desastre = contiene[df["target"] == 0].mean()
    filas_probabilidad.append(
        {
            "termino": termino,
            "log_odds": np.log((p_desastre + epsilon) / (p_no_desastre + epsilon)),
        }
    )
tabla_log_odds = pd.DataFrame(filas_probabilidad).sort_values("log_odds")

fig, ax = plt.subplots(figsize=(9, 8))
colores = ["#4c72b0" if v < 0 else "#dd8452" for v in tabla_log_odds["log_odds"]]
ax.barh(tabla_log_odds["termino"], tabla_log_odds["log_odds"], color=colores)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("log-odds (negativo = no desastre, positivo = desastre)")
ax.set_title("Log-odds de n-gramas frecuentes por clase")
guardar("log_odds_ngramas.png")

# --- Analisis de sentimiento (VADER) ---
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()


def limpiar_para_sentimiento(texto):
    texto = str(texto)
    texto = pd.Series([texto]).str.replace(r"http\S+|www\.\S+", " ", regex=True).iloc[0]
    texto = pd.Series([texto]).str.replace(r"@\w+", " ", regex=True).iloc[0]
    return texto.replace("#", "").strip()


df["text_sentimiento"] = df["text"].apply(limpiar_para_sentimiento)
puntajes_vader = df["text_sentimiento"].apply(sia.polarity_scores).apply(pd.Series)
df = pd.concat([df, puntajes_vader.add_prefix("vader_")], axis=1)


def etiquetar_sentimiento(compound, umbral=0.05):
    if compound >= umbral:
        return "positivo"
    if compound <= -umbral:
        return "negativo"
    return "neutro"


df["sentimiento"] = df["vader_compound"].apply(etiquetar_sentimiento)
df["negatividad"] = -df["vader_compound"]

# 9. Distribucion de sentimiento por clase
tabla_sentimiento_target = pd.crosstab(df["target"], df["sentimiento"], normalize="index").round(3) * 100
tabla_sentimiento_target.index = ["No desastre (0)", "Desastre (1)"]
ax = tabla_sentimiento_target.plot(kind="bar", figsize=(7, 5), color=["#c44e52", "#8172b3", "#55a868"])
ax.set_ylabel("Porcentaje de tweets (%)")
ax.set_xlabel("Clase")
ax.set_title("Distribucion de sentimiento (VADER) por clase")
plt.xticks(rotation=0)
plt.legend(title="Sentimiento")
guardar("sentimiento_por_clase.png")

# 10. Negatividad por clase (boxplot)
ax = df.boxplot(column="negatividad", by="target", grid=False, figsize=(6, 4))
ax.set_title("Negatividad por clase")
ax.set_xlabel("target")
ax.set_ylabel("negatividad")
plt.suptitle("")
guardar("negatividad_por_clase.png")

print("\nTodas las figuras se generaron en:", FIGURES_DIR)
