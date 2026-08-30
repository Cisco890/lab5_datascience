"""Preprocesamiento y modelo base compartidos entre los notebooks del laboratorio 5.

Este módulo centraliza la limpieza de texto y el entrenamiento del modelo
preliminar (TF-IDF + Regresión Logística) definidos en `lab5_avances.ipynb`,
para que los notebooks siguientes (función de clasificación, análisis de
sentimiento, etc.) puedan reconstruir la misma línea base sin duplicar todo
el análisis exploratorio.
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U0001F900-\U0001F9FF"
    "\U00002B00-\U00002BFF"
    "]+",
    flags=re.UNICODE,
)
EMOTICON_PATTERN = re.compile(r"(?<![:/])[:;=8][\-o\*']?[\)\]\(\[dDpP\}\{@\|\\]")

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def limpiar_texto(texto):
    """Limpieza usada para el modelo de bolsa de palabras (ver lab5_avances.ipynb)."""
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"@\w+", " ", texto)
    texto = EMOJI_PATTERN.sub(" ", texto)
    texto = EMOTICON_PATTERN.sub(" ", texto)
    texto = texto.replace("#", "")
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"_+", " ", texto)
    tokens = [token for token in texto.split() if token not in STOP_WORDS]
    return " ".join(tokens)


def cargar_datos(ruta="data/train.csv"):
    """Carga train.csv y agrega la columna `text_clean`."""
    df = pd.read_csv(ruta)
    df["text_clean"] = df["text"].apply(limpiar_texto)
    return df


def entrenar_modelo_base(df, test_size=0.20, random_state=42):
    """Reproduce el modelo preliminar de lab5_avances.ipynb (TF-IDF + Regresión Logística)."""
    X = df["text_clean"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    modelo = LogisticRegression(max_iter=1000, random_state=random_state)
    modelo.fit(X_train_tfidf, y_train)

    return {
        "tfidf": tfidf,
        "modelo": modelo,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_tfidf": X_train_tfidf,
        "X_test_tfidf": X_test_tfidf,
    }
