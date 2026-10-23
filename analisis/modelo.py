"""🤖 Modelo predictivo de churn (regresión logística, scikit-learn).

Entrena con data/processed/features_modelo.csv (variables al 30-jun-2026, etiqueta jul–sep 2026:
sin fuga temporal). Se evalúa en un 25 % de clientes no vistos y se compara con una línea base.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = Path(__file__).resolve().parents[1] / "data" / "processed" / "features_modelo.csv"
CATEGORICAS = ["genero", "ciudad", "segmento", "canal_adquisicion", "dispositivo_preferido"]


def cargar_features(path: Path = FEATURES) -> pd.DataFrame:
    return pd.read_csv(path)


def entrenar(feat: pd.DataFrame, seed: int = 42, umbral: float = 0.5) -> dict:
    if feat.churn.nunique() < 2 or len(feat) < 100:
        raise ValueError("Se necesitan al menos 100 clientes y ambas clases para entrenar el modelo.")
    X, y = feat.drop(columns=["cliente_id", "churn"]), feat.churn.astype(int)
    cats = [c for c in CATEGORICAS if c in X]
    nums = [c for c in X.columns if c not in cats]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("esc", StandardScaler())]), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats)])
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, C=1.0))])
    Xtr, Xte, ytr, yte, idtr, idte = train_test_split(X, y, feat.cliente_id, test_size=.25, random_state=seed,
                                                      stratify=y)
    pipe.fit(Xtr, ytr)
    prob = pipe.predict_proba(Xte)[:, 1]
    pred = (prob >= umbral).astype(int)
    mayoritaria = int(ytr.mean() >= .5)
    metricas = {
        "auc": roc_auc_score(yte, prob), "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0), "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0), "brier": brier_score_loss(yte, prob),
        "accuracy_linea_base": accuracy_score(yte, np.full(len(yte), mayoritaria)),
        "auc_linea_base": 0.5, "n_entrenamiento": len(ytr), "n_prueba": len(yte),
        "tasa_churn_prueba": float(yte.mean()), "umbral": umbral,
    }
    names = pipe.named_steps["pre"].get_feature_names_out()
    coef = pipe.named_steps["lr"].coef_[0]
    imp = pd.DataFrame({"variable": names, "coeficiente": coef})
    imp["original"] = imp.variable.str.replace(r"^(num|cat)__", "", regex=True)
    imp["original"] = imp.original.map(lambda v: next((c for c in cats if v.startswith(c + "_")),
                                                       v.replace("missingindicator_", "")))
    imp = imp.reindex(imp.coeficiente.abs().sort_values(ascending=False).index).reset_index(drop=True)
    fpr, tpr, _ = roc_curve(yte, prob)
    pruebas = pd.DataFrame({"cliente_id": idte.values, "real": yte.values, "probabilidad": prob, "prediccion": pred})
    pruebas = pruebas.merge(feat[["cliente_id", "segmento", "ciudad", "canal_adquisicion", "miembro_fidelidad"]],
                            on="cliente_id")
    pruebas["decil_riesgo"] = pd.qcut(pruebas.probabilidad.rank(method="first"), 10, labels=range(1, 11)).astype(int)
    calib = pruebas.groupby("decil_riesgo").agg(clientes=("real", "size"), prob_media=("probabilidad", "mean"),
                                                churn_real=("real", "mean")).reset_index()
    return {"pipeline": pipe, "metricas": metricas, "importancia": imp,
            "matriz": confusion_matrix(yte, pred, labels=[0, 1]),
            "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr}), "pruebas": pruebas, "calibracion": calib}
