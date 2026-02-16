# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import pipeline

try:
    from scripts.csv_validator import validar_csv
except ModuleNotFoundError:
    from csv_validator import validar_csv

MODEL_NAME = "facebook/bart-large-mnli"
DEFAULT_DATASET = Path("data/assunto.csv")
DEFAULT_LABELS = ["economia", "saude", "educacao", "seguranca"]


def avaliar(dataset_path: str | Path = DEFAULT_DATASET) -> dict:
    dataset_path = Path(dataset_path)
    df = validar_csv(
        caminho_csv=dataset_path,
        schema_esperado={"texto": str, "rotulo": str},
        aliases_colunas={"rotulo": ["label", "rotulo"]},
        colunas_obrigatorias=["texto", "rotulo"],
    )

    clf = pipeline("zero-shot-classification", model=MODEL_NAME)
    labels = sorted(set(df["rotulo"].astype(str).tolist()) | set(DEFAULT_LABELS))

    y_true = df["rotulo"].astype(str).tolist()
    y_pred = []

    for texto in df["texto"].astype(str).tolist():
        resultado = clf(texto, candidate_labels=labels)
        y_pred.append(str(resultado["labels"][0]))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    return {
        "accuracy": acc,
        "f1": f1,
        "labels": labels,
        "confusion_matrix": cm_df,
        "total": int(len(df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste de classificacao de assunto")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Caminho do CSV com colunas texto, rotulo",
    )
    args = parser.parse_args()

    resultado = avaliar(args.dataset)

    print("=== Assunto (Zero-shot) ===")
    print(f"Amostras: {resultado['total']}")
    print(f"Accuracy: {resultado['accuracy']:.4f}")
    print(f"F1-score (weighted): {resultado['f1']:.4f}")
    print("\nMatriz de confusao:")
    print(resultado["confusion_matrix"].to_string())


if __name__ == "__main__":
    main()
