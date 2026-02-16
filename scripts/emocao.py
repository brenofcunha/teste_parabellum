# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import pipeline

try:
    from scripts.csv_validator import validar_csv
except ModuleNotFoundError:
    from csv_validator import validar_csv

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_DATASET = Path("data/emocao.csv")


def avaliar(dataset_path: str | Path = DEFAULT_DATASET) -> dict:
    dataset_path = Path(dataset_path)
    df = validar_csv(
        caminho_csv=dataset_path,
        schema_esperado={"texto": str, "rotulo": str},
        aliases_colunas={"rotulo": ["label", "rotulo"]},
        colunas_obrigatorias=["texto", "rotulo"],
    )

    clf = pipeline("text-classification", model=MODEL_NAME, top_k=1)
    resultados = clf(df["texto"].astype(str).tolist())
    predicoes = [
        (resultado[0] if isinstance(resultado, list) else resultado)["label"]
        for resultado in resultados
    ]

    y_true = df["rotulo"].astype(str).tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predicoes,
        average="weighted",
        zero_division=0,
    )

    relatorio = classification_report(y_true, predicoes, zero_division=0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "report": relatorio,
        "total": int(len(df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste de classificador de emocao")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Caminho do CSV com colunas texto, rotulo",
    )
    args = parser.parse_args()

    resultado = avaliar(args.dataset)

    print("=== Emocao ===")
    print(f"Amostras: {resultado['total']}")
    print(f"Precision (weighted): {resultado['precision']:.4f}")
    print(f"Recall (weighted): {resultado['recall']:.4f}")
    print(f"F1-score (weighted): {resultado['f1']:.4f}")
    print("\nClassification report:")
    print(resultado["report"])


if __name__ == "__main__":
    main()
