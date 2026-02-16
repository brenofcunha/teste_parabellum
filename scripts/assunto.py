# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import pipeline

try:
    from scripts.csv_validator import validar_csv
except ModuleNotFoundError:
    from csv_validator import validar_csv

MODEL_NAME = "facebook/bart-large-mnli"
DEFAULT_DATASET = Path("data/assunto.csv")
DEFAULT_LABELS = ["economia", "saude", "educacao", "seguranca"]


def _normalizar_label(valor: str) -> str:
    return str(valor).strip().lower()


def avaliar_modelo(df) -> dict:
    textos = df["texto"].astype(str).tolist()
    y_true = [_normalizar_label(v) for v in df["rotulo"].astype(str).tolist()]

    clf = pipeline("zero-shot-classification", model=MODEL_NAME)
    labels = sorted(set(y_true) | set(DEFAULT_LABELS))

    y_pred: list[str] = []
    for texto in textos:
        resultado = clf(texto, candidate_labels=labels, multi_label=False)
        top1 = resultado["labels"][0] if resultado.get("labels") else ""
        y_pred.append(_normalizar_label(top1))

    classes_sem_predicao = [lab for lab in sorted(set(y_true)) if lab not in set(y_pred)]
    warnings: list[str] = []
    if classes_sem_predicao:
        warnings.append(f"Classes sem nenhuma predicao: {', '.join(classes_sem_predicao)}")

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    f1_por_classe = {
        classe: float(report_dict.get(classe, {}).get("f1-score", 0.0))
        for classe in labels
    }

    prediction_samples: list[dict] = []
    erros_exemplos: list[dict] = []
    for idx, texto in enumerate(textos):
        sample = {
            "index": idx,
            "texto": texto,
            "rotulo_real": y_true[idx],
            "rotulo_predito": y_pred[idx],
            "acerto": y_true[idx] == y_pred[idx],
        }
        prediction_samples.append(sample)
        if not sample["acerto"] and len(erros_exemplos) < 5:
            erros_exemplos.append(sample)

    for sample in prediction_samples:
        print(
            f"[assunto] idx={sample['index']} real={sample['rotulo_real']} pred={sample['rotulo_predito']} "
            f"acerto={sample['acerto']}"
        )

    return {
        "task": "assunto",
        "total": int(len(df)),
        "labels": labels,
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "f1_por_classe": f1_por_classe,
        },
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "prediction_samples": prediction_samples,
        "erros_exemplos": erros_exemplos,
        "warnings": warnings,
    }


def avaliar(dataset_path: str | Path = DEFAULT_DATASET) -> dict:
    dataset_path = Path(dataset_path)
    df = validar_csv(
        caminho_csv=dataset_path,
        schema_esperado={"texto": str, "rotulo": str},
        aliases_colunas={"rotulo": ["label", "rotulo"]},
        colunas_obrigatorias=["texto", "rotulo"],
    )
    return avaliar_modelo(df)


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
    print(f"Accuracy: {resultado['metrics']['accuracy']:.4f}")
    print(f"F1-score (macro): {resultado['metrics']['f1_macro']:.4f}")
    print(f"F1-score (weighted): {resultado['metrics']['f1_weighted']:.4f}")
    print("\nClassification report:")
    print(resultado["classification_report_text"])


if __name__ == "__main__":
    main()
