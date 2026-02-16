# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import pipeline

try:
    from scripts.csv_validator import validar_csv
except ModuleNotFoundError:
    from csv_validator import validar_csv

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_DATASET = Path("data/emocao.csv")


def _normalizar_label(valor: str) -> str:
    return str(valor).strip().lower()


def _extrair_top1(previsao: object) -> str:
    if isinstance(previsao, dict):
        return _normalizar_label(previsao.get("label", ""))
    if isinstance(previsao, list) and previsao:
        primeiro = previsao[0]
        if isinstance(primeiro, dict):
            return _normalizar_label(primeiro.get("label", ""))
    return _normalizar_label(str(previsao))


def _extrair_multilabel(previsao: object, threshold: float = 0.35) -> set[str]:
    labels_preditos: set[str] = set()
    if isinstance(previsao, list):
        for item in previsao:
            if isinstance(item, dict) and float(item.get("score", 0.0)) >= threshold:
                labels_preditos.add(_normalizar_label(item.get("label", "")))
    if not labels_preditos:
        labels_preditos.add(_extrair_top1(previsao))
    return labels_preditos


def avaliar_modelo(df) -> dict:
    textos = df["texto"].astype(str).tolist()
    y_true_raw = df["rotulo"].astype(str).tolist()
    y_true_norm = [_normalizar_label(rot) for rot in y_true_raw]
    tarefa_multilabel = any("|" in rot for rot in y_true_norm)

    warnings: list[str] = []
    erros_exemplos: list[dict] = []

    clf = pipeline("text-classification", model=MODEL_NAME, top_k=None if tarefa_multilabel else 1)
    previsoes = clf(textos)

    prediction_samples: list[dict] = []

    if tarefa_multilabel:
        y_true_sets = [set(part.strip() for part in rot.split("|")) for rot in y_true_norm]
        y_pred_sets = [_extrair_multilabel(prev) for prev in previsoes]

        labels = sorted({label for conj in y_true_sets for label in conj} | {label for conj in y_pred_sets for label in conj})
        mlb = MultiLabelBinarizer(classes=labels)
        y_true_bin = mlb.fit_transform(y_true_sets)
        y_pred_bin = mlb.transform(y_pred_sets)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_bin,
            y_pred_bin,
            average="macro",
            zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_bin,
            y_pred_bin,
            average="weighted",
            zero_division=0,
        )

        report_dict = classification_report(
            y_true_bin,
            y_pred_bin,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        )
        report_text = classification_report(
            y_true_bin,
            y_pred_bin,
            target_names=labels,
            zero_division=0,
        )

        y_true_print = ["|".join(sorted(item)) for item in y_true_sets]
        y_pred_print = ["|".join(sorted(item)) for item in y_pred_sets]
        confusion = None
    else:
        y_pred_norm = [_extrair_top1(prev) for prev in previsoes]
        labels = sorted(set(y_true_norm) | set(y_pred_norm))
        classes_sem_predicao = [lab for lab in sorted(set(y_true_norm)) if lab not in set(y_pred_norm)]
        if classes_sem_predicao:
            warnings.append(f"Classes sem nenhuma predicao: {', '.join(classes_sem_predicao)}")

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_norm,
            y_pred_norm,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_norm,
            y_pred_norm,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        report_dict = classification_report(
            y_true_norm,
            y_pred_norm,
            labels=labels,
            zero_division=0,
            output_dict=True,
        )
        report_text = classification_report(
            y_true_norm,
            y_pred_norm,
            labels=labels,
            zero_division=0,
        )
        confusion = confusion_matrix(y_true_norm, y_pred_norm, labels=labels).tolist()
        y_true_print = y_true_norm
        y_pred_print = y_pred_norm

    for idx, texto in enumerate(textos):
        sample = {
            "index": idx,
            "texto": texto,
            "rotulo_real": y_true_print[idx],
            "rotulo_predito": y_pred_print[idx],
            "acerto": y_true_print[idx] == y_pred_print[idx],
        }
        prediction_samples.append(sample)
        if not sample["acerto"] and len(erros_exemplos) < 5:
            erros_exemplos.append(sample)

    for sample in prediction_samples:
        print(
            f"[emocao] idx={sample['index']} real={sample['rotulo_real']} pred={sample['rotulo_predito']} "
            f"acerto={sample['acerto']}"
        )

    return {
        "task": "emocao",
        "total": int(len(df)),
        "task_mode": "multilabel" if tarefa_multilabel else "multiclasse",
        "labels": labels,
        "metrics": {
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
        },
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": confusion,
        "y_true": y_true_print,
        "y_pred": y_pred_print,
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
    print(f"Precision (macro): {resultado['metrics']['precision_macro']:.4f}")
    print(f"Recall (macro): {resultado['metrics']['recall_macro']:.4f}")
    print(f"F1-score (macro): {resultado['metrics']['f1_macro']:.4f}")
    print(f"F1-score (weighted): {resultado['metrics']['f1_weighted']:.4f}")
    if resultado["warnings"]:
        print("Alertas:")
        for aviso in resultado["warnings"]:
            print(f"- {aviso}")
    print("\nClassification report:")
    print(resultado["classification_report_text"])


if __name__ == "__main__":
    main()
