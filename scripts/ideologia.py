# -*- coding: utf-8 -*-
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

try:
    from scripts.csv_validator import validar_csv
except ModuleNotFoundError:
    from csv_validator import validar_csv

MODEL_NAME = "distilbert-base-uncased"
DEFAULT_DATASET = Path("data/ideologia.csv")
DEFAULT_MODEL_DIR = Path("models/ideologia_model")


class IdeologiaDataset(Dataset):
    def __init__(self, textos: list[str], labels: list[float] | list[int], tokenizer) -> None:
        self.textos = textos
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.textos)

    def __getitem__(self, idx: int) -> dict:
        texto = self.textos[idx]
        item = self.tokenizer(texto, truncation=True)
        label_bruto = self.labels[idx]
        if isinstance(label_bruto, float):
            label = torch.tensor(label_bruto, dtype=torch.float32)
        else:
            label = torch.tensor(label_bruto, dtype=torch.long)
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": label,
        }


def _normalizar_label(valor: str) -> str:
    return str(valor).strip().lower()


def detectar_tarefa(rotulos: pd.Series, override: str = "auto") -> str:
    if override in {"regressao", "classificacao"}:
        return override

    numericos = pd.to_numeric(rotulos, errors="coerce")
    if numericos.isna().any():
        return "classificacao"

    if numericos.between(-1, 1).all():
        valores = numericos.to_numpy(dtype=float)
        unicos = np.unique(valores)
        sao_discretos = np.all(np.isclose(unicos, np.round(unicos))) and len(unicos) <= 5
        return "classificacao" if sao_discretos else "regressao"

    unicos = np.unique(numericos.to_numpy(dtype=float))
    sao_discretos = np.all(np.isclose(unicos, np.round(unicos))) and len(unicos) <= 10
    return "classificacao" if sao_discretos else "regressao"


def baseline_wordfish(
    x_train: list[str],
    y_train: np.ndarray,
    x_test: list[str],
    tarefa: str,
) -> np.ndarray:
    vetor = CountVectorizer(ngram_range=(1, 2), min_df=1)
    xtr = vetor.fit_transform(x_train)
    xte = vetor.transform(x_test)

    if tarefa == "regressao":
        modelo = LinearRegression()
        modelo.fit(xtr, y_train.astype(float))
        return modelo.predict(xte).astype(float)

    modelo = LogisticRegression(max_iter=500)
    modelo.fit(xtr, y_train.astype(int))
    return modelo.predict(xte).astype(float)


def _treinar(
    model,
    train_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    epochs: int,
) -> None:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()


def _prever(model, test_loader: DataLoader, device: torch.device, tarefa: str) -> np.ndarray:
    model.to(device)
    model.eval()
    preds: list[float] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            infer_batch = {k: v for k, v in batch.items() if k != "labels"}
            logits = model(**infer_batch).logits
            if tarefa == "regressao":
                valores = logits.squeeze(-1).detach().cpu().numpy().astype(float).tolist()
            else:
                valores = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(float).tolist()
            preds.extend(valores)
    return np.array(preds, dtype=float)


def _avaliar_regressao(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    serie_true = pd.Series(y_true)
    serie_pred = pd.Series(y_pred)
    spearman = float(serie_true.corr(serie_pred, method="spearman")) if len(serie_true) > 1 else 0.0
    if np.isnan(spearman):
        spearman = 0.0

    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def avaliar_modelo(
    textos_teste: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tarefa: str,
    classes: list[str] | None = None,
) -> dict:
    warnings: list[str] = []

    if tarefa == "regressao":
        metrics = _avaliar_regressao(y_true.astype(float), y_pred.astype(float))
        prediction_samples: list[dict] = []
        erros_exemplos: list[dict] = []

        for idx, texto in enumerate(textos_teste):
            erro_abs = abs(float(y_true[idx]) - float(y_pred[idx]))
            sample = {
                "index": idx,
                "texto": texto,
                "rotulo_real": float(y_true[idx]),
                "rotulo_predito": float(y_pred[idx]),
                "erro_absoluto": float(erro_abs),
            }
            prediction_samples.append(sample)

        ordenados = sorted(prediction_samples, key=lambda x: x["erro_absoluto"], reverse=True)
        erros_exemplos = ordenados[:5]

        for sample in prediction_samples:
            print(
                f"[ideologia] idx={sample['index']} real={sample['rotulo_real']:.4f} "
                f"pred={sample['rotulo_predito']:.4f} erro={sample['erro_absoluto']:.4f}"
            )

        if len(textos_teste) < 10:
            warnings.append("Conjunto de teste pequeno; metricas podem oscilar bastante.")

        return {
            "task": "ideologia",
            "task_type": "regressao",
            "total_teste": int(len(textos_teste)),
            "metrics": metrics,
            "y_true": y_true.astype(float).tolist(),
            "y_pred": y_pred.astype(float).tolist(),
            "prediction_samples": prediction_samples,
            "erros_exemplos": erros_exemplos,
            "warnings": warnings,
        }

    classes = classes or sorted(set(y_true.astype(int).tolist()) | set(y_pred.astype(int).tolist()))
    y_true_cls = [str(int(v)) for v in y_true.tolist()]
    y_pred_cls = [str(int(v)) for v in y_pred.tolist()]
    labels = [str(c) for c in classes]

    classes_sem_predicao = [lab for lab in sorted(set(y_true_cls)) if lab not in set(y_pred_cls)]
    if classes_sem_predicao:
        warnings.append(f"Classes sem nenhuma predicao: {', '.join(classes_sem_predicao)}")

    report_dict = classification_report(
        y_true_cls,
        y_pred_cls,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true_cls,
        y_pred_cls,
        labels=labels,
        zero_division=0,
    )

    prediction_samples = []
    erros_exemplos = []
    for idx, texto in enumerate(textos_teste):
        sample = {
            "index": idx,
            "texto": texto,
            "rotulo_real": y_true_cls[idx],
            "rotulo_predito": y_pred_cls[idx],
            "acerto": y_true_cls[idx] == y_pred_cls[idx],
        }
        prediction_samples.append(sample)
        if not sample["acerto"] and len(erros_exemplos) < 5:
            erros_exemplos.append(sample)

    for sample in prediction_samples:
        print(
            f"[ideologia] idx={sample['index']} real={sample['rotulo_real']} pred={sample['rotulo_predito']} "
            f"acerto={sample['acerto']}"
        )

    return {
        "task": "ideologia",
        "task_type": "classificacao",
        "total_teste": int(len(textos_teste)),
        "labels": labels,
        "metrics": {
            "accuracy": float(accuracy_score(y_true_cls, y_pred_cls)),
            "f1_macro": float(f1_score(y_true_cls, y_pred_cls, labels=labels, average="macro", zero_division=0)),
        },
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "y_true": y_true_cls,
        "y_pred": y_pred_cls,
        "prediction_samples": prediction_samples,
        "erros_exemplos": erros_exemplos,
        "warnings": warnings,
    }


def avaliar(
    dataset_path: str | Path = DEFAULT_DATASET,
    modo: str = "completo",
    tarefa: str = "auto",
    epochs: int = 4,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    usar_wordfish: bool = False,
) -> dict:
    dataset_path = Path(dataset_path)
    model_dir = Path(model_dir)
    df = validar_csv(
        caminho_csv=dataset_path,
        schema_esperado={"texto": str, "rotulo": float},
        aliases_colunas={"rotulo": ["label", "rotulo"]},
        colunas_obrigatorias=["texto", "rotulo"],
    )

    df = df.dropna(subset=["texto", "rotulo"]).copy()
    if len(df) < 2:
        raise ValueError("Dataset precisa de pelo menos 2 amostras para split treino/teste.")

    tarefa_detectada = detectar_tarefa(df["rotulo"], override=tarefa)

    x = df["texto"].astype(str).tolist()
    y_raw = df["rotulo"]

    if tarefa_detectada == "regressao":
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise ValueError("Rotulos invalidos para regressao em 'rotulo'.")
        stratify = None
        num_labels = 1
        problem_type = "regression"
        id2label = {0: "score_ideologico"}
        label2id = {"score_ideologico": 0}
        classes = None
    else:
        classes = sorted({_normalizar_label(v) for v in y_raw.astype(str).tolist()})
        class_to_id = {classe: idx for idx, classe in enumerate(classes)}
        y = y_raw.astype(str).map(lambda v: class_to_id[_normalizar_label(v)]).to_numpy(dtype=int)
        stratify = y if len(np.unique(y)) > 1 else None
        num_labels = len(classes)
        problem_type = "single_label_classification"
        id2label = {idx: classe for classe, idx in class_to_id.items()}
        label2id = class_to_id

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if modo == "completo":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            problem_type=problem_type,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    elif modo == "rapido":
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Modelo salvo nao encontrado em '{model_dir}'. Rode com --modo completo primeiro."
            )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        raise ValueError("Modo invalido. Use 'completo' ou 'rapido'.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_dataset = IdeologiaDataset(
        textos=x_train,
        labels=y_train.tolist(),
        tokenizer=tokenizer,
    )
    test_dataset = IdeologiaDataset(
        textos=x_test,
        labels=y_test.tolist(),
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    if modo == "completo":
        _treinar(
            model=model,
            train_loader=train_loader,
            device=device,
            learning_rate=learning_rate,
            epochs=epochs,
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    y_pred = _prever(model=model, test_loader=test_loader, device=device, tarefa=tarefa_detectada)

    resultado_avaliacao = avaliar_modelo(
        textos_teste=x_test,
        y_true=y_test.astype(float),
        y_pred=y_pred.astype(float),
        tarefa=tarefa_detectada,
        classes=classes,
    )

    if usar_wordfish:
        y_baseline = baseline_wordfish(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            tarefa=tarefa_detectada,
        )
        if tarefa_detectada == "regressao":
            baseline_metricas = _avaliar_regressao(y_test.astype(float), y_baseline.astype(float))
        else:
            baseline_metricas = {
                "accuracy": float(accuracy_score(y_test.astype(int), y_baseline.astype(int))),
                "f1_macro": float(f1_score(y_test.astype(int), y_baseline.astype(int), average="macro", zero_division=0)),
            }
    else:
        baseline_metricas = None

    return {
        "task": "ideologia",
        "task_type": tarefa_detectada,
        "mode": modo,
        "model_name": MODEL_NAME,
        "model_dir": str(model_dir),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "total": int(len(df)),
        "metrics": resultado_avaliacao["metrics"],
        "y_true": resultado_avaliacao["y_true"],
        "y_pred": resultado_avaliacao["y_pred"],
        "prediction_samples": resultado_avaliacao["prediction_samples"],
        "erros_exemplos": resultado_avaliacao["erros_exemplos"],
        "warnings": resultado_avaliacao["warnings"],
        "labels": resultado_avaliacao.get("labels"),
        "classification_report_text": resultado_avaliacao.get("classification_report_text"),
        "classification_report_dict": resultado_avaliacao.get("classification_report_dict"),
        "wordfish_baseline": baseline_metricas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treino e avaliacao da tarefa de ideologia (classificacao ou regressao)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Caminho do CSV com colunas texto, rotulo",
    )
    parser.add_argument(
        "--modo",
        type=str,
        default="completo",
        choices=["completo", "rapido"],
        help="completo: treina+avalia | rapido: carrega modelo salvo e apenas avalia",
    )
    parser.add_argument(
        "--tarefa",
        type=str,
        default="auto",
        choices=["auto", "regressao", "classificacao"],
        help="Detecta automaticamente ou forca tipo da tarefa.",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Numero de epocas (sugestao: 3 a 5).")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate do AdamW.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size de treino/inferencia.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Diretorio para salvar/carregar o modelo treinado.",
    )
    parser.add_argument(
        "--wordfish-baseline",
        action="store_true",
        help="Calcula baseline estatistico simples inspirado em Wordfish.",
    )
    args = parser.parse_args()

    resultado = avaliar(
        dataset_path=args.dataset,
        modo=args.modo,
        tarefa=args.tarefa,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
        usar_wordfish=args.wordfish_baseline,
    )

    print("=== Ideologia ===")
    print(f"Tarefa: {resultado['task_type']}")
    print(f"Modo: {resultado['mode']}")
    print(f"Modelo base: {resultado['model_name']}")
    print(f"Modelo salvo: {resultado['model_dir']}")
    print(f"Amostras: {resultado['total']} (treino={resultado['train_size']}, teste={resultado['test_size']})")
    for chave, valor in resultado["metrics"].items():
        print(f"{chave}: {valor:.4f}")
    if resultado["warnings"]:
        print("Alertas:")
        for aviso in resultado["warnings"]:
            print(f"- {aviso}")
    if resultado["classification_report_text"]:
        print("\nClassification report:")
        print(resultado["classification_report_text"])
    if resultado["wordfish_baseline"] is not None:
        print("\nBaseline (Wordfish simplificado):")
        for chave, valor in resultado["wordfish_baseline"].items():
            print(f"{chave}: {valor:.4f}")


if __name__ == "__main__":
    main()
