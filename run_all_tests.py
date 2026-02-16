# -*- coding: utf-8 -*-
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts import assunto, emocao, ideologia, sentimento


VENV_DIR_NAME = ".venv"


def is_running_in_project_venv() -> bool:
    executable = Path(sys.executable).resolve()
    parts = {p.lower() for p in executable.parts}
    return VENV_DIR_NAME.lower() in parts


def exibir_status_ambiente() -> None:
    print("\n================ STATUS DO AMBIENTE ================")
    print(f"Python em uso: {sys.executable}")

    if is_running_in_project_venv():
        print("[OK] Ambiente virtual ativo detectado: .venv")
        return

    print("[ALERTA] O Python em uso nao pertence ao .venv do projeto.")
    print("[ALERTA] Recomendado recriar/ativar .venv para evitar conflitos globais.")


def _serializar_para_json(valor):
    if isinstance(valor, dict):
        return {k: _serializar_para_json(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [_serializar_para_json(v) for v in valor]
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        if np.isnan(valor):
            return None
        return float(valor)
    if isinstance(valor, float) and math.isnan(valor):
        return None
    return valor


def _salvar_plot_matriz_confusao(labels: list[str], matriz: list[list[int]], titulo: str, output_path: Path) -> None:
    arr = np.array(matriz, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(titulo)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", color="black", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _salvar_plot_ideologia(y_true: list[float], y_pred: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, c="#1f77b4", alpha=0.8)

    if y_true and y_pred:
        minimo = min(min(y_true), min(y_pred))
        maximo = max(max(y_true), max(y_pred))
        ax.plot([minimo, maximo], [minimo, maximo], linestyle="--", color="gray", linewidth=1)

    ax.set_title("Ideologia: Real x Predito")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predito")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _salvar_heatmap_correlacao(metricas_df: pd.DataFrame, output_path: Path) -> None:
    corr = metricas_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_title("Heatmap de Correlacao entre Metricas")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            valor = corr.values[i, j]
            texto = "nan" if np.isnan(valor) else f"{valor:.2f}"
            ax.text(j, i, texto, ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def gerar_plots(resultados: dict, plots_dir: Path) -> dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}

    for tarefa in ["emocao", "sentimento", "assunto"]:
        info = resultados[tarefa]
        if info.get("confusion_matrix") and info.get("labels"):
            nome = f"confusion_{tarefa}.png"
            caminho = plots_dir / nome
            _salvar_plot_matriz_confusao(
                labels=info["labels"],
                matriz=info["confusion_matrix"],
                titulo=f"Matriz de Confusao - {tarefa.title()}",
                output_path=caminho,
            )
            plot_paths[tarefa] = f"plots/{nome}"

    ideologia_info = resultados["ideologia"]
    if ideologia_info.get("task_type") == "regressao":
        nome = "ideologia_real_vs_predito.png"
        caminho = plots_dir / nome
        _salvar_plot_ideologia(
            y_true=[float(v) for v in ideologia_info.get("y_true", [])],
            y_pred=[float(v) for v in ideologia_info.get("y_pred", [])],
            output_path=caminho,
        )
        plot_paths["ideologia"] = f"plots/{nome}"

    resumo_metricas = {
        "emocao": {
            "f1_macro": resultados["emocao"]["metrics"].get("f1_macro"),
            "f1_weighted": resultados["emocao"]["metrics"].get("f1_weighted"),
        },
        "sentimento": {
            "accuracy": resultados["sentimento"]["metrics"].get("accuracy"),
            "f1_macro": resultados["sentimento"]["metrics"].get("f1_macro"),
            "f1_weighted": resultados["sentimento"]["metrics"].get("f1_weighted"),
        },
        "assunto": {
            "accuracy": resultados["assunto"]["metrics"].get("accuracy"),
            "f1_macro": resultados["assunto"]["metrics"].get("f1_macro"),
            "f1_weighted": resultados["assunto"]["metrics"].get("f1_weighted"),
        },
        "ideologia": resultados["ideologia"].get("metrics", {}),
    }
    metricas_df = pd.DataFrame(resumo_metricas).T
    metricas_df = metricas_df.apply(pd.to_numeric, errors="coerce")

    heatmap_nome = "metricas_correlacao.png"
    heatmap_path = plots_dir / heatmap_nome
    _salvar_heatmap_correlacao(metricas_df, heatmap_path)
    plot_paths["heatmap"] = f"plots/{heatmap_nome}"

    return plot_paths


def salvar_historico(resultados: dict) -> None:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_path = models_dir / "results.json"

    registro = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "venv_ativo": is_running_in_project_venv(),
        "resultados": _serializar_para_json(resultados),
    }

    historico = []
    if results_path.exists():
        try:
            historico = json.loads(results_path.read_text(encoding="utf-8"))
            if not isinstance(historico, list):
                historico = []
        except json.JSONDecodeError:
            historico = []

    historico.append(registro)
    results_path.write_text(
        json.dumps(historico, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Historico salvo em {results_path}")


def _tabela_metricas_markdown(resultados: dict) -> str:
    linhas = [
        "| Tarefa | Metrica | Valor |",
        "|---|---|---|",
    ]

    for tarefa in ["emocao", "sentimento", "assunto", "ideologia"]:
        metricas = resultados[tarefa].get("metrics", {})
        for metrica, valor in metricas.items():
            if isinstance(valor, dict):
                for sub_metrica, sub_valor in valor.items():
                    linhas.append(f"| {tarefa} | {metrica}.{sub_metrica} | {sub_valor} |")
            else:
                linhas.append(f"| {tarefa} | {metrica} | {valor} |")

    return "\n".join(linhas)


def gerar_relatorio_markdown(resultados: dict, plot_paths: dict[str, str]) -> None:
    models_dir = Path("models")
    report_path = models_dir / "results_report.md"

    warnings: list[str] = []
    for tarefa in ["emocao", "sentimento", "assunto", "ideologia"]:
        warnings.extend([f"[{tarefa}] {w}" for w in resultados[tarefa].get("warnings", [])])
        if resultados[tarefa].get("total", 0) < 10:
            warnings.append(f"[{tarefa}] dataset pequeno; interpretar metricas com cautela.")

    resumo = []
    resumo.append("# Relatorio de Avaliacao")
    resumo.append("")
    resumo.append("## Resumo por tarefa")
    resumo.append("")
    resumo.append(f"- Emocao: F1 macro={resultados['emocao']['metrics'].get('f1_macro', 0):.4f}, F1 weighted={resultados['emocao']['metrics'].get('f1_weighted', 0):.4f}")
    resumo.append(f"- Sentimento: Accuracy={resultados['sentimento']['metrics'].get('accuracy', 0):.4f}, F1 macro={resultados['sentimento']['metrics'].get('f1_macro', 0):.4f}")
    resumo.append(f"- Assunto: Accuracy={resultados['assunto']['metrics'].get('accuracy', 0):.4f}, F1 macro={resultados['assunto']['metrics'].get('f1_macro', 0):.4f}")

    if resultados["ideologia"].get("task_type") == "regressao":
        resumo.append(
            "- Ideologia: "
            f"Pearson={resultados['ideologia']['metrics'].get('pearson', 0):.4f}, "
            f"Spearman={resultados['ideologia']['metrics'].get('spearman', 0):.4f}, "
            f"MAE={resultados['ideologia']['metrics'].get('mae', 0):.4f}, "
            f"RMSE={resultados['ideologia']['metrics'].get('rmse', 0):.4f}"
        )
    else:
        resumo.append(
            "- Ideologia (classificacao): "
            f"Accuracy={resultados['ideologia']['metrics'].get('accuracy', 0):.4f}, "
            f"F1 macro={resultados['ideologia']['metrics'].get('f1_macro', 0):.4f}"
        )

    resumo.append("")
    resumo.append("## Interpretacao")
    resumo.append("")
    resumo.append("- O desempenho em classificacao e fortemente afetado pelo tamanho pequeno dos conjuntos de teste.")
    resumo.append("- Classes nao previstas tendem a reduzir macro-F1, mesmo quando accuracy parece alta.")
    resumo.append("- Na regressao de ideologia, MAE/RMSE medem erro absoluto; Pearson/Spearman medem alinhamento de tendencia.")

    resumo.append("")
    resumo.append("## Onde foi bem e onde foi mal")
    resumo.append("")
    for tarefa in ["emocao", "sentimento", "assunto", "ideologia"]:
        erros = resultados[tarefa].get("erros_exemplos", [])
        if erros:
            resumo.append(f"- {tarefa}: houve erros de previsao; exemplos abaixo ajudam na depuracao.")
        else:
            resumo.append(f"- {tarefa}: sem erros nos exemplos avaliados.")

    resumo.append("")
    resumo.append("## Sugestoes de melhoria")
    resumo.append("")
    resumo.append("- Aumentar o tamanho e diversidade dos dados rotulados.")
    resumo.append("- Balancear classes sub-representadas para reduzir vies de predicao.")
    resumo.append("- Ajustar threshold/estrategia de decisao para classes ambiguas.")
    resumo.append("- Para ideologia, validar calibracao com mais pontos e validacao cruzada.")

    resumo.append("")
    resumo.append("## Tabela completa de metricas")
    resumo.append("")
    resumo.append(_tabela_metricas_markdown(resultados))

    resumo.append("")
    resumo.append("## Alertas")
    resumo.append("")
    if warnings:
        for aviso in warnings:
            resumo.append(f"- {aviso}")
    else:
        resumo.append("- Nenhum alerta adicional.")

    resumo.append("")
    resumo.append("## Exemplos de erros")
    resumo.append("")
    for tarefa in ["emocao", "sentimento", "assunto", "ideologia"]:
        resumo.append(f"### {tarefa.title()}")
        erros = resultados[tarefa].get("erros_exemplos", [])
        if not erros:
            resumo.append("- Sem erros para listar.")
            continue
        for erro in erros[:5]:
            if "acerto" in erro:
                resumo.append(
                    f"- idx={erro['index']} real={erro['rotulo_real']} predito={erro['rotulo_predito']} texto=\"{erro['texto']}\""
                )
            else:
                resumo.append(
                    f"- idx={erro['index']} real={erro['rotulo_real']:.4f} predito={erro['rotulo_predito']:.4f} "
                    f"erro={erro['erro_absoluto']:.4f} texto=\"{erro['texto']}\""
                )

    resumo.append("")
    resumo.append("## Graficos")
    resumo.append("")
    if "emocao" in plot_paths:
        resumo.append("### Matriz de confusao - Emocao")
        resumo.append(f"![Emocao]({plot_paths['emocao']})")
        resumo.append("")
    if "sentimento" in plot_paths:
        resumo.append("### Matriz de confusao - Sentimento")
        resumo.append(f"![Sentimento]({plot_paths['sentimento']})")
        resumo.append("")
    if "assunto" in plot_paths:
        resumo.append("### Matriz de confusao - Assunto")
        resumo.append(f"![Assunto]({plot_paths['assunto']})")
        resumo.append("")
    if "ideologia" in plot_paths:
        resumo.append("### Dispersao real x predito - Ideologia")
        resumo.append(f"![Ideologia]({plot_paths['ideologia']})")
        resumo.append("")

    resumo.append("### Heatmap de correlacao")
    resumo.append(f"![Correlacao]({plot_paths['heatmap']})")

    report_path.write_text("\n".join(resumo), encoding="utf-8")
    print(f"[OK] Relatorio salvo em {report_path}")


def main() -> None:
    resultados = {}

    exibir_status_ambiente()

    print("\n===== EXECUTANDO TESTE: EMOCAO =====")
    resultados["emocao"] = emocao.avaliar()

    print("\n===== EXECUTANDO TESTE: SENTIMENTO =====")
    resultados["sentimento"] = sentimento.avaliar()

    print("\n===== EXECUTANDO TESTE: IDEOLOGIA =====")
    resultados["ideologia"] = ideologia.avaliar()

    print("\n===== EXECUTANDO TESTE: ASSUNTO =====")
    resultados["assunto"] = assunto.avaliar()

    print("\n================ RESUMO FINAL ================")
    print(
        "Emocao -> "
        f"F1 macro: {resultados['emocao']['metrics'].get('f1_macro', 0):.4f}, "
        f"F1 weighted: {resultados['emocao']['metrics'].get('f1_weighted', 0):.4f}"
    )
    print(
        "Sentimento -> "
        f"Accuracy: {resultados['sentimento']['metrics'].get('accuracy', 0):.4f}, "
        f"F1 macro: {resultados['sentimento']['metrics'].get('f1_macro', 0):.4f}"
    )
    print(
        "Assunto -> "
        f"Accuracy: {resultados['assunto']['metrics'].get('accuracy', 0):.4f}, "
        f"F1 macro: {resultados['assunto']['metrics'].get('f1_macro', 0):.4f}"
    )
    if resultados["ideologia"].get("task_type") == "regressao":
        print(
            "Ideologia -> "
            f"Pearson: {resultados['ideologia']['metrics'].get('pearson', 0):.4f}, "
            f"Spearman: {resultados['ideologia']['metrics'].get('spearman', 0):.4f}, "
            f"MAE: {resultados['ideologia']['metrics'].get('mae', 0):.4f}, "
            f"RMSE: {resultados['ideologia']['metrics'].get('rmse', 0):.4f}"
        )
    else:
        print(
            "Ideologia (classificacao) -> "
            f"Accuracy: {resultados['ideologia']['metrics'].get('accuracy', 0):.4f}, "
            f"F1 macro: {resultados['ideologia']['metrics'].get('f1_macro', 0):.4f}"
        )

    plots_dir = Path("models") / "plots"
    plot_paths = gerar_plots(resultados=resultados, plots_dir=plots_dir)
    salvar_historico(resultados)
    gerar_relatorio_markdown(resultados, plot_paths)


if __name__ == "__main__":
    main()

