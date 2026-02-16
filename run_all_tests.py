# -*- coding: utf-8 -*-
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    print("[ALERTA] Comandos sugeridos:")
    print("         powershell .\\activate_venv.ps1")
    print("         .\\install_requirements.bat")


def exibir_contexto_execucao() -> None:
    print("\n================ CONTEXTO TECNICO DO PROJETO ================")
    print("Objetivo: validar 4 tarefas de NLP com modelos Transformers.")
    print("\nEstrutura principal:")
    print("- Scripts de teste: scripts/emocao.py, scripts/sentimento.py, scripts/ideologia.py, scripts/assunto.py")
    print("- Dados de entrada (CSV):")
    print("  data/emocao.csv")
    print("  data/sentimento.csv")
    print("  data/ideologia.csv")
    print("  data/assunto.csv")
    print("\nMetricas por tarefa:")
    print("- Emocao: Precision, Recall, F1 (classificacao)")
    print("- Sentimento: classification_report (Precision, Recall, F1)")
    print("- Ideologia: Pearson, MAE, R2 (regressao)")
    print("- Assunto: matriz de confusao, Accuracy, F1")
    print("\nFluxo esperado de uso:")
    print("1) powershell .\\activate_venv.ps1")
    print("2) .\\install_requirements.bat")
    print("3) python validate_scripts.py")
    print("4) python run_all_tests.py")
    print("\nObservacao:")
    print("- Os resultados desta execucao sao exibidos no console.")
    print("- Os resultados tambem sao salvos em models/results.json.")


def salvar_historico(resultados: dict) -> None:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_path = models_dir / "results.json"

    registro = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "venv_ativo": is_running_in_project_venv(),
        "resultados": {
            "emocao": {
                "precision": resultados["emocao"]["precision"],
                "recall": resultados["emocao"]["recall"],
                "f1": resultados["emocao"]["f1"],
                "total": resultados["emocao"]["total"],
            },
            "sentimento": {
                "precision": resultados["sentimento"]["precision"],
                "recall": resultados["sentimento"]["recall"],
                "f1": resultados["sentimento"]["f1"],
                "total": resultados["sentimento"]["total"],
            },
            "ideologia": {
                "pearson": resultados["ideologia"]["pearson"],
                "mae": resultados["ideologia"]["mae"],
                "r2": resultados["ideologia"]["r2"],
                "total": resultados["ideologia"]["total"],
            },
            "assunto": {
                "accuracy": resultados["assunto"]["accuracy"],
                "f1": resultados["assunto"]["f1"],
                "total": resultados["assunto"]["total"],
            },
        },
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


def main() -> None:
    resultados = {}

    exibir_status_ambiente()
    exibir_contexto_execucao()

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
        f"Precision: {resultados['emocao']['precision']:.4f}, "
        f"Recall: {resultados['emocao']['recall']:.4f}, "
        f"F1: {resultados['emocao']['f1']:.4f}"
    )
    print(
        "Sentimento -> "
        f"Precision: {resultados['sentimento']['precision']:.4f}, "
        f"Recall: {resultados['sentimento']['recall']:.4f}, "
        f"F1: {resultados['sentimento']['f1']:.4f}"
    )
    print(
        "Ideologia -> "
        f"Pearson: {resultados['ideologia']['pearson']:.4f}, "
        f"MAE: {resultados['ideologia']['mae']:.4f}, "
        f"R2: {resultados['ideologia']['r2']:.4f}"
    )
    print(
        "Assunto -> "
        f"Accuracy: {resultados['assunto']['accuracy']:.4f}, "
        f"F1: {resultados['assunto']['f1']:.4f}"
    )

    salvar_historico(resultados)


if __name__ == "__main__":
    main()
