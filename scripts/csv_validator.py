# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CsvValidationIssue:
    line: int | None
    error_type: str
    message: str


class CsvValidationError(Exception):
    def __init__(self, arquivo: Path, issues: list[CsvValidationIssue]) -> None:
        self.arquivo = arquivo
        self.issues = issues
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        linhas = [f"Validacao CSV falhou para '{self.arquivo}'."]
        for issue in self.issues:
            origem = f"linha {issue.line}" if issue.line is not None else "arquivo"
            linhas.append(f"- [{issue.error_type}] {origem}: {issue.message}")
        return "\n".join(linhas)


def _normalizar_aliases(aliases_colunas: dict[str, list[str]] | None) -> dict[str, set[str]]:
    aliases = aliases_colunas or {}
    normalizado: dict[str, set[str]] = {}
    for canonical, alternatives in aliases.items():
        chave = canonical.strip().lower()
        valores = {chave}
        valores.update(item.strip().lower() for item in alternatives)
        normalizado[chave] = valores
    return normalizado


def _detectar_delimitador(conteudo: str, delimitador_esperado: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(conteudo[:4096], delimiters=[",", ";", "\t", "|"])
        return str(dialect.delimiter)
    except csv.Error:
        return delimitador_esperado


def _validar_numero_colunas(conteudo: str) -> list[CsvValidationIssue]:
    rows = list(csv.reader(conteudo.splitlines()))
    if not rows:
        return [CsvValidationIssue(line=None, error_type="ARQUIVO_VAZIO", message="Arquivo CSV vazio.")]

    total_colunas = len(rows[0])
    if total_colunas == 0:
        return [
            CsvValidationIssue(
                line=1,
                error_type="CABECALHO_INVALIDO",
                message="Cabecalho sem colunas.",
            )
        ]

    issues: list[CsvValidationIssue] = []
    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != total_colunas:
            issues.append(
                CsvValidationIssue(
                    line=idx,
                    error_type="NUM_COLUNAS_INVALIDO",
                    message=f"Esperado {total_colunas}, encontrado {len(row)}.",
                )
            )
    return issues


def _linhas_tipo_invalido(series: pd.Series, expected_type: type) -> list[int]:
    invalidas: list[int] = []
    for idx, value in series.items():
        if pd.isna(value):
            continue
        if expected_type is str:
            if not isinstance(value, str):
                invalidas.append(int(idx) + 2)
            continue
        if expected_type is int:
            try:
                numero = float(value)
                if not numero.is_integer():
                    invalidas.append(int(idx) + 2)
            except (TypeError, ValueError):
                invalidas.append(int(idx) + 2)
            continue
        if expected_type is float:
            try:
                float(value)
            except (TypeError, ValueError):
                invalidas.append(int(idx) + 2)
    return invalidas


def validar_csv(
    caminho_csv: str | Path,
    schema_esperado: dict[str, type] | None = None,
    aliases_colunas: dict[str, list[str]] | None = None,
    colunas_obrigatorias: list[str] | None = None,
    delimitador_esperado: str = ",",
) -> pd.DataFrame:
    caminho = Path(caminho_csv)
    schema = {chave.strip().lower(): tipo for chave, tipo in (schema_esperado or {}).items()}
    obrigatorias = [col.strip().lower() for col in (colunas_obrigatorias or [])]
    aliases = _normalizar_aliases(aliases_colunas)

    try:
        if not caminho.exists() or not caminho.is_file():
            raise CsvValidationError(
                caminho,
                [CsvValidationIssue(line=None, error_type="ARQUIVO_NAO_ENCONTRADO", message="Arquivo inexistente.")],
            )

        try:
            conteudo = caminho.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise CsvValidationError(
                caminho,
                [
                    CsvValidationIssue(
                        line=None,
                        error_type="ENCODING_INVALIDO",
                        message=f"O arquivo precisa estar em UTF-8: {exc}",
                    )
                ],
            ) from exc
        except OSError as exc:
            raise CsvValidationError(
                caminho,
                [CsvValidationIssue(line=None, error_type="LEITURA_FALHOU", message=str(exc))],
            ) from exc

        issues: list[CsvValidationIssue] = []
        delimitador_detectado = _detectar_delimitador(conteudo, delimitador_esperado)
        if delimitador_detectado != delimitador_esperado:
            issues.append(
                CsvValidationIssue(
                    line=1,
                    error_type="DELIMITADOR_INVALIDO",
                    message=f"Detectado '{delimitador_detectado}', esperado '{delimitador_esperado}'.",
                )
            )

        issues.extend(_validar_numero_colunas(conteudo))
        if issues:
            raise CsvValidationError(caminho, issues)

        try:
            df = pd.read_csv(
                caminho,
                sep=delimitador_esperado,
                encoding="utf-8",
                on_bad_lines="error",
            )
        except Exception as exc:
            raise CsvValidationError(
                caminho,
                [
                    CsvValidationIssue(
                        line=None,
                        error_type="ERRO_PANDAS",
                        message=f"Falha na leitura do CSV: {exc}",
                    )
                ],
            ) from exc

        rename_map: dict[str, str] = {}
        colunas_destino: set[str] = set()
        for coluna in df.columns:
            coluna_norm = str(coluna).strip().lower()
            destino = coluna_norm
            for canonica, aceitaveis in aliases.items():
                if coluna_norm in aceitaveis:
                    destino = canonica
                    break

            if destino in colunas_destino and coluna != destino:
                raise CsvValidationError(
                    caminho,
                    [
                        CsvValidationIssue(
                            line=1,
                            error_type="COLUNA_DUPLICADA_APOS_ALIAS",
                            message=f"Mais de uma coluna mapeada para '{destino}'.",
                        )
                    ],
                )
            rename_map[coluna] = destino
            colunas_destino.add(destino)

        df = df.rename(columns=rename_map)

        faltantes = [col for col in obrigatorias if col not in df.columns]
        if faltantes:
            raise CsvValidationError(
                caminho,
                [
                    CsvValidationIssue(
                        line=1,
                        error_type="COLUNA_OBRIGATORIA_AUSENTE",
                        message=f"Ausentes: {', '.join(faltantes)}.",
                    )
                ],
            )

        for coluna in obrigatorias:
            nulos = df[coluna].isna()
            if coluna in schema and schema[coluna] is str:
                nulos = nulos | df[coluna].astype(str).str.strip().eq("")
            if nulos.any():
                linhas = [int(i) + 2 for i in df.index[nulos].tolist()[:10]]
                sufixo = "..." if int(nulos.sum()) > 10 else ""
                raise CsvValidationError(
                    caminho,
                    [
                        CsvValidationIssue(
                            line=None,
                            error_type="VALOR_NULO",
                            message=f"Coluna '{coluna}' com nulos/vazios nas linhas: {linhas}{sufixo}.",
                        )
                    ],
                )

        for coluna, expected_type in schema.items():
            if coluna not in df.columns:
                continue
            if expected_type not in {str, int, float}:
                raise CsvValidationError(
                    caminho,
                    [
                        CsvValidationIssue(
                            line=1,
                            error_type="TIPO_NAO_SUPORTADO",
                            message=f"Tipo '{expected_type}' nao suportado em '{coluna}'.",
                        )
                    ],
                )

            invalidas = _linhas_tipo_invalido(df[coluna], expected_type)
            if invalidas:
                sufixo = "..." if len(invalidas) > 10 else ""
                raise CsvValidationError(
                    caminho,
                    [
                        CsvValidationIssue(
                            line=None,
                            error_type="TIPO_INVALIDO",
                            message=(
                                f"Coluna '{coluna}' esperava {expected_type.__name__}. "
                                f"Linhas invalidas: {invalidas[:10]}{sufixo}."
                            ),
                        )
                    ],
                )

        return df

    except CsvValidationError as exc:
        print("[ERRO] Falha na validacao de CSV.")
        print(str(exc))
        raise
