# -*- coding: utf-8 -*-
import compileall
import sys
from pathlib import Path


def main() -> None:
    scripts_dir = Path("scripts")

    if not scripts_dir.exists() or not scripts_dir.is_dir():
        print("[ERRO] Pasta 'scripts' nao encontrada.")
        sys.exit(1)

    print("[INFO] Validando sintaxe dos arquivos Python em 'scripts'...")
    sucesso = compileall.compile_dir(str(scripts_dir), force=True, quiet=1)

    if sucesso:
        print("[OK] Validacao concluida: nenhum erro de sintaxe encontrado em 'scripts'.")
        return

    print("[ERRO] Foram encontrados erros de sintaxe em 'scripts'.")
    print("[DICA] Revise colchetes, parenteses e listas nao fechadas.")
    sys.exit(1)


if __name__ == "__main__":
    main()
