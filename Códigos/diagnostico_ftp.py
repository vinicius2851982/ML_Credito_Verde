"""
diagnostico_ftp.py  v2  [OBSOLETO — NAO FAZ PARTE DO PIPELINE]
------------------------------------------------------------------
OBSOLETO: utilitario de diagnostico da epoca em que se tentou baixar a RAIS/
CAGED por FTP. O pipeline atual le ARQUIVOS LOCAIS (ver etl_social.py); este
script foi mantido apenas como registro historico e nao deve ser executado.
------------------------------------------------------------------
Explora a estrutura real do FTP do MTE usando cwd + nlst()
(sem path no NLST — evita problemas com espacos e restricoes FTP).

Uso:
  python diagnostico_ftp.py
"""

import ftplib
import os
import sys

FTP_HOST = "ftp.mtps.gov.br"

def conectar():
    ftp = ftplib.FTP(FTP_HOST, timeout=60)
    ftp.login()
    ftp.set_pasv(True)
    return ftp

def listar_cwd(ftp, path, prefixo="", max_items=20):
    """Lista diretorio usando cwd + nlst() sem argumento (fix para paths com espaco)."""
    try:
        ftp.cwd(path)
        nomes = ftp.nlst()
        nomes = [os.path.basename(n) for n in nomes]
        print(f"{prefixo}[OK] {path} -> {len(nomes)} itens")
        for n in nomes[:max_items]:
            print(f"{prefixo}     {n}")
        if len(nomes) > max_items:
            print(f"{prefixo}     ... (+{len(nomes)-max_items} mais)")
        return nomes
    except Exception as e:
        print(f"{prefixo}[ERRO] {path}: {e}")
        return []

def main():
    print("=" * 65)
    print(" DIAGNOSTICO FTP v2 - ftp.mtps.gov.br (cwd + nlst fix)")
    print("=" * 65)

    try:
        ftp = conectar()
        print(f"[OK] Conectado a {FTP_HOST}\n")
    except Exception as e:
        print(f"[ERRO] Nao foi possivel conectar: {e}")
        sys.exit(1)

    # 1. Raiz
    print("\n--- /pdet/microdados ---")
    listar_cwd(ftp, "/pdet/microdados")

    # 2. RAIS
    print("\n--- RAIS root ---")
    anos_rais = listar_cwd(ftp, "/pdet/microdados/RAIS")

    for ano in ["2023", "2022", "2021", "2020"]:
        if ano in anos_rais:
            print(f"\n--- RAIS/{ano} ---")
            arqs = listar_cwd(ftp, f"/pdet/microdados/RAIS/{ano}", max_items=10)
            if arqs:
                break   # basta ver um ano que funcione

    # 3. NOVO CAGED
    print("\n--- NOVO CAGED root ---")
    itens_nc = listar_cwd(ftp, "/pdet/microdados/NOVO CAGED")

    if itens_nc:
        # Tenta o primeiro item para descobrir a estrutura
        primeiro = itens_nc[0]
        print(f"\n--- NOVO CAGED/{primeiro} (primeiro item) ---")
        subitens = listar_cwd(ftp, f"/pdet/microdados/NOVO CAGED/{primeiro}", max_items=5)
        if subitens:
            # Ve se tem sub-subdiretorio
            print(f"\n--- NOVO CAGED/{primeiro}/{subitens[0]} (sub-subdir) ---")
            listar_cwd(ftp, f"/pdet/microdados/NOVO CAGED/{primeiro}/{subitens[0]}", max_items=5)

        # Testa 202001 diretamente
        print(f"\n--- NOVO CAGED/202001 (sem subfolder de ano) ---")
        listar_cwd(ftp, "/pdet/microdados/NOVO CAGED/202001", max_items=5)

    # 4. CAGED (antigo)
    print("\n--- CAGED (antigo) root ---")
    listar_cwd(ftp, "/pdet/microdados/CAGED", max_items=10)

    # 5. Colunas do arquivo RAIS se existir localmente
    dados_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Dados", "rais_estab"
    )
    txt_encontrado = None
    if os.path.exists(dados_dir):
        for f in os.listdir(dados_dir):
            if f.upper().endswith(".TXT"):
                txt_encontrado = os.path.join(dados_dir, f)
                break

    if txt_encontrado:
        print(f"\n--- COLUNAS: {os.path.basename(txt_encontrado)} ---")
        try:
            import pandas as pd
            for enc in ["latin1", "utf-8-sig", "utf-8"]:
                for sep in [";", "|", "\t"]:
                    try:
                        df = pd.read_csv(txt_encontrado, sep=sep, encoding=enc,
                                         nrows=2, dtype=str)
                        if len(df.columns) > 5:
                            print(f"  sep='{sep}'  enc='{enc}'  colunas={len(df.columns)}")
                            for i, c in enumerate(df.columns):
                                print(f"  {i:>3}. '{c}'")
                            break
                    except Exception:
                        continue
                else:
                    continue
                break
        except Exception as e:
            print(f"  [ERRO] {e}")
    else:
        print(f"\n[INFO] Nenhum arquivo RAIS TXT encontrado em {dados_dir}")
        print("       (o .7z pode nao ter sido extraido, ou TXTs foram deletados)")

    ftp.quit()
    print("\n[OK] Diagnostico v2 concluido.")
    print("\nSe os paths NOVO CAGED estiverem corretos, rode etl_social.py novamente.")

if __name__ == "__main__":
    main()
