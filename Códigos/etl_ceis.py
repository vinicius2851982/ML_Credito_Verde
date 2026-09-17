"""
etl_ceis.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Processa o Cadastro de Empresas Inidoneas e Suspensas (CEIS) do
Portal da Transparencia (CGU). Extrai features de sancoes por CNPJ:

  - ceis_total_sancoes  : quantidade total de sancoes registradas
  - ceis_sancoes_ativas : sancoes com DATA FINAL SANCAO nula ou futura
  - ceis_sancionado     : flag binaria (1 = aparece no CEIS)

O CEIS e tratado como filtro de ALTO RISCO no modelo:
  - sancoes_ativas > 0 -> risco elevado (possivel bloqueio de credito)
  - sancoes apenas historicas -> penalizacao menor no score

LAYOUT DO ARQUIVO (sep=";", encoding=latin1, com cabecalho):
  CPF OU CNPJ DO SANCIONADO  -> documento identificador
  TIPO DE PESSOA             -> F=Fisica, J=Juridica
  CATEGORIA DA SANCAO        -> tipo da sancao
  DATA INICIO SANCAO         -> dd/mm/aaaa
  DATA FINAL SANCAO          -> dd/mm/aaaa (vazio = indefinido/ativa)
  ORGAO SANCIONADOR
  UF ORGAO SANCIONADOR
  ESFERA ORGAO SANCIONADOR   -> Federal / Estadual / Municipal
"""

import os
import re
import glob
import pandas as pd
from datetime import date

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_CEIS  = os.path.join(BASE_DIR, "Dados", "ceis")
OUTPUT_PATH = os.path.join(BASE_DIR, "Dados", "stg_ceis.csv")

PADRAO_NAO_DIGITO = re.compile(r"\D")
HOJE = pd.Timestamp(date.today())


def localizar_csv_ceis(pasta: str) -> str:
    """Localiza o CSV do CEIS na pasta extraida."""
    csvs = sorted(glob.glob(os.path.join(pasta, "*.csv")))
    if not csvs:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {pasta}\n"
            "Execute preparar_dados_publicos.py para extrair o CEIS.zip."
        )
    if len(csvs) > 1:
        print(f"[WARN] Multiplos CSVs encontrados. Usando: {csvs[-1]}")
    return csvs[-1]


def ler_ceis(caminho: str) -> pd.DataFrame:
    for enc in ["latin1", "utf-8-sig", "utf-8"]:
        try:
            df = pd.read_csv(
                caminho, sep=";", dtype=str,
                encoding=enc, on_bad_lines="skip",
            )
            print(f"[OK]  Lido com encoding='{enc}'. Linhas: {len(df):,}")
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Falha ao ler o CEIS com todos os encodings tentados.")


def limpar_cnpj(valor) -> str:
    if pd.isna(valor):
        return ""
    s = PADRAO_NAO_DIGITO.sub("", str(valor))
    return s.zfill(14) if 11 < len(s) < 14 else s


def processar_ceis(df: pd.DataFrame) -> pd.DataFrame:
    # Identifica coluna de documento (aceita variações de nome)
    col_doc = next(
        (c for c in df.columns if "CPF" in c.upper() and "CNPJ" in c.upper()), None
    )
    if col_doc is None:
        raise KeyError(f"Coluna de CPF/CNPJ nao encontrada. Colunas: {list(df.columns)}")
    print(f"[OK]  Coluna de documento: '{col_doc}'")

    # Limpa e filtra PJs
    df["CNPJ_LIMPO"] = df[col_doc].apply(limpar_cnpj)
    df = df[df["CNPJ_LIMPO"].str.len() == 14].copy()
    print(f"[OK]  {len(df):,} registros com CNPJ valido (PJ).")

    # Identifica coluna de data final da sancao
    col_fim = next(
        (c for c in df.columns if "DATA" in c.upper() and "FINAL" in c.upper()), None
    )

    # Sancao ativa: data final nula OU data final >= hoje
    if col_fim:
        df["_data_fim"] = pd.to_datetime(df[col_fim], dayfirst=True, errors="coerce")
        df["_ativa"] = df["_data_fim"].isna() | (df["_data_fim"] >= HOJE)
    else:
        print("[WARN] Coluna de data final da sancao nao encontrada. Todas marcadas como ativas.")
        df["_ativa"] = True

    # Agrega por CNPJ
    resultado = (
        df.groupby("CNPJ_LIMPO", as_index=False)
        .agg(
            ceis_total_sancoes=("CNPJ_LIMPO", "count"),
            ceis_sancoes_ativas=("_ativa", "sum"),
        )
        .assign(ceis_sancionado=1)
    )
    resultado["ceis_sancoes_ativas"] = resultado["ceis_sancoes_ativas"].astype(int)

    print(f"[OK]  {len(resultado):,} CNPJs unicos no CEIS.")
    print(f"[OK]  {resultado['ceis_sancoes_ativas'].gt(0).sum():,} com sancoes ATIVAS.")
    return resultado


def main() -> None:
    print("=" * 70)
    print(" ETL CEIS - Portal da Transparencia (CGU)")
    print("=" * 70)

    caminho = localizar_csv_ceis(PASTA_CEIS)
    print(f"\n[1/3] Lendo: {caminho}")
    df = ler_ceis(caminho)

    print("\n[2/3] Processando sancoes por CNPJ...")
    resultado = processar_ceis(df)

    print("\n[3/3] Exportando...")
    resultado.to_csv(OUTPUT_PATH, sep=";", index=False, encoding="utf-8-sig")
    print(f"[OK]  Salvo: {OUTPUT_PATH}")

    print("\n--- Amostra ---")
    print(resultado.head().to_string(index=False))
    print("\nProximo passo: merge_bases.py")


if __name__ == "__main__":
    main()
