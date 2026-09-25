"""
recorte.py  — datas de corte do estudo (fonte unica da verdade)
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

O trabalho adota DOIS recortes distintos, que nao devem ser confundidos:

  RECORTE NORMATIVO  — 31/05/2026
      Fixa o arcabouco vigente: Taxonomia Verde da FEBRABAN (versao Dez/2020),
      TSB (Decreto 12.705/2025), PRSAC (Res. CMN 4.945/2021) e demais normas.
      A FEBRABAN substituiu a Taxonomia Verde pela Taxonomia de Financas
      Sustentaveis em julho/2026, POSTERIOR a este recorte — por isso a versao
      utilizada era a vigente.

  CORTE DOS DADOS    — 31/08/2026
      Posicao das bases publicas. Nenhum fato posterior a esta data pode
      entrar na analise.

POR QUE ISTO E UM MODULO
  As bases do IBAMA e do CEIS sao CUMULATIVAS: o arquivo publicado hoje traz
  registros posteriores ao corte. Sem filtro explicito, a base incorporaria
  fatos que nao existiam no recorte declarado — erro silencioso, que nao
  aparece em nenhuma metrica. Centralizar as datas evita divergencia entre
  scripts.

Uso:
    from recorte import CORTE_DADOS, HOJE, filtrar_ate_corte
    df = filtrar_ate_corte(df, "DAT_AUTO_INFRACAO")
"""

import pandas as pd

# --- recortes -------------------------------------------------------------
RECORTE_NORMATIVO = pd.Timestamp(2026, 5, 31)
CORTE_DADOS = pd.Timestamp(2026, 8, 31)

# Referencia para idade da empresa: dia seguinte ao corte, de modo que
# "tempo de atividade" seja medido exatamente na posicao dos dados.
HOJE = CORTE_DADOS + pd.Timedelta(days=1)

# Rotulos para relatorios e para a secao de Metodologia
RECORTE_NORMATIVO_STR = "31/05/2026"
CORTE_DADOS_STR = "31/08/2026"
PERIODO_FONTE = "Agosto 2026"      # usado nas fontes de tabelas e figuras


def filtrar_ate_corte(df, coluna, corte=CORTE_DADOS, verbose=True):
    """Remove registros posteriores ao corte dos dados.

    Linhas com data ausente sao MANTIDAS: em bases cumulativas a ausencia
    costuma indicar registro antigo sem preenchimento, e descarta-las
    introduziria perda seletiva. O volume de nulos e reportado para que a
    decisao seja auditavel.
    """
    if coluna not in df.columns:
        if verbose:
            print(f"      [recorte] coluna '{coluna}' ausente — filtro nao aplicado")
        return df
    datas = pd.to_datetime(df[coluna], errors="coerce")
    posteriores = datas > corte
    nulos = int(datas.isna().sum())
    n_fora = int(posteriores.sum())
    if verbose:
        print(f"      [recorte] {coluna}: {n_fora:,} registros posteriores a "
              f"{corte:%d/%m/%Y} removidos | {nulos:,} sem data (mantidos)")
    return df.loc[~posteriores].copy()


def resumo():
    return (f"Recorte normativo: {RECORTE_NORMATIVO_STR} | "
            f"Corte dos dados: {CORTE_DADOS_STR}")


if __name__ == "__main__":
    print(resumo())
    print(f"Referencia de idade (HOJE): {HOJE:%d/%m/%Y}")
