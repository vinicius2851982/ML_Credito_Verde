"""
etl_b3_esg.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Constroi um BENCHMARK ESG SETORIAL a partir de dados REAIS de empresas
listadas (ISE B3 / ESG Workspace) e o atribui as PMEs pela CNAE.

LOGICA (proxy grande -> pequena, mesma do RAIS/CAGED):
  Empresas de capital aberto DIVULGAM ESG (questionario ISE B3); PMEs nao.
  Agregamos o desempenho ESG por SETOR economico B3 e o usamos como
  referencia setorial para as PMEs do mesmo setor (via crosswalk CNAE).

FONTE: Dados/ESG_Workspace/respostas_questionario_carteira_2026.xlsx
  Aba "Score Base 2026-27": score por empresa em 6 dimensoes
    E (Ambiental) = Meio Ambiente + Mudanca do Clima (CDP)
    S (Social)    = Capital Humano + Capital Social
    G (Governanca)= Governanca + Modelo de Negocios (fora do modelo ML)
  Aba "Score ISE B3 2026-27": SETOR economico B3 por empresa
  (carteiras 2023-2025 usadas para o score geral historico)

SAIDA: Dados/stg_b3_esg.csv  (uma linha por DIVISAO CNAE de 2 digitos)
  cnae_divisao | esg_ambiental_setor | esg_social_setor |
  esg_geral_setor | b3_setor | origem_esg
"""

import os
import glob
import unicodedata
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS    = os.path.join(BASE_DIR, "Dados")
ESG_DIR  = os.path.join(DADOS, "ESG_Workspace")
QUEST    = os.path.join(ESG_DIR, "respostas_questionario_carteira_2026.xlsx")
OUTPUT   = os.path.join(DADOS, "stg_b3_esg.csv")


def _ascii(s) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").upper().strip()


def achar_col(cols, *tokens):
    """Coluna cujo nome (sem acento) contem todos os tokens."""
    toks = [_ascii(t) for t in tokens]
    for c in cols:
        ca = _ascii(c)
        if all(t in ca for t in toks):
            return c
    return None


# ------------------------------------------------------------------
# CROSSWALK: Setor economico B3 -> divisoes CNAE (2 digitos)
# Mapeia os 10 setores B3 para as divisoes CNAE que melhor representam.
# Base: classificacao setorial B3 x CNAE 2.0 (IBGE). Aproximacao macro.
# ------------------------------------------------------------------
B3_PARA_CNAE = {
    "PETROLEO, GAS E BIOCOMBUSTIVEIS": ["06", "19", "09"],
    "MATERIAIS BASICOS":               ["05", "07", "08", "17", "20", "23", "24"],
    "BENS INDUSTRIAIS":                ["25", "28", "29", "30", "33", "41", "42", "43",
                                        "49", "50", "51", "52", "53", "22", "27"],
    "CONSUMO NAO CICLICO":             ["01", "02", "03", "10", "11", "12", "21", "47"],  # agro, alimentos, farma
    "CONSUMO CICLICO":                 ["13", "14", "15", "16", "18", "31", "32", "45",
                                        "46", "55", "56", "77", "79", "90", "93", "95"],
    "SAUDE":                           ["86", "87", "88", "75"],
    "TECNOLOGIA DA INFORMACAO":        ["62", "63", "26", "58", "59", "60"],
    "TELECOMUNICACOES":                ["61"],
    "UTILIDADE PUBLICA":               ["35", "36", "37", "38", "39"],
    "FINANCEIRO E OUTROS":             ["64", "65", "66", "68", "69", "70", "71", "72",
                                        "73", "74", "78", "80", "81", "82", "84", "85",
                                        "94", "96", "97"],
}

# Inverte: cnae_divisao -> setor B3
CNAE_PARA_B3 = {}
for setor, divisoes in B3_PARA_CNAE.items():
    for d in divisoes:
        CNAE_PARA_B3[d] = setor


def carregar_scores_empresa() -> pd.DataFrame:
    """Le Score Base (dimensoes) + Score ISE (setor) e junta por razao social."""
    xl = pd.ExcelFile(QUEST)

    # --- Score Base: dimensoes (header na 2a linha) ---
    sb = pd.read_excel(xl, "Score Base 2026-27", header=1, dtype=str)
    sb = sb.rename(columns={sb.columns[0]: "RAZAO_SOCIAL"})
    col_cap_hum = achar_col(sb.columns, "CAPITAL", "HUMANO")
    col_cap_soc = achar_col(sb.columns, "CAPITAL", "SOCIAL")
    col_amb     = achar_col(sb.columns, "MEIO", "AMBIENTE")
    col_clima   = achar_col(sb.columns, "CLIMA") or achar_col(sb.columns, "CDP")
    col_gov     = achar_col(sb.columns, "GOVERNAN")
    col_neg     = achar_col(sb.columns, "MODELO", "NEGOCIO")

    for c in [col_cap_hum, col_cap_soc, col_amb, col_clima, col_gov, col_neg]:
        if c:
            sb[c] = pd.to_numeric(sb[c].astype(str).str.replace(",", ".", regex=False),
                                  errors="coerce")

    sb["RAZAO_SOCIAL"] = sb["RAZAO_SOCIAL"].astype(str).map(_ascii)
    sb = sb.dropna(subset=["RAZAO_SOCIAL"])
    sb = sb[sb["RAZAO_SOCIAL"].str.len() > 2]

    # Pilares (media das dimensoes que compoem cada um)
    sb["E"] = sb[[c for c in [col_amb, col_clima] if c]].mean(axis=1)
    sb["S"] = sb[[c for c in [col_cap_hum, col_cap_soc] if c]].mean(axis=1)
    sb["G"] = sb[[c for c in [col_gov, col_neg] if c]].mean(axis=1)

    # --- Score ISE: setor economico ---
    si = pd.read_excel(xl, "Score ISE B3 2026-27", header=0, dtype=str)
    si = si.rename(columns={si.columns[0]: "RAZAO_SOCIAL"})
    col_setor = achar_col(si.columns, "SETOR")
    col_score = achar_col(si.columns, "SCORE")
    si["RAZAO_SOCIAL"] = si["RAZAO_SOCIAL"].astype(str).map(_ascii)
    si["b3_setor_topo"] = si[col_setor].astype(str).str.split("/").str[0].map(_ascii)
    si["score_geral"]   = pd.to_numeric(si[col_score].astype(str).str.replace(",", ".", regex=False),
                                        errors="coerce")
    si = si[["RAZAO_SOCIAL", "b3_setor_topo", "score_geral"]].dropna(subset=["RAZAO_SOCIAL"])

    # --- Junta ---
    df = sb.merge(si, on="RAZAO_SOCIAL", how="left")
    n_setor = df["b3_setor_topo"].notna().sum()
    print(f"[B3] {len(df)} empresas | {n_setor} com setor identificado")
    return df


def agregar_por_setor(df: pd.DataFrame) -> pd.DataFrame:
    """Media de E, S, G e score geral por setor economico B3."""
    agg = (df.dropna(subset=["b3_setor_topo"])
             .groupby("b3_setor_topo")
             .agg(esg_ambiental_setor=("E", "mean"),
                  esg_social_setor=("S", "mean"),
                  esg_governanca_setor=("G", "mean"),
                  esg_geral_setor=("score_geral", "mean"),
                  n_empresas=("RAZAO_SOCIAL", "count"))
             .reset_index())
    # score_geral vem em escala 0-100; E/S/G em 0-1. Normaliza geral p/ 0-1.
    if agg["esg_geral_setor"].max() > 1.5:
        agg["esg_geral_setor"] = (agg["esg_geral_setor"] / 100).round(4)
    for c in ["esg_ambiental_setor", "esg_social_setor", "esg_governanca_setor"]:
        agg[c] = agg[c].round(4)
    print("\n[B3] Benchmark ESG por setor economico:")
    print(agg.to_string(index=False))
    return agg


def expandir_para_cnae(agg: pd.DataFrame) -> pd.DataFrame:
    """Atribui o benchmark setorial a cada divisao CNAE via crosswalk."""
    setor_idx = agg.set_index("b3_setor_topo")
    medias_globais = {
        "esg_ambiental_setor": agg["esg_ambiental_setor"].mean(),
        "esg_social_setor":    agg["esg_social_setor"].mean(),
        "esg_governanca_setor":agg["esg_governanca_setor"].mean(),
        "esg_geral_setor":     agg["esg_geral_setor"].mean(),
    }

    linhas = []
    todas_divisoes = [f"{i:02d}" for i in range(1, 100)]
    for div in todas_divisoes:
        setor = CNAE_PARA_B3.get(div)
        if setor and setor in setor_idx.index:
            r = setor_idx.loc[setor]
            linhas.append({
                "cnae_divisao": div,
                "esg_ambiental_setor": r["esg_ambiental_setor"],
                "esg_social_setor":    r["esg_social_setor"],
                "esg_governanca_setor":r["esg_governanca_setor"],
                "esg_geral_setor":     r["esg_geral_setor"],
                "b3_setor":            setor,
                "origem_esg":          "ise_setor",
            })
        else:
            linhas.append({
                "cnae_divisao": div,
                **{k: round(v, 4) for k, v in medias_globais.items()},
                "b3_setor": setor or "NAO_MAPEADO",
                "origem_esg": "media_global",
            })
    out = pd.DataFrame(linhas)
    n_real = (out["origem_esg"] == "ise_setor").sum()
    print(f"\n[B3] {n_real}/{len(out)} divisoes CNAE com benchmark setorial ISE "
          f"(restante = media global).")
    return out


def main():
    print("=" * 70)
    print(" ETL B3 ESG - Benchmark ESG setorial (ISE B3) -> CNAE")
    print("=" * 70)
    if not os.path.exists(QUEST):
        raise SystemExit(f"Arquivo nao encontrado: {QUEST}")

    df = carregar_scores_empresa()
    agg = agregar_por_setor(df)
    out = expandir_para_cnae(agg)

    out.to_csv(OUTPUT, sep=";", index=False, encoding="utf-8-sig")
    print(f"\n[OK] Salvo: {OUTPUT}")
    print("\n  E/S/G estao em escala 0-1 (fracao do score maximo da dimensao).")
    print("  Integra no merge_bases.py por cnae_divisao.")
    print("\n  NOTA: cobertura ISE e macro-setorial (10 setores, ~89 listadas).")
    print("  Setores com poucas listadas (ex.: agro) herdam media do setor B3.")
    print("  Aprofundamento futuro: extrair indicadores de relatorios individuais")
    print("  de sustentabilidade (ex.: 3tentos) por NLP/manual — fora do escopo atual.")
    print("\nProximo passo: merge_bases.py (incluir features B3 ESG)")


if __name__ == "__main__":
    main()
