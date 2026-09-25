"""
analise_descritiva.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Analise descritiva de TODAS as bases integradas (item 5 do Resultado
Preliminar). Produz estatisticas e figuras por pilar, SEM interpretacao
(a leitura critica e do autor). Gera:

  - tabela_descritiva[_sufixo].csv     (describe das variaveis numericas)
  - plots/descritiva[_sufixo]/*.png    (paineis por pilar)
  - relatorio_descritiva[_sufixo].txt  (destaques quantitativos)

USO (entrada parametrizavel — permite descricao INICIAL das bases e FINAL):
  python analise_descritiva.py                       # base pontuada (final)
  python analise_descritiva.py base_analitica_febraban.csv inicial
    -> descreve a base integrada ANTES da rubrica/score (caracterizacao inicial
       das bases de dados); colunas de score/faixa ausentes sao puladas.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS    = os.path.join(BASE_DIR, "Dados")

# Entrada e sufixo de saida parametrizaveis por linha de comando:
#   argv[1] = nome do CSV de entrada (em Dados/); default base_analitica_scored.csv
#   argv[2] = sufixo dos arquivos de saida (ex.: 'inicial'); default '' (final)
_INPUT_NOME = sys.argv[1] if len(sys.argv) > 1 else "base_analitica_scored.csv"
_SUFIXO     = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""
INPUT    = os.path.join(DADOS, _INPUT_NOME)
PLOTS    = os.path.join(DADOS, "plots", f"descritiva{_SUFIXO}")
TAB_CSV  = os.path.join(DADOS, f"tabela_descritiva{_SUFIXO}.csv")
REL      = os.path.join(DADOS, f"relatorio_descritiva{_SUFIXO}.txt")
os.makedirs(PLOTS, exist_ok=True)

# Porte RFB -> nome legivel (a base febraban/raw tem 'porte' mas nao 'porte_nome').
PORTE_NOME = {"01": "Micro", "03": "Pequena", "05": "Media+", "00": "ND"}

AZUL = "#1A237E"; CLARO = "#90CAF9"; HOJE = pd.Timestamp(2026, 9, 1)   # dia seguinte ao corte dos dados (31/08/2026); ver recorte.py

# Variaveis por pilar (rotulo amigavel)
GRUPOS = {
    "Cadastro (RFB)": {
        "capital_social_log": "Capital social (log)",
        "tempo_empresa_anos": "Tempo de empresa (anos)",
    },
    "Ambiental (IBAMA)": {
        "qtd_infracoes": "Qtd. infracoes",
        "valor_total_multas_log": "Multas (log)",
        "anos_desde_ultima_infracao": "Anos s/ infracao",
    },
    "Social setorial (RAIS/CAGED)": {
        "salario_medio_sm": "Salario setor (SM)",
        "tempo_emprego_medio": "Tempo emprego (meses)",
        "pct_vinculo_ativo": "% vinculo ativo",
        "rotatividade": "Rotatividade",
        "saldo_setor_taxa": "Saldo setor",
    },
    "Equidade (RAIS)": {
        "gap_genero_ajustado_cnae": "Gap genero ajustado",
        "gap_raca_ajustado_cnae": "Gap raca ajustado",
        "pct_chefia_feminina_cnae": "% chefia feminina",
        "dispersao_salarial_cnae": "Dispersao salarial (CV)",
    },
    "Benchmark ESG (B3/ISE)": {
        "esg_ambiental_setor": "ESG ambiental setor",
        "esg_social_setor": "ESG social setor",
    },
    "Rubrica / Score": {
        "score_e": "Score E (0-100)",
        "score_s": "Score S (0-100)",
        "score_enquadramento": "Score enquadramento",
        "score_socioambiental": "Score 0-1000",
    },
}


def carregar():
    print(f"[1/4] Carregando {os.path.basename(INPUT)} ...")
    df = pd.read_csv(INPUT, sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)
    # porte legivel (base raw/febraban so tem 'porte'; rotulada/scored ja tem porte_nome)
    if "porte_nome" not in df.columns and "porte" in df.columns:
        df["porte_nome"] = df["porte"].astype(str).str.strip().map(PORTE_NOME).fillna("ND")
    # tempo de empresa
    s = df.get("data_inicio_atividade", pd.Series("", index=df.index)).astype(str)
    s = s.str.replace(r"[^0-9]", "", regex=True).str.slice(0, 8)
    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    df["tempo_empresa_anos"] = ((HOJE - dt).dt.days / 365.25).clip(lower=0)
    if "capital_social_log" not in df.columns:
        df["capital_social_log"] = np.log1p(pd.to_numeric(df.get("capital_social"), errors="coerce").fillna(0))
    # numericas
    todas = [c for g in GRUPOS.values() for c in g]
    for c in todas:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"      {len(df):,} registros")
    return df


def tabela_descritiva(df):
    print("[2/4] Tabela descritiva (describe) ...")
    cols = [c for g in GRUPOS.values() for c in g if c in df.columns]
    desc = df[cols].describe(percentiles=[.25, .5, .75]).T
    desc = desc.round(3)
    desc.to_csv(TAB_CSV, sep=";", encoding="utf-8-sig")
    return desc


def paineis(df):
    print("[3/4] Gerando paineis por pilar ...")
    for titulo, mapa in GRUPOS.items():
        cols = [c for c in mapa if c in df.columns]
        if not cols:
            continue
        n = len(cols); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for i, c in enumerate(cols):
            s = df[c].dropna()
            # recorte de outliers extremos para visualizacao (1-99%)
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            sv = s[(s >= lo) & (s <= hi)] if hi > lo else s
            axes[i].hist(sv, bins=40, color=CLARO, edgecolor=AZUL, linewidth=0.4)
            axes[i].axvline(s.median(), color=AZUL, ls="--", lw=1.5)
            axes[i].set_title(mapa[c], fontsize=10)
            axes[i].grid(axis="y", alpha=0.3)
        for j in range(len(cols), len(axes)):
            axes[j].axis("off")
        fig.suptitle(titulo, fontsize=12, fontweight="bold")
        plt.tight_layout()
        nome = titulo.split(" (")[0].replace("/", "_").replace(" ", "_").lower()
        cam = os.path.join(PLOTS, f"desc_{nome}.png")
        plt.savefig(cam, dpi=140, bbox_inches="tight"); plt.close()
        print(f"      {os.path.basename(cam)}")

    # categoricas: porte e faixa
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    if "porte_nome" in df.columns:
        vc = df["porte_nome"].value_counts()
        axes[0].bar(vc.index.astype(str), vc.values, color=CLARO, edgecolor=AZUL)
        axes[0].set_title("Distribuicao por porte"); axes[0].tick_params(axis="x", rotation=20)
    if "faixa" in df.columns:
        ordem = ["Verde-A", "Verde-B", "Amarelo", "Vermelho", "Vetado", "Fora de escopo"]
        vc = df["faixa"].value_counts().reindex(ordem).dropna()
        cores = {"Verde-A": "#1B5E20", "Verde-B": "#66BB6A", "Amarelo": "#F9A825",
                 "Vermelho": "#C62828", "Vetado": "#4A148C", "Fora de escopo": "#757575"}
        axes[1].bar(vc.index.astype(str), vc.values, color=[cores.get(x, "#999") for x in vc.index])
        axes[1].set_title("Distribuicao por faixa de enquadramento"); axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    cam = os.path.join(PLOTS, "desc_categoricas.png")
    plt.savefig(cam, dpi=140, bbox_inches="tight"); plt.close()
    print(f"      {os.path.basename(cam)}")


def relatorio(df, desc):
    print("[4/4] Relatorio de destaques ...")
    n = len(df)
    L = ["=" * 64, " ANALISE DESCRITIVA DAS BASES — DESTAQUES", "=" * 64,
         f"  Registros: {n:,}", ""]
    def pct(mask): return f"{int(mask.sum()):,} ({mask.mean():.2%})"
    if "porte_nome" in df:
        L.append("  Porte:")
        for k, v in df["porte_nome"].value_counts().items():
            L.append(f"    {k:10s}: {v:>10,} ({v/n:.1%})")
    if "qtd_infracoes" in df:
        L.append(f"  Com autuacao IBAMA      : {pct(df['qtd_infracoes'] > 0)}")
    if "embargado" in df:
        L.append(f"  Embargo IBAMA ativo     : {pct(pd.to_numeric(df['embargado'],errors='coerce').fillna(0) > 0)}")
    if "ceis_sancoes_ativas" in df:
        L.append(f"  CEIS sancao ativa       : {pct(pd.to_numeric(df['ceis_sancoes_ativas'],errors='coerce').fillna(0) > 0)}")
    L.append("")
    L.append("  Medias setoriais (Social/Equidade):")
    for c, lbl in [("salario_medio_sm", "Salario (SM)"), ("rotatividade", "Rotatividade"),
                   ("gap_genero_ajustado_cnae", "Gap genero ajustado"),
                   ("gap_raca_ajustado_cnae", "Gap raca ajustado"),
                   ("pct_chefia_feminina_cnae", "% chefia feminina")]:
        if c in df:
            L.append(f"    {lbl:22s}: {df[c].mean():.4f}")
    # Camada de ATIVIDADE — Taxonomia FEBRABAN (se presente na base)
    if "flag_economia_verde" in df.columns:
        fe = pd.to_numeric(df["flag_economia_verde"], errors="coerce").fillna(0)
        L.append("")
        L.append("  Atividade verde (Taxonomia FEBRABAN, por subclasse CNAE):")
        L.append(f"    Economia verde (qualquer eixo): {pct(fe > 0)}")
        if "fbb_ev_eixo" in df.columns:
            for eixo in ["Social", "Ambiental", "Social + Ambiental"]:
                m = df["fbb_ev_eixo"].fillna("") == eixo
                L.append(f"      eixo {eixo:18s}: {int(m.sum()):>10,} ({m.mean():.2%})")
        if "flag_risco_ambiental" in df.columns:
            ra = pd.to_numeric(df["flag_risco_ambiental"], errors="coerce").fillna(0)
            L.append(f"    Exposicao a risco ambiental    : {pct(ra > 0)}")
        if "flag_clima" in df.columns:
            cl = pd.to_numeric(df["flag_clima"], errors="coerce").fillna(0)
            L.append(f"    Exposicao a mudancas climaticas: {pct(cl > 0)}")
    if "score_socioambiental" in df:
        sc = df.loc[df["score_socioambiental"].astype(float) >= 0, "score_socioambiental"].astype(float)
        L.append("")
        L.append(f"  Score 0-1000: media {sc.mean():.1f} | mediana {sc.median():.1f}")
    if "faixa" in df:
        L.append("  Faixas:")
        for k, v in df["faixa"].value_counts().items():
            L.append(f"    {k:14s}: {v:>10,} ({v/n:.1%})")
    L += ["", f"  Tabela completa: {os.path.basename(TAB_CSV)}",
          f"  Figuras: plots/descritiva/", "=" * 64]
    print("\n".join(L))
    with open(REL, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    print("=" * 64)
    print(" ANALISE DESCRITIVA DE TODAS AS BASES (item 5)")
    print("=" * 64)
    df = carregar()
    desc = tabela_descritiva(df)
    paineis(df)
    relatorio(df, desc)
    print(f"\n  Salvos: {os.path.basename(TAB_CSV)}, {os.path.basename(REL)}, plots/descritiva/*.png")


if __name__ == "__main__":
    main()
