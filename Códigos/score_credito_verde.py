"""
score_credito_verde.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Gera o Score Socioambiental final (0-1000) para cada PME, simulando um
bureau de credito verde para micro, pequenas e medias empresas.

DE ONDE VEM O SCORE (importante — apos a virada metodologica):
  O score E a RUBRICA DE ENQUADRAMENTO (Fase 1, rubrica_enquadramento.py),
  ancorada em criterios oficiais (Taxonomia Sustentavel Brasileira + PRSAC/
  Res. CMN 4.945 + Sustainability Bond Framework do BNDES). NAO e o resultado
  de um clustering circular.
    score_socioambiental = round( score_enquadramento(0-100) * 10 )  -> 0-1000

  A classificacao supervisionada (Fase 2, classificacao.py) NAO gera o score:
  ela e a FERRAMENTA OPERACIONAL que preve a faixa a partir do cadastro barato
  (AUC ~0,92), evidenciando a reducao de horas-homem na originacao
  (Metrica de Gestao). Por isso este script nao re-roda o modelo: pontua a
  base pela rubrica (exato), sem risco de incompatibilidade de features.

LOGICA DAS FAIXAS (politica de concessao, mapeada da rubrica):
  Verde-A        -> Excelente  (Credito Verde Premium, taxa preferencial)
  Verde-B        -> Bom        (Credito Verde Padrao)
  Amarelo        -> Regular    (Credito Verde Condicional, comprovacao extra)
  Vermelho       -> Restrito   (nao elegivel no momento)
  Vetado         -> Penalizacao severa (embargo/CEIS nao compensados) -> score 0
  Fora de escopo -> CNAE categoricamente nao-elegivel (Anexo 2 BNDES)

OUTPUT:
  Dados/base_analitica_scored.csv
  Dados/plots/distribuicao_score.png
  Dados/plots/score_por_porte.png
  Dados/simulacao_portfolio.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS     = os.path.join(BASE_DIR, "Dados")
PLOTS_DIR = os.path.join(DADOS, "plots")

INPUT_ROTULADO = os.path.join(DADOS, "base_analitica_rotulada.csv")
OUTPUT_SCORED  = os.path.join(DADOS, "base_analitica_scored.csv")
OUTPUT_SIMUL   = os.path.join(DADOS, "simulacao_portfolio.txt")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Mapeamento faixa da rubrica -> politica de concessao
# ------------------------------------------------------------------
FAIXA_POLITICA = {
    "Verde-A":        "Excelente — Credito Verde Premium",
    "Verde-B":        "Bom — Credito Verde Padrao",
    "Amarelo":        "Regular — Credito Verde Condicional",
    "Vermelho":       "Restrito — Analise caso a caso",
    "Vetado":         "Vetado — Penalizacao severa (G)",
    "Fora de escopo": "Fora de escopo — CNAE nao elegivel",
}
ORDEM_POLITICA = list(FAIXA_POLITICA.values())
# Faixas consideradas elegiveis a credito verde (aprovadas)
FAIXAS_APROVADAS = ["Verde-A", "Verde-B"]

PORTE_LABEL = {"01": "Micro (ME)", "03": "Pequeno (EPP)", "05": "Medio/Demais"}
TICKET = {"01": 50_000, "03": 150_000, "05": 400_000}   # hipoteses (R$)


# ==================================================================
# ETAPAS
# ==================================================================

def carregar() -> pd.DataFrame:
    print("[1/5] Carregando base rotulada (rubrica) ...")
    df = pd.read_csv(INPUT_ROTULADO, sep=";", dtype=str, encoding="utf-8-sig",
                     low_memory=False)
    print(f"      {len(df):,} registros")
    if "score_enquadramento" not in df.columns or "faixa" not in df.columns:
        raise SystemExit("ERRO: rode rubrica_enquadramento.py antes (faltam score_enquadramento/faixa).")
    df["score_enquadramento"] = pd.to_numeric(df["score_enquadramento"], errors="coerce")
    return df


def calcular_score(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Convertendo rubrica em score 0-1000 ...")
    # score 0-1000 a partir da rubrica (0-100). Negativos (Vetado) -> 0.
    base = df["score_enquadramento"].clip(lower=0, upper=100).fillna(0)
    df["score_socioambiental"] = np.round(base * 10).astype(int).clip(0, 1000)

    # Vetado -> 0 ; Fora de escopo -> sentinela -1 (nao pontua)
    df.loc[df["faixa"] == "Vetado", "score_socioambiental"] = 0
    df.loc[df["faixa"] == "Fora de escopo", "score_socioambiental"] = -1

    df["faixa_elegibilidade"] = df["faixa"].map(FAIXA_POLITICA).fillna("Indefinido")
    df["aprovado_credito_verde"] = df["faixa"].isin(FAIXAS_APROVADAS).astype(int)

    n_apr = int(df["aprovado_credito_verde"].sum())
    print(f"      Aprovados a credito verde (Verde-A/B): {n_apr:,} ({n_apr/len(df):.1%})")
    return df


def plotar(df: pd.DataFrame) -> None:
    print("[3/5] Gerando graficos ...")
    AZUL = "#1A237E"; CLARO = "#BBDEFB"
    cores = {"Excelente — Credito Verde Premium": "#1B5E20",
             "Bom — Credito Verde Padrao":        "#66BB6A",
             "Regular — Credito Verde Condicional":"#F9A825",
             "Restrito — Analise caso a caso":    "#C62828",
             "Vetado — Penalizacao severa (G)":   "#4A148C",
             "Fora de escopo — CNAE nao elegivel":"#757575"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = df.loc[df["score_socioambiental"] >= 0, "score_socioambiental"]
    axes[0].hist(sc, bins=50, color=CLARO, edgecolor=AZUL, linewidth=0.5)
    axes[0].axvline(sc.median(), color=AZUL, linestyle="--", linewidth=2,
                    label=f"Mediana: {sc.median():.0f}")
    axes[0].set_title("Distribuicao do Score Socioambiental (0-1000)")
    axes[0].set_xlabel("Score"); axes[0].set_ylabel("Empresas")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    cont = df["faixa_elegibilidade"].value_counts().reindex(ORDEM_POLITICA, fill_value=0)
    labels = [f.split(" — ")[0] for f in ORDEM_POLITICA]
    axes[1].barh(labels, cont.values, color=[cores.get(f, "#999") for f in ORDEM_POLITICA])
    axes[1].invert_yaxis(); axes[1].set_title("Distribuicao por faixa (politica de concessao)")
    axes[1].set_xlabel("Empresas"); axes[1].grid(axis="x", alpha=0.3)
    total = len(df)
    for i, n in enumerate(cont.values):
        axes[1].text(n + total*0.005, i, f"{n:,} ({n/total:.1%})", va="center", fontsize=9)
    plt.tight_layout()
    cam = os.path.join(PLOTS_DIR, "distribuicao_score.png")
    plt.savefig(cam, dpi=150, bbox_inches="tight"); plt.close()
    print(f"      Salvo: {os.path.basename(cam)}")

    # score por porte
    if "porte" in df.columns:
        df["porte_lbl"] = df["porte"].astype(str).str.strip().map(
            {"01": "Micro", "03": "Pequena", "05": "Media+"}).fillna("ND")
        g = df[df["score_socioambiental"] >= 0].groupby("porte_lbl")["score_socioambiental"]\
              .agg(["mean", "median", "count"])
        if not g.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            x = range(len(g))
            ax2.bar(x, g["mean"], color=CLARO, edgecolor=AZUL, linewidth=1.2, label="Media")
            ax2.scatter(x, g["median"], color=AZUL, s=80, zorder=5, label="Mediana")
            ax2.set_xticks(list(x)); ax2.set_xticklabels(g.index)
            ax2.set_ylabel("Score Socioambiental"); ax2.set_ylim(0, 1050)
            ax2.set_title("Score medio e mediano por porte"); ax2.legend()
            ax2.grid(axis="y", alpha=0.3)
            for i, (idx, row) in enumerate(g.iterrows()):
                ax2.text(i, row["mean"] + 15, f"{row['mean']:.0f}\n(n={int(row['count']):,})",
                         ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            cam2 = os.path.join(PLOTS_DIR, "score_por_porte.png")
            plt.savefig(cam2, dpi=150, bbox_inches="tight"); plt.close()
            print(f"      Salvo: {os.path.basename(cam2)}")


def simular(df: pd.DataFrame) -> None:
    print("[4/5] Simulando portfolio ...")
    total = len(df)
    L = ["=" * 70,
         " SIMULACAO DE PORTFOLIO — Bureau de Credito Verde",
         " TCC MBA USP/Esalq | Autor: Helio Vinicius Moreira Ribeiro",
         "=" * 70,
         "\nUNIVERSO", f"  Total de PMEs: {total:,}",
         "\nDISTRIBUICAO POR FAIXA (politica de concessao)"]
    cont = df["faixa_elegibilidade"].value_counts().reindex(ORDEM_POLITICA, fill_value=0)
    for f in ORDEM_POLITICA:
        n = int(cont[f]); L.append(f"  {f:<45s}: {n:>8,} ({n/total:.1%})")

    n_apr = int(df["aprovado_credito_verde"].sum())
    L.append(f"\nAPROVACAO A CREDITO VERDE (faixas Verde-A e Verde-B)")
    L.append(f"  Aprovados: {n_apr:,} ({n_apr/total:.1%})")
    if "porte" in df.columns:
        L.append("\n  Por porte (aprovados) e carteira estimada:")
        for cod, lbl in PORTE_LABEL.items():
            m = (df["porte"].astype(str).str.strip() == cod) & (df["aprovado_credito_verde"] == 1)
            n_p = int(m.sum()); cart = n_p * TICKET[cod]
            L.append(f"    {lbl:<16s}: {n_p:>8,} empresas | carteira ~ R$ {cart/1e6:,.1f}M "
                     f"(ticket R$ {TICKET[cod]:,})")

    sc = df.loc[df["score_socioambiental"] >= 0, "score_socioambiental"]
    L += ["\nESTATISTICAS DO SCORE (exclui 'Fora de escopo')",
          f"  Media   : {sc.mean():.1f}", f"  Mediana : {sc.median():.1f}",
          f"  Desv-pad: {sc.std():.1f}",
          f"  P25/P75 : {sc.quantile(.25):.0f} / {sc.quantile(.75):.0f}",
          f"  Min/Max : {sc.min():.0f} / {sc.max():.0f}"]

    L += ["\nPOLITICA DE CONCESSAO — COMPROVACOES COMPLEMENTARES (pre-enquadramento)",
          "  Alem da faixa, no modelo 'Empreender Clima' (COP30) o tomador anexa",
          "  comprovacoes que aproximam seu perfil ao padrao verde das grandes",
          "  empresas (B3/ISE). Exemplos operacionalizaveis (ver framework):",
          "   - Notas de compra de fornecedores em CNAE/NCM verde (embalagem",
          "     biodegradavel, reciclagem, energia renovavel) — criterio E6;",
          "   - Relatorio de igualdade salarial (Lei 14.611/2023) — criterio S2;",
          "   - Certificacoes de cadeia (FSC, RTRS, BONSUCRO, MSC) — Anexo 1 BNDES.",
          "\nNOTA METODOLOGICA",
          "  Score = rubrica de enquadramento (0-100) x 10, ancorada em Taxonomia",
          "  Sustentavel Brasileira, PRSAC (Res. CMN 4.945/2021) e Sustainability",
          "  Bond Framework do BNDES (2021). Penalizacao por conduta (embargo IBAMA,",
          "  CEIS/Lista Suja) e compensavel, com teto em 'Amarelo'. Vies geografico",
          "  mitigado: localizacao so soma (bonus afirmativo IDH baixo), nunca penaliza.",
          "  A classificacao supervisionada (cadastro -> faixa, AUC ~0,92) e a",
          "  ferramenta de pre-triagem automatizada (Metrica de Gestao).",
          "=" * 70]
    with open(OUTPUT_SIMUL, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"      Simulacao: {os.path.basename(OUTPUT_SIMUL)}")
    print(f"      Aprovados (Verde-A/B): {n_apr:,} ({n_apr/total:.1%})")


def main() -> None:
    print("=" * 70)
    print(" SCORE SOCIOAMBIENTAL — Bureau de Credito Verde (0-1000)")
    print(" Score = RUBRICA de enquadramento | Fase 2 = ferramenta de pre-triagem")
    print("=" * 70)
    df = carregar()
    df = calcular_score(df)
    plotar(df)
    simular(df)
    print(f"[5/5] Exportando {os.path.basename(OUTPUT_SCORED)} ...")
    df.to_csv(OUTPUT_SCORED, sep=";", index=False, encoding="utf-8-sig")

    sc = df.loc[df["score_socioambiental"] >= 0, "score_socioambiental"]
    print("\n" + "=" * 70)
    print(" RESUMO")
    print("=" * 70)
    print(f"  PMEs pontuadas      : {len(df):,}")
    print(f"  Score medio         : {sc.mean():.1f}  | mediano: {sc.median():.1f}")
    print(f"  Aprovados (Verde-A/B): {int(df['aprovado_credito_verde'].sum()):,} "
          f"({df['aprovado_credito_verde'].mean():.1%})")
    print(f"\n  Arquivo: {OUTPUT_SCORED}")
    print("\nPipeline completo (Fase 1 rubrica -> Fase 2 pre-triagem -> Score).")


if __name__ == "__main__":
    main()
