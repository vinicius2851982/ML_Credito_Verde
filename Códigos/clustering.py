"""
clustering.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

ANALISE EXPLORATORIA (K-Means) sobre a base_analitica.csv.
ATENCAO: apos a virada metodologica, este script NAO e mais a fonte do
rotulo. O rotulo oficial vem da rubrica (rubrica_enquadramento.py). Aqui o
K-Means apenas evidencia agrupamentos naturais (apoio descritivo) e grava
em base_clustering_exploratorio.csv (sem sobrescrever a rubrica).

PILARES NO MODELO (conforme objetivo geral do TCC):
  E — Ambiental : qtd_infracoes, valor_total_multas_log,
                  anos_desde_ultima_infracao
  S — Social    : qtd_vinculos_ativos, salario_medio_sm,
                  rotatividade, intensidade_emprego_cnae
  Contexto      : capital_social_log, porte_num

  G — Governanca: NAO entra no modelo ML.
                  CEIS e embargos = filtros de VETO pre-cluster.
                  Empresas com veto_governanca=1 recebem score=0
                  diretamente, sem necessidade de ML.

PIPELINE:
  [1] Carrega base_analitica.csv
  [2] Aplica VETO (veto_governanca=1 -> excluido do ML, score=0)
  [3] Selecao e pre-processamento de features (E + S)
  [4] MiniBatchKMeans com k = 2..5
  [5] Metricas: Inertia (Cotovelo) + Silhouette Score
  [6] Rotulacao por perfil de risco ponderado
  [7] Exporta base_analitica_rotulada.csv + graficos PNG

OUTPUT:
  Dados/base_analitica_rotulada.csv
  Dados/plots/elbow_silhouette.png
  Dados/plots/perfil_clusters.png
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS     = os.path.join(BASE_DIR, "Dados")
PLOTS_DIR = os.path.join(DADOS, "plots")

INPUT  = os.path.join(DADOS, "base_analitica.csv")
# IMPORTANTE (pos-virada metodologica): o ROTULO oficial vem da rubrica
# (rubrica_enquadramento.py -> base_analitica_rotulada.csv). O K-Means aqui e
# apenas ANALISE EXPLORATORIA e grava em arquivo SEPARADO para NAO sobrescrever
# a rubrica.
OUTPUT = os.path.join(DADOS, "base_clustering_exploratorio.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
SEED        = 42
K_RANGE     = range(2, 6)
SILH_SAMPLE = 50_000          # amostra p/ Silhouette (evita OOM com 2.7M)

# Features do modelo ML (E + S apenas — sem G)
# Pilar S = perfil SETORIAL por CNAE (RAIS Vinculos + NOVO CAGED), pois nenhuma
# fonte publica do MTE traz CNPJ. Atribuido a cada firma pela sua CNAE.
FEATURES_CORE = [
    # Pilar E — Ambiental (IBAMA + benchmark ESG setorial B3)
    "qtd_infracoes",
    "valor_total_multas_log",      # log1p(valor_total_multas)
    "anos_desde_ultima_infracao",  # negativo: + tempo sem infracao = menor risco
    "esg_ambiental_setor",         # [E] negativo: setor com melhor ESG ambiental = menor risco
    # Pilar S — Social setorial (RAIS Vinculos + CAGED + benchmark ESG B3, por CNAE)
    "salario_medio_sm",            # [S] negativo: salario setorial maior = melhor condicao
    "tempo_emprego_medio",         # [S] negativo: maior estabilidade = menor risco social
    "pct_vinculo_ativo",           # [S] negativo: mais formalizacao/permanencia = menor risco
    "rotatividade",                # [S] positivo: alta rotatividade = precarizacao
    "saldo_setor_taxa",            # [S] negativo: setor criando vagas = menor risco
    "intensidade_emprego_cnae",    # [S] proxy CNAE (contexto)
    "esg_social_setor",            # [S] negativo: setor com melhor ESG social = menor risco
    # Contexto financeiro
    "capital_social_log",          # negativo: + capital = menor risco
    "porte_num",
]

# Pesos para score de risco por cluster (positivo = aumenta risco)
PESOS_RISCO = {
    # Pilar E
    "qtd_infracoes":               1.5,
    "valor_total_multas_log":      1.0,
    "anos_desde_ultima_infracao": -0.8,   # mais tempo sem infracao = menor risco
    "esg_ambiental_setor":        -0.8,   # setor com melhor ESG ambiental = menor risco
    # Pilar S (setorial)
    "salario_medio_sm":           -0.9,   # salario setorial maior = melhor pratica social
    "tempo_emprego_medio":        -0.7,   # maior estabilidade = menor risco social
    "pct_vinculo_ativo":          -0.5,   # mais formalizacao = menor risco
    "rotatividade":                1.2,   # alta rotatividade = risco social
    "saldo_setor_taxa":           -0.4,   # setor em expansao = menor risco
    "intensidade_emprego_cnae":   -0.3,   # mais intensivo = mais relevante socialmente
    "esg_social_setor":           -0.6,   # setor com melhor ESG social = menor risco
    # Contexto
    "capital_social_log":         -0.3,
    "porte_num":                  -0.1,
}

PORTE_MAP = {"00": 0, "01": 1, "03": 2, "05": 3}
ANOS_SEM_INFRACAO = 30.0   # imputado para empresa sem historico IBAMA


# ==================================================================
# ETAPAS
# ==================================================================

def carregar_base() -> pd.DataFrame:
    print(f"[1/7] Carregando {INPUT} ...")
    df = pd.read_csv(INPUT, sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)
    print(f"      {len(df):,} registros | {len(df.columns)} colunas")
    return df


def aplicar_veto(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa empresas com veto de governanca (score final = 0)
    das elegíveis ao modelo ML.

    Criterio de veto (Pilar G — NAO e feature ML):
      - embargado = 1        (embargo IBAMA ativo)
      - ceis_sancoes_ativas > 0  (sancao CEIS ativa)
    """
    print("\n[2/7] Aplicando filtro de veto (Pilar G — Governanca)...")

    # Converte colunas de veto
    for col in ["embargado", "ceis_sancoes_ativas", "veto_governanca"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Determina veto
    if "veto_governanca" in df.columns:
        mask_veto = df["veto_governanca"] == 1
    else:
        # Fallback: calcula veto localmente
        embargado    = df.get("embargado", pd.Series(0, index=df.index))
        ceis_ativas  = df.get("ceis_sancoes_ativas", pd.Series(0, index=df.index))
        mask_veto = (embargado == 1) | (ceis_ativas > 0)
        df["veto_governanca"] = mask_veto.astype(int)

    vetadas  = df[mask_veto].copy()
    elegíveis = df[~mask_veto].copy()

    print(f"      Empresas vetadas (score=0): {len(vetadas):,} "
          f"({len(vetadas)/len(df):.2%})")
    print(f"      Empresas elegíveis (entram no ML): {len(elegíveis):,} "
          f"({len(elegíveis)/len(df):.2%})")

    return elegíveis, vetadas


def preparar_features(df: pd.DataFrame):
    """
    Cria features derivadas, imputa NaN e aplica StandardScaler.
    Retorna: (df_enriquecido, X_normalizado, lista_features_usadas)
    """
    print("\n[3/7] Preparando features (Pilares E + S)...")

    # Converte numericos
    cols_numericas = [
        "qtd_infracoes", "valor_total_multas", "anos_desde_ultima_infracao",
        "capital_social", "salario_medio_sm", "tempo_emprego_medio",
        "pct_vinculo_ativo", "rotatividade", "saldo_setor_taxa",
        "intensidade_emprego_cnae", "esg_ambiental_setor", "esg_social_setor",
    ]
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Imputacoes
    df["anos_desde_ultima_infracao"] = (
        df["anos_desde_ultima_infracao"].fillna(ANOS_SEM_INFRACAO)
    )
    df["capital_social"] = df["capital_social"].fillna(0)

    # Log1p para distribuicoes heavy-tail
    df["valor_total_multas_log"] = np.log1p(df["valor_total_multas"].fillna(0))
    df["capital_social_log"]     = np.log1p(df["capital_social"].fillna(0))

    # Imputacao Social: mediana por porte (para CNPJs sem dados RAIS/CAGED)
    if "porte" in df.columns:
        df["porte_num"] = df["porte"].str.strip().map(PORTE_MAP).fillna(0).astype(float)
    else:
        df["porte_num"] = 0.0

    for col in ["salario_medio_sm", "tempo_emprego_medio", "pct_vinculo_ativo",
                "rotatividade", "saldo_setor_taxa", "intensidade_emprego_cnae",
                "esg_ambiental_setor", "esg_social_setor"]:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med if pd.notna(med) else 0)

    # Usa apenas features presentes no DataFrame
    features_usadas = [f for f in FEATURES_CORE if f in df.columns]
    print(f"      Features ({len(features_usadas)}): {features_usadas}")

    # Verifica features ausentes (S opcional pode nao estar disponivel)
    ausentes = [f for f in FEATURES_CORE if f not in df.columns]
    if ausentes:
        print(f"      [WARN] Features ausentes (nao entram): {ausentes}")
        print(f"             Execute etl_social.py e merge_bases.py para completo Pilar S.")

    X_raw = df[features_usadas].fillna(0).values.astype(np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print(f"      Shape apos StandardScaler: {X.shape}")
    return df, X, features_usadas


def rodar_kmeans(X: np.ndarray) -> dict:
    """
    MiniBatchKMeans para k=2..5.
    Silhouette em amostra de 50k para evitar OOM.
    """
    print("\n[4/7] MiniBatchKMeans (k = 2 a 5)...")

    rng = np.random.default_rng(SEED)
    idx_silh = rng.choice(len(X), size=min(SILH_SAMPLE, len(X)), replace=False)

    resultados = {}
    for k in K_RANGE:
        print(f"      k={k} ...", end=" ", flush=True)

        modelo = MiniBatchKMeans(
            n_clusters=k,
            random_state=SEED,
            batch_size=10_000,
            n_init=10,
            max_iter=300,
        )
        labels  = modelo.fit_predict(X)
        inertia = modelo.inertia_

        silh = silhouette_score(X[idx_silh], labels[idx_silh], sample_size=None)

        resultados[k] = {
            "modelo":     modelo,
            "labels":     labels,
            "inertia":    inertia,
            "silhouette": silh,
        }
        print(f"inertia={inertia:>14,.0f} | silhouette={silh:.4f}")

    return resultados


def plotar_metricas(resultados: dict) -> None:
    print("\n[5/7] Plotando metricas (Cotovelo + Silhouette)...")

    ks       = list(resultados.keys())
    inertias = [resultados[k]["inertia"]    for k in ks]
    silhs    = [resultados[k]["silhouette"] for k in ks]

    AZUL_ESCURO = "#1A237E"
    AZUL_CLARO  = "#BBDEFB"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ks, inertias, marker="o", color=AZUL_ESCURO, linewidth=2.5, markersize=8)
    axes[0].fill_between(ks, inertias, alpha=0.10, color=AZUL_ESCURO)
    axes[0].set_title("Metodo do Cotovelo", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Numero de clusters (k)", fontsize=11)
    axes[0].set_ylabel("Inertia (WCSS)", fontsize=11)
    axes[0].grid(alpha=0.3)

    cores_bar = [AZUL_CLARO if s < max(silhs) else AZUL_ESCURO for s in silhs]
    axes[1].bar(ks, silhs, color=cores_bar, edgecolor=AZUL_ESCURO, linewidth=1.2)
    axes[1].set_title("Silhouette Score por k", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Numero de clusters (k)", fontsize=11)
    axes[1].set_ylabel("Silhouette Score", fontsize=11)
    axes[1].set_ylim(0, max(silhs) * 1.25)
    axes[1].grid(axis="y", alpha=0.3)
    for k, s in zip(ks, silhs):
        axes[1].text(k, s + 0.002, f"{s:.4f}", ha="center", va="bottom", fontsize=10)

    k_melhor = max(resultados, key=lambda k: resultados[k]["silhouette"])
    plt.suptitle(
        f"Selecao do k otimo — K-Means | Pilares E+S | n={len(resultados[k_melhor]['labels']):,}",
        fontsize=13, y=1.02
    )
    plt.tight_layout()

    caminho = os.path.join(PLOTS_DIR, "elbow_silhouette.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Salvo: {caminho}")


def escolher_k(resultados: dict) -> int:
    k_otimo = max(resultados, key=lambda k: resultados[k]["silhouette"])
    print(f"\n      k otimo pelo Silhouette: k={k_otimo} "
          f"(silhouette={resultados[k_otimo]['silhouette']:.4f})")
    return k_otimo


def calcular_score_risco(perfil: pd.DataFrame, features: list) -> pd.Series:
    score = pd.Series(0.0, index=perfil.index)
    for col in features:
        if col not in PESOS_RISCO or col not in perfil.columns:
            continue
        col_vals  = perfil[col]
        col_range = col_vals.max() - col_vals.min()
        col_norm  = (col_vals - col_vals.min()) / col_range if col_range > 1e-9 \
                    else pd.Series(0.0, index=perfil.index)
        score += PESOS_RISCO[col] * col_norm
    return score


def rotular_clusters(df: pd.DataFrame, labels: np.ndarray,
                     features: list, k: int):
    df = df.copy()
    df["cluster"] = labels

    perfil = df.groupby("cluster")[features].mean()
    score  = calcular_score_risco(perfil, features)
    ranking = score.sort_values()   # menor = menor risco

    ROTULOS = {
        2: ["Baixo", "Alto"],
        3: ["Baixo", "Medio", "Alto"],
        4: ["Baixo", "Medio-Baixo", "Medio-Alto", "Alto"],
        5: ["Muito Baixo", "Baixo", "Medio", "Alto", "Muito Alto"],
    }
    rotulos_k = ROTULOS.get(k, [str(i) for i in range(k)])
    mapa_risco = {cl: rotulos_k[i] for i, cl in enumerate(ranking.index)}
    df["risco_label"] = df["cluster"].map(mapa_risco)

    print(f"\n      Mapa cluster -> risco (score ponderado E+S):")
    for cl in sorted(mapa_risco):
        risco = mapa_risco[cl]
        n     = (df["cluster"] == cl).sum()
        s     = score[cl]
        print(f"        Cluster {cl} -> {risco:12s} | {n:>10,} ({n/len(df):.1%}) | score={s:.3f}")

    return df, perfil, mapa_risco


def plotar_perfil(perfil: pd.DataFrame, mapa_risco: dict, features: list) -> None:
    perfil_plot = perfil[features].copy()
    perfil_plot.index = [
        f"Cluster {i} — {mapa_risco.get(i, '?')}"
        for i in perfil_plot.index
    ]

    perfil_norm = (perfil_plot - perfil_plot.min()) / (
        perfil_plot.max() - perfil_plot.min() + 1e-9
    )

    NOMES_CURTOS = {
        "qtd_infracoes":               "Qtd Infracoes [E]",
        "valor_total_multas_log":      "Multas log [E]",
        "anos_desde_ultima_infracao":  "Anos s/ Infracao [E]",
        "salario_medio_sm":            "Salario Setor SM [S]",
        "tempo_emprego_medio":         "Tempo Emprego [S]",
        "pct_vinculo_ativo":           "% Vinculo Ativo [S]",
        "rotatividade":                "Rotatividade [S]",
        "saldo_setor_taxa":            "Saldo Setor [S]",
        "intensidade_emprego_cnae":    "Int. CNAE [S]",
        "esg_ambiental_setor":         "ESG Amb. B3 [E]",
        "esg_social_setor":            "ESG Soc. B3 [S]",
        "capital_social_log":          "Capital log",
        "porte_num":                   "Porte",
    }
    perfil_norm.columns = [NOMES_CURTOS.get(c, c) for c in perfil_norm.columns]
    perfil_plot.columns = perfil_norm.columns

    n_rows = len(perfil_norm)
    fig, ax = plt.subplots(figsize=(15, max(3, n_rows * 2.2)))

    im = ax.imshow(perfil_norm.values, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(perfil_norm.columns)))
    ax.set_xticklabels(perfil_norm.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(perfil_norm.index, fontsize=11)
    ax.set_title(
        "Perfil medio dos clusters — Pilares E+S (intensidade 0=min, 1=max)",
        fontsize=13, pad=12
    )

    for i in range(n_rows):
        for j in range(len(perfil_norm.columns)):
            val = perfil_plot.iloc[i, j]
            txt = f"{val:.3f}" if abs(val) < 100 else f"{val:,.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if 0.25 < perfil_norm.values[i, j] < 0.75 else "white")

    plt.colorbar(im, ax=ax, label="Intensidade normalizada (por coluna)")
    plt.tight_layout()

    caminho = os.path.join(PLOTS_DIR, "perfil_clusters.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Salvo: {caminho}")


# ==================================================================
# MAIN
# ==================================================================

def main() -> None:
    print("=" * 70)
    print(" CLUSTERING K-MEANS — Bureau de Credito Verde")
    print(" Pilares: E (Ambiental) + S (Social) | G = veto pre-cluster")
    print("=" * 70)

    # [1] Carrega
    df_total = carregar_base()

    # [2] Veto de Governanca (CEIS + embargados)
    df_elegiveis, df_vetadas = aplicar_veto(df_total)
    gc.collect()

    # [3] Features E + S
    df_elegiveis, X, features = preparar_features(df_elegiveis)
    gc.collect()

    # [4] K-Means
    resultados = rodar_kmeans(X)
    gc.collect()

    # [5] Metricas
    plotar_metricas(resultados)

    # [6] Rotula
    print("\n[6/7] Rotulando clusters com k otimo...")
    k_otimo = escolher_k(resultados)
    labels  = resultados[k_otimo]["labels"]
    df_elegiveis, perfil, mapa_risco = rotular_clusters(
        df_elegiveis, labels, features, k_otimo
    )
    plotar_perfil(perfil, mapa_risco, features)
    gc.collect()

    # [7] Reconstroi base completa: elegiveis rotulados + vetados (score=0)
    print(f"\n[7/7] Exportando {OUTPUT} ...")

    df_vetadas = df_vetadas.copy()
    df_vetadas["cluster"]     = -1          # cluster -1 = vetado (nao entra no ML)
    df_vetadas["risco_label"] = "Vetado"    # Governanca: inelegivel ao credito verde

    df_final = pd.concat([df_elegiveis, df_vetadas], ignore_index=True)
    df_final.to_csv(OUTPUT, sep=";", index=False, encoding="utf-8-sig")

    # Resumo
    print("\n" + "=" * 70)
    print(" RESUMO DO CLUSTERING")
    print("=" * 70)
    print(f"  k otimo                 : {k_otimo}")
    print(f"  Silhouette Score        : {resultados[k_otimo]['silhouette']:.4f}")
    print(f"  Inertia                 : {resultados[k_otimo]['inertia']:>16,.0f}")
    print(f"  Total registros         : {len(df_final):,}")
    print(f"\n  Distribuicao por risco (incluindo vetados):")
    dist = df_final["risco_label"].value_counts()
    for risco, n in dist.items():
        print(f"    {str(risco):15s}: {n:>10,} ({n/len(df_final):.1%})")
    print(f"\n  NOTA: empresas 'Vetado' receberao score final = 0 em score_credito_verde.py")
    print(f"        (inelegiveis por embargo IBAMA ou sancao CEIS ativa)")
    print(f"\n  Salvo em  : {OUTPUT}")
    print(f"  Graficos  : {PLOTS_DIR}")
    print("\nNOTA: este arquivo e EXPLORATORIO. O rotulo oficial do pipeline vem")
    print("      de rubrica_enquadramento.py (base_analitica_rotulada.csv).")


if __name__ == "__main__":
    main()
