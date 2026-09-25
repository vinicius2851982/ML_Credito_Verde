"""
rubrica_enquadramento.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

FASE 1 (substitui o clustering como FONTE DO ROTULO).

Por que existe:
  O K-Means gerava o rotulo de risco a partir das MESMAS variaveis que a
  Fase 2 depois reaprendia -> circularidade -> AUC ~ 1,0 sem significado.
  Aqui o rotulo passa a vir de uma RUBRICA DE CRITERIOS OFICIAIS de credito
  sustentavel/ESG, comparada DENTRO de grupos de pares (CNAE divisao x porte).
  Definido FORA dos dados da PME -> nao e circular.

Ancoragem (ver "20260605 - DECISOES_TCC - claude.md"):
  - ATIVIDADE verde: Taxonomia Verde da FEBRABAN (subclasse CNAE, escopo E+S),
    via colunas fbb_* de construir_base_febraban.py. Legitimidade: TSB
    (Decreto 12.705/2025, em harmonizacao c/ a FEBRABAN); BNDES e componente.
  - CONDUTA da firma (camada distinta): IBAMA, social RAIS/CAGED, veto CEIS.
    Modelo de DUAS CAMADAS -> trava anti-dupla-contagem entre o [Social] da
    atividade (bonus S) e o pilar S de conduta (merito S).

Principios (ver "20260604 - Framework_Criterios_Credito_Verde_PME - claude.md" + "20260605 - DECISOES_TCC - claude.md"):
  - BONUS de atividade FEBRABAN por eixo: Ambiental->E, Social->S, ambos->E+S.
  - EXPOSICAO (risco amb./climatico da atividade) != demerito -> nao penaliza
    por padrao (toggle PENALIZAR_EXPOSICAO, default False).
  - Pesos por PORTE: micro/pequena -> Social preponderante; E cresce c/ porte.
  - PENALIZACAO compensavel (IBAMA/CEIS reduzem score; nao eliminam; teto Amarelo).
  - EXCLUSAO categorica (CNAE fora de escopo: fossil, jogos...) != penalizacao.
  - BONUS geografico AFIRMATIVO (IDH baixo soma; NUNCA subtrai) -> anti-vies.
  - HOOK de cadeia (% fornecedores verdes): inerte ate existir dado por firma.
  - G (Governanca) nao e feature: entra como penalizacao/contrapartida.

Faixas de saida (4 + Vetado + Fora de escopo):
  Verde-A > Verde-B > Amarelo > Vermelho | Vetado | Fora de escopo

OUTPUT:
  Dados/base_analitica_rotulada.csv   (coluna risco_label = faixa; +scores)
  Dados/plots/rubrica_distribuicao.png
  Dados/relatorio_rubrica.txt
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
INPUT     = os.path.join(DADOS, "base_analitica_febraban.csv")
OUTPUT    = os.path.join(DADOS, "base_analitica_rotulada.csv")
RELATORIO = os.path.join(DADOS, "relatorio_rubrica.txt")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==================================================================
# CONSTANTES DA RUBRICA  (todas calibraveis; ancoradas no framework)
# ==================================================================

# Porte RFB: 01=Micro  03=Pequena(EPP)  05=Demais(media+)  00=Nao informado
PORTE_NOME = {"01": "Micro", "03": "Pequena", "05": "Media+", "00": "ND"}

# Pesos (S, E) por porte — micro: Social preponderante (§6.2 do framework).
PESOS_PORTE = {
    "Micro":   {"S": 0.70, "E": 0.30},
    "Pequena": {"S": 0.60, "E": 0.40},
    "Media+":  {"S": 0.50, "E": 0.50},
    "ND":      {"S": 0.60, "E": 0.40},
}

# --- Pilar E: merito ambiental SETORIAL (varia entre CNAE -> rank GLOBAL) ---
# (col, peso, sentido)  sentido=+1 maior e melhor ; -1 menor e melhor
E_MERITO_SETOR = [
    ("esg_ambiental_setor",      0.65, +1),   # benchmark B3/ISE ambiental
    ("intensidade_emprego_cnae", 0.10, +1),   # contexto setorial
]
# --- Pilar E: conduta INDIVIDUAL (varia dentro do CNAE -> rank por COORTE) ---
E_CONDUTA = [
    ("anos_desde_ultima_infracao", 0.25, +1), # + tempo sem infracao = melhor
]

# --- Pilar S: merito social SETORIAL (rank GLOBAL) ---
# Hooks de equidade (gap g/raca ajustado, % chefia feminina) entram aqui
# quando etl_social.py exportar essas colunas (peso ja reservado).
S_MERITO_SETOR = [
    ("esg_social_setor",   0.30, +1),   # benchmark B3/ISE social
    ("salario_medio_sm",   0.20, +1),   # remuneracao setorial
    ("tempo_emprego_medio",0.15, +1),   # estabilidade
    ("pct_vinculo_ativo",  0.15, +1),   # formalizacao
    ("rotatividade",       0.10, -1),   # menor rotatividade = melhor
    ("saldo_setor_taxa",   0.10, +1),   # setor gerando vagas
]
# Colunas de EQUIDADE (S2/S4) — entram automaticamente se existirem na base.
# (col, peso, sentido). Pesos serao RENORMALIZADOS junto com S_MERITO_SETOR.
S_EQUIDADE_OPCIONAL = [
    ("pct_chefia_feminina_cnae", 0.20, +1),   # % mulheres em chefia (CBO 1-2)
    ("gap_genero_ajustado_cnae", 0.20, -1),   # gap salarial ajustado (0 = ideal)
    ("gap_raca_ajustado_cnae",   0.15, -1),
    ("dispersao_salarial_cnae",  0.10, -1),   # razao topo/base (menor = melhor)
]

# --- Penalizacoes por conduta (compensaveis; em pontos na escala 0-100) ---
PEN_EMBARGO_IBAMA   = 30.0    # embargado = 1
PEN_INFRACAO_MAX    = 30.0    # teto p/ multas/infracoes (log)
PEN_CEIS_ATIVA      = 35.0    # ceis_sancoes_ativas > 0
PEN_CEIS_HISTORICO  = 15.0    # ceis_sancionado (sem sancao ativa)

# --- Exclusao categorica (Fora de escopo) — Anexo 2 BNDES + fossil ---
# Divisao CNAE (2 digitos). Conservador (granularidade de divisao):
#   05 carvao, 06 petroleo/gas, 19 coque/petroleo (FOSSIL);
#   92 jogos de azar/apostas.
# (armas, moteis, garimpo exigem subclasse CNAE -> refinar depois)
EXCLUSAO_CNAE_DIVISAO = {"05", "06", "19", "92"}

# --- Atividade verde — Taxonomia Verde da FEBRABAN (camada de ATIVIDADE) ---
# Usa as colunas fbb_* ja presentes na base (construir_base_febraban.py), no nivel de
# SUBCLASSE CNAE. Escopo ESG: contribuicao AMBIENTAL bonifica E; SOCIAL bonifica S;
# "Social + Ambiental" bonifica os dois. Escala 0-100 (comparavel as penalidades).
BONUS_FEBRABAN = {"Alta": 8.0, "Moderada": 4.0}

def bonus_atividade_febraban(df: pd.DataFrame):
    """(bonus_e, bonus_s) por firma a partir da economia verde FEBRABAN.
    Anti-dupla-contagem: bonus da ATIVIDADE (proposito do setor), distinto do
    merito de CONDUTA da firma."""
    rot  = df.get("fbb_economia_verde", pd.Series("", index=df.index)).fillna("").astype(str)
    eixo = df.get("fbb_ev_eixo",        pd.Series("", index=df.index)).fillna("").astype(str)
    nivel = rot.str.extract(r"^(Alta|Moderada)", expand=False).map(BONUS_FEBRABAN).fillna(0.0)
    tem_amb = eixo.str.contains("Ambiental", na=False).astype(float)
    tem_soc = eixo.str.contains("Social",    na=False).astype(float)
    return nivel * tem_amb, nivel * tem_soc

# --- Eixo dinamico de cadeia (maturidade como trajetoria) — HOOK ---
# Bonus pelo % de fornecedores verdes (0-1). A base NAO tem esse dado por firma;
# fica inerte ate a coluna existir. Documentar como mecanismo proposto no TCC.
COL_CADEIA_VERDE = "pct_fornecedores_verdes"   # ajustar se/quando existir
BONUS_CADEIA_MAX = 8.0

# --- Exposicao (risco ambiental/climatico) da ATIVIDADE — FEBRABAN ---
# Exposicao NAO e demerito de conduta. Por padrao so carregamos as colunas fbb_* (uteis
# na Fase 2/relatorios), SEM penalizar o merito. Ligue o toggle so se quiser modular.
PENALIZAR_EXPOSICAO = False
PEN_EXPOSICAO_AMBIENTAL = 5.0
PEN_EXPOSICAO_CLIMATICA = 5.0

# --- Bonus geografico AFIRMATIVO (IDH municipal abaixo da media) ---
# So SOMA, nunca subtrai (anti-vies, §6.3.1). Hook: requer coluna idh_municipio.
BONUS_IDH_BAIXO = 5.0

# Coorte de comparacao entre pares: CNAE divisao x porte
COORTE = ["cnae_divisao", "porte_nome"]

# Faixas relativas DENTRO do porte (quantis do score entre pares):
#   topo 20% Verde-A | 20-50% Verde-B | 50-80% Amarelo | 80-100% Vermelho
FAIXA_QUANTIS = [0.20, 0.50, 0.80]
FAIXA_NOMES   = ["Verde-A", "Verde-B", "Amarelo", "Vermelho"]

# Teto suave: embargo IBAMA / CEIS ativo nao passam desta faixa (compensavel,
# mas sem a melhor taxa verde). Decisao do Helio.
TETO_CONDUTA_FAIXA = "Amarelo"


# ==================================================================
# FUNCOES
# ==================================================================

def carregar() -> pd.DataFrame:
    print(f"[1/6] Carregando {os.path.basename(INPUT)} ...")
    df = pd.read_csv(INPUT, sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)
    print(f"      {len(df):,} registros | {len(df.columns)} colunas")
    return df


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Preparando colunas e coortes ...")
    num_cols = [
        "qtd_infracoes", "valor_total_multas", "anos_desde_ultima_infracao",
        "capital_social", "salario_medio_sm", "tempo_emprego_medio",
        "pct_vinculo_ativo", "rotatividade", "saldo_setor_taxa",
        "intensidade_emprego_cnae", "esg_ambiental_setor", "esg_social_setor",
        "embargado", "ceis_sancoes_ativas", "ceis_sancionado", "veto_governanca",
    ]
    # colunas opcionais de equidade / IDH (entram se o ETL ja as produziu)
    for c, _, _ in S_EQUIDADE_OPCIONAL:
        num_cols.append(c)
    num_cols += ["idh_municipio"]

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # porte legivel
    df["porte"] = df["porte"].astype(str).str.strip()
    df["porte_nome"] = df["porte"].map(PORTE_NOME).fillna("ND")

    # cnae_divisao garantida
    if "cnae_divisao" not in df.columns or df["cnae_divisao"].isna().all():
        df["cnae_divisao"] = df["cnae_fiscal_principal"].astype(str).str[:2]
    df["cnae_divisao"] = df["cnae_divisao"].astype(str).str.zfill(2)

    return df


def _rank_global(s: pd.Series, sentido: int) -> pd.Series:
    """Rank percentil global em [0,1]; sentido=-1 inverte (menor=melhor)."""
    r = s.rank(pct=True, method="average")
    return r if sentido > 0 else (1.0 - r)


def _rank_coorte(df: pd.DataFrame, col: str, sentido: int) -> pd.Series:
    """Rank percentil DENTRO da coorte (CNAE x porte)."""
    r = df.groupby(COORTE)[col].rank(pct=True, method="average")
    r = r.fillna(0.5)   # coorte com 1 elemento -> neutro
    return r if sentido > 0 else (1.0 - r)


def _merito(df: pd.DataFrame, specs, escopo: str) -> pd.Series:
    """Combina specs (col,peso,sentido) em score 0-100. escopo: 'global'|'coorte'."""
    specs = [(c, w, s) for (c, w, s) in specs if c in df.columns]
    if not specs:
        return pd.Series(0.0, index=df.index)
    soma_pesos = sum(w for _, w, _ in specs)
    acc = pd.Series(0.0, index=df.index)
    for col, peso, sentido in specs:
        rk = _rank_global(df[col], sentido) if escopo == "global" \
             else _rank_coorte(df, col, sentido)
        acc += (peso / soma_pesos) * rk
    return 100.0 * acc


def calcular_scores(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Calculando scores E e S (semelhanca ao benchmark verde) ...")

    # ---------- Pilar E ----------
    e_setor   = _merito(df, E_MERITO_SETOR, "global")
    e_conduta = _merito(df, E_CONDUTA, "coorte")
    w_set = sum(w for _, w, _ in E_MERITO_SETOR)
    w_con = sum(w for _, w, _ in E_CONDUTA)
    e_merito = (w_set * e_setor + w_con * e_conduta) / (w_set + w_con)

    # bonus atividade verde (afirmativo) — Taxonomia FEBRABAN (E e S por eixo)
    bonus_e_ativ, bonus_s_ativ = bonus_atividade_febraban(df)

    # penalizacao ambiental (compensavel)
    pen_e = pd.Series(0.0, index=df.index)
    if "embargado" in df:
        pen_e += np.where(df["embargado"].fillna(0) == 1, PEN_EMBARGO_IBAMA, 0.0)
    if "qtd_infracoes" in df:
        qi = df["qtd_infracoes"].fillna(0).clip(lower=0)
        mu = df.get("valor_total_multas", pd.Series(0.0, index=df.index)).fillna(0)
        sev = np.log1p(qi) + 0.3 * np.log1p(mu / 1000.0)
        pen_e += np.minimum(PEN_INFRACAO_MAX, 6.0 * sev)

    if PENALIZAR_EXPOSICAO and "fbb_risco_ambiental" in df:
        exp_amb = df["fbb_risco_ambiental"].fillna("").astype(str).str.contains("Alta", na=False)
        pen_e = pen_e + np.where(exp_amb, PEN_EXPOSICAO_AMBIENTAL, 0.0)
    df["score_e"] = (e_merito + bonus_e_ativ - pen_e).clip(lower=-50, upper=100)

    # ---------- Pilar S ----------
    s_specs = S_MERITO_SETOR + [(c, w, s) for (c, w, s) in S_EQUIDADE_OPCIONAL
                                if c in df.columns]
    s_merito = _merito(df, s_specs, "global")

    pen_s = pd.Series(0.0, index=df.index)
    if "ceis_sancoes_ativas" in df:
        pen_s += np.where(df["ceis_sancoes_ativas"].fillna(0) > 0, PEN_CEIS_ATIVA, 0.0)
    if "ceis_sancionado" in df:
        # historico sem sancao ativa
        ativa = df.get("ceis_sancoes_ativas", pd.Series(0.0, index=df.index)).fillna(0) > 0
        hist  = (df["ceis_sancionado"].fillna(0) > 0) & (~ativa)
        pen_s += np.where(hist, PEN_CEIS_HISTORICO, 0.0)

    df["score_s"] = (s_merito + bonus_s_ativ - pen_s).clip(lower=-50, upper=100)

    # ---------- Total ponderado por porte ----------
    wS = df["porte_nome"].map(lambda p: PESOS_PORTE.get(p, PESOS_PORTE["ND"])["S"])
    wE = df["porte_nome"].map(lambda p: PESOS_PORTE.get(p, PESOS_PORTE["ND"])["E"])
    bonus_cadeia = pd.Series(0.0, index=df.index)
    if COL_CADEIA_VERDE in df.columns:
        pct = pd.to_numeric(df[COL_CADEIA_VERDE], errors="coerce").fillna(0.0).clip(0, 1)
        bonus_cadeia = pct * BONUS_CADEIA_MAX
    else:
        print(f"  [cadeia] coluna '{COL_CADEIA_VERDE}' ausente — eixo de cadeia inerte (0).")
    score = wS * df["score_s"] + wE * df["score_e"] + bonus_cadeia
    if PENALIZAR_EXPOSICAO and "fbb_clima" in df:
        exp_cli = df["fbb_clima"].fillna("").astype(str).str.contains("Alta", na=False)
        score = score - np.where(exp_cli, PEN_EXPOSICAO_CLIMATICA, 0.0)

    # bonus geografico afirmativo (so soma)
    if "idh_municipio" in df.columns and df["idh_municipio"].notna().any():
        media_idh = df["idh_municipio"].mean()
        score += np.where(df["idh_municipio"] < media_idh, BONUS_IDH_BAIXO, 0.0)

    df["score_enquadramento"] = score.clip(lower=-50, upper=100)
    return df


def classificar_faixas(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/6] Atribuindo faixas (regua relativa aos pares por porte) ...")

    df["faixa"] = pd.NA

    # 1) Fora de escopo (exclusao categorica)
    mask_fora = df["cnae_divisao"].isin(EXCLUSAO_CNAE_DIVISAO)
    df.loc[mask_fora, "faixa"] = "Fora de escopo"

    # 2) Vetado (penalizacao severa -> score negativo) entre os em escopo
    mask_vet = (~mask_fora) & (df["score_enquadramento"] < 0)
    df.loc[mask_vet, "faixa"] = "Vetado"

    # 3) Demais: quantis do score DENTRO do porte
    rest = df["faixa"].isna()
    if rest.any():
        def faixa_por_quantil(grp: pd.Series) -> pd.Series:
            qs = grp.quantile(FAIXA_QUANTIS).values
            # bins: -inf < q20 < q50 < q80 < +inf  (invertido: score alto=Verde-A)
            cortes = [-np.inf, qs[0], qs[1], qs[2], np.inf]
            # labels do pior->melhor p/ pd.cut, depois invertemos a leitura
            cat = pd.cut(grp, bins=cortes, labels=FAIXA_NOMES[::-1],
                         include_lowest=True)
            return cat.astype(object)
        df.loc[rest, "faixa"] = (
            df.loc[rest]
              .groupby("porte_nome")["score_enquadramento"]
              .transform(lambda g: faixa_por_quantil(g))
        )
    df["faixa"] = df["faixa"].fillna("Vermelho")

    # 4) TETO SUAVE (decisao do Helio): conduta com embargo IBAMA ou CEIS ATIVO
    #    nao elimina (continua compensavel), mas NAO pode pegar a melhor taxa
    #    verde -> teto em "Amarelo". Honra o "nao mata" sem a contradicao de
    #    uma empresa embargada alcancar Verde-A.
    emb  = df.get("embargado", pd.Series(0.0, index=df.index)).fillna(0)
    ceis = df.get("ceis_sancoes_ativas", pd.Series(0.0, index=df.index)).fillna(0)
    mask_teto = ((emb == 1) | (ceis > 0)) & df["faixa"].isin(["Verde-A", "Verde-B"])
    n_teto = int(mask_teto.sum())
    df.loc[mask_teto, "faixa"] = TETO_CONDUTA_FAIXA
    if n_teto:
        print(f"      Teto suave aplicado (embargo/CEIS ativo -> {TETO_CONDUTA_FAIXA}): {n_teto:,}")

    # compatibilidade com a Fase 2 (classificacao.py consome risco_label)
    df["risco_label"] = df["faixa"]
    return df


def relatar(df: pd.DataFrame) -> None:
    print("[5/6] Gerando relatorio e grafico ...")
    linhas = []
    def log(t=""):
        print(t); linhas.append(t)

    log("=" * 64)
    log(" RUBRICA DE ENQUADRAMENTO VERDE — RESUMO")
    log("=" * 64)
    log(f"  Registros: {len(df):,}")
    log("")
    log("  Distribuicao por faixa:")
    dist = df["faixa"].value_counts()
    ordem = ["Verde-A", "Verde-B", "Amarelo", "Vermelho", "Vetado", "Fora de escopo"]
    for f in ordem:
        if f in dist:
            n = int(dist[f]); log(f"    {f:16s}: {n:>10,} ({n/len(df):.1%})")
    log("")
    log("  Score de enquadramento (0-100):")
    s = df["score_enquadramento"]
    log(f"    min={s.min():.1f}  p25={s.quantile(.25):.1f}  "
        f"mediana={s.median():.1f}  p75={s.quantile(.75):.1f}  max={s.max():.1f}")
    log("")
    log("  Faixa media por porte:")
    piv = df.pivot_table(index="porte_nome", values=["score_e","score_s",
                          "score_enquadramento"], aggfunc="mean")
    for porte, row in piv.iterrows():
        log(f"    {porte:8s}  E={row['score_e']:5.1f}  S={row['score_s']:5.1f}  "
            f"Total={row['score_enquadramento']:5.1f}")
    log("")
    eq = [c for c, _, _ in S_EQUIDADE_OPCIONAL if c in df.columns]
    log(f"  Colunas de equidade presentes: {eq if eq else 'NENHUMA (rodar etl_social atualizado)'}")
    idh = "SIM" if "idh_municipio" in df.columns and df["idh_municipio"].notna().any() else "NAO"
    log(f"  Bonus IDH aplicado: {idh}")
    log("=" * 64)

    with open(RELATORIO, "w", encoding="utf-8") as fh:
        fh.write("\n".join(linhas))

    # grafico
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cores = {"Verde-A": "#1B5E20", "Verde-B": "#66BB6A", "Amarelo": "#F9A825",
             "Vermelho": "#C62828", "Vetado": "#4A148C", "Fora de escopo": "#757575"}
    ord_p = [f for f in ordem if f in dist]
    axes[0].bar(ord_p, [dist[f] for f in ord_p],
                color=[cores[f] for f in ord_p])
    axes[0].set_title("Distribuicao por faixa de enquadramento")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Empresas")

    axes[1].hist(df["score_enquadramento"], bins=50, color="#1A237E", alpha=0.85)
    axes[1].axvline(0, color="#C62828", linestyle="--", linewidth=1)
    axes[1].set_title("Distribuicao do score (0-100)")
    axes[1].set_xlabel("score_enquadramento")
    plt.tight_layout()
    cam = os.path.join(PLOTS_DIR, "rubrica_distribuicao.png")
    plt.savefig(cam, dpi=150, bbox_inches="tight"); plt.close()
    log(f"  Grafico: {cam}")


def main() -> None:
    print("=" * 64)
    print(" RUBRICA DE ENQUADRAMENTO VERDE (Fase 1 — fonte do rotulo)")
    print(" Atividade: Taxonomia FEBRABAN (E+S) | Legitimidade: TSB (Dec. 12.705/2025)")
    print("=" * 64)
    df = carregar()
    df = preparar(df)
    df = calcular_scores(df)
    df = classificar_faixas(df)
    relatar(df)
    print(f"[6/6] Exportando {os.path.basename(OUTPUT)} ...")
    df.to_csv(OUTPUT, sep=";", index=False, encoding="utf-8-sig")
    print(f"      Salvo: {OUTPUT}")
    print("\nProximo passo: classificacao.py (prever a faixa a partir do cadastro)")


if __name__ == "__main__":
    main()
