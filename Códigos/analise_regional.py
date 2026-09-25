"""
analise_regional.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Distribuicao da amostra por Regiao/UF e ANALISE DE VIES GEOGRAFICO do score.

Pergunta que responde (debate anti-vies):
  "Como garantir que uma empresa do Norte/Nordeste nao tera score pior que
   uma do Sudeste apenas por sua localizacao?"

Salvaguardas de DESENHO ja existentes (verificadas aqui empiricamente):
  1. UF/municipio NAO sao features da Fase 2 (classificacao.py) - o modelo
     nao "ve" a regiao.
  2. A rubrica pontua por rank RELATIVO dentro de coortes CNAE x porte e as
     metricas sociais sao SETORIAIS NACIONAIS (por CNAE) - nao ha penalidade
     regional direta.
  3. Localizacao so pode SOMAR (bonus afirmativo IDH baixo) - nunca subtrai.
     [NOTA: o bonus esta INATIVO nesta rodada - municipio da RFB usa codigo
      SIAFI (4 dig.) e o IDHM usa codigo IBGE (7 dig.); a conversao SIAFI->IBGE
      e necessaria para ativa-lo. Registrado como pendencia.]

O que resta verificar EMPIRICAMENTE (este script):
  - Vies INDIRETO: a composicao setorial/porte difere por regiao; o modelo
    cadastral pode aprender proxies regionais via CNAE. Medimos:
      a) distribuicao da amostra por Regiao/UF;
      b) score medio e % de aprovacao (Verde-A/B) da RUBRICA por regiao;
      c) FASE 2 no conjunto de teste, por regiao: taxa de aprovacao prevista
         vs real, sensibilidade (recall) e taxa de falsos positivos - se forem
         proximas entre regioes, ha igualdade de oportunidade (equalized odds).

OUTPUT:
  Dados/relatorio_regional.txt
  Dados/tabela_regional_uf.csv | tabela_regional_regiao.csv
  Dados/plots/regional/*.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS    = os.path.join(BASE_DIR, "Dados")
PLOTS    = os.path.join(DADOS, "plots", "regional")
INPUT    = os.path.join(DADOS, "base_analitica_scored.csv")
# modelo_melhor.pkl = XGBoost da Fase 2 vigente (cadastral). NAO usar o
# legado modelo_xgb.pkl (formato antigo, features socioambientais).
MODELO   = os.path.join(DADOS, "modelos", "modelo_melhor.pkl")
REL      = os.path.join(DADOS, "relatorio_regional.txt")
TAB_UF   = os.path.join(DADOS, "tabela_regional_uf.csv")
TAB_REG  = os.path.join(DADOS, "tabela_regional_regiao.csv")
TAB_FAIR = os.path.join(DADOS, "tabela_regional_fairness.csv")
os.makedirs(PLOTS, exist_ok=True)

SEED = 42; TEST_SIZE = 0.20
HOJE = pd.Timestamp(2026, 9, 1)   # dia seguinte ao corte dos dados (31/08/2026); ver recorte.py
PORTE_MAP = {"00": 0, "01": 1, "03": 2, "05": 3}

REGIAO = {
    **{u: "Norte"        for u in ["AC","AP","AM","PA","RO","RR","TO"]},
    **{u: "Nordeste"     for u in ["AL","BA","CE","MA","PB","PE","PI","RN","SE"]},
    **{u: "Centro-Oeste" for u in ["DF","GO","MT","MS"]},
    **{u: "Sudeste"      for u in ["ES","MG","RJ","SP"]},
    **{u: "Sul"          for u in ["PR","RS","SC"]},
}
ORD_REG = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
APROVA  = {"Verde-A", "Verde-B"}

AZUL = "#1A237E"; CLARO = "#90CAF9"; VERDE = "#1B5E20"


def carregar():
    print("[1/5] Carregando base pontuada (colunas necessarias) ...")
    cols = ["uf", "porte", "capital_social", "data_inicio_atividade",
            "cnae_divisao", "cnae_fiscal_principal", "faixa",
            "score_e", "score_s", "score_enquadramento", "score_socioambiental"]
    df = pd.read_csv(INPUT, sep=";", dtype=str, encoding="utf-8-sig",
                     usecols=lambda c: c in cols, low_memory=False)
    for c in ["score_e", "score_s", "score_enquadramento", "score_socioambiental",
              "capital_social"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["uf"] = df["uf"].astype(str).str.strip().str.upper()
    df["regiao"] = df["uf"].map(REGIAO).fillna("ND")
    df["aprovado"] = df["faixa"].isin(APROVA).astype(int)
    print(f"      {len(df):,} registros")
    return df


def tabelas_distribuicao(df):
    print("[2/5] Distribuicao da amostra e score por Regiao/UF ...")
    def agg(g):
        return pd.Series({
            "empresas": len(g),
            "pct_amostra": len(g) / len(df),
            "score_e_med": g["score_e"].mean(),
            "score_s_med": g["score_s"].mean(),
            "score_total_med": g["score_enquadramento"].mean(),
            "score_1000_med": g.loc[g["score_socioambiental"] >= 0,
                                    "score_socioambiental"].mean(),
            "pct_verde_a": (g["faixa"] == "Verde-A").mean(),
            "pct_aprovados": g["aprovado"].mean(),
        })
    t_reg = df.groupby("regiao").apply(agg, include_groups=False).reindex(ORD_REG + ["ND"]).dropna(how="all")
    t_uf  = df.groupby(["regiao", "uf"]).apply(agg, include_groups=False)
    t_reg.round(4).to_csv(TAB_REG, sep=";", encoding="utf-8-sig")
    t_uf.round(4).to_csv(TAB_UF, sep=";", encoding="utf-8-sig")
    return t_reg, t_uf


def fase2_fairness(df):
    """Fase 2 por regiao no TESTE (mesmo split da classificacao): igualdade de
    oportunidade = recall e FPR proximos entre regioes (binario Verde-A/B)."""
    print("[3/5] Fairness da Fase 2 por regiao (conjunto de teste) ...")
    from sklearn.model_selection import train_test_split

    el = df[~df["faixa"].isin(["Vetado", "Fora de escopo"])].copy()
    el["capital_social_log"] = np.log1p(el["capital_social"].fillna(0))
    el["porte_num"] = el["porte"].astype(str).str.strip().map(PORTE_MAP).fillna(0).astype(float)
    s = el["data_inicio_atividade"].astype(str).str.replace(r"[^0-9]", "", regex=True).str.slice(0, 8)
    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    tempo = (HOJE - dt).dt.days / 365.25
    med = tempo[tempo > 0].median()
    el["tempo_empresa_anos"] = tempo.where(tempo > 0).fillna(med)
    if "cnae_divisao" not in el.columns or el["cnae_divisao"].isna().all():
        el["cnae_divisao"] = el["cnae_fiscal_principal"].astype(str).str[:2]
    el["cnae_divisao"] = el["cnae_divisao"].astype(str).str.zfill(2)

    X_num = el[["porte_num", "capital_social_log", "tempo_empresa_anos"]].fillna(0)
    X_cat = pd.get_dummies(el["cnae_divisao"], prefix="cnae")
    X_df = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)

    with open(MODELO, "rb") as f:
        payload = pickle.load(f)
    feats = payload["features"]; modelo = payload["modelo"]
    X_df = X_df.reindex(columns=feats, fill_value=0.0)

    # alvo identico ao da classificacao (faixa 4 classes) p/ reproduzir o split
    RANK = {"Verde-A": 0, "Verde-B": 1, "Amarelo": 2, "Vermelho": 3}
    y4 = el["faixa"].map(RANK).fillna(2).astype(int).values

    idx = np.arange(len(el))
    _, idx_te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, stratify=y4)
    Xte = X_df.values[idx_te].astype(np.float64)
    el_te = el.iloc[idx_te].copy()

    pred4 = modelo.predict_proba(Xte).argmax(axis=1)
    el_te["aprov_prev"] = (pred4 <= 1).astype(int)     # Verde-A/B previsto
    el_te["aprov_real"] = (y4[idx_te] <= 1).astype(int)

    linhas = []
    for reg in ORD_REG:
        g = el_te[el_te["regiao"] == reg]
        if not len(g):
            continue
        vp = ((g.aprov_prev == 1) & (g.aprov_real == 1)).sum()
        fn = ((g.aprov_prev == 0) & (g.aprov_real == 1)).sum()
        fp = ((g.aprov_prev == 1) & (g.aprov_real == 0)).sum()
        vn = ((g.aprov_prev == 0) & (g.aprov_real == 0)).sum()
        linhas.append({
            "regiao": reg, "n_teste": len(g),
            "aprov_real": g.aprov_real.mean(),
            "aprov_prevista": g.aprov_prev.mean(),
            "recall (sensibilidade)": vp / max(vp + fn, 1),
            "fpr (falso verde)": fp / max(fp + vn, 1),
            "acuracia": (vp + vn) / len(g),
        })
    fair = pd.DataFrame(linhas).set_index("regiao")
    fair.round(4).to_csv(TAB_FAIR, sep=";", encoding="utf-8-sig")
    return fair


def graficos(df, t_reg, fair):
    print("[4/5] Graficos ...")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    t = t_reg.loc[[r for r in ORD_REG if r in t_reg.index]]
    axes[0].bar(t.index, t["empresas"], color=CLARO, edgecolor=AZUL)
    axes[0].set_title("Amostra por regiao (n)"); axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(t.index, 100 * t["pct_aprovados"], color="#A5D6A7", edgecolor=VERDE)
    axes[1].axhline(100 * df["aprovado"].mean(), ls="--", color=VERDE,
                    label=f"Brasil: {100*df['aprovado'].mean():.1f}%")
    axes[1].set_title("% aprovados (Verde-A/B) — rubrica"); axes[1].legend()
    axes[1].tick_params(axis="x", rotation=20)
    f = fair.loc[[r for r in ORD_REG if r in fair.index]]
    x = np.arange(len(f)); w = 0.38
    axes[2].bar(x - w/2, f["recall (sensibilidade)"], w, label="Recall", color=AZUL)
    axes[2].bar(x + w/2, f["fpr (falso verde)"], w, label="FPR", color="#EF9A9A")
    axes[2].set_xticks(x); axes[2].set_xticklabels(f.index, rotation=20)
    axes[2].set_title("Fase 2 (teste): recall e FPR por regiao"); axes[2].legend()
    plt.tight_layout()
    cam = os.path.join(PLOTS, "regional_painel.png")
    plt.savefig(cam, dpi=150, bbox_inches="tight"); plt.close()
    print(f"      Salvo: {cam}")


def relatorio(df, t_reg, t_uf, fair):
    print("[5/5] Relatorio ...")
    L = ["=" * 70,
         " ANALISE REGIONAL — DISTRIBUICAO DA AMOSTRA E VIES GEOGRAFICO",
         "=" * 70, ""]
    L.append("1) DISTRIBUICAO DA AMOSTRA POR REGIAO")
    for reg, r in t_reg.iterrows():
        L.append(f"   {reg:13s}: {int(r.empresas):>10,} ({r.pct_amostra:6.1%})  "
                 f"score={r.score_total_med:5.1f}  aprovados={r.pct_aprovados:6.1%}")
    L.append("")
    L.append("2) RUBRICA — dispersao entre regioes")
    sc = t_reg.loc[[r for r in ORD_REG if r in t_reg.index], "score_total_med"]
    ap = t_reg.loc[[r for r in ORD_REG if r in t_reg.index], "pct_aprovados"]
    L.append(f"   score medio: min={sc.min():.1f} ({sc.idxmin()}) | max={sc.max():.1f} ({sc.idxmax()}) | amplitude={sc.max()-sc.min():.1f} p.")
    L.append(f"   %aprovados : min={ap.min():.1%} ({ap.idxmin()}) | max={ap.max():.1%} ({ap.idxmax()}) | amplitude={(ap.max()-ap.min())*100:.1f} p.p.")
    L.append("")
    L.append("3) FASE 2 (teste) — IGUALDADE DE OPORTUNIDADE POR REGIAO")
    L.append(fair.round(4).to_string())
    rc = fair["recall (sensibilidade)"]; fp = fair["fpr (falso verde)"]
    L.append("")
    L.append(f"   recall: amplitude entre regioes = {(rc.max()-rc.min())*100:.1f} p.p.")
    L.append(f"   FPR   : amplitude entre regioes = {(fp.max()-fp.min())*100:.1f} p.p.")
    L.append("")
    L += ["4) SALVAGUARDAS DE DESENHO (por que o vies direto nao ocorre)",
          "   - UF/municipio NAO sao features da Fase 2 (modelo nao 've' regiao).",
          "   - Metricas sociais sao setoriais NACIONAIS (CNAE); rubrica rankeia",
          "     dentro de coortes CNAE x porte (pares), nao entre regioes.",
          "   - Localizacao so SOMA (bonus afirmativo IDH baixo), nunca subtrai.",
          "",
          "5) PENDENCIA HONESTA",
          "   - Bonus IDH INATIVO nesta rodada: 'municipio' (RFB) usa codigo SIAFI",
          "     (4 dig.) e o IDHM usa IBGE (7 dig.) -> requer tabela de conversao",
          "     SIAFI->IBGE para ativar. Sem ele, o desenho ja e neutro (nao pune);",
          "     com ele, torna-se afirmativo (soma p/ municipios de IDH baixo).",
          "=" * 70]
    txt = "\n".join(L)
    print(txt)
    with open(REL, "w", encoding="utf-8") as f:
        f.write(txt)


def main():
    print("=" * 70)
    print(" ANALISE REGIONAL E DE VIES GEOGRAFICO")
    print("=" * 70)
    df = carregar()
    t_reg, t_uf = tabelas_distribuicao(df)
    fair = fase2_fairness(df)
    graficos(df, t_reg, fair)
    relatorio(df, t_reg, t_uf, fair)


if __name__ == "__main__":
    main()
