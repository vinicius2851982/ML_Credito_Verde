"""
estimativas_populacionais.py  (passo 14c — expansao da amostra para a populacao)
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Aplica o PESO AMOSTRAL e converte os achados da amostra em ESTIMATIVAS
POPULACIONAIS com intervalo de confianca. E o passo que transforma o
trabalho de "descricao de uma amostra" em INFERENCIA sobre o universo das
empresas ativas no pais — e que define a APLICABILIDADE pratica do modelo
(mercado enderecavel de credito verde).

DESENHO
  Amostragem aleatoria simples (AAS) sem reposicao de estabelecimentos
  ATIVOS, fracao f = n/N. Como a selecao foi equiprobabilistica e a
  aderencia por regiao e setor foi verificada (analise_amostral.py), o peso
  e UNIFORME:

      w = N / n        (cada registro representa w unidades da populacao)

  Nao se aplicou pos-estratificacao porque os desvios observados entre
  amostra e populacao ficaram abaixo de 0,05 p.p. — o ganho de precisao
  seria irrelevante diante da complexidade adicional.

ESTIMADORES (AAS sem reposicao)
  proporcao : p = y / n
  variancia : Var(p) = (1 - f) * p(1-p) / (n - 1)      <- (1-f) = correcao
                                                          para populacao finita
  total     : Y = N * p        IC(Y) = N * IC(p)

  A correcao (1-f) reduz o erro porque 10% da populacao foi de fato
  observada; ignora-la superestimaria a incerteza.

OBSERVACAO SOBRE A MODELAGEM
  O peso e uniforme, logo NAO altera o ajuste dos modelos da Fase 2 (ponderar
  todas as observacoes igualmente equivale a nao ponderar). O peso afeta a
  LEITURA dos resultados, nao o treino.

OUTPUT:
  Dados/estimativas_populacionais.csv
  Dados/relatorio_estimativas_populacionais.txt
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(BASE_DIR, "Dados")
SCORED = os.path.join(DADOS, "base_analitica_scored.csv")
POP_CSV = os.path.join(DADOS, "parametros_populacionais.csv")
SAIDA_CSV = os.path.join(DADOS, "estimativas_populacionais.csv")
REL = os.path.join(DADOS, "relatorio_estimativas_populacionais.txt")

Z = 1.959964            # 95%
N_PADRAO = 27_647_482   # fallback: populacao apurada em 25/05/2026

REGIAO = {
    **{u: "Norte" for u in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
    **{u: "Nordeste" for u in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{u: "Centro-Oeste" for u in ["DF", "GO", "MT", "MS"]},
    **{u: "Sudeste" for u in ["ES", "MG", "RJ", "SP"]},
    **{u: "Sul" for u in ["PR", "RS", "SC"]},
}
ORD_REG = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
PORTE_NOME = {"01": "Micro", "03": "Pequena", "05": "Media+", "00": "Nao informado"}


def ler_N():
    """Le a populacao apurada por analise_amostral.py; usa fallback se ausente."""
    if os.path.exists(POP_CSV):
        try:
            d = pd.read_csv(POP_CSV, sep=";")
            v = d.loc[d["indicador"] == "populacao_ativos", "valor"]
            if len(v):
                return int(float(v.iloc[0]))
        except Exception:
            pass
    print(f"[AVISO] {os.path.basename(POP_CSV)} ausente — usando N padrao.")
    return N_PADRAO


def estimar(y, n, N):
    """Proporcao e total estimados, com IC 95% e correcao p/ populacao finita."""
    if n <= 1:
        return dict(zip(("p", "p_li", "p_ls", "total", "tot_li", "tot_ls", "erro_pp"),
                        (np.nan,) * 7))
    f = n / N
    p = y / n
    var = max((1 - f) * p * (1 - p) / (n - 1), 0.0)
    se = np.sqrt(var)
    li, ls = max(p - Z * se, 0.0), min(p + Z * se, 1.0)
    return {"p": p, "p_li": li, "p_ls": ls,
            "total": N * p, "tot_li": N * li, "tot_ls": N * ls,
            "erro_pp": Z * se * 100}


def carregar():
    print("[1/4] Carregando base pontuada ...")
    cols = ["uf", "porte", "faixa", "flag_economia_verde", "fbb_ev_eixo",
            "flag_risco_ambiental", "flag_clima", "score_socioambiental",
            "qtd_infracoes", "embargado", "ceis_sancoes_ativas"]
    df = pd.read_csv(SCORED, sep=";", dtype=str, encoding="utf-8-sig",
                     usecols=lambda c: c in cols, low_memory=False)
    for c in ("flag_economia_verde", "flag_risco_ambiental", "flag_clima",
              "score_socioambiental", "qtd_infracoes", "embargado",
              "ceis_sancoes_ativas"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["regiao"] = df["uf"].astype(str).str.strip().str.upper().map(REGIAO).fillna("ND")
    df["porte_nome"] = df["porte"].astype(str).str.strip().str.zfill(2).map(PORTE_NOME).fillna("ND")
    df["aprovado"] = df["faixa"].isin(["Verde-A", "Verde-B"]).astype(int)
    print(f"      {len(df):,} registros")
    return df


def main():
    print("=" * 70)
    print(" ESTIMATIVAS POPULACIONAIS — expansao pelo peso amostral")
    print("=" * 70)
    df = carregar()
    n = len(df); N = ler_N(); w = N / n; f = n / N

    print(f"[2/4] Peso amostral: w = {w:.4f} (fracao f = {f:.4%})")

    linhas = []

    def add(dominio, indicador, y, n_dom=None, N_dom=None):
        nn = n if n_dom is None else n_dom
        NN = N if N_dom is None else N_dom
        e = estimar(y, nn, NN)
        linhas.append({"dominio": dominio, "indicador": indicador,
                       "amostra": int(y), "n_dominio": int(nn), **e})

    print("[3/4] Estimando ...")
    # --- indicadores globais ---
    ev = (df["flag_economia_verde"].fillna(0) > 0).sum()
    add("Brasil", "Economia verde (qualquer eixo)", ev)
    for eixo, rot in (("Social", "Economia verde — eixo Social"),
                      ("Ambiental", "Economia verde — eixo Ambiental"),
                      ("Social + Ambiental", "Economia verde — ambos os eixos")):
        add("Brasil", rot, (df["fbb_ev_eixo"] == eixo).sum())
    add("Brasil", "Exposicao a risco ambiental", (df["flag_risco_ambiental"].fillna(0) > 0).sum())
    add("Brasil", "Exposicao a mudancas climaticas", (df["flag_clima"].fillna(0) > 0).sum())
    for fx in ["Verde-A", "Verde-B", "Amarelo", "Vermelho"]:
        add("Brasil", f"Faixa {fx}", (df["faixa"] == fx).sum())
    add("Brasil", "Elegiveis a credito verde (Verde-A/B)", int(df["aprovado"].sum()))
    add("Brasil", "Com autuacao do IBAMA", int((df["qtd_infracoes"].fillna(0) > 0).sum()))
    add("Brasil", "Com embargo ambiental ativo", int((df["embargado"].fillna(0) > 0).sum()))
    add("Brasil", "Com sancao de integridade ativa", int((df["ceis_sancoes_ativas"].fillna(0) > 0).sum()))

    # --- por regiao (dominio = subpopulacao) ---
    for reg in ORD_REG:
        g = df[df["regiao"] == reg]
        if g.empty:
            continue
        N_reg = len(g) * w      # tamanho estimado da subpopulacao
        add(f"Regiao: {reg}", "Elegiveis a credito verde", int(g["aprovado"].sum()),
            len(g), N_reg)
        add(f"Regiao: {reg}", "Economia verde", int((g["flag_economia_verde"].fillna(0) > 0).sum()),
            len(g), N_reg)

    # --- por porte ---
    for por in ["Micro", "Pequena", "Media+"]:
        g = df[df["porte_nome"] == por]
        if g.empty:
            continue
        N_por = len(g) * w
        add(f"Porte: {por}", "Elegiveis a credito verde", int(g["aprovado"].sum()),
            len(g), N_por)
        add(f"Porte: {por}", "Economia verde", int((g["flag_economia_verde"].fillna(0) > 0).sum()),
            len(g), N_por)

    t = pd.DataFrame(linhas)
    t.to_csv(SAIDA_CSV, sep=";", index=False, encoding="utf-8-sig",
             float_format="%.6f")

    # --- relatorio ---
    print("[4/4] Relatorio ...")
    L = ["=" * 78,
         " ESTIMATIVAS POPULACIONAIS — amostra expandida pelo peso",
         "=" * 78, "",
         "DESENHO AMOSTRAL",
         f"  Populacao (N) — estabelecimentos ativos : {N:,}",
         f"  Amostra (n)                             : {n:,}",
         f"  Fracao amostral (f = n/N)               : {f:.4%}",
         f"  Peso amostral (w = N/n)                 : {w:.4f}",
         "  Cada registro da amostra representa cerca de 10 empresas do universo.",
         "  Estimadores de AAS sem reposicao, com correcao (1-f) para populacao",
         "  finita; intervalos de confianca de 95%.", ""]

    glob = t[t["dominio"] == "Brasil"]
    L += ["ESTIMATIVAS PARA O BRASIL (empresas ativas)", "-" * 78,
          f"  {'Indicador':<42}{'Estimativa':>14}{'IC 95%':>22}"]
    for _, r in glob.iterrows():
        ic = f"[{r.tot_li/1e6:,.2f} - {r.tot_ls/1e6:,.2f}] mi"
        L.append(f"  {r.indicador:<42}{r.total/1e6:>11,.2f} mi{ic:>22}")
    L.append("")

    L += ["APLICABILIDADE — mercado enderecavel", "-" * 78]
    eleg = glob[glob["indicador"] == "Elegiveis a credito verde (Verde-A/B)"]
    if len(eleg):
        r = eleg.iloc[0]
        L += [f"  Empresas potencialmente elegiveis a credito verde no Brasil:",
              f"    {r.total:,.0f}  (IC 95%: {r.tot_li:,.0f} a {r.tot_ls:,.0f})",
              f"    equivalente a {r.p:.1%} do universo de empresas ativas",
              "",
              "  Leitura: e a ordem de grandeza do publico que poderia ser pre-triado",
              "  sem custo de auditoria — dimensao que sustenta a relevancia pratica",
              "  da proposta.", ""]

    for bloco, titulo in (("Regiao:", "POR REGIAO"), ("Porte:", "POR PORTE")):
        sub = t[t["dominio"].str.startswith(bloco)]
        if sub.empty:
            continue
        L += [f"{titulo} (estimativas em milhares de empresas)", "-" * 78,
              f"  {'Dominio':<22}{'Indicador':<26}{'Estimativa':>14}{'Erro':>10}"]
        for _, r in sub.iterrows():
            L.append(f"  {r.dominio:<22}{r.indicador:<26}"
                     f"{r.total/1e3:>11,.1f} mil{r.erro_pp:>8.2f} pp")
        L.append("")

    L += ["VALIDADE E LIMITES DE APLICACAO", "-" * 78,
          "  1. As estimativas se aplicam aos estabelecimentos ATIVOS no cadastro da",
          "     Receita Federal na posicao consultada; nao alcancam empresas baixadas,",
          "     suspensas ou informais.",
          "  2. O peso e uniforme por construcao (AAS equiprobabilistica), portanto nao",
          "     altera o ajuste dos modelos — afeta a leitura dos resultados.",
          "  3. Os intervalos refletem apenas a incerteza AMOSTRAL. Nao incorporam erro",
          "     de medida das fontes nem a incerteza da propria rubrica de enquadramento.",
          "  4. A expansao e valida para o instante do recorte; nao projeta o futuro.",
          "=" * 78]

    txt = "\n".join(L)
    print("\n" + txt)
    with open(REL, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(f"\nSalvos: {SAIDA_CSV}\n        {REL}")


if __name__ == "__main__":
    main()
