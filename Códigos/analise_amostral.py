"""
analise_amostral.py  (passo 0b — fundamentacao estatistica da amostra)
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Fundamenta ESTATISTICAMENTE o desenho amostral, em vez de justifica-lo por
limitacao computacional. Produz:

  1. PARAMETROS POPULACIONAIS — varre os arquivos brutos da RFB em blocos
     (apenas contadores; memoria minima) e apura N e a distribuicao da
     populacao de estabelecimentos ATIVOS por UF/regiao, secao CNAE e porte.

  2. ADERENCIA DA AMOSTRA — compara a distribuicao amostral com a
     populacional por estrato e aplica teste qui-quadrado de aderencia,
     verificando se a amostra e PROPORCIONAL aos grupos (estratificacao
     proporcional verificada ex-post).

  3. DIMENSIONAMENTO — calcula o erro amostral obtido (global e por estrato)
     e o n minimo exigido para a precisao-alvo, com correcao para populacao
     finita. Justifica o n adotado pela precisao NOS ESTRATOS DE ANALISE
     (regiao x porte x setor), nao pela media global.

Uso:
    python -B Códigos/analise_amostral.py            # completo (le a populacao)
    python -B Códigos/analise_amostral.py --rapido   # so amostra (sem varrer 15 GB)

OUTPUT:
  Dados/parametros_populacionais.csv
  Dados/tabela_aderencia_amostral.csv
  Dados/relatorio_amostral.txt
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(BASE_DIR, "Dados")
PASTA_ESTAB = os.path.join(DADOS, "rfb_estabelecimentos")
PASTA_EMPRESAS = os.path.join(DADOS, "rfb_empresas")
AMOSTRA = os.path.join(DADOS, "base_analitica_scored.csv")

POP_CSV = os.path.join(DADOS, "parametros_populacionais.csv")
ADER_CSV = os.path.join(DADOS, "tabela_aderencia_amostral.csv")
REL = os.path.join(DADOS, "relatorio_amostral.txt")

CHUNK = 300_000
SIT_ATIVA = "02"
Z = 1.959964            # normal padrao, 95%
ERRO_ALVO = 0.01        # 1 p.p. — precisao exigida no MENOR estrato de analise
P_CONSERV = 0.5         # variancia maxima

REGIAO = {
    **{u: "Norte" for u in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
    **{u: "Nordeste" for u in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{u: "Centro-Oeste" for u in ["DF", "GO", "MT", "MS"]},
    **{u: "Sudeste" for u in ["ES", "MG", "RJ", "SP"]},
    **{u: "Sul" for u in ["PR", "RS", "SC"]},
}
ORD_REG = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
PORTE_NOME = {"01": "Micro", "03": "Pequena", "05": "Media+", "00": "Nao informado"}


def _secao_cnae(div):
    """Divisao CNAE (2 digitos) -> secao (letra), conforme CNAE 2.0."""
    try:
        d = int(div)
    except (TypeError, ValueError):
        return "ND"
    faixas = [(1, 3, "A"), (5, 9, "B"), (10, 33, "C"), (35, 35, "D"), (36, 39, "E"),
              (41, 43, "F"), (45, 47, "G"), (49, 53, "H"), (55, 56, "I"), (58, 63, "J"),
              (64, 66, "K"), (68, 68, "L"), (69, 75, "M"), (77, 82, "N"), (84, 84, "O"),
              (85, 85, "P"), (86, 88, "Q"), (90, 93, "R"), (94, 96, "S"), (97, 97, "T"),
              (99, 99, "U")]
    for ini, fim, letra in faixas:
        if ini <= d <= fim:
            return letra
    return "ND"


def varrer_populacao():
    """Conta a populacao de estabelecimentos ATIVOS por UF e secao CNAE.
    Coleta tambem os cnpj_basico ativos (uint32), necessarios para apurar o
    porte SOMENTE das empresas com estabelecimento ativo."""
    arquivos = sorted(glob.glob(os.path.join(PASTA_ESTAB, "*.ESTABELE")))
    if not arquivos:
        print("[AVISO] arquivos brutos nao encontrados — use --rapido")
        return None
    print(f"[1/5] Varrendo {len(arquivos)} arquivos da RFB (apenas contadores) ...")
    c_uf, c_secao, total = {}, {}, 0
    blocos_cnpj = []
    for i, arq in enumerate(arquivos, 1):
        nome = os.path.basename(arq)
        lidos = 0
        for chunk in pd.read_csv(arq, sep=";", header=None, dtype=str,
                                 encoding="latin1", chunksize=CHUNK,
                                 usecols=[0, 5, 11, 19],
                                 names=["cnpj_basico", "situacao", "cnae", "uf"],
                                 on_bad_lines="skip", low_memory=False):
            ativos = chunk[chunk["situacao"].str.strip() == SIT_ATIVA]
            if ativos.empty:
                continue
            total += len(ativos); lidos += len(ativos)
            for uf, n in ativos["uf"].str.strip().str.upper().value_counts().items():
                c_uf[uf] = c_uf.get(uf, 0) + int(n)
            sec = ativos["cnae"].str.strip().str.zfill(7).str[:2].map(_secao_cnae)
            for s, n in sec.value_counts().items():
                c_secao[s] = c_secao.get(s, 0) + int(n)
            cb = pd.to_numeric(ativos["cnpj_basico"], errors="coerce").dropna()
            blocos_cnpj.append(cb.astype(np.uint32).values)
        print(f"      [{i}/{len(arquivos)}] {nome}: {lidos:,} ativos (acum. {total:,})")
    if blocos_cnpj:
        todos = np.concatenate(blocos_cnpj)
        # cnpjs unicos + quantos estabelecimentos ATIVOS cada empresa possui
        cnpjs, freq = np.unique(todos, return_counts=True)
        del todos
    else:
        cnpjs, freq = np.array([], np.uint32), np.array([], np.int64)
    print(f"      empresas distintas com estabelecimento ativo: {len(cnpjs):,}")
    if len(cnpjs):
        print(f"      estabelecimentos ativos por empresa (media): {total/len(cnpjs):.4f}")
    return {"total": total, "uf": c_uf, "secao": c_secao,
            "cnpjs_ativos": cnpjs, "freq_ativos": freq.astype(np.int64),
            "n_empresas": len(cnpjs)}


def varrer_porte(cnpjs_ativos, freq_ativos=None):
    """Conta o porte na populacao, na MESMA UNIDADE da amostra.

    A unidade amostral e o ESTABELECIMENTO (a AAS sorteou estabelecimentos).
    O porte, porem, e atributo da EMPRESA. Contar empresas distintas por porte
    compararia unidades diferentes e produziria desvio espurio: empresas com
    varios estabelecimentos aparecem mais vezes na amostra.

    Solucao: ponderar cada empresa pelo seu numero de ESTABELECIMENTOS ATIVOS
    (freq_ativos), obtendo a distribuicao populacional de estabelecimentos por
    porte — diretamente comparavel a amostra.

    Retorna as duas contagens: por estabelecimento (comparavel) e por empresa.
    """
    arquivos = sorted(glob.glob(os.path.join(PASTA_EMPRESAS, "*.EMPRECSV")))
    if not arquivos or cnpjs_ativos is None or len(cnpjs_ativos) == 0:
        return None
    print(f"[2/5] Varrendo {len(arquivos)} arquivos de Empresas "
          f"(porte, filtrado por {len(cnpjs_ativos):,} CNPJ ativos) ...")
    c_estab, c_empresa, total = {}, {}, 0
    for i, arq in enumerate(arquivos, 1):
        for chunk in pd.read_csv(arq, sep=";", header=None, dtype=str,
                                 encoding="latin1", chunksize=CHUNK,
                                 usecols=[0, 5], names=["cnpj_basico", "porte"],
                                 on_bad_lines="skip", low_memory=False):
            cb = pd.to_numeric(chunk["cnpj_basico"], errors="coerce")
            ok = cb.notna()
            if not ok.any():
                continue
            vals = cb[ok].astype(np.uint32).values
            idx = np.clip(np.searchsorted(cnpjs_ativos, vals), 0, len(cnpjs_ativos) - 1)
            pertence = cnpjs_ativos[idx] == vals
            if not pertence.any():
                continue
            sel = chunk.loc[ok].loc[pertence]
            # fillna antes do zfill: porte nulo viraria NaN (float) e quebraria
            # a ordenacao por misturar tipos no array
            portes = (sel["porte"].fillna("00").astype(str)
                      .str.strip().replace("", "00").str.zfill(2).values)
            total += len(sel)
            # contagem por EMPRESA
            for p, n in pd.Series(portes).value_counts().items():
                c_empresa[str(p)] = c_empresa.get(str(p), 0) + int(n)
            # contagem por ESTABELECIMENTO (pondera pela frequencia de ativos)
            if freq_ativos is not None:
                pesos = freq_ativos[idx[pertence]]
                for p in pd.unique(portes):
                    c_estab[str(p)] = c_estab.get(str(p), 0) + int(pesos[portes == p].sum())
        print(f"      [{i}/{len(arquivos)}] acum. {total:,}")
    return {"total": total, "porte": c_estab or c_empresa,
            "porte_empresa": c_empresa,
            "unidade": "estabelecimento" if c_estab else "empresa"}


def carregar_amostra():
    print("[3/5] Carregando a amostra analisada ...")
    df = pd.read_csv(AMOSTRA, sep=";", dtype=str, encoding="utf-8-sig",
                     usecols=lambda c: c in ("uf", "porte", "cnae_divisao",
                                             "cnae_fiscal_principal"),
                     low_memory=False)
    df["uf"] = df["uf"].astype(str).str.strip().str.upper()
    df["regiao"] = df["uf"].map(REGIAO).fillna("ND")
    df["porte_nome"] = df["porte"].astype(str).str.strip().str.zfill(2).map(PORTE_NOME).fillna("ND")
    if "cnae_divisao" in df.columns:
        div = df["cnae_divisao"].astype(str).str.zfill(2)
    else:
        div = df["cnae_fiscal_principal"].astype(str).str.zfill(7).str[:2]
    df["secao"] = div.map(_secao_cnae)
    print(f"      amostra: {len(df):,} registros")
    return df


def erro_amostral(n, N=None, p=P_CONSERV, z=Z):
    """Margem de erro para proporcao, com correcao para populacao finita."""
    if n <= 0:
        return np.nan
    e = z * np.sqrt(p * (1 - p) / n)
    if N and N > n:
        e *= np.sqrt((N - n) / (N - 1))
    return e


def n_minimo(N, erro=ERRO_ALVO, p=P_CONSERV, z=Z):
    """Tamanho minimo de amostra para proporcao com populacao finita."""
    n0 = (z ** 2) * p * (1 - p) / (erro ** 2)
    return n0 / (1 + (n0 - 1) / N) if N else n0


def comparar(pop_cont, amo_series, rotulo, ordem=None):
    """Tabela populacao x amostra + qui-quadrado de aderencia."""
    pop = pd.Series(pop_cont, dtype=float)
    amo = amo_series.value_counts().astype(float)
    idx = ordem if ordem else sorted(set(pop.index) | set(amo.index))
    pop = pop.reindex(idx).fillna(0); amo = amo.reindex(idx).fillna(0)
    pop_pct = pop / pop.sum(); amo_pct = amo / amo.sum()
    esperado = pop_pct * amo.sum()
    mask = esperado >= 5
    qui, pval = (stats.chisquare(amo[mask], esperado[mask])
                 if mask.sum() > 1 else (np.nan, np.nan))
    # V de Cramer: tamanho de efeito, insensivel ao n.
    # Com n na casa dos milhoes, o qui-quadrado rejeita H0 para desvios
    # irrelevantes na pratica; o V mede a MAGNITUDE da diferenca.
    gl = max(int(mask.sum()) - 1, 1)
    cramer = np.sqrt(qui / (amo.sum() * gl)) if not np.isnan(qui) else np.nan
    t = pd.DataFrame({
        "populacao": pop.astype("int64"),
        "pop_%": pop_pct,
        "amostra": amo.astype("int64"),
        "amo_%": amo_pct,
        "dif_pp": (amo_pct - pop_pct) * 100,
        "fracao_%": np.where(pop > 0, amo / pop.replace(0, np.nan) * 100, np.nan),
        "erro_amostral_pp": [erro_amostral(n, N) * 100
                             for n, N in zip(amo.values, pop.values)],
    })
    t.index.name = rotulo
    return t, qui, pval, cramer


def main():
    rapido = "--rapido" in sys.argv
    print("=" * 68)
    print(" FUNDAMENTACAO ESTATISTICA DO DESENHO AMOSTRAL")
    print("=" * 68)

    amo = carregar_amostra() if rapido else None
    pop = None if rapido else varrer_populacao()
    porte_pop = None if rapido else varrer_porte(
        pop.get("cnpjs_ativos") if pop else None,
        pop.get("freq_ativos") if pop else None)
    if amo is None:
        amo = carregar_amostra()

    L = ["=" * 68, " DESENHO AMOSTRAL — FUNDAMENTACAO ESTATISTICA", "=" * 68, ""]
    n = len(amo)

    if pop:
        N = pop["total"]
        frac = n / N
        L += ["1) POPULACAO E FRACAO AMOSTRAL",
              f"   Populacao (estabelecimentos ATIVOS na RFB): {N:,}",
              f"   Amostra analisada                        : {n:,}",
              f"   Fracao amostral                          : {frac:.4%}",
              f"   Erro amostral GLOBAL (95%, p=0,5)        : +/- {erro_amostral(n, N)*100:.4f} p.p.",
              ""]
        nmin = n_minimo(N, ERRO_ALVO)
        L += ["2) DIMENSIONAMENTO",
              f"   n minimo p/ erro de {ERRO_ALVO*100:.0f} p.p. (global): {nmin:,.0f}",
              f"   n adotado                              : {n:,}",
              f"   razao n_adotado / n_minimo             : {n/nmin:,.1f}x",
              "   Observacao: o n NAO foi dimensionado para a media global, e sim",
              "   para garantir precisao nos ESTRATOS de analise (regiao x porte x",
              "   setor). Ver secao 4.", ""]

        # regiao
        c_reg = {}
        for uf, q in pop["uf"].items():
            c_reg[REGIAO.get(uf, "ND")] = c_reg.get(REGIAO.get(uf, "ND"), 0) + q
        t_reg, q_reg, p_reg, v_reg = comparar(c_reg, amo["regiao"], "regiao", ORD_REG)
        L += ["3) ADERENCIA POR REGIAO (proporcionalidade aos grupos)",
              t_reg.round(4).to_string(), ""]
        if not np.isnan(q_reg):
            L += [f"   Qui-quadrado de aderencia: X2={q_reg:,.1f}  p-valor={p_reg:.4g}",
                  f"   V de Cramer (tamanho de efeito): {v_reg:.5f}",
                  f"   Maior desvio absoluto    : {t_reg['dif_pp'].abs().max():.4f} p.p.", ""]

        t_sec, q_sec, p_sec, v_sec = comparar(pop["secao"], amo["secao"], "secao_cnae")
        L += ["4) ADERENCIA POR SECAO CNAE (menor estrato de analise)",
              t_sec.round(4).to_string(), "",
              f"   Menor estrato amostral : {int(t_sec['amostra'].min()):,} registros",
              f"   Pior erro amostral     : +/- {t_sec['erro_amostral_pp'].max():.3f} p.p.",
              f"   Maior desvio            : {t_sec['dif_pp'].abs().max():.4f} p.p.", ""]
        if not np.isnan(q_sec):
            L += [f"   Qui-quadrado: X2={q_sec:,.1f}  p-valor={p_sec:.4g}",
                  f"   V de Cramer: {v_sec:.5f}", ""]

        pd.concat({"regiao": t_reg, "secao": t_sec}).to_csv(ADER_CSV, sep=";",
                                                            encoding="utf-8-sig")
        pd.DataFrame({"indicador": ["populacao_ativos", "amostra", "fracao"],
                      "valor": [N, n, frac]}).to_csv(POP_CSV, sep=";", index=False,
                                                     encoding="utf-8-sig")

    if porte_pop:
        c_porte = {PORTE_NOME.get(k, k): v for k, v in porte_pop["porte"].items()}
        t_por, q_por, p_por, v_por = comparar(c_porte, amo["porte_nome"], "porte")
        L += [f"5) ADERENCIA POR PORTE (unidade: {porte_pop.get('unidade','?')})",
              "   Nota: a unidade amostral e o ESTABELECIMENTO; a populacao de porte",
              "   foi ponderada pelo numero de estabelecimentos ativos de cada empresa,",
              "   para que as unidades sejam comparaveis.",
              "", t_por.round(4).to_string(), ""]
        if not np.isnan(q_por):
            L += [f"   Qui-quadrado: X2={q_por:,.1f}  p-valor={p_por:.4g}",
                  f"   V de Cramer: {v_por:.5f}",
                  f"   Maior desvio absoluto: {t_por['dif_pp'].abs().max():.4f} p.p.", ""]
        if porte_pop.get("porte_empresa"):
            ce = {PORTE_NOME.get(k, k): v for k, v in porte_pop["porte_empresa"].items()}
            tot_e = sum(ce.values())
            L += ["   Referencia — mesma populacao contada por EMPRESA (nao comparavel",
                  "   diretamente a amostra, registrada apenas para transparencia):"]
            for k in sorted(ce):
                L.append(f"     {k:<16}: {ce[k]:>12,}  ({ce[k]/tot_e:6.2%})")
            L.append("")
        if pop and pop.get("n_empresas"):
            L += [f"   Estabelecimentos ativos : {pop['total']:,}",
                  f"   Empresas distintas      : {pop['n_empresas']:,}",
                  f"   Razao estab/empresa     : {pop['total']/pop['n_empresas']:.4f}", ""]

    # precisao nos estratos cruzados da amostra (sempre)
    # exclui estratos residuais (UF/porte nao identificados), que nao sao
    # dominio de analise do trabalho
    val = amo[(amo["regiao"] != "ND") & (~amo["porte_nome"].isin(["ND", "Nao informado"]))]
    cross = val.groupby(["regiao", "porte_nome"]).size().sort_values()
    n_res = len(amo) - len(val)
    erro_menor = erro_amostral(cross.min()) * 100
    L += ["6) PRECISAO NOS ESTRATOS DE ANALISE (regiao x porte)",
          f"   registros em estratos validos : {len(val):,}",
          f"   registros residuais excluidos : {n_res:,} (UF ou porte nao identificado)",
          f"   estratos: {len(cross)} | menor: {cross.min():,} | mediana: {int(cross.median()):,}",
          f"   erro amostral no MENOR estrato: +/- {erro_menor:.3f} p.p.",
          "",
          "   Tres menores estratos:"]
    for (reg, por), q in cross.head(3).items():
        L.append(f"     {reg} x {por:<8}: {q:>9,}  (+/- {erro_amostral(q)*100:.3f} p.p.)")
    atende = erro_menor <= ERRO_ALVO * 100
    L += ["",
          f"   Criterio: erro <= {ERRO_ALVO*100:.0f} p.p. em todo estrato de analise -> "
          f"{'ATENDIDO' if atende else 'NAO ATENDIDO'}",
          "",
          "   Leitura: o n foi dimensionado para sustentar inferencia nos ESTRATOS",
          "   (regiao x porte x setor), e nao apenas na media global — e o que",
          "   justifica estatisticamente a ordem de grandeza adotada.", "=" * 68]

    txt = "\n".join(L)
    print("\n" + txt)
    with open(REL, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\nRelatorio: {REL}")


if __name__ == "__main__":
    main()
