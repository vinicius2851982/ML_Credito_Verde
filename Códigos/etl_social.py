"""
etl_social.py  (v2 — leitura local, agregacao por CNAE)
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Constroi o pilar Social (S) do Score Socioambiental.

DESCOBERTA METODOLOGICA (importante):
  Nenhuma fonte publica do MTE traz CNPJ — RAIS (ESTAB e VINCULOS) e
  NOVO CAGED sao anonimizados no nivel da firma por sigilo legal.
  Portanto, o pilar Social e construido como PERFIL SETORIAL por CNAE
  e atribuido a cada empresa pela sua CNAE 2.0 (subclasse/divisao).
  Esta abordagem usa dado REAL (nao proxy estimado) e e defensavel:
  caracteriza as condicoes do mercado de trabalho formal do setor.

FONTES (lidas de arquivos LOCAIS ja baixados via Windows Explorer):

  FONTE 1 — RAIS Vinculos (estoque anual; qualidade do emprego)
    Pasta: Dados/rais_estab/{ANO}/Legado/RAIS_VINC_PUB_*.7z
    Por CNAE: salario_medio_sm, tempo_emprego_medio, pct_vinculo_ativo

  FONTE 2 — NOVO CAGED Movimentacao (fluxo; dinamica do emprego)
    Pasta: Dados/caged_novo/{ANO}/{ANOMES}/CAGEDMOV{ANOMES}.7z
    Por CNAE: rotatividade, saldo_setor_taxa  (janela 36 meses)

  FONTE 3 — Proxy CNAE (tabela interna, contexto/fallback)
    intensidade_emprego_cnae (0.0 a 1.0)

DEPENDENCIAS: pip install py7zr

OUTPUT: Dados/stg_social.csv  (uma linha por CNPJ da base analitica)
  CNPJ_LIMPO | salario_medio_sm | tempo_emprego_medio | pct_vinculo_ativo |
  rotatividade | saldo_setor_taxa | intensidade_emprego_cnae |
  cnae_origem_social   (qualidade do match: subclasse/divisao/mediana)
"""

import os
import gc
import re
import glob
import shutil
import tempfile
import unicodedata
import warnings

import numpy as np
import pandas as pd

try:
    import py7zr
except ImportError:
    raise SystemExit("Biblioteca py7zr nao instalada. Execute: pip install py7zr")

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS       = os.path.join(BASE_DIR, "Dados")
PASTA_RAIS  = os.path.join(DADOS, "rais_estab")     # contem {ANO}/Legado/RAIS_VINC_PUB_*.7z
PASTA_CAGED = os.path.join(DADOS, "caged_novo")     # contem {ANO}/{ANOMES}/CAGEDMOV*.7z
OUTPUT_PATH = os.path.join(DADOS, "stg_social.csv")
CKPT_RAIS   = os.path.join(DADOS, "ckpt_social_rais_cnae.csv")
CKPT_CAGED  = os.path.join(DADOS, "ckpt_social_caged_cnae.csv")
CKPT_EQUID  = os.path.join(DADOS, "ckpt_social_equidade_cnae.csv")
TMP_DIR     = os.path.join(tempfile.gettempdir(), "etl_social_tmp")

os.makedirs(TMP_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
CHUNKSIZE        = 1_000_000     # RAIS/CAGED sao grandes — chunks grandes
N_MESES_CAGED    = 36            # janela comportamental (Thomas, Edelman & Crook, 2002)
INTENSIDADE_DEFAULT = 0.40

# Equidade (criterios S2/S4) — extraidas da RAIS Vinculos por CNAE subclasse.
# Ancoradas na Lei 14.611/2023 (igualdade salarial) e na definicao social do
# BNDES (financiamento de MPME liderada por mulheres).
EQUIDADE_COLS = ["gap_genero_ajustado_cnae", "gap_raca_ajustado_cnae",
                 "pct_chefia_feminina_cnae", "dispersao_salarial_cnae"]
TEMPO_BINS    = [-1, 12, 36, 60, 120, 1e12]   # faixas de tempo de emprego (meses)
CBO_CHEFIA_GG = "1"   # CBO grande grupo 1 (dirigentes/gerentes) = chefia
RACA_BRANCA   = {2}            # codigo RAIS: 2 = Branca
RACA_NEGRA    = {4, 8}         # 4 = Preta, 8 = Parda (foco do gap racial)

# Proxy CNAE — intensidade de mao de obra por divisao (2 digitos). Fallback/contexto.
INTENSIDADE_CNAE = {
    "47": 0.92, "56": 0.90, "85": 0.88, "96": 0.86, "86": 0.85, "88": 0.83,
    "81": 0.82, "78": 0.80, "43": 0.79, "93": 0.78, "79": 0.76, "55": 0.75,
    "46": 0.72, "49": 0.68, "74": 0.65, "82": 0.63, "77": 0.62, "45": 0.60,
    "72": 0.58, "75": 0.57, "41": 0.56, "58": 0.55, "62": 0.52, "63": 0.50,
    "10": 0.48, "14": 0.47, "31": 0.46, "13": 0.45, "15": 0.44, "16": 0.43,
    "32": 0.42, "25": 0.41, "11": 0.40, "27": 0.39, "28": 0.38, "01": 0.37,
    "22": 0.36, "24": 0.35, "26": 0.33, "29": 0.32, "23": 0.31, "33": 0.30,
    "42": 0.28, "02": 0.27, "50": 0.25, "51": 0.24, "52": 0.23, "71": 0.22,
    "08": 0.22, "17": 0.21, "18": 0.21, "20": 0.20, "69": 0.18, "64": 0.17,
    "65": 0.17, "68": 0.16, "70": 0.16, "66": 0.15, "61": 0.14, "35": 0.13,
    "36": 0.13, "06": 0.12, "19": 0.12, "07": 0.11, "05": 0.10, "03": 0.10,
}


# ------------------------------------------------------------------
# Utilitarios de deteccao de colunas (robusto a acentos/encoding)
# ------------------------------------------------------------------
def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").upper().strip()


def detectar_coluna(cols, tokens, evitar=None):
    """
    Retorna a coluna cujo nome (sem acento, maiusculo) contem TODOS os
    tokens e NENHUM dos tokens em 'evitar'. Prefere o nome mais curto
    (mais especifico) entre os candidatos.
    """
    tokens  = [_ascii(t) for t in tokens]
    evitar  = [_ascii(e) for e in (evitar or [])]
    cands = []
    for c in cols:
        ca = _ascii(c)
        if all(t in ca for t in tokens) and not any(e in ca for e in evitar):
            cands.append(c)
    if not cands:
        return None
    return min(cands, key=lambda x: len(x))   # mais curto = mais especifico


def extrair_7z(arquivo: str, destino: str) -> list[str]:
    os.makedirs(destino, exist_ok=True)
    with py7zr.SevenZipFile(arquivo, mode="r") as z:
        nomes = z.getnames()
        z.extractall(path=destino)
    return [os.path.join(destino, n) for n in nomes]


def ler_csv_robusto_cabecalho(txt_path):
    """Detecta separador e encoding lendo so o cabecalho. Retorna (cols, sep, enc).
    Inclui ',' pois o RAIS 2024+ (.COMT) usa virgula com campos entre aspas."""
    for enc in ["latin1", "utf-8-sig", "utf-8"]:
        for sep in [";", ",", "|", "\t"]:
            try:
                df = pd.read_csv(txt_path, sep=sep, encoding=enc, nrows=3,
                                 dtype=str, on_bad_lines="skip")
                if len(df.columns) > 5:
                    return list(df.columns), sep, enc
            except Exception:
                continue
    return None, ";", "latin1"


def arquivos_dados(extraidos: list[str]) -> list[str]:
    """Retorna os arquivos de dados extraidos (qualquer extensao, exceto .7z).
    O MTE usa .txt (2023) e .COMT (2024+) — nao filtra por extensao fixa."""
    return [t for t in extraidos
            if os.path.isfile(t) and not t.upper().endswith((".7Z", ".PDF"))]


def _num(serie: pd.Series, sep: str) -> pd.Series:
    """
    Converte string numerica para float ciente do formato decimal.
      - sep == ','  -> arquivo .COMT (2024+): decimal e PONTO. Remove virgulas (milhar).
      - sep != ','  -> arquivo .txt (2023) e CAGED: decimal e VIRGULA. Remove pontos (milhar).
    """
    s = serie.astype(str).str.strip()
    if sep == ",":   # decimal = ponto
        s = s.str.replace(",", "", regex=False)
    else:            # decimal = virgula
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


# ==================================================================
# FASE 1 — RAIS Vinculos (estoque por CNAE)
# ==================================================================

def _descobrir_ano_rais() -> tuple[int, list[str]]:
    """
    Encontra o ano mais recente sob PASTA_RAIS que contenha arquivos
    RAIS_VINC_PUB_*.7z. Retorna (ano, lista_de_arquivos_7z).
    """
    todos = glob.glob(os.path.join(PASTA_RAIS, "**", "RAIS_VINC_PUB_*.7z"), recursive=True)
    # Ignora o arquivo "NI" (Nao Informado, residual) e arquivos vazios
    todos = [f for f in todos
             if "_NI" not in os.path.basename(f).upper()
             and os.path.getsize(f) > 1_000_000]
    if not todos:
        return 0, []

    # Extrai o ano do caminho (primeira ocorrencia de 20XX entre 2018-2026)
    def ano_do_path(p):
        m = re.findall(r"(20[12]\d)", p)
        return max((int(x) for x in m), default=0)

    por_ano = {}
    for f in todos:
        a = ano_do_path(f)
        por_ano.setdefault(a, []).append(f)

    ano = max(por_ano)
    # Dedup por nome de regiao (mantem 1 por regiao)
    arquivos = {}
    for f in por_ano[ano]:
        arquivos[os.path.basename(f).upper()] = f
    return ano, sorted(arquivos.values())


def processar_rais_vinc(txt_path: str) -> pd.DataFrame | None:
    """Le um arquivo RAIS_VINC em chunks e agrega por CNAE subclasse."""
    cols, sep, enc = ler_csv_robusto_cabecalho(txt_path)
    if cols is None:
        print("       [WARN] cabecalho ilegivel; pulando.")
        return None

    col_cnae  = detectar_coluna(cols, ["CNAE", "SUBCLASSE"]) or detectar_coluna(cols, ["CNAE", "CLASSE"])
    # "Vl Rem Media (SM)" (2024+) ou "Vl Remun Media (SM)" (2023) — token "REM" cobre ambos.
    col_sal   = detectar_coluna(cols, ["VL", "REM", "MEDIA", "SM"], evitar=["FAIXA", "NOM"])
    col_tempo = detectar_coluna(cols, ["TEMPO", "EMPREGO"], evitar=["FAIXA"])
    col_ativo = detectar_coluna(cols, ["VINCULO", "ATIVO"], evitar=["ABANDON"])

    if not col_cnae:
        print("       [WARN] coluna CNAE nao encontrada; pulando.")
        return None

    usecols = [c for c in [col_cnae, col_sal, col_tempo, col_ativo] if c]
    partes = []
    reader = pd.read_csv(txt_path, sep=sep, encoding=enc, dtype=str,
                         on_bad_lines="skip", chunksize=CHUNKSIZE, usecols=usecols)

    for chunk in reader:
        ren = {col_cnae: "CNAE"}
        if col_sal:   ren[col_sal]   = "SAL"
        if col_tempo: ren[col_tempo] = "TEMPO"
        if col_ativo: ren[col_ativo] = "ATIVO"
        chunk = chunk.rename(columns=ren)

        chunk["CNAE"] = chunk["CNAE"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(7)
        chunk = chunk[chunk["CNAE"].str.len() == 7]
        # Decimal: arquivo com sep ',' (.COMT 2024+) usa ponto decimal;
        #          arquivo com sep ';' (.txt 2023) usa virgula decimal.
        if "SAL" in chunk:   chunk["SAL"]   = _num(chunk["SAL"], sep)
        if "TEMPO" in chunk: chunk["TEMPO"] = _num(chunk["TEMPO"], sep)
        if "ATIVO" in chunk:
            chunk["ATIVO"] = pd.to_numeric(chunk["ATIVO"], errors="coerce").fillna(0)
            chunk["ATIVO"] = (chunk["ATIVO"] > 0).astype(int)

        g = chunk.groupby("CNAE").agg(
            sal_sum=("SAL", "sum")     if "SAL" in chunk   else ("CNAE", "size"),
            sal_n=("SAL", "count")     if "SAL" in chunk   else ("CNAE", "size"),
            tempo_sum=("TEMPO", "sum") if "TEMPO" in chunk else ("CNAE", "size"),
            tempo_n=("TEMPO", "count") if "TEMPO" in chunk else ("CNAE", "size"),
            ativo_sum=("ATIVO", "sum") if "ATIVO" in chunk else ("CNAE", "size"),
            n_total=("CNAE", "size"),
        )
        partes.append(g)
        del chunk; gc.collect()

    if not partes:
        return None
    return pd.concat(partes).groupby(level=0).sum()


def fase1_rais() -> pd.DataFrame:
    """Agrega RAIS Vinculos por CNAE: salario_medio_sm, tempo_emprego_medio, pct_vinculo_ativo."""
    if os.path.exists(CKPT_RAIS):
        print(f"[SKIP] Checkpoint RAIS-CNAE existe: {CKPT_RAIS}")
        return pd.read_csv(CKPT_RAIS, sep=";", dtype={"CNAE": str}, encoding="utf-8-sig")

    ano, arquivos = _descobrir_ano_rais()
    if not arquivos:
        print("[WARN] Nenhum RAIS_VINC_PUB_*.7z encontrado em", PASTA_RAIS)
        return pd.DataFrame(columns=["CNAE", "salario_medio_sm",
                                     "tempo_emprego_medio", "pct_vinculo_ativo"])

    print(f"[RAIS] Ano {ano} | {len(arquivos)} arquivos regionais.")
    acumulado = None

    for i, arq in enumerate(arquivos, 1):
        nome = os.path.basename(arq)
        print(f"  ({i}/{len(arquivos)}) {nome} ({os.path.getsize(arq)/1e6:.0f} MB) ...", flush=True)
        try:
            txts = arquivos_dados(extrair_7z(arq, TMP_DIR))
        except Exception as e:
            print(f"       [ERR] extracao: {e}")
            continue

        for txt in txts:
            g = processar_rais_vinc(txt)
            if g is not None:
                acumulado = g if acumulado is None else acumulado.add(g, fill_value=0)
            try: os.remove(txt)
            except Exception: pass
        gc.collect()

    if acumulado is None or acumulado.empty:
        print("[WARN] RAIS sem dados processados.")
        return pd.DataFrame(columns=["CNAE", "salario_medio_sm",
                                     "tempo_emprego_medio", "pct_vinculo_ativo"])

    df = acumulado.reset_index()
    df["salario_medio_sm"]    = (df["sal_sum"]   / df["sal_n"].replace(0, np.nan)).round(2)
    df["tempo_emprego_medio"] = (df["tempo_sum"] / df["tempo_n"].replace(0, np.nan)).round(1)
    df["pct_vinculo_ativo"]   = (df["ativo_sum"] / df["n_total"].replace(0, np.nan)).round(3)
    df = df[["CNAE", "salario_medio_sm", "tempo_emprego_medio", "pct_vinculo_ativo"]]

    print(f"[RAIS] {len(df):,} CNAEs (subclasses) com perfil de emprego.")
    df.to_csv(CKPT_RAIS, sep=";", index=False, encoding="utf-8-sig")
    return df


# ==================================================================
# FASE 1B — Equidade (S2/S4) a partir da RAIS Vinculos
# ==================================================================

def processar_rais_equidade(txt_path: str) -> dict | None:
    """
    Le um RAIS_VINC e devolve 4 agregados por CNAE subclasse (one-pass):
      G = (CNAE, SEXO, ESCOL, TBAND) -> sal_sum, n      [gap genero ajustado]
      R = (CNAE, RACA_GRP, ESCOL, TBAND) -> sal_sum, n  [gap racial ajustado]
      C = (CNAE, SEXO) -> chefia_n, n                   [% chefia feminina]
      D = (CNAE) -> sal_sum, sal_sumsq, n               [dispersao salarial (CV)]
    'ajustado' = comparacao DENTRO de estratos de escolaridade x faixa de tempo
    (controla experiencia/qualificacao; gap residual ~ 0 = padrao desejado).
    """
    cols, sep, enc = ler_csv_robusto_cabecalho(txt_path)
    if cols is None:
        return None
    col_cnae  = detectar_coluna(cols, ["CNAE", "SUBCLASSE"]) or detectar_coluna(cols, ["CNAE", "CLASSE"])
    col_sal   = detectar_coluna(cols, ["VL", "REM", "MEDIA", "SM"], evitar=["FAIXA", "NOM"])
    col_tempo = detectar_coluna(cols, ["TEMPO", "EMPREGO"], evitar=["FAIXA"])
    col_sexo  = detectar_coluna(cols, ["SEXO"])
    col_raca  = detectar_coluna(cols, ["RACA", "COR"])
    col_escol = detectar_coluna(cols, ["ESCOLARIDADE"])
    col_cbo   = detectar_coluna(cols, ["CBO"], evitar=["FAMILIA"])

    if not (col_cnae and col_sal and col_sexo):
        print("       [WARN] equidade: faltam CNAE/SAL/SEXO; pulando arquivo.")
        return None

    usecols = [c for c in [col_cnae, col_sal, col_tempo, col_sexo, col_raca,
                           col_escol, col_cbo] if c]
    G_parts, R_parts, C_parts, D_parts = [], [], [], []
    reader = pd.read_csv(txt_path, sep=sep, encoding=enc, dtype=str,
                         on_bad_lines="skip", chunksize=CHUNKSIZE, usecols=usecols)
    for chunk in reader:
        ren = {col_cnae: "CNAE", col_sal: "SAL", col_sexo: "SEXO"}
        if col_tempo: ren[col_tempo] = "TEMPO"
        if col_raca:  ren[col_raca]  = "RACA"
        if col_escol: ren[col_escol] = "ESCOL"
        if col_cbo:   ren[col_cbo]   = "CBO"
        chunk = chunk.rename(columns=ren)

        chunk["CNAE"] = chunk["CNAE"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(7)
        chunk = chunk[chunk["CNAE"].str.len() == 7]
        chunk["SAL"]  = _num(chunk["SAL"], sep)
        chunk = chunk[chunk["SAL"] > 0]
        chunk["SEXO"] = pd.to_numeric(chunk["SEXO"], errors="coerce")
        if "TEMPO" in chunk:
            chunk["TEMPO"] = _num(chunk["TEMPO"], sep)
            chunk["TBAND"] = pd.cut(chunk["TEMPO"], bins=TEMPO_BINS, labels=False).astype("Int64")
        else:
            chunk["TBAND"] = 0
        chunk["ESCOL"] = (pd.to_numeric(chunk["ESCOL"], errors="coerce").astype("Int64")
                          if "ESCOL" in chunk else 0)
        if chunk.empty:
            continue

        # D — dispersao (CV) por CNAE
        chunk["SAL2"] = chunk["SAL"] ** 2
        D_parts.append(chunk.groupby("CNAE").agg(
            sal_sum=("SAL", "sum"), sal_sq=("SAL2", "sum"), n=("SAL", "size")))

        # G — gap genero ajustado (estratos escolaridade x tempo)
        gk = chunk.dropna(subset=["SEXO"])
        gk = gk[gk["SEXO"].isin([1, 2])]
        gk["SEXO"] = gk["SEXO"].astype(int)
        if not gk.empty:
            G_parts.append(gk.groupby(["CNAE", "SEXO", "ESCOL", "TBAND"]).agg(
                sal_sum=("SAL", "sum"), n=("SAL", "size")))

        # C — chefia (CBO grande grupo 1) por sexo
        if "CBO" in chunk:
            cc = gk.copy()
            gg = cc["CBO"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6).str[0]
            cc["CHEFIA"] = (gg == CBO_CHEFIA_GG).astype(int)
            C_parts.append(cc.groupby(["CNAE", "SEXO"]).agg(
                chefia_n=("CHEFIA", "sum"), n=("CHEFIA", "size")))

        # R — gap racial ajustado (branca x negra), estratos escolaridade x tempo
        if "RACA" in chunk:
            rc = chunk.copy()
            rc["RC"] = pd.to_numeric(rc["RACA"], errors="coerce")
            rc["RGRP"] = np.where(rc["RC"].isin(list(RACA_BRANCA)), "B",
                          np.where(rc["RC"].isin(list(RACA_NEGRA)), "N", None))
            rc = rc[rc["RGRP"].notna()]
            if not rc.empty:
                R_parts.append(rc.groupby(["CNAE", "RGRP", "ESCOL", "TBAND"]).agg(
                    sal_sum=("SAL", "sum"), n=("SAL", "size")))
        del chunk; gc.collect()

    def _comb(parts):
        if not parts:
            return None
        return pd.concat(parts).groupby(level=list(range(parts[0].index.nlevels))).sum()

    return {"G": _comb(G_parts), "R": _comb(R_parts),
            "C": _comb(C_parts), "D": _comb(D_parts)}


def _gap_ajustado(frame: pd.DataFrame, grp_col: str, alto: str, baixo: str) -> pd.Series:
    """Gap ajustado por CNAE: media ponderada dos gaps DENTRO de cada estrato
    (escolaridade x faixa de tempo). gap = (media_alto - media_baixo)/media_alto.
    Positivo => grupo 'baixo' (mulheres/negros) ganha menos. 0 => paridade."""
    df = frame.reset_index()
    df["media"] = df["sal_sum"] / df["n"].replace(0, np.nan)
    idx = ["CNAE", "ESCOL", "TBAND"]
    a = df[df[grp_col] == alto].set_index(idx)[["media", "n"]].rename(
        columns={"media": "m_a", "n": "n_a"})
    b = df[df[grp_col] == baixo].set_index(idx)[["media", "n"]].rename(
        columns={"media": "m_b", "n": "n_b"})
    j = a.join(b, how="inner").reset_index()
    if j.empty:
        return pd.Series(dtype=float)
    j["gap"] = (j["m_a"] - j["m_b"]) / j["m_a"]
    j["w"]   = j["n_a"] + j["n_b"]
    g = j.groupby("CNAE").apply(
        lambda x: np.average(x["gap"], weights=x["w"]) if x["w"].sum() > 0 else np.nan)
    return g.round(4)


def fase1b_equidade() -> pd.DataFrame:
    """Calcula, por CNAE subclasse: gap genero/raca ajustado, % chefia feminina, dispersao."""
    if os.path.exists(CKPT_EQUID):
        print(f"[SKIP] Checkpoint equidade existe: {CKPT_EQUID}")
        return pd.read_csv(CKPT_EQUID, sep=";", dtype={"CNAE": str}, encoding="utf-8-sig")

    ano, arquivos = _descobrir_ano_rais()
    if not arquivos:
        print("[WARN] equidade: nenhum RAIS_VINC encontrado.")
        return pd.DataFrame(columns=["CNAE"] + EQUIDADE_COLS)

    print(f"[EQUIDADE] Ano {ano} | {len(arquivos)} arquivos (sexo/raca/CBO/escolaridade).")
    accG = accR = accC = accD = None
    def _merge(acc, novo):
        if novo is None: return acc
        if acc is None:  return novo
        n = acc.index.nlevels
        return pd.concat([acc, novo]).groupby(level=list(range(n))).sum()

    for i, arq in enumerate(arquivos, 1):
        print(f"  ({i}/{len(arquivos)}) {os.path.basename(arq)} ...", flush=True)
        try:
            txts = arquivos_dados(extrair_7z(arq, TMP_DIR))
        except Exception as e:
            print(f"       [ERR] extracao: {e}"); continue
        for txt in txts:
            d = processar_rais_equidade(txt)
            if d:
                accG = _merge(accG, d["G"]); accR = _merge(accR, d["R"])
                accC = _merge(accC, d["C"]); accD = _merge(accD, d["D"])
            try: os.remove(txt)
            except Exception: pass
        gc.collect()

    if accD is None:
        print("[WARN] equidade sem dados.")
        return pd.DataFrame(columns=["CNAE"] + EQUIDADE_COLS)

    # Monta tabela por CNAE
    out = pd.DataFrame({"CNAE": accD.reset_index()["CNAE"]}).drop_duplicates().set_index("CNAE")

    # Dispersao (coeficiente de variacao)
    d = accD.reset_index()
    mean = d["sal_sum"] / d["n"].replace(0, np.nan)
    var  = (d["sal_sq"] / d["n"].replace(0, np.nan)) - mean ** 2
    d["dispersao_salarial_cnae"] = (np.sqrt(var.clip(lower=0)) / mean).round(4)
    out = out.join(d.set_index("CNAE")["dispersao_salarial_cnae"])

    # Gaps ajustados
    if accG is not None:
        out = out.join(_gap_ajustado(accG, "SEXO", 1, 2).rename("gap_genero_ajustado_cnae"))
    if accR is not None:
        out = out.join(_gap_ajustado(accR, "RGRP", "B", "N").rename("gap_raca_ajustado_cnae"))

    # % chefia feminina
    if accC is not None:
        c = accC.reset_index()
        piv = c.pivot_table(index="CNAE", columns="SEXO", values="chefia_n",
                            aggfunc="sum", fill_value=0)
        masc = piv.get(1, 0); fem = piv.get(2, 0)
        tot = (masc + fem).replace(0, np.nan)
        out["pct_chefia_feminina_cnae"] = (fem / tot).round(4)

    out = out.reset_index()
    for c in EQUIDADE_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = out[["CNAE"] + EQUIDADE_COLS]
    print(f"[EQUIDADE] {len(out):,} CNAEs com indicadores de equidade.")
    out.to_csv(CKPT_EQUID, sep=";", index=False, encoding="utf-8-sig")
    return out


# ==================================================================
# FASE 2 — NOVO CAGED (fluxo por CNAE)
# ==================================================================

def _descobrir_meses_caged() -> list[str]:
    """Lista os CAGEDMOV*.7z locais, dedup por mes, retorna os N_MESES mais recentes."""
    arqs = glob.glob(os.path.join(PASTA_CAGED, "**", "CAGEDMOV*.7z"), recursive=True)
    por_mes = {}
    for f in arqs:
        m = re.search(r"CAGEDMOV(\d{6})", os.path.basename(f).upper())
        if m:
            por_mes[m.group(1)] = f      # ultima ocorrencia vence (ok)
    meses = sorted(por_mes)
    recentes = meses[-N_MESES_CAGED:] if len(meses) > N_MESES_CAGED else meses
    return [por_mes[m] for m in recentes]


def processar_caged_mes(txt_path: str) -> pd.DataFrame | None:
    """Agrega por CNAE: admissoes e demissoes (saldomovimentacao +1/-1)."""
    cols, sep, enc = ler_csv_robusto_cabecalho(txt_path)
    if cols is None:
        return None
    col_cnae  = detectar_coluna(cols, ["SUBCLASSE"]) or detectar_coluna(cols, ["CNAE"])
    col_saldo = detectar_coluna(cols, ["SALDO", "MOVIMENTACAO"]) or detectar_coluna(cols, ["SALDOMOV"])
    if not col_cnae or not col_saldo:
        return None

    partes = []
    reader = pd.read_csv(txt_path, sep=sep, encoding=enc, dtype=str,
                         on_bad_lines="skip", chunksize=CHUNKSIZE,
                         usecols=[col_cnae, col_saldo])
    for chunk in reader:
        chunk = chunk.rename(columns={col_cnae: "CNAE", col_saldo: "SALDO"})
        chunk["CNAE"] = chunk["CNAE"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(7)
        chunk = chunk[chunk["CNAE"].str.len() == 7]
        chunk["SALDO"] = pd.to_numeric(chunk["SALDO"], errors="coerce").fillna(0)
        chunk["ADM"] = (chunk["SALDO"] > 0).astype(int)
        chunk["DEM"] = (chunk["SALDO"] < 0).astype(int)
        partes.append(chunk.groupby("CNAE")[["ADM", "DEM"]].sum())
        del chunk; gc.collect()

    if not partes:
        return None
    return pd.concat(partes).groupby(level=0).sum()


def fase2_caged() -> pd.DataFrame:
    """Agrega CAGED por CNAE: rotatividade e saldo_setor_taxa (janela 36 meses)."""
    if os.path.exists(CKPT_CAGED):
        print(f"[SKIP] Checkpoint CAGED-CNAE existe: {CKPT_CAGED}")
        return pd.read_csv(CKPT_CAGED, sep=";", dtype={"CNAE": str}, encoding="utf-8-sig")

    arquivos = _descobrir_meses_caged()
    if not arquivos:
        print("[WARN] Nenhum CAGEDMOV*.7z encontrado em", PASTA_CAGED)
        return pd.DataFrame(columns=["CNAE", "rotatividade", "saldo_setor_taxa"])

    print(f"[CAGED] {len(arquivos)} meses (janela {N_MESES_CAGED}m): "
          f"{re.search(r'(\d{6})', os.path.basename(arquivos[0])).group(1)} a "
          f"{re.search(r'(\d{6})', os.path.basename(arquivos[-1])).group(1)}")
    acumulado = None

    for i, arq in enumerate(arquivos, 1):
        mes = re.search(r"CAGEDMOV(\d{6})", os.path.basename(arq).upper()).group(1)
        try:
            txts = arquivos_dados(extrair_7z(arq, TMP_DIR))
        except Exception as e:
            print(f"  [ERR] {mes}: {e}")
            continue
        for txt in txts:
            g = processar_caged_mes(txt)
            if g is not None:
                acumulado = g if acumulado is None else acumulado.add(g, fill_value=0)
            try: os.remove(txt)
            except Exception: pass
        if i % 6 == 0:
            print(f"       [{i}/{len(arquivos)}] meses ...", flush=True)
        gc.collect()

    if acumulado is None or acumulado.empty:
        print("[WARN] CAGED sem dados processados.")
        return pd.DataFrame(columns=["CNAE", "rotatividade", "saldo_setor_taxa"])

    df = acumulado.reset_index()
    total = (df["ADM"] + df["DEM"]).replace(0, np.nan)
    df["rotatividade"]     = (df["DEM"] / total).fillna(0).round(3)
    df["saldo_setor_taxa"] = ((df["ADM"] - df["DEM"]) / total).fillna(0).round(3)
    df = df[["CNAE", "rotatividade", "saldo_setor_taxa"]]

    print(f"[CAGED] {len(df):,} CNAEs com dinamica de emprego.")
    df.to_csv(CKPT_CAGED, sep=";", index=False, encoding="utf-8-sig")
    return df


# ==================================================================
# FASE 3 — Perfil por CNAE + atribuicao por CNPJ
# ==================================================================

def fase3_atribuir(rais_cnae: pd.DataFrame, caged_cnae: pd.DataFrame,
                   equid_cnae: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Constroi a tabela setorial (subclasse + divisao) e atribui a cada CNPJ
    da base analitica pela sua CNAE. Fallback: subclasse -> divisao -> mediana.
    """
    print("\n[4/4] Construindo perfil setorial e atribuindo por CNPJ ...")

    # Tabela por subclasse (7 digitos)
    sub = pd.merge(rais_cnae, caged_cnae, on="CNAE", how="outer")
    if equid_cnae is not None and not equid_cnae.empty:
        equid_cnae = equid_cnae.copy()
        equid_cnae["CNAE"] = equid_cnae["CNAE"].astype(str).str.zfill(7)
        sub = sub.merge(equid_cnae, on="CNAE", how="outer")
    sub["CNAE"] = sub["CNAE"].astype(str).str.zfill(7)

    feats = ["salario_medio_sm", "tempo_emprego_medio", "pct_vinculo_ativo",
             "rotatividade", "saldo_setor_taxa"]
    feats += [c for c in EQUIDADE_COLS if c in sub.columns]
    for f in feats:
        if f not in sub.columns:
            sub[f] = np.nan

    # Tabela por divisao (2 digitos) — media ponderada simples (mean) para fallback
    sub["divisao"] = sub["CNAE"].str[:2]
    div = sub.groupby("divisao")[feats].mean().reset_index()

    # Medianas globais (ultimo fallback)
    medianas = {f: sub[f].median() for f in feats}

    # Salva a tabela setorial (util para auditoria e para o texto do TCC)
    sub.to_csv(os.path.join(DADOS, "stg_social_por_cnae.csv"),
               sep=";", index=False, encoding="utf-8-sig")

    # Carrega CNPJs do staging RFB (existe ANTES do merge — evita dependencia circular)
    base_path = os.path.join(DADOS, "stg_rfb_consolidado.csv")
    if not os.path.exists(base_path):
        print("[WARN] stg_rfb_consolidado.csv ausente. Salvei apenas a tabela por CNAE.")
        print("       Rode os ETLs da RFB e reexecute para atribuir por CNPJ.")
        return pd.DataFrame()

    col_cnae_rfb = None
    base = pd.read_csv(base_path, sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)
    for c in ["cnae_fiscal_principal", "cnae_principal", "cnae", "CNAE"]:
        if c in base.columns:
            col_cnae_rfb = c
            break
    if "CNPJ" not in base.columns or col_cnae_rfb is None:
        print(f"[WARN] stg_rfb_consolidado.csv sem CNPJ/CNAE (cols={list(base.columns)[:8]}...).")
        return pd.DataFrame()

    base = base[["CNPJ", col_cnae_rfb]].drop_duplicates("CNPJ")
    base["CNPJ"] = base["CNPJ"].astype(str).str.zfill(14)
    base["cnae_sub"] = base[col_cnae_rfb].astype(str).str.replace(r"\D", "", regex=True).str.zfill(7)
    base["cnae_div"] = base["cnae_sub"].str[:2]

    # 1) match por subclasse (presenca do CNAE na tabela setorial)
    cnaes_sub = set(sub["CNAE"])
    out = base.merge(sub[["CNAE"] + feats], left_on="cnae_sub", right_on="CNAE", how="left")
    out["cnae_origem_social"] = ""
    out.loc[out["cnae_sub"].isin(cnaes_sub), "cnae_origem_social"] = "subclasse"

    # 2) fallback por divisao (para CNAEs sem match na subclasse)
    falta = out["cnae_origem_social"] == ""
    if falta.any():
        div_map = div.set_index("divisao")
        cnaes_div = set(div["divisao"])
        for f in feats:
            out.loc[falta, f] = out.loc[falta, "cnae_div"].map(div_map[f])
        mask_div = falta & out["cnae_div"].isin(cnaes_div)
        out.loc[mask_div, "cnae_origem_social"] = "divisao"

    # 3) fallback final por mediana global
    falta = out["cnae_origem_social"] == ""
    if falta.any():
        for f in feats:
            out.loc[falta, f] = medianas.get(f, 0)
        out.loc[falta, "cnae_origem_social"] = "mediana"

    # Proxy CNAE (contexto)
    out["intensidade_emprego_cnae"] = out["cnae_div"].map(INTENSIDADE_CNAE).fillna(INTENSIDADE_DEFAULT)

    resultado = out[["CNPJ"] + feats + ["intensidade_emprego_cnae", "cnae_origem_social"]].copy()
    resultado = resultado.rename(columns={"CNPJ": "CNPJ_LIMPO"})
    # arredonda
    for f in feats:
        resultado[f] = pd.to_numeric(resultado[f], errors="coerce").round(3)

    return resultado


# ==================================================================
# MAIN
# ==================================================================

def main() -> None:
    print("=" * 70)
    print(" ETL SOCIAL v2 - Pilar S (perfil setorial por CNAE)")
    print("=" * 70)
    print(f"  RAIS_VINC : {PASTA_RAIS}")
    print(f"  CAGED     : {PASTA_CAGED}")
    print(f"  Janela CAGED: {N_MESES_CAGED} meses")

    print("\n[1/4] RAIS Vinculos -> perfil de emprego por CNAE ...")
    rais_cnae = fase1_rais()
    gc.collect()

    print("\n[2/4] RAIS Vinculos -> equidade (genero/raca/chefia/dispersao) por CNAE ...")
    equid_cnae = fase1b_equidade()
    gc.collect()

    print("\n[3/4] NOVO CAGED -> dinamica de emprego por CNAE ...")
    caged_cnae = fase2_caged()
    gc.collect()

    resultado = fase3_atribuir(rais_cnae, caged_cnae, equid_cnae)

    # Limpa temporarios
    try:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    except Exception:
        pass

    if resultado.empty:
        print("\n[OK] Tabela setorial salva (stg_social_por_cnae.csv).")
        print("     Rode merge_bases.py e depois reexecute para atribuir por CNPJ.")
        return

    print(f"\n[OK] Exportando {OUTPUT_PATH} ...")
    resultado.to_csv(OUTPUT_PATH, sep=";", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(" RESUMO DO PILAR SOCIAL (setorial por CNAE)")
    print("=" * 70)
    print(f"  Registros (CNPJs)          : {len(resultado):,}")
    print(f"  Salario medio setorial (SM): {resultado['salario_medio_sm'].mean():.2f}")
    print(f"  Tempo emprego medio (meses): {resultado['tempo_emprego_medio'].mean():.1f}")
    print(f"  % vinculo ativo medio      : {resultado['pct_vinculo_ativo'].mean():.1%}")
    print(f"  Rotatividade media         : {resultado['rotatividade'].mean():.3f}")
    print(f"  Saldo setor (taxa) media   : {resultado['saldo_setor_taxa'].mean():+.3f}")
    for c in EQUIDADE_COLS:
        if c in resultado.columns and resultado[c].notna().any():
            print(f"  {c:27s}: {pd.to_numeric(resultado[c], errors='coerce').mean():.4f}")
    print(f"\n  Qualidade do match CNAE:")
    for origem, n in resultado["cnae_origem_social"].value_counts().items():
        print(f"    {str(origem):12s}: {n:>10,} ({n/len(resultado):.1%})")
    print(f"\n  Salvo em: {OUTPUT_PATH}")
    print("\nProximo passo: merge_bases.py (reprocessar com pilar Social)")


if __name__ == "__main__":
    main()
