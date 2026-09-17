"""
etl_ibama.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Crédito Verde e IA para PMEs
Autor: Hélio Vinícius Moreira Ribeiro
Sprint Dia 2 - ETL de Autuações IBAMA

Objetivo:
    Consolidar a base bruta de autuações do IBAMA em uma camada
    "staging" (stg_ibama_infracoes_pmes.csv), pronta para a etapa
    de feature engineering do modelo de Risco Socioambiental.

Premissas do TCC:
    - Universo de análise: Pessoas Jurídicas (CNPJs com 14 dígitos).
    - PMEs serão isoladas em etapa posterior (cruzando com base
      da Receita Federal); aqui já descartamos Pessoas Físicas.
"""

import io
import os
import re
import sys
import zipfile
import requests
import pandas as pd

# ------------------------------------------------------------------
# Configurações de caminho
# ------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "Dados", "raw")
INPUT_PATH  = os.path.join(BASE_DIR, "Dados", "ibama_autuacoes.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "Dados", "stg_ibama_infracoes_pmes.csv")

# URLs oficiais — portal https://dadosabertos.ibama.gov.br
# Dataset: fiscalizacao-auto-de-infracao
URL_AUTUACOES = (
    "https://dadosabertos.ibama.gov.br/dados/SIFISC/"
    "auto_infracao/auto_infracao/auto_infracao_csv.zip"
)
# Dataset: fiscalizacao-termo-de-embargo
URL_EMBARGOS = (
    "https://dadosabertos.ibama.gov.br/dados/SIFISC/"
    "termo_embargo/termo_embargo/termo_embargo_csv.zip"
)

INPUT_EMBARGOS  = os.path.join(BASE_DIR, "Dados", "ibama_embargos.csv")
OUTPUT_EMBARGOS = os.path.join(BASE_DIR, "Dados", "stg_ibama_embargos.csv")

# ------------------------------------------------------------------
# 0. Download automático (só executa se o arquivo ainda não existir)
# ------------------------------------------------------------------
def _baixar_zip_csv(url: str, destino: str, label: str) -> bool:
    """
    Baixa um ZIP do IBAMA em streaming, extrai o primeiro CSV e salva em 'destino'.
    Retorna True se bem-sucedido.
    """
    headers = {"User-Agent": "TCC-MBA-USPEsalq/1.0 (pesquisa academica)"}
    print(f"[...] Baixando {label}:\n      {url}")
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=120)
        if resp.status_code != 200:
            print(f"[WARN] HTTP {resp.status_code} para {label}.")
            return False

        # Lê o ZIP inteiro em memória (streaming por chunks para não travar)
        buf = io.BytesIO()
        total = 0
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            buf.write(chunk)
            total += len(chunk)
            print(f"\r[...] {total / 1e6:.1f} MB baixados...", end="", flush=True)
        print()
        buf.seek(0)

        with zipfile.ZipFile(buf) as z:
            csvs = sorted([n for n in z.namelist() if n.lower().endswith(".csv")])
            if not csvs:
                print(f"[WARN] Nenhum CSV dentro do ZIP de {label}.")
                return False

            print(f"[OK]  {len(csvs)} arquivo(s) CSV no ZIP.")

            if len(csvs) == 1:
                # ZIP simples — extrai direto
                with z.open(csvs[0]) as f_in, open(destino, "wb") as f_out:
                    f_out.write(f_in.read())
            else:
                # ZIP com múltiplos CSVs (ex.: um por ano) — concatena tudo
                print(f"[...] Concatenando {len(csvs)} arquivos...")
                partes = []
                for i, nome_csv in enumerate(csvs, 1):
                    try:
                        with z.open(nome_csv) as f:
                            chunk = pd.read_csv(
                                f, sep=";", dtype=str,
                                encoding="latin1", on_bad_lines="skip",
                            )
                        if not chunk.empty:
                            partes.append(chunk)
                        print(f"\r[...] {i}/{len(csvs)} — {nome_csv} ({len(chunk):,} linhas)",
                              end="", flush=True)
                    except Exception as e:
                        print(f"\n[WARN] Pulando {nome_csv}: {e}")
                print()
                if not partes:
                    print(f"[WARN] Todos os arquivos vieram vazios.")
                    return False
                df_total = pd.concat(partes, ignore_index=True)
                df_total.to_csv(destino, sep=";", index=False, encoding="utf-8-sig")
                print(f"[OK]  Total concatenado: {len(df_total):,} linhas")

        tamanho_mb = os.path.getsize(destino) / 1e6
        print(f"[OK]  Salvo: {destino}  ({tamanho_mb:.1f} MB)")
        return True

    except (requests.RequestException, zipfile.BadZipFile) as e:
        print(f"[WARN] Erro ao baixar {label}: {e}")
        return False


def baixar_ibama(destino_autuacoes: str, destino_embargos: str) -> bool:
    """
    Baixa autuações e embargos do IBAMA (dadosabertos.ibama.gov.br).
    Pula arquivos que já existem em disco.
    """
    os.makedirs(os.path.dirname(destino_autuacoes), exist_ok=True)

    ok_aut = True
    ok_emb = True

    if os.path.exists(destino_autuacoes):
        print(f"[OK]  Autuações já existem, pulando download: {destino_autuacoes}")
    else:
        ok_aut = _baixar_zip_csv(URL_AUTUACOES, destino_autuacoes, "Autuações IBAMA")
        if not ok_aut:
            print(f"""
[ERRO] Falha no download de autuações.
  Download manual:
    https://dadosabertos.ibama.gov.br/dataset/fiscalizacao-auto-de-infracao
  Salvar como: {destino_autuacoes}
""")

    if os.path.exists(destino_embargos):
        print(f"[OK]  Embargos já existem, pulando download: {destino_embargos}")
    else:
        ok_emb = _baixar_zip_csv(URL_EMBARGOS, destino_embargos, "Termos de Embargo IBAMA")
        if not ok_emb:
            print(f"""
[ERRO] Falha no download de embargos.
  Download manual:
    https://dadosabertos.ibama.gov.br/dataset/fiscalizacao-termo-de-embargo
  Salvar como: {destino_embargos}
""")

    return ok_aut and ok_emb

# Mapeamento dinâmico de colunas.
# O IBAMA muda nomes entre anos; detectamos por padrão de substring.
# Formato: nome_padrão → lista de substrings para buscar (case-insensitive, AND lógico).
MAPA_COLUNAS = {
    # Colunas obrigatórias
    "CPF_CNPJ_INFRATOR": [["CPF_CNPJ_INFRAT"], ["CPF", "CNPJ"], ["INFRAT"]],
    "DAT_AUTO_INFRACAO":  [["DAT_HORA_AUTO"], ["DAT_AUTO"], ["DAT", "AUTO"], ["DAT", "INFRACAO"], ["DT_AUTO"]],
    "VALOR_MULTA":        [["VAL_AUTO_INF"], ["VAL", "AUTO"], ["VAL", "MULTA"], ["VLR", "MULTA"]],
    # Colunas opcionais — ausência não para o pipeline
    "TIPO_INFRACAO":      [["TIPO_INFRA"], ["DES_INFRA"], ["DES_AUTO_INFRA"]],
    "STATUS_DEBITO":      [["DES_STATUS_FORM"], ["STATUS_DEBIT"], ["SIT_DEBIT"], ["DS_SIT_AUTO"]],
}


def detectar_coluna(colunas_disponiveis: list[str], grupos_substrings: list[list[str]]) -> str | None:
    """
    Retorna a primeira coluna disponível que satisfaz QUALQUER grupo de substrings.
    Cada grupo é uma lista de substrings que TODAS devem estar presentes (AND).
    """
    cols_upper = {c.upper(): c for c in colunas_disponiveis}
    for grupo in grupos_substrings:
        for col_upper, col_orig in cols_upper.items():
            if all(s.upper() in col_upper for s in grupo):
                return col_orig
    return None


def mapear_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia colunas do DataFrame para os nomes padrão do ETL,
    usando detecção por substring. Imprime mapeamento encontrado.
    """
    renomear = {}
    for nome_padrao, grupos in MAPA_COLUNAS.items():
        encontrada = detectar_coluna(list(df.columns), grupos)
        if encontrada and encontrada != nome_padrao:
            renomear[encontrada] = nome_padrao
            print(f"[OK]  Coluna mapeada: '{encontrada}' -> '{nome_padrao}'")
        elif encontrada:
            print(f"[OK]  Coluna já no padrão: '{nome_padrao}'")
        else:
            print(f"[WARN] Coluna não encontrada para: '{nome_padrao}'")
    return df.rename(columns=renomear) if renomear else df


# ------------------------------------------------------------------
# 1. Leitura segura do CSV
# ------------------------------------------------------------------
def ler_csv_ibama(caminho: str) -> pd.DataFrame:
    """
    Lê o CSV bruto do IBAMA tentando primeiro latin1 e, em caso de
    falha, utf-8. O IBAMA historicamente publica em latin1 com sep=';'.
    on_bad_lines='skip' evita que linhas mal formatadas derrubem o ETL.
    """
    encodings = ["latin1", "utf-8"]
    ultimo_erro = None
    for enc in encodings:
        try:
            df = pd.read_csv(
                caminho,
                sep=";",
                encoding=enc,
                dtype=str,            # Lemos tudo como string e tipamos depois.
                on_bad_lines="skip",  # Pula linhas quebradas.
                low_memory=False,
            )
            print(f"[OK] Arquivo lido com encoding='{enc}'. "
                  f"Linhas brutas: {len(df):,}")
            return df
        except UnicodeDecodeError as e:
            ultimo_erro = e
            print(f"[WARN] Falha com encoding='{enc}', tentando próximo...")
    raise RuntimeError(f"Não foi possível ler o arquivo: {ultimo_erro}")


# ------------------------------------------------------------------
# 2. Filtragem de colunas essenciais
# ------------------------------------------------------------------
def filtrar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém apenas as colunas padronizadas pelo MAPA_COLUNAS.
    Se alguma estiver ausente após o mapeamento, avisa e segue com as disponíveis.
    """
    esperadas = list(MAPA_COLUNAS.keys())
    presentes = [c for c in esperadas if c in df.columns]
    ausentes = set(esperadas) - set(presentes)
    if ausentes:
        print(f"[WARN] Colunas ausentes apos mapeamento: {ausentes}")
    return df[presentes].copy()


# ------------------------------------------------------------------
# 3. Sanitização de CNPJs
# ------------------------------------------------------------------
# Pré-compilamos o regex por performance (é executado em milhões de linhas).
PADRAO_NAO_DIGITO = re.compile(r"\D")


def limpar_documento(valor) -> str:
    """
    Remove qualquer caractere não numérico (pontos, traços, barras,
    espaços). Trata NaN/None devolvendo string vazia.
    """
    if pd.isna(valor):
        return ""
    return PADRAO_NAO_DIGITO.sub("", str(valor))


def isolar_cnpjs_pj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica a limpeza e mantém apenas registros com EXATAMENTE 14
    dígitos. Isso descarta:
        - CPFs (11 dígitos) -> Pessoas Físicas, fora do escopo PME.
        - Valores corrompidos / parciais.
        - CNPJs com zeros à esquerda perdidos (defensivo: zfill 14
          aplicado antes da validação para recuperar casos onde a
          planilha tratou o campo como número e comeu o zero).
    """
    df = df.copy()
    df["CNPJ_LIMPO"] = df["CPF_CNPJ_INFRATOR"].apply(limpar_documento)

    # Recupera CNPJs que perderam zeros à esquerda (ex.: "1234567000199"
    # com 13 dígitos vira "01234567000199"). CPFs com 10 dígitos NÃO
    # passam a virar CNPJ, pois o filtro abaixo exige exatamente 14.
    df["CNPJ_LIMPO"] = df["CNPJ_LIMPO"].apply(
        lambda x: x.zfill(14) if 11 < len(x) < 14 else x
    )

    antes = len(df)
    df = df[df["CNPJ_LIMPO"].str.len() == 14].copy()
    depois = len(df)
    print(f"[OK] Sanitização de CNPJ: {antes:,} -> {depois:,} "
          f"linhas ({antes - depois:,} descartadas como PF/inválidos).")
    return df


# ------------------------------------------------------------------
# 4. Tipagem de valores e datas
# ------------------------------------------------------------------
def tipar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte VALOR_MULTA (string com vírgula decimal BR) para float
    e DAT_AUTO_INFRACAO para datetime. Erros viram NaN/NaT em vez de
    derrubarem o pipeline.
    """
    df = df.copy()

    if "VALOR_MULTA" in df.columns:
        # Padrão BR: "1.234.567,89" -> remove pontos de milhar e troca vírgula.
        df["VALOR_MULTA"] = (
            df["VALOR_MULTA"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["VALOR_MULTA"] = pd.to_numeric(df["VALOR_MULTA"], errors="coerce")

    if "DAT_AUTO_INFRACAO" in df.columns:
        # dayfirst=True porque o IBAMA usa dd/mm/aaaa.
        df["DAT_AUTO_INFRACAO"] = pd.to_datetime(
            df["DAT_AUTO_INFRACAO"], dayfirst=True, errors="coerce"
        )

    return df


# ------------------------------------------------------------------
# 5. Agregação por CNPJ (features de reincidência)
# ------------------------------------------------------------------
def agregar_por_cnpj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria, por CNPJ, as features que alimentarão o modelo:
        - qtd_infracoes:        contagem de autuações.
        - valor_total_multas:   soma do valor das multas.
        - infracao_mais_recente: data da última autuação.
    """
    agregado = (
        df.groupby("CNPJ_LIMPO", as_index=False)
          .agg(
              qtd_infracoes=("CNPJ_LIMPO", "count"),
              valor_total_multas=("VALOR_MULTA", "sum"),
              infracao_mais_recente=("DAT_AUTO_INFRACAO", "max"),
          )
          .sort_values("valor_total_multas", ascending=False)
    )
    print(f"[OK] Agregação concluída: {len(agregado):,} CNPJs únicos.")
    return agregado


# ------------------------------------------------------------------
# 6. Exportação
# ------------------------------------------------------------------
def exportar(df: pd.DataFrame, caminho: str) -> None:
    """Salva em UTF-8 com BOM (excel-friendly) e separador ';'."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df.to_csv(caminho, sep=";", index=False, encoding="utf-8-sig")
    print(f"[OK] Arquivo gerado: {caminho}")


# ------------------------------------------------------------------
# ETL Embargos (pipeline secundário)
# ------------------------------------------------------------------
# Colunas candidatas para o CNPJ/CPF no arquivo de embargos.
# O IBAMA usa nomes ligeiramente diferentes entre datasets.
COLS_CNPJ_EMBARGO = [
    "CPF_CNPJ_INFRATOR",
    "CPF_CNPJ_INTERESSADO",
    "NUM_CPF_CNPJ",
    "CPF_CNPJ",
]

def processar_embargos(caminho: str) -> pd.DataFrame:
    """
    Lê o CSV de Termos de Embargo, identifica a coluna de CNPJ/CPF
    automaticamente, filtra apenas PJs (14 dígitos) e cria uma flag
    binária 'embargado' = 1 por CNPJ.

    O embargo é tratado como FILTRO ELIMINATÓRIO no modelo:
    CNPJ com embargo ativo → bloqueio de crédito verde imediato.
    """
    print(f"[OK]  Lendo embargos: {caminho}")
    for enc in ["latin1", "utf-8"]:
        try:
            df = pd.read_csv(caminho, sep=";", dtype=str,
                             encoding=enc, on_bad_lines="skip", low_memory=False)
            break
        except UnicodeDecodeError:
            continue

    print(f"[OK]  {len(df):,} registros brutos de embargo. Colunas: {list(df.columns)}")

    # Detecta coluna de documento automaticamente
    col_doc = next((c for c in COLS_CNPJ_EMBARGO if c in df.columns), None)
    if col_doc is None:
        # Busca qualquer coluna que contenha "CNPJ" ou "CPF"
        candidatas = [c for c in df.columns if "CNPJ" in c.upper() or "CPF" in c.upper()]
        if not candidatas:
            print("[WARN] Nenhuma coluna de CNPJ/CPF encontrada nos embargos.")
            print(f"       Colunas disponíveis: {list(df.columns)}")
            return pd.DataFrame(columns=["CNPJ_LIMPO", "embargado"])
        col_doc = candidatas[0]

    print(f"[OK]  Coluna de documento utilizada: '{col_doc}'")

    df["CNPJ_LIMPO"] = df[col_doc].apply(limpar_documento)
    df["CNPJ_LIMPO"] = df["CNPJ_LIMPO"].apply(
        lambda x: x.zfill(14) if 11 < len(x) < 14 else x
    )
    df = df[df["CNPJ_LIMPO"].str.len() == 14].copy()
    print(f"[OK]  {len(df):,} registros de embargo com CNPJ válido (PJ).")

    # Um CNPJ pode ter múltiplos termos de embargo — basta marcar presença
    resultado = (
        df[["CNPJ_LIMPO"]]
        .drop_duplicates()
        .assign(embargado=1)
    )
    print(f"[OK]  {len(resultado):,} CNPJs únicos com histórico de embargo.")
    return resultado


# ------------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print(" ETL IBAMA - Bureau de Crédito Verde (TCC MBA USP/Esalq)")
    print(" Fontes: dadosabertos.ibama.gov.br")
    print("=" * 70)

    # ── 0. Download ────────────────────────────────────────────────
    print("\n[0] Verificando / baixando dados do IBAMA...")
    ok = baixar_ibama(INPUT_PATH, INPUT_EMBARGOS)
    if not ok:
        print("[ERRO] Um ou mais downloads falharam. Veja instruções acima.")
        sys.exit(1)

    # ── Pipeline A: Autuações ──────────────────────────────────────
    print("\n" + "-" * 70)
    print(" PIPELINE A — Autuações (features de reincidência)")
    print("-" * 70)

    print(f"\n[A1/5] Lendo arquivo bruto: {INPUT_PATH}")
    df = ler_csv_ibama(INPUT_PATH)
    print(f"       Colunas brutas: {list(df.columns)}")

    print("\n[A2/5] Mapeando nomes de colunas (detecção dinâmica)...")
    df = mapear_colunas(df)

    print("\n[A3/6] Filtrando colunas essenciais...")
    df = filtrar_colunas(df)

    print("\n[A4/6] Sanitizando CPF/CNPJ e isolando Pessoas Jurídicas...")
    df = isolar_cnpjs_pj(df)

    print("\n[A5/6] Tipando valores (multa) e datas...")
    df = tipar_colunas(df)

    print("\n[A6/6] Agregando métricas de reincidência por CNPJ...")
    agregado = agregar_por_cnpj(df)
    exportar(agregado, OUTPUT_PATH)

    print("\n--- Top 5 por valor total de multas ---")
    print(agregado.head().to_string(index=False))

    # ── Pipeline B: Embargos ───────────────────────────────────────
    print("\n" + "-" * 70)
    print(" PIPELINE B — Embargos (filtro eliminatório binário)")
    print("-" * 70)

    embargos = processar_embargos(INPUT_EMBARGOS)
    exportar(embargos, OUTPUT_EMBARGOS)

    print("\n--- Amostra embargos ---")
    print(embargos.head().to_string(index=False))

    # ── Resumo ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" RESUMO DOS OUTPUTS")
    print("=" * 70)
    print(f"  Autuações : {OUTPUT_PATH}")
    print(f"              {len(agregado):,} CNPJs | colunas: {list(agregado.columns)}")
    print(f"  Embargos  : {OUTPUT_EMBARGOS}")
    print(f"              {len(embargos):,} CNPJs | colunas: {list(embargos.columns)}")
    print("\nProximo passo: etl_receita.py -> etl_ceis.py -> feature_engineering.py")


if __name__ == "__main__":
    main()
