"""
etl_receita.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Crédito Verde e IA para PMEs
Autor: Hélio Vinícius Moreira Ribeiro

FASE 1 — Amostragem de Estabelecimentos
    Lê os 10 arquivos .ESTABELE em chunks de 200.000 linhas (nunca
    carrega o arquivo inteiro na RAM). Filtra situação cadastral ativa
    e realiza amostragem aleatória simples de 10% por chunk, com semente
    determinística (reprodutível). Salva checkpoint intermediário e o
    conjunto de cnpj_basico selecionados.

FASE 2 — Enriquecimento com dados de Empresas
    Lê os arquivos de Empresas também em chunks, filtrando apenas os
    cnpj_basico presentes na Fase 1. Faz join left com a amostra de
    estabelecimentos e exporta a base consolidada.

LAYOUTS OFICIAIS RFB (CSV, sep=";", sem cabeçalho, encoding=latin1):

  Estabelecimentos (30 colunas):
    0  cnpj_basico             8 dígitos
    1  cnpj_ordem              4 dígitos
    2  cnpj_dv                 2 dígitos
    3  identificador           1=matriz, 2=filial
    4  nome_fantasia
    5  situacao_cadastral      02=ativa
    6  data_situacao_cadastral
    7  motivo_situacao_cadastral
    8  nome_cidade_exterior
    9  pais
    10 data_inicio_atividade
    11 cnae_fiscal_principal
    12 cnae_fiscal_secundaria
    13 tipo_logradouro  14 logradouro  15 numero
    16 complemento  17 bairro  18 cep
    19 uf  20 municipio
    21 ddd_1  22 telefone_1  23 ddd_2  24 telefone_2
    25 ddd_fax  26 fax
    27 correio_eletronico
    28 situacao_especial
    29 data_situacao_especial

  Empresas (7 colunas):
    0  cnpj_basico             8 dígitos (chave de join)
    1  razao_social
    2  natureza_juridica       4 dígitos (tabela RFB)
    3  qualificacao_responsavel
    4  capital_social          decimal com vírgula
    5  porte                   00=n/i, 01=ME, 03=EPP, 05=demais
    6  ente_federativo_responsavel

METODOLOGIA DE AMOSTRAGEM (registrar em Material e Métodos):
    "Foi utilizada amostra aleatória simples de 10% do Cadastro
     Nacional de Pessoa Jurídica — base Estabelecimentos, processada
     em lotes de 200.000 registros com semente por lote determinística
     (semente_arquivo + índice_lote), abrangendo somente estabelecimentos
     com situação cadastral ativa (código 02). Os dados de empresa
     (razao_social, natureza_juridica, capital_social, porte) foram
     obtidos por cruzamento com a base Empresas via cnpj_basico."
"""

import os
import gc
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_RFB      = os.path.join(BASE_DIR, "Dados", "rfb_estabelecimentos")
PASTA_EMPRESAS = os.path.join(BASE_DIR, "Dados", "rfb_empresas")
SAIDA_ESTAB    = os.path.join(BASE_DIR, "Dados", "stg_rfb_amostra_estabelecimentos.csv")
SAIDA_FINAL    = os.path.join(BASE_DIR, "Dados", "stg_rfb_consolidado.csv")

FRAC_AMOSTRA = 0.10
SEMENTE_BASE = 42
CHUNKSIZE    = 50_000       # linhas por lote — conservador para 16 GB RAM com 94% uso
RELATORIO_A_CADA = 20       # imprime progresso a cada N chunks

# Índices das colunas que realmente precisamos do arquivo Estabelecimentos (30 colunas total).
# Ler só 9 colunas reduz o consumo de RAM ~70% por lote.
USECOLS_ESTAB = [0, 1, 2, 3, 5, 10, 11, 19, 20]
#                basico ordem dv  ident situa data_ini cnae  uf  mun

COLS_ESTAB = [
    "cnpj_basico", "CNPJ", "identificador", "situacao_cadastral",
    "cnae_fiscal_principal", "data_inicio_atividade", "uf", "municipio",
]
COLS_EMPRESA = [
    "razao_social", "natureza_juridica", "capital_social", "porte",
]

NOMES_ESTAB = [
    "cnpj_basico", "cnpj_ordem", "cnpj_dv", "identificador",
    "nome_fantasia", "situacao_cadastral", "data_situacao_cadastral",
    "motivo_situacao_cadastral", "nome_cidade_exterior", "pais",
    "data_inicio_atividade", "cnae_fiscal_principal", "cnae_fiscal_secundaria",
    "tipo_logradouro", "logradouro", "numero", "complemento", "bairro",
    "cep", "uf", "municipio", "ddd_1", "telefone_1", "ddd_2", "telefone_2",
    "ddd_fax", "fax", "correio_eletronico", "situacao_especial",
    "data_situacao_especial",
]
NOMES_EMPRESA = [
    "cnpj_basico", "razao_social", "natureza_juridica",
    "qualificacao_responsavel", "capital_social", "porte",
    "ente_federativo_responsavel",
]


# ------------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------------
def abrir_reader_rfb(caminho: str, nomes: list[str], usecols=None):
    """Abre um TextFileReader em chunks. Tenta latin1, depois utf-8."""
    kwargs = dict(
        sep=";", header=None, names=nomes, dtype=str,
        on_bad_lines="skip", chunksize=CHUNKSIZE,
        usecols=usecols,
    )
    try:
        return pd.read_csv(caminho, encoding="latin1", **kwargs)
    except Exception:
        return pd.read_csv(caminho, encoding="utf-8", **kwargs)


def verificar_espaco_disco(pasta: str, minimo_gb: float = 5.0) -> None:
    """Avisa se o disco livre estiver abaixo do mínimo recomendado."""
    livre_gb = shutil.disk_usage(pasta).free / 1e9
    if livre_gb < minimo_gb:
        print(f"[AVISO] Disco com apenas {livre_gb:.1f} GB livres em {pasta}.")
        print(f"        Os arquivos de saída podem precisar de até {minimo_gb:.0f} GB.")
        print("        Considere liberar espaço antes de continuar.")


def localizar_arquivos(pasta: str, sufixo_upper: str) -> list[str]:
    resultado = []
    if not os.path.isdir(pasta):
        return resultado
    for arq in os.listdir(pasta):
        if sufixo_upper in arq.upper() and not arq.lower().endswith(".zip"):
            resultado.append(os.path.join(pasta, arq))
    return sorted(resultado)


def localizar_arquivos_empresas(pasta: str) -> list[str]:
    resultado = []
    if not os.path.isdir(pasta):
        return resultado
    for arq in os.listdir(pasta):
        if "EMPR" in arq.upper() and not arq.lower().endswith(".zip"):
            resultado.append(os.path.join(pasta, arq))
    return sorted(resultado)


# ------------------------------------------------------------------
# FASE 1 — Amostragem de Estabelecimentos (leitura em chunks)
# ------------------------------------------------------------------
def processar_arquivo_estabele(
    caminho: str, semente_arquivo: int, saida_csv: str, primeiro_arquivo: bool
) -> int:
    """
    Lê um .ESTABELE em chunks de CHUNKSIZE linhas.
    Por chunk: filtra ativos → amostra 10% → grava diretamente no CSV de saída.
    Sem acumular nada em memória. Retorna total de linhas gravadas.
    """
    nome = os.path.basename(caminho)
    tamanho_mb = os.path.getsize(caminho) / 1e6
    print(f"     Arquivo: {nome}  ({tamanho_mb:.0f} MB)")

    total_lido = 0
    total_ativos = 0
    total_amostrado = 0
    n_chunk = 0

    # usecols com os nomes correspondentes às posições USECOLS_ESTAB
    nomes_uteis = [NOMES_ESTAB[c] for c in USECOLS_ESTAB]
    reader = abrir_reader_rfb(caminho, nomes=nomes_uteis, usecols=USECOLS_ESTAB)

    for n_chunk, chunk in enumerate(reader):
        total_lido += len(chunk)

        ativos = chunk[chunk["situacao_cadastral"].str.strip() == "02"]
        total_ativos += len(ativos)

        if len(ativos) == 0:
            del chunk, ativos
            continue

        amostra = ativos.sample(frac=FRAC_AMOSTRA, random_state=semente_arquivo + n_chunk).copy()

        # Monta CNPJ completo de 14 dígitos
        amostra["CNPJ"] = (
            amostra["cnpj_basico"].str.strip().str.zfill(8)
            + amostra["cnpj_ordem"].str.strip().str.zfill(4)
            + amostra["cnpj_dv"].str.strip().str.zfill(2)
        )

        # Grava direto no CSV — sem guardar em memória
        escrever_header = primeiro_arquivo and (n_chunk == 0)
        amostra[COLS_ESTAB].to_csv(
            saida_csv,
            sep=";",
            index=False,
            encoding="utf-8-sig",
            mode="w" if escrever_header else "a",
            header=escrever_header,
        )

        total_amostrado += len(amostra)

        # Libera memória do lote imediatamente
        del chunk, ativos, amostra
        gc.collect()

        if (n_chunk + 1) % RELATORIO_A_CADA == 0:
            print(
                f"     ... lote {n_chunk + 1:>4} | "
                f"lido: {total_lido/1e6:>6.1f}M | "
                f"ativos: {total_ativos/1e6:>5.1f}M | "
                f"amostrados: {total_amostrado:>8,}",
                flush=True,
            )

    print(
        f"     [OK] lotes: {n_chunk + 1} | "
        f"lido: {total_lido/1e6:.1f}M | "
        f"ativos: {total_ativos/1e6:.1f}M | "
        f"amostrados: {total_amostrado:,}"
    )
    return total_amostrado


def fase1_amostrar_estabelecimentos() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(" FASE 1 — Amostragem de Estabelecimentos (AAS 10% por chunk)")
    print(f" Lote: {CHUNKSIZE:,} linhas | Colunas lidas: {len(USECOLS_ESTAB)}/30 | Semente base: {SEMENTE_BASE}")
    print(" Escrita incremental — sem acúmulo em RAM.")
    print("=" * 70)

    arquivos = localizar_arquivos(PASTA_RFB, "ESTABELE")
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo .ESTABELE encontrado em {PASTA_RFB}\n"
            "Execute preparar_dados_publicos.py primeiro."
        )

    print(f"[OK]  {len(arquivos)} arquivo(s) encontrado(s).\n")
    total_gravado = 0

    for i, arq in enumerate(arquivos, 1):
        print(f"  [{i}/{len(arquivos)}]")
        n = processar_arquivo_estabele(
            arq,
            semente_arquivo=SEMENTE_BASE + i * 1000,
            saida_csv=SAIDA_ESTAB,
            primeiro_arquivo=(i == 1),
        )
        total_gravado += n
        gc.collect()

    print(f"\n[OK]  Total gravado no checkpoint: {total_gravado:,} registros.")
    print(f"[OK]  Checkpoint: {SAIDA_ESTAB}")

    # Lê o checkpoint de volta para o join (Fase 2)
    # O arquivo já está no disco; carrega em memória só agora, uma única vez.
    print("[...] Carregando checkpoint para Fase 2 ...")
    df_estab = pd.read_csv(SAIDA_ESTAB, sep=";", dtype=str, encoding="utf-8-sig")
    antes = len(df_estab)
    df_estab = df_estab.drop_duplicates(subset=["CNPJ"])
    if antes - len(df_estab):
        print(f"[OK]  {antes - len(df_estab):,} CNPJ(s) duplicado(s) removido(s).")
    print(f"[OK]  {len(df_estab):,} estabelecimentos únicos na amostra.")
    return df_estab


# ------------------------------------------------------------------
# FASE 2 — Enriquecimento com dados de Empresas (leitura em chunks)
# ------------------------------------------------------------------
def processar_arquivo_empresa(caminho: str, cnpjs_alvo: set) -> pd.DataFrame:
    """
    Lê um arquivo de Empresas em chunks, retendo apenas os registros
    cujo cnpj_basico está no conjunto amostrado na Fase 1.
    Carrega apenas as 5 colunas necessárias (de 7 disponíveis).
    """
    # Empresas tem 7 colunas; queremos só: basico(0), razao(1), nat_jur(2), capital(4), porte(5)
    usecols_emp = [0, 1, 2, 4, 5]
    nomes_uteis = [NOMES_EMPRESA[c] for c in usecols_emp]

    nome = os.path.basename(caminho)
    tamanho_mb = os.path.getsize(caminho) / 1e6
    print(f"     Arquivo: {nome}  ({tamanho_mb:.0f} MB)")

    partes = []
    total_lido = 0
    total_encontrado = 0
    n_chunk = 0

    reader = abrir_reader_rfb(caminho, nomes=nomes_uteis, usecols=usecols_emp)

    for n_chunk, chunk in enumerate(reader):
        total_lido += len(chunk)
        chunk["cnpj_basico"] = chunk["cnpj_basico"].str.strip().str.zfill(8)
        filtrado = chunk[chunk["cnpj_basico"].isin(cnpjs_alvo)].copy()

        if len(filtrado) > 0:
            partes.append(filtrado[["cnpj_basico"] + COLS_EMPRESA])
            total_encontrado += len(filtrado)

        del chunk, filtrado
        gc.collect()

        if (n_chunk + 1) % RELATORIO_A_CADA == 0:
            print(
                f"     ... lote {n_chunk + 1:>4} | "
                f"lido: {total_lido/1e6:>6.1f}M | "
                f"encontrados: {total_encontrado:>8,}",
                flush=True,
            )

    if not partes:
        return pd.DataFrame(columns=["cnpj_basico"] + COLS_EMPRESA)

    df = pd.concat(partes, ignore_index=True)
    print(f"     [OK] lotes: {n_chunk + 1} | lido: {total_lido/1e6:.1f}M | encontrados: {len(df):,}")
    return df


def fase2_enriquecer_com_empresas(df_estab: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(" FASE 2 — Enriquecimento com dados de Empresas")
    print(f" Tamanho do lote: {CHUNKSIZE:,} linhas")
    print("=" * 70)

    arquivos = localizar_arquivos_empresas(PASTA_EMPRESAS)
    if not arquivos:
        print(f"[WARN] Nenhum arquivo de Empresas encontrado em {PASTA_EMPRESAS}")
        print("       Execute preparar_dados_publicos.py com Empresas.zip disponível.")
        print("       Exportando base de estabelecimentos sem enriquecimento.")
        return df_estab

    cnpjs_alvo = set(df_estab["cnpj_basico"].str.strip().str.zfill(8).unique())
    print(f"[OK]  {len(cnpjs_alvo):,} cnpj_basico únicos a localizar nas Empresas.")
    print(f"[OK]  {len(arquivos)} arquivo(s) de Empresas encontrado(s).\n")

    partes_empresa = []

    for i, arq in enumerate(arquivos, 1):
        print(f"  [{i}/{len(arquivos)}]")
        df_emp = processar_arquivo_empresa(arq, cnpjs_alvo)
        partes_empresa.append(df_emp)

    print("\n[...] Consolidando dados de empresas ...")
    df_empresa = pd.concat(partes_empresa, ignore_index=True)
    df_empresa = df_empresa.drop_duplicates(subset=["cnpj_basico"])

    # Converte capital_social: "1.234,56" → 1234.56
    df_empresa["capital_social"] = (
        df_empresa["capital_social"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df_empresa["capital_social"] = pd.to_numeric(
        df_empresa["capital_social"], errors="coerce"
    )

    print(f"[OK]  {len(df_empresa):,} empresas únicas consolidadas.")

    # Join: estabelecimento ← empresa (via cnpj_basico, left join)
    df_estab = df_estab.copy()
    df_estab["cnpj_basico"] = df_estab["cnpj_basico"].str.strip().str.zfill(8)
    df_final = df_estab.merge(df_empresa, on="cnpj_basico", how="left")

    cobertura = df_final["razao_social"].notna().sum()
    print(
        f"[OK]  {cobertura:,} de {len(df_final):,} estabelecimentos enriquecidos "
        f"({cobertura / len(df_final):.1%} de cobertura)."
    )
    return df_final


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print(" ETL Receita Federal — Estabelecimentos + Empresas")
    print(f" Amostragem: {int(FRAC_AMOSTRA * 100)}% | Chunk: {CHUNKSIZE:,} linhas | Semente base: {SEMENTE_BASE}")
    print("=" * 70)

    verificar_espaco_disco(BASE_DIR, minimo_gb=5.0)

    df_estab = fase1_amostrar_estabelecimentos()
    gc.collect()
    df_final  = fase2_enriquecer_com_empresas(df_estab)
    del df_estab
    gc.collect()

    df_final.to_csv(SAIDA_FINAL, sep=";", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"[OK]  Registros finais  : {len(df_final):,}")
    print(f"[OK]  Colunas           : {list(df_final.columns)}")
    print(f"[OK]  Exportado         : {SAIDA_FINAL}")
    print("\nPróximos passos:")
    print("  python Códigos/etl_ibama.py")
    print("  python Códigos/etl_ceis.py")


if __name__ == "__main__":
    main()
