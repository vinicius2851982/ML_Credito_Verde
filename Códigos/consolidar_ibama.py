"""
consolidar_ibama.py  (preparo dos arquivos do IBAMA para o ETL)
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

O IBAMA distribui os autos de infracao em um ZIP com um CSV POR ANO
(auto_infracao_1977.csv ... auto_infracao_2026.csv). O ETL espera um unico
arquivo consolidado. Este script faz a ponte:

  1. Le cada CSV anual de dentro do ZIP, sem extrair tudo para o disco.
  2. DESCARTA anos inteiros posteriores ao corte (economiza leitura).
  3. Concatena em Dados/ibama_autuacoes.csv, preservando o cabecalho uma vez.
  4. Copia o arquivo de embargos para Dados/ibama_embargos.csv.

O filtro fino por data (dentro do ano do corte) permanece no etl_ibama.py,
que e onde as colunas ja estao normalizadas.

Os arquivos anteriores sao preservados com sufixo .anterior — a substituicao
de uma base de 700 MB nao deve ser irreversivel por descuido.

Uso:
    python -B Códigos/consolidar_ibama.py
"""

import os
import shutil
import zipfile

from recorte import CORTE_DADOS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(BASE_DIR, "Dados")
RAW = os.path.join(DADOS, "raw", "ibama")

ZIP_AUTOS = os.path.join(RAW, "auto_infracao_csv.zip")
CSV_EMBARGO = os.path.join(RAW, "termo_de_embargo.csv")
SAIDA_AUTOS = os.path.join(DADOS, "ibama_autuacoes.csv")
SAIDA_EMBARGO = os.path.join(DADOS, "ibama_embargos.csv")

ANO_CORTE = CORTE_DADOS.year


def preservar(caminho):
    """Renomeia o arquivo existente em vez de sobrescrever."""
    if os.path.exists(caminho):
        anterior = caminho + ".anterior"
        if os.path.exists(anterior):
            os.remove(anterior)
        os.rename(caminho, anterior)
        print(f"      anterior preservado: {os.path.basename(anterior)}")


def consolidar_autos():
    if not os.path.exists(ZIP_AUTOS):
        print(f"[ERRO] nao encontrado: {ZIP_AUTOS}")
        print("       rode antes: python -B Códigos/atualizar_dados_ibama.py")
        return False

    with zipfile.ZipFile(ZIP_AUTOS) as z:
        nomes = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
        # descarta anos integralmente posteriores ao corte
        selecionados, descartados = [], []
        for n in nomes:
            digitos = "".join(c for c in os.path.basename(n) if c.isdigit())
            ano = int(digitos[-4:]) if len(digitos) >= 4 else 0
            (descartados if ano > ANO_CORTE else selecionados).append(n)

        print(f"[1/2] Autos de infracao: {len(selecionados)} arquivos anuais")
        if descartados:
            print(f"      descartados por serem posteriores a {ANO_CORTE}: "
                  f"{', '.join(os.path.basename(d) for d in descartados)}")

        preservar(SAIDA_AUTOS)
        escritos = 0
        with open(SAIDA_AUTOS, "wb") as saida:
            for i, nome in enumerate(selecionados):
                with z.open(nome) as f:
                    primeira = f.readline()          # cabecalho
                    if i == 0:
                        saida.write(primeira)
                    shutil.copyfileobj(f, saida, length=1 << 20)
                escritos += 1
                if escritos % 10 == 0 or escritos == len(selecionados):
                    print(f"      {escritos}/{len(selecionados)} consolidados")
    mb = os.path.getsize(SAIDA_AUTOS) / 1e6
    print(f"      salvo: {SAIDA_AUTOS} ({mb:,.1f} MB)")
    return True


def copiar_embargos():
    if not os.path.exists(CSV_EMBARGO):
        print(f"[ERRO] nao encontrado: {CSV_EMBARGO}")
        return False
    print("[2/2] Termos de embargo")
    preservar(SAIDA_EMBARGO)
    shutil.copy2(CSV_EMBARGO, SAIDA_EMBARGO)
    mb = os.path.getsize(SAIDA_EMBARGO) / 1e6
    print(f"      salvo: {SAIDA_EMBARGO} ({mb:,.1f} MB)")
    return True


def main():
    print("=" * 66)
    print(" CONSOLIDACAO DOS DADOS DO IBAMA")
    print(f" Corte dos dados: {CORTE_DADOS:%d/%m/%Y} "
          f"(anos > {ANO_CORTE} descartados aqui; filtro fino no ETL)")
    print("=" * 66)
    ok = consolidar_autos()
    ok = copiar_embargos() and ok
    if ok:
        print("\nProximo passo: python -B Códigos/etl_ibama.py")
        print("O ETL aplica o filtro por data e agrega por CNPJ.")


if __name__ == "__main__":
    main()
