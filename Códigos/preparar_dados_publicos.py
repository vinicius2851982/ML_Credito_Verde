"""
preparar_dados_publicos.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Crédito Verde e IA para PMEs
Autor: Hélio Vinícius Moreira Ribeiro
Sprint Dia 2 - Organização das bases públicas complementares

CONTEXTO (importante para a defesa do TCC):
    O download programático das bases da Receita Federal (CNPJ) e
    do Portal da Transparência (CEIS) tornou-se inviável em 2026:

      - CEIS: protegido por AWS WAF com CAPTCHA.
      - RFB:  migrou de FTP para Nextcloud (SERPRO+), com URLs
              dinâmicas que mudam a cada publicação mensal.

    Por isso, este script NÃO baixa os arquivos — ele assume que
    o pesquisador baixou manualmente via browser e os colocou em
    /Dados/raw/. O script extrai, organiza e valida.

COMO USAR:
    1) Baixe manualmente:
       a) RFB Estabelecimentos:
          https://dados.gov.br/dados/conjuntos-dados/
          cadastro-nacional-da-pessoa-juridica-cnpj
          -> escolha UM arquivo "Estabelecimentos*.zip"
       b) CEIS:
          https://portaldatransparencia.gov.br/download-de-dados/ceis
          -> botão "Baixar" do mês corrente

    2) Coloque os dois ZIPs em /Dados/raw/

    3) Rode: python Códigos/preparar_dados_publicos.py
"""

import os
import sys
import glob
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_RAW = os.path.join(BASE_DIR, "Dados", "raw")
PASTA_RFB = os.path.join(BASE_DIR, "Dados", "rfb_estabelecimentos")
PASTA_EMPRESAS = os.path.join(BASE_DIR, "Dados", "rfb_empresas")
PASTA_CEIS = os.path.join(BASE_DIR, "Dados", "ceis")


# ------------------------------------------------------------------
# Localização dos ZIPs em /Dados/raw/
# ------------------------------------------------------------------
def localizar_zip(pasta: str, padroes: list[str]) -> str | None:
    """
    Procura, em /Dados/raw/, o primeiro ZIP que case com qualquer
    um dos padrões (case-insensitive). Retorna o caminho ou None.
    """
    if not os.path.isdir(pasta):
        return None
    arquivos = glob.glob(os.path.join(pasta, "*.zip"))
    arquivos += glob.glob(os.path.join(pasta, "*.ZIP"))
    for arq in arquivos:
        nome = os.path.basename(arq).lower()
        if any(p.lower() in nome for p in padroes):
            return arq
    return None


def extrair_e_listar(zip_path: str, destino: str) -> list[str]:
    """Extrai um ZIP e devolve a lista de arquivos extraídos."""
    os.makedirs(destino, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(destino)
        return z.namelist()


def extrair_subzip_rfb(zip_externo: str, destino: str) -> list[str]:
    """
    Lida com o formato zip-de-zips da RFB:
    Estabelecimentos.zip contém Estabelecimentos0..9.zip.
    Extrai todos os sub-ZIPs e seus conteúdos para que o etl_receita.py
    possa amostrar aleatoriamente 10% de cada arquivo (AAS da população).
    Retorna a lista de arquivos .ESTABELE finais extraídos.
    """
    os.makedirs(destino, exist_ok=True)
    arquivos_finais = []

    with zipfile.ZipFile(zip_externo, "r") as z_ext:
        membros = z_ext.namelist()
        sub_zips = sorted([m for m in membros if m.lower().endswith(".zip")])
        arquivos_diretos = [m for m in membros if not m.lower().endswith(".zip")]

        # Caso o ZIP já contenha os arquivos finais (download direto de sub-zip individual)
        if arquivos_diretos and not sub_zips:
            z_ext.extractall(destino)
            return arquivos_diretos

        if not sub_zips:
            print("[WARN] Nenhum sub-ZIP encontrado dentro do arquivo externo.")
            return []

        print(f"[INFO] Zip-de-zips: {len(sub_zips)} sub-arquivos encontrados.")
        print("[INFO] Extraindo todos para amostragem aleatória simples de 10% no etl_receita.py.")

        for i, sub in enumerate(sub_zips, 1):
            print(f"  [{i}/{len(sub_zips)}] Extraindo {sub} ...", end=" ", flush=True)
            z_ext.extract(sub, destino)
            sub_path = os.path.join(destino, sub)

            with zipfile.ZipFile(sub_path, "r") as z_sub:
                z_sub.extractall(destino)
                arquivos_finais.extend(z_sub.namelist())

            os.remove(sub_path)
            print("OK")

    return arquivos_finais


# ------------------------------------------------------------------
# Instruções de erro (quando o arquivo não está em /raw)
# ------------------------------------------------------------------
INSTRUCAO_RFB = """
[FALTANDO] Estabelecimentos.zip (Receita Federal)

  1) Abra no browser:
     https://dados.gov.br/dados/conjuntos-dados/cadastro-nacional-da-pessoa-juridica-cnpj
  2) Baixe o arquivo "Estabelecimentos.zip" (zip-de-zips, ~4,7 GB).
     Este script extrai os 10 sub-arquivos para que etl_receita.py
     possa realizar amostragem aleatória simples de 10% da população.
  3) Mova o ZIP para: {pasta}
  4) Rode este script novamente.
"""

INSTRUCAO_EMPRESAS = """
[FALTANDO] Empresas.zip (Receita Federal)

  1) Abra no browser:
     https://dados.gov.br/dados/conjuntos-dados/cadastro-nacional-da-pessoa-juridica-cnpj
  2) Baixe o arquivo "Empresas.zip" (zip-de-zips, ~1,5 GB).
     Contém: razao_social, natureza_juridica, capital_social, porte.
  3) Mova o ZIP para: {pasta}
  4) Rode este script novamente.
"""

INSTRUCAO_CEIS = """
[FALTANDO] CEIS.zip (Portal da Transparência)

  1) Abra no browser:
     https://portaldatransparencia.gov.br/download-de-dados/ceis
  2) Clique em "Baixar" do mês mais recente (resolva o CAPTCHA).
  3) Mova o ZIP para: {pasta}
     (Pode renomear para 'ceis.zip' ou deixar o nome original.)
  4) Rode este script novamente.
"""


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
def processar_rfb() -> bool:
    print("\n" + "=" * 70)
    print(" [1/3] Receita Federal - Estabelecimentos")
    print("=" * 70)
    zip_path = localizar_zip(PASTA_RAW, ["estabelec"])
    if not zip_path:
        print(INSTRUCAO_RFB.format(pasta=PASTA_RAW))
        return False

    print(f"[OK]  ZIP encontrado: {zip_path}")

    # Verifica se é zip-de-zips (Estabelecimentos.zip) ou zip direto (um sub-arquivo)
    with zipfile.ZipFile(zip_path, "r") as z:
        membros = z.namelist()
        tem_subzips = any(m.lower().endswith(".zip") for m in membros)

    if tem_subzips:
        print(f"[INFO] Formato zip-de-zips: {len(membros)} sub-arquivo(s) detectado(s).")
        print("[INFO] Extraindo todos. etl_receita.py fará AAS de 10% por arquivo.")
        arquivos = extrair_subzip_rfb(zip_path, PASTA_RFB)
    else:
        arquivos = extrair_e_listar(zip_path, PASTA_RFB)

    print(f"[OK]  {len(arquivos)} arquivo(s) extraído(s) em {PASTA_RFB}:")
    for a in arquivos[:10]:
        caminho = os.path.join(PASTA_RFB, a)
        if os.path.isfile(caminho):
            tamanho = os.path.getsize(caminho)
            print(f"      - {a}  ({tamanho/1e6:.1f} MB)")
        else:
            print(f"      - {a}")

    # Sanity check: a RFB entrega um arquivo .ESTABELE sem extensão csv.
    estabele = [a for a in arquivos if "ESTABELE" in a.upper()]
    if not estabele:
        print("[WARN] Nenhum arquivo .ESTABELE encontrado. "
              "Confirme que baixou o arquivo de Estabelecimentos (não Empresas ou Sócios).")
    return True


def processar_empresas() -> bool:
    print("\n" + "=" * 70)
    print(" [2/3] Receita Federal - Empresas")
    print("=" * 70)
    zip_path = localizar_zip(PASTA_RAW, ["empres"])
    if not zip_path:
        print(INSTRUCAO_EMPRESAS.format(pasta=PASTA_RAW))
        return False

    print(f"[OK]  ZIP encontrado: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        membros = z.namelist()
        tem_subzips = any(m.lower().endswith(".zip") for m in membros)

    if tem_subzips:
        print(f"[INFO] Formato zip-de-zips: {len(membros)} sub-arquivo(s) detectado(s).")
        arquivos = extrair_subzip_rfb(zip_path, PASTA_EMPRESAS)
    else:
        arquivos = extrair_e_listar(zip_path, PASTA_EMPRESAS)

    print(f"[OK]  {len(arquivos)} arquivo(s) extraído(s) em {PASTA_EMPRESAS}:")
    for a in arquivos[:10]:
        caminho = os.path.join(PASTA_EMPRESAS, a)
        if os.path.isfile(caminho):
            tamanho = os.path.getsize(caminho)
            print(f"      - {a}  ({tamanho/1e6:.1f} MB)")
        else:
            print(f"      - {a}")

    empr = [a for a in arquivos if "EMPR" in a.upper()]
    if not empr:
        print("[WARN] Nenhum arquivo de Empresas reconhecido. "
              "Confirme que baixou 'Empresas.zip' (não Estabelecimentos ou Sócios).")
    return True


def processar_ceis() -> bool:
    print("\n" + "=" * 70)
    print(" [3/3] Portal da Transparência - CEIS")
    print("=" * 70)
    zip_path = localizar_zip(PASTA_RAW, ["ceis"])
    if not zip_path:
        print(INSTRUCAO_CEIS.format(pasta=PASTA_RAW))
        return False

    print(f"[OK]  ZIP encontrado: {zip_path}")
    arquivos = extrair_e_listar(zip_path, PASTA_CEIS)
    print(f"[OK]  {len(arquivos)} arquivo(s) extraído(s) em {PASTA_CEIS}:")
    for a in arquivos:
        tamanho = os.path.getsize(os.path.join(PASTA_CEIS, a))
        print(f"      - {a}  ({tamanho/1e6:.1f} MB)")
    return True


def main() -> None:
    print("=" * 70)
    print(" Preparação de Dados Públicos - Bureau de Crédito Verde")
    print(" TCC MBA USP/Esalq")
    print("=" * 70)

    os.makedirs(PASTA_RAW, exist_ok=True)
    print(f"\nPasta de ZIPs brutos: {PASTA_RAW}")

    ok_rfb = processar_rfb()
    ok_empresas = processar_empresas()
    ok_ceis = processar_ceis()

    print("\n" + "=" * 70)
    if ok_rfb and ok_empresas and ok_ceis:
        print(" Tudo pronto. Próximo passo: etl_receita.py → etl_ceis.py")
        sys.exit(0)
    else:
        print(" Faltam arquivos. Veja as instruções acima e rode de novo.")
        sys.exit(1)


if __name__ == "__main__":
    main()
