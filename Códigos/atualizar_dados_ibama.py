"""
atualizar_dados_ibama.py  (coleta automatizada — IBAMA)
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Baixa os dados abertos do IBAMA direto do repositorio de origem, resolvendo
as URLs pela API CKAN do portal (as URLs finais ficam em blob storage e
mudam; resolver pela API evita link quebrado).

POR QUE ESTE SCRIPT EXISTE
  O portal do IBAMA passou a responder 403/502 em acesso direto ao diretorio,
  mas a API CKAN permanece aberta e entrega as URLs dos arquivos. Assim, esta
  fonte CONTINUA automatizavel — ao contrario de RFB e CEIS, que exigem
  download manual (repositorio privado e protecao por CAPTCHA).

CORTE TEMPORAL
  Os arquivos do IBAMA sao CUMULATIVOS: a versao publicada hoje contem
  registros posteriores ao recorte do estudo. Por isso o download guarda a
  data de referencia e o ETL deve filtrar por DATA_CORTE — do contrario a
  base incluiria fatos posteriores a 31/08/2026 e violaria o recorte.

Uso:
    python -B Códigos/atualizar_dados_ibama.py
    python -B Códigos/atualizar_dados_ibama.py --verificar   # so consulta

OUTPUT:
  Dados/raw/ibama/auto_infracao_csv.zip
  Dados/raw/ibama/termo_de_embargo.csv
  Dados/raw/ibama/_manifesto_download.json
"""

import os
import sys
import json
import time
import urllib.request as req
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTINO = os.path.join(BASE_DIR, "Dados", "raw", "ibama")
MANIFESTO = os.path.join(DESTINO, "_manifesto_download.json")

API = "https://dadosabertos.ibama.gov.br/api/3/action/package_show?id={}"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Recurso principal de cada dataset, identificado pelo final da URL.
ALVOS = {
    "auto_infracao": {
        "dataset": "fiscalizacao-auto-de-infracao",
        "sufixo": "auto_infracao/auto_infracao_csv.zip",
        "arquivo": "auto_infracao_csv.zip",
    },
    "termo_embargo": {
        "dataset": "fiscalizacao-termo-de-embargo",
        "sufixo": "TERMO_EMBARGO/termo_de_embargo.csv",
        "arquivo": "termo_de_embargo.csv",
        # O arquivo principal de embargos NAO esta catalogado na API (ela lista
        # apenas os complementares: itens, coordenadas, anexos, enquadramento).
        # A URL abaixo foi verificada por requisicao direta e serve de fallback.
        "url_direta": ("https://stibamadadosabertosprd.blob.core.windows.net/"
                       "dados-abertos/dados/TERMOS_DE_EMBARGO/TERMO_EMBARGO/"
                       "termo_de_embargo.csv"),
    },
}

DATA_CORTE = "2026-08-31"   # registros posteriores devem ser descartados no ETL


def resolver(dataset, sufixo):
    """Descobre a URL vigente do recurso pela API CKAN."""
    with req.urlopen(req.Request(API.format(dataset), headers=UA), timeout=60) as r:
        pacote = json.load(r)["result"]
    for recurso in pacote.get("resources", []):
        url = recurso.get("url", "")
        if url.endswith(sufixo):
            return url, pacote.get("metadata_modified", "")
    return None, pacote.get("metadata_modified", "")


def cabecalho(url):
    with req.urlopen(req.Request(url, headers=UA, method="HEAD"), timeout=90) as r:
        return (int(r.headers.get("Content-Length", 0)),
                r.headers.get("Last-Modified", "?"))


def baixar(url, destino, tamanho_esperado):
    """Download em streaming com barra de progresso simples."""
    tmp = destino + ".parcial"
    baixado = 0
    inicio = time.time()
    with req.urlopen(req.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        while True:
            bloco = r.read(1 << 20)          # 1 MiB
            if not bloco:
                break
            f.write(bloco)
            baixado += len(bloco)
            if tamanho_esperado:
                pct = 100 * baixado / tamanho_esperado
                vel = baixado / max(time.time() - inicio, 0.1) / 1e6
                print(f"\r      {pct:5.1f}%  {baixado/1e6:7.1f} MB  ({vel:.1f} MB/s)",
                      end="", flush=True)
    print()
    if tamanho_esperado and baixado < tamanho_esperado * 0.95:
        os.remove(tmp)
        raise IOError(f"download incompleto: {baixado:,} de {tamanho_esperado:,} bytes")
    os.replace(tmp, destino)
    return baixado


def main():
    somente_verificar = "--verificar" in sys.argv
    os.makedirs(DESTINO, exist_ok=True)
    print("=" * 68)
    print(" ATUALIZACAO DOS DADOS DO IBAMA")
    print(f" Recorte do estudo: ate {DATA_CORTE} (filtro aplicado no ETL)")
    print("=" * 68)

    manifesto = {"baixado_em": datetime.now(timezone.utc).isoformat(),
                 "data_corte": DATA_CORTE, "arquivos": {}}

    for nome, cfg in ALVOS.items():
        print(f"\n[{nome}]")
        try:
            url, modificado = resolver(cfg["dataset"], cfg["sufixo"])
        except Exception as e:
            print(f"   ERRO ao consultar a API: {type(e).__name__}: {e}")
            url, modificado = None, ""
        if not url and cfg.get("url_direta"):
            url = cfg["url_direta"]
            print("   recurso ausente no catalogo da API — usando URL direta verificada")
        if not url:
            print("   recurso nao encontrado — verifique o sufixo configurado")
            continue

        # O catalogo do IBAMA contem URLs obsoletas que respondem 404. Quando o
        # HEAD falha, cai-se para a URL direta verificada.
        tamanho = last_mod = None
        for tentativa in (url, cfg.get("url_direta")):
            if not tentativa or (tentativa == url and tamanho is not None):
                continue
            try:
                tamanho, last_mod = cabecalho(tentativa)
                url = tentativa
                break
            except Exception as e:
                print(f"   indisponivel ({type(e).__name__}): {tentativa[-58:]}")
        if tamanho is None:
            print("   nenhuma URL respondeu — verifique o catalogo do IBAMA")
            continue

        print(f"   origem     : {url}")
        print(f"   tamanho    : {tamanho/1e6:,.1f} MB")
        print(f"   publicado  : {last_mod}")
        print(f"   dataset mod: {modificado[:10]}")

        destino = os.path.join(DESTINO, cfg["arquivo"])
        if somente_verificar:
            print("   (--verificar: download nao realizado)")
            continue
        if os.path.exists(destino) and abs(os.path.getsize(destino) - tamanho) < 1024:
            print("   ja presente e com o mesmo tamanho — pulando")
        else:
            print("   baixando ...")
            baixar(url, destino, tamanho)
            print(f"   salvo em   : {destino}")

        manifesto["arquivos"][nome] = {
            "url": url, "arquivo": cfg["arquivo"],
            "bytes": tamanho, "last_modified": last_mod,
            "dataset_modificado": modificado,
        }

    if not somente_verificar and manifesto["arquivos"]:
        with open(MANIFESTO, "w", encoding="utf-8") as f:
            json.dump(manifesto, f, ensure_ascii=False, indent=2)
        print(f"\nManifesto salvo: {MANIFESTO}")
        print("\nO manifesto registra a proveniencia (URL, tamanho e data de")
        print("publicacao) de cada arquivo — util para a secao de Metodologia.")

    print("\n" + "=" * 68)
    print(" FONTES QUE NAO PERMITEM COLETA AUTOMATIZADA")
    print("=" * 68)
    print(" Receita Federal (CNPJ): repositorio migrado para Nextcloud privado")
    print("   (SERPRO+); os caminhos publicos retornam 404. Baixar manualmente")
    print("   em dados.gov.br e salvar os ZIP em Dados/raw/.")
    print(" CEIS (CGU): protegido por WAF com CAPTCHA; o download direto")
    print("   retorna 403. Baixar manualmente no Portal da Transparencia.")
    print("\n Apos colocar os arquivos em Dados/raw/, rode preparar_dados_publicos.py")


if __name__ == "__main__":
    main()
