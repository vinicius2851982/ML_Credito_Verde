"""
etl_dados_abertos.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Coleta bases abertas COMPLEMENTARES (item 1 do Resultado Preliminar).
Entram como CONTEXTO descritivo / benchmark, NAO como features do modelo
(coerente com a mitigacao de vies geografico).

FONTES (APIs publicas):
  - IPEA (OData4)  : IDHM municipal (Atlas DH) -> contexto geografico;
                     habilita o bonus afirmativo de IDH na rubrica (futuro).
  - BNDES (CKAN)   : desembolsos a MPMEs (por porte e por produto) -> calibra
                     os setores/portes efetivamente financiados.
  - IBGE (Agregados v3) e BCB (SICOR): tentativa best-effort (contexto/benchmark
                     do credito rural verde); se o endpoint falhar, registra e segue.

OUTPUT (Dados/):
  stg_idhm_municipio.csv, stg_bndes_desembolsos_porte.csv,
  stg_bndes_desembolsos_produto.csv  (+ best-effort: stg_pnad_regional / stg_sicor)
"""

import os
import ssl
import json
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS    = os.path.join(BASE_DIR, "Dados")

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
HDR = {"User-Agent": "Mozilla/5.0 (TCC-CreditoVerde)"}


def _get_json(url, timeout=90):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.load(r)


def _download(url, destino, timeout=120):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        dados = r.read()
    with open(destino, "wb") as f:
        f.write(dados)
    return len(dados)


# ------------------------------------------------------------------
def ipea_idhm():
    """IDHM municipal (ano mais recente: 2010) -> stg_idhm_municipio.csv."""
    print("[IPEA] IDHM municipal (Atlas DH) ...")
    url = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='ADH_IDHM')"
    v = _get_json(url).get("value", [])
    # mantem o ano mais recente por municipio
    melhor = {}
    for x in v:
        ter = str(x.get("TERCODIGO", "")).strip()
        ano = str(x.get("VALDATA", ""))[:4]
        val = x.get("VALVALOR")
        if not ter or val is None:
            continue
        if ter not in melhor or ano > melhor[ter][0]:
            melhor[ter] = (ano, val)
    out = os.path.join(DADOS, "stg_idhm_municipio.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        f.write("municipio_ibge;ano;idhm\n")
        for ter, (ano, val) in sorted(melhor.items()):
            f.write(f"{ter};{ano};{val}\n")
    print(f"       {len(melhor):,} municipios | salvo: {os.path.basename(out)}")
    return len(melhor)


def bndes_desembolsos_mpme():
    """Baixa tabelas agregadas de desembolsos a MPMEs (por porte e por produto)."""
    print("[BNDES] desembolsos a MPMEs (CKAN) ...")
    meta = _get_json("https://dadosabertos.bndes.gov.br/api/3/action/package_show?id=desembolsos-mpme")
    recursos = meta["result"]["resources"]
    alvos = {
        "por porte de empresa - desembolsos mpme": "stg_bndes_desembolsos_porte.csv",
        "(indiretas) e produto - desembolsos mpme": "stg_bndes_desembolsos_produto.csv",
    }
    feitos = set()
    for r in recursos:
        if str(r.get("format", "")).upper() != "CSV":
            continue
        nome = str(r.get("name", "")).lower()
        for chave, destino in alvos.items():
            if chave in nome and destino not in feitos:
                try:
                    n = _download(r["url"], os.path.join(DADOS, destino))
                    print(f"       {destino} ({n/1024:.0f} KB)")
                    feitos.add(destino)
                except Exception as e:
                    print(f"       [ERR] {destino}: {e}")
    return len(feitos)


def ibge_pnad_regional():
    """Best-effort: rendimento medio mensal real (PNAD Continua) por UF — agregado IBGE."""
    print("[IBGE] PNAD Continua — rendimento por UF (best-effort) ...")
    # Agregado 5439 (rendimento medio real) / variavel; pode mudar. Tenta e segue.
    cands = [
        "https://servicodados.ibge.gov.br/api/v3/agregados/6387/periodos/-1/variaveis?localidades=N3[all]",
        "https://servicodados.ibge.gov.br/api/v3/agregados/5439/periodos/-1/variaveis?localidades=N3[all]",
    ]
    for url in cands:
        try:
            d = _get_json(url, timeout=60)
            out = os.path.join(DADOS, "stg_pnad_regional.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            print(f"       salvo: {os.path.basename(out)} (agregado respondeu)")
            return 1
        except Exception as e:
            print(f"       [skip] {url.split('agregados/')[1][:8]}: {type(e).__name__}")
    print("       PNAD nao coletada (endpoint a confirmar) — registrar como pendencia.")
    return 0


def bcb_sicor():
    """Best-effort: SICOR credito rural verde (Olinda). Varios nomes de servico possiveis."""
    print("[BCB] SICOR credito rural (best-effort) ...")
    cands = [
        "https://olinda.bcb.gov.br/olinda/servico/SICOR_OutrasOpFinanceiras/versao/v1/odata/",
        "https://olinda.bcb.gov.br/olinda/servico/SICOR/versao/v1/odata/",
        "https://olinda.bcb.gov.br/olinda/servico/SICOR_DADOS_ABERTOS/versao/v2/odata/",
    ]
    for url in cands:
        try:
            d = _get_json(url, timeout=40)
            recs = [v.get("name") or v.get("url") for v in d.get("value", [])]
            svc = url.split("servico/")[1].split("/")[0]
            print(f"       servico OK: {svc} | {len(recs)} recursos")
            # baixa uma amostra de um recurso sem filtros, se houver
            recurso = next((r for r in recs if r and "SemFiltros" in r), recs[0] if recs else None)
            if recurso:
                amostra = f"{url}{recurso}?$top=1000&$format=json"
                try:
                    dd = _get_json(amostra, timeout=90).get("value", [])
                    out = os.path.join(DADOS, "stg_sicor_amostra.json")
                    with open(out, "w", encoding="utf-8") as f:
                        json.dump(dd, f, ensure_ascii=False)
                    print(f"       amostra '{recurso}' ({len(dd)} linhas) -> {os.path.basename(out)}")
                    return 1
                except Exception as e:
                    print(f"       servico OK mas amostra falhou ({type(e).__name__}); recursos: {recs[:5]}")
                    return 1
        except Exception as e:
            print(f"       [skip] {url.split('servico/')[1].split('/')[0]}: {type(e).__name__}")
    print("       SICOR via Olinda nao confirmado nesta sessao — alternativa: CSV da Matriz")
    print("       de Dados do Credito Rural (MDCR) no site do BCB. Registrar como pendencia.")
    return 0


def main():
    print("=" * 64)
    print(" ETL DADOS ABERTOS COMPLEMENTARES (contexto — nao-feature)")
    print("=" * 64)
    res = {}
    for nome, fn in [("IPEA IDHM", ipea_idhm), ("BNDES MPME", bndes_desembolsos_mpme),
                     ("IBGE PNAD", ibge_pnad_regional), ("BCB SICOR", bcb_sicor)]:
        try:
            res[nome] = fn()
        except Exception as e:
            print(f"  [ERRO] {nome}: {type(e).__name__} {str(e)[:80]}")
            res[nome] = 0
    print("\n" + "=" * 64)
    print(" RESUMO DA COLETA")
    print("=" * 64)
    for k, v in res.items():
        status = "OK" if v else "PENDENTE"
        print(f"  {k:12s}: {status}")
    print("\n  Uso: contexto descritivo e benchmark (NAO features do modelo).")
    print("  IDHM municipal habilita o bonus geografico afirmativo (requer crosswalk")
    print("  codigo RFB->IBGE do municipio para ligar a base).")


if __name__ == "__main__":
    main()
