"""
etl_fornecedores_verdes.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

[LEGADO — FORA DA CADEIA OFICIAL] (ver "20260605 - DECISOES_TCC - claude.md" §7)
  A classificacao da ATIVIDADE verde migrou para a Taxonomia Verde da FEBRABAN
  (subclasse CNAE, 1.331 itens, escopo E+S) em construir_base_febraban.py. O
  dicionario CNAE-verde de 6 divisoes BNDES abaixo (stg_cnae_verde.csv) cobria
  so ~0,5% da base e ignorava o eixo Social -> NAO e mais consumido pela rubrica.
  Mantido por rastreabilidade e como demonstracao CONCEITUAL do eixo de cadeia
  (criterio E6 / funcao pct_compras_verdes) ate haver dado de fornecedor por firma.

Criterio E6 — Cadeia de fornecimento sustentavel.

Produz dois artefatos, ancorados nas 6 categorias VERDES do Sustainability
Bond Framework do BNDES (Energia Renovavel, Eficiencia Energetica, Agua e
Saneamento, Prevencao e Controle de Poluicao, Transporte Limpo, Recursos
Naturais e Uso da Terra):

  (1) DICIONARIO CNAE-VERDE  -> stg_cnae_verde.csv
      Classifica a ATIVIDADE PROPRIA da empresa (CNAE da RFB) como verde.
      Usado pela rubrica como bonus de atividade verde (refina o criterio E6).
      Dado REAL: o CNAE de cada PME vem da Receita Federal.

  (2) FUNCAO % DE COMPRAS VERDES  -> pct_compras_verdes(notas)
      Mecanismo operacional do pre-enquadramento (plataforma 'Empreender
      Clima'): dada a lista de NOTAS DE COMPRA da PME (CNPJ/CNAE do
      fornecedor e/ou NCM do produto e valor), calcula o percentual de
      compras de fornecedores/insumos verdes (ex.: embalagem biodegradavel,
      reciclagem, energia renovavel).
      OBS DE INTEGRIDADE: nao ha NF-e publica por PME; a funcao e demonstrada
      com um EXEMPLO ILUSTRATIVO (claramente rotulado), NAO com dado de
      pesquisa. Em producao, a PME anexa as proprias notas no pre-enquadramento.

OUTPUT: Dados/stg_cnae_verde.csv
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS    = os.path.join(BASE_DIR, "Dados")
OUTPUT   = os.path.join(DADOS, "stg_cnae_verde.csv")

# ------------------------------------------------------------------
# (1) Dicionario CNAE-verde por DIVISAO (2 digitos), mapeado as 6
#     categorias verdes do BNDES. nivel: 'pleno' (atividade ambiental
#     inequivoca) ou 'parcial' (verde conforme tecnologia/certificacao).
#     peso_bonus = pontos somados ao pilar E na rubrica (afirmativo).
# ------------------------------------------------------------------
CNAE_VERDE_DIVISAO = {
    # divisao : (categoria_bndes, nivel, peso_bonus)
    "36": ("Agua e Saneamento",            "pleno",   8.0),  # captacao/tratamento de agua
    "37": ("Agua e Saneamento",            "pleno",   8.0),  # esgoto
    "38": ("Prevencao e Controle de Poluicao", "pleno", 8.0),  # residuos / reciclagem
    "39": ("Prevencao e Controle de Poluicao", "pleno", 8.0),  # descontaminacao
    "35": ("Energia Renovavel/Eficiencia", "parcial", 4.0),  # energia eletrica/gas (porcao renovavel)
    "02": ("Recursos Naturais e Uso da Terra", "parcial", 3.0),  # producao florestal (manejo sustentavel)
}
# NOTA: transporte rodoviario (div. 49) NAO entra para evitar super-credito
# ao frete a diesel; 'transporte limpo' exige distincao por classe/tecnologia
# (ferroviario, eletrico), refinamento futuro. Energia (35) e agro/floresta
# (02) entram como 'parcial' pois so a porcao renovavel/sustentavel e elegivel.

# ------------------------------------------------------------------
# (2) NCM-verde (prefixos ILUSTRATIVOS p/ a demonstracao do mecanismo).
#     Em producao, usar a Taxonomia Sustentavel Brasileira por NCM.
# ------------------------------------------------------------------
NCM_VERDE_PREFIXOS = {
    "4819": "Embalagem de papel/papelao (reciclavel)",
    "3923": "Embalagem plastica (avaliar biodegradavel/reciclada)",
    "4707": "Papel/cartao para reciclagem",
    "8541": "Celulas/paineis fotovoltaicos",
    "8502": "Grupos geradores (eolico/renovavel)",
}


def construir_dicionario_cnae_verde() -> pd.DataFrame:
    rows = []
    for div, (cat, nivel, peso) in CNAE_VERDE_DIVISAO.items():
        rows.append({"cnae_divisao": div, "categoria_bndes": cat,
                     "nivel_verde": nivel, "peso_bonus": peso, "flag_verde": 1})
    df = pd.DataFrame(rows).sort_values("cnae_divisao").reset_index(drop=True)
    return df


def pct_compras_verdes(notas: pd.DataFrame,
                       col_cnae: str = "cnae_fornecedor",
                       col_ncm: str = "ncm",
                       col_valor: str = "valor") -> dict:
    """
    Dada a lista de notas de compra da PME, retorna metricas de cadeia verde.
    'verde' = fornecedor com CNAE em divisao verde OU produto com NCM verde.
    Retorna % por valor e por quantidade de notas.
    """
    d = notas.copy()
    div_verde = set(CNAE_VERDE_DIVISAO.keys())
    ncm_verde = tuple(NCM_VERDE_PREFIXOS.keys())

    def _div(x):
        return str(x).replace(".", "").zfill(2)[:2] if pd.notna(x) else ""
    cnae_ok = d[col_cnae].map(_div).isin(div_verde) if col_cnae in d else False
    ncm_ok  = d[col_ncm].astype(str).str.startswith(ncm_verde) if col_ncm in d else False
    d["verde"] = (cnae_ok | ncm_ok).astype(int)

    valor = d[col_valor] if col_valor in d else pd.Series(1.0, index=d.index)
    total_val = valor.sum()
    return {
        "n_notas": int(len(d)),
        "n_notas_verdes": int(d["verde"].sum()),
        "pct_notas_verdes": round(d["verde"].mean(), 4) if len(d) else 0.0,
        "pct_valor_verde": round(float((valor * d["verde"]).sum() / total_val), 4)
                           if total_val else 0.0,
    }


def _exemplo_ilustrativo() -> None:
    """EXEMPLO ILUSTRATIVO — nao e dado de pesquisa. Apenas demonstra o calculo."""
    notas = pd.DataFrame({
        "fornecedor":     ["A", "B", "C", "D", "E"],
        "cnae_fornecedor":["3811200", "4711301", "3530100", "4646002", "3839401"],
        "ncm":            ["4819000", "2106909", "8541430", "3304990", "3915000"],
        "valor":          [1000.0, 5000.0, 8000.0, 2000.0, 1500.0],
    })
    res = pct_compras_verdes(notas)
    print("  [EXEMPLO ILUSTRATIVO — nao e dado de pesquisa]")
    print(f"    Notas: {res['n_notas']} | verdes: {res['n_notas_verdes']} "
          f"| % notas verdes: {res['pct_notas_verdes']:.1%} "
          f"| % valor verde: {res['pct_valor_verde']:.1%}")


def main() -> None:
    print("=" * 64)
    print(" DICIONARIO CNAE/NCM-VERDE (criterio E6) — base: BNDES")
    print("=" * 64)
    df = construir_dicionario_cnae_verde()
    df.to_csv(OUTPUT, sep=";", index=False, encoding="utf-8-sig")
    print(f"[OK] {len(df)} divisoes CNAE verdes mapeadas as categorias BNDES:")
    for _, r in df.iterrows():
        print(f"    {r['cnae_divisao']}  {r['nivel_verde']:8s} (+{r['peso_bonus']:.0f})  {r['categoria_bndes']}")
    print(f"\n  Salvo: {OUTPUT}")
    print("\n  Demonstracao da funcao de % de compras verdes (pre-enquadramento):")
    _exemplo_ilustrativo()
    print("\nProximo passo: rubrica_enquadramento.py (le stg_cnae_verde.csv no bonus E6)")


if __name__ == "__main__":
    main()
