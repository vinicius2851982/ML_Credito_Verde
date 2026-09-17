"""
merge_bases.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Funde todas as camadas staging em um unico DataFrame analitico,
pronto para clustering e classificacao.

ARQUITETURA DE PILARES (conforme objetivo geral do TCC):
  E — Ambiental   : IBAMA autuacoes + IBAMA embargos
  S — Social      : RAIS Estabelecimentos + NOVO CAGED (etl_social.py)
  G — Governanca  : CEIS — NAO integra o modelo ML.
                    Tratado como FILTRO DE VETO / contrapartida contratual.

REGRA DE VETO (G):
  veto_governanca = 1  SE  embargado = 1  OU  ceis_sancoes_ativas > 0
  -> Score final = 0, empresa inelegivel ao credito verde.
  -> Excluida do clustering/classificacao (nao precisa de ML).

POPULACAO-ALVO:
  Micro Empresas (ME, RFB 01) e Empresas de Pequeno Porte (EPP, RFB 03).
  Medias empresas (50-249 vinculos via RAIS) sao mantidas com nota metodologica.
  Grandes empresas (>= 250 vinculos) sao excluidas do escopo.

BASES UTILIZADAS:
  Base            Chave    Features geradas
  --------------- -------- ------------------------------------------------
  RFB             CNPJ     cnae, porte, uf, municipio, capital_social
  IBAMA aut.      CNPJ     qtd_infracoes, valor_total_multas, anos_desde_ultima_infracao
  IBAMA emb.      CNPJ     embargado (flag binaria -> veto E)
  CEIS            CNPJ     ceis_sancoes_ativas (-> veto G | NAO entra no ML)
  Social (RAIS+   CNPJ     qtd_vinculos_ativos, salario_medio_sm,
  CAGED opcional)          saldo_empregos, rotatividade,
                           intensidade_emprego_cnae (proxy CNAE)

OUTPUT: Dados/base_analitica.csv
"""

import os
import pandas as pd
from datetime import date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS    = os.path.join(BASE_DIR, "Dados")

ARQUIVOS = {
    "rfb":       os.path.join(DADOS, "stg_rfb_consolidado.csv"),
    "ibama_aut": os.path.join(DADOS, "stg_ibama_infracoes_pmes.csv"),
    "ibama_emb": os.path.join(DADOS, "stg_ibama_embargos.csv"),
    "ceis":      os.path.join(DADOS, "stg_ceis.csv"),
    # Social e opcional: sera integrado apos etl_social.py concluir
    "social":    os.path.join(DADOS, "stg_social.csv"),
    # B3 ESG opcional: benchmark ESG setorial das listadas (etl_b3_esg.py)
    "b3_esg":    os.path.join(DADOS, "stg_b3_esg.csv"),
}
OUTPUT = os.path.join(DADOS, "base_analitica.csv")

HOJE = pd.Timestamp(date.today())

# Limite de vinculos para excluir grandes empresas (IBGE: >= 250 = grande)
LIMITE_GRANDE = 250


def verificar_inputs() -> None:
    obrigatorios = ["rfb", "ibama_aut", "ibama_emb", "ceis"]
    faltando = [k for k in obrigatorios if not os.path.exists(ARQUIVOS[k])]
    if faltando:
        raise FileNotFoundError(
            f"Arquivos ausentes: {faltando}\n"
            "Execute os ETLs correspondentes antes de rodar o merge."
        )
    social_ok = os.path.exists(ARQUIVOS["social"])
    print(f"[INFO] stg_social.csv {'ENCONTRADO' if social_ok else 'NAO ENCONTRADO — pillar S sera proxy CNAE apenas'}")
    return social_ok


def ler(chave: str) -> pd.DataFrame:
    caminho = ARQUIVOS[chave]
    df = pd.read_csv(caminho, sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)
    print(f"[OK]  {chave:12s}: {len(df):>10,} linhas | colunas: {list(df.columns)}")
    return df


def padronizar_cnpj(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].astype(str).str.strip().str.zfill(14)
    return df


def main() -> None:
    print("=" * 70)
    print(" MERGE DE BASES - Bureau de Credito Verde")
    print(" Pilares: E (Ambiental) + S (Social) | G = filtro de veto")
    print("=" * 70)

    social_disponivel = verificar_inputs()

    # ------------------------------------------------------------------
    # 1. Base RFB (ancora da analise)
    # ------------------------------------------------------------------
    print("\n[1/7] Carregando base RFB (ancora)...")
    rfb = ler("rfb")
    rfb = padronizar_cnpj(rfb, "CNPJ")
    rfb = rfb.drop_duplicates(subset=["CNPJ"])

    cols_rfb = [c for c in [
        "CNPJ", "cnpj_basico", "identificador", "cnae_fiscal_principal",
        "data_inicio_atividade", "uf", "municipio",
        "razao_social", "natureza_juridica", "capital_social", "porte",
    ] if c in rfb.columns]
    rfb = rfb[cols_rfb]

    # Porte: mantemos ME(01) + EPP(03) + "Demais"(05)
    # Grandes serao separadas apos join com RAIS (pelo n. de vinculos)
    if "porte" in rfb.columns:
        n_antes = len(rfb)
        rfb = rfb[rfb["porte"].str.strip().isin(["01", "03", "05"])]
        print(f"[OK]  {len(rfb):,} estabelecimentos ativos ME/EPP/Demais "
              f"(excluidos {n_antes - len(rfb):,} sem porte definido).")
    else:
        print(f"[OK]  {len(rfb):,} estabelecimentos ativos.")

    # ------------------------------------------------------------------
    # 2. IBAMA Autuacoes  [Pilar E]
    # ------------------------------------------------------------------
    print("\n[2/7] Juntando IBAMA - Autuacoes [Pilar E]...")
    ibama_aut = ler("ibama_aut")
    ibama_aut = padronizar_cnpj(ibama_aut, "CNPJ_LIMPO")
    ibama_aut = ibama_aut.rename(columns={"CNPJ_LIMPO": "CNPJ"})

    ibama_aut["infracao_mais_recente"] = pd.to_datetime(
        ibama_aut["infracao_mais_recente"], errors="coerce"
    )
    ibama_aut["anos_desde_ultima_infracao"] = (
        (HOJE - ibama_aut["infracao_mais_recente"]).dt.days / 365.25
    ).round(1)

    for col in ["qtd_infracoes", "valor_total_multas"]:
        ibama_aut[col] = pd.to_numeric(ibama_aut[col], errors="coerce")

    ibama_aut = ibama_aut[["CNPJ", "qtd_infracoes", "valor_total_multas",
                            "anos_desde_ultima_infracao"]]

    rfb = rfb.merge(ibama_aut, on="CNPJ", how="left")
    rfb[["qtd_infracoes", "valor_total_multas"]] = (
        rfb[["qtd_infracoes", "valor_total_multas"]].fillna(0)
    )
    match = rfb["qtd_infracoes"].gt(0).sum()
    print(f"[OK]  {match:,} estabelecimentos com historico de autuacao IBAMA.")

    # ------------------------------------------------------------------
    # 3. IBAMA Embargos  [Pilar E -> tambem aciona veto G]
    # ------------------------------------------------------------------
    print("\n[3/7] Juntando IBAMA - Embargos [Pilar E / Veto]...")
    ibama_emb = ler("ibama_emb")
    ibama_emb = padronizar_cnpj(ibama_emb, "CNPJ_LIMPO")
    ibama_emb = ibama_emb.rename(columns={"CNPJ_LIMPO": "CNPJ"})
    ibama_emb["embargado"] = pd.to_numeric(ibama_emb["embargado"], errors="coerce")

    rfb = rfb.merge(ibama_emb, on="CNPJ", how="left")
    rfb["embargado"] = rfb["embargado"].fillna(0).astype(int)
    print(f"[OK]  {rfb['embargado'].sum():,} estabelecimentos com embargo IBAMA ativo.")

    # ------------------------------------------------------------------
    # 4. CEIS  [Pilar G — VETO APENAS, nao entra no modelo ML]
    # ------------------------------------------------------------------
    print("\n[4/7] Juntando CEIS [Pilar G — somente filtro de veto]...")
    print("      ATENCAO: ceis_sancoes_ativas gera veto_governanca=1")
    print("               Esses registros serao excluidos do clustering/classificacao.")

    ceis = ler("ceis")
    ceis = padronizar_cnpj(ceis, "CNPJ_LIMPO")
    ceis = ceis.rename(columns={"CNPJ_LIMPO": "CNPJ"})

    # Seleciona apenas a coluna de sancoes ativas (unica necessaria para o veto)
    ceis_cols = ["CNPJ"]
    for col in ["ceis_sancoes_ativas", "ceis_total_sancoes", "ceis_sancionado"]:
        if col in ceis.columns:
            ceis[col] = pd.to_numeric(ceis[col], errors="coerce")
            ceis_cols.append(col)
    ceis = ceis[ceis_cols]

    rfb = rfb.merge(ceis, on="CNPJ", how="left")
    for col in ["ceis_sancoes_ativas", "ceis_total_sancoes", "ceis_sancionado"]:
        if col in rfb.columns:
            rfb[col] = rfb[col].fillna(0).astype(int)

    print(f"[OK]  {rfb.get('ceis_sancoes_ativas', pd.Series([0])).gt(0).sum():,} "
          f"estabelecimentos com sancoes CEIS ativas.")

    # ------------------------------------------------------------------
    # 4b. Veto de Governanca
    # ------------------------------------------------------------------
    ceis_ativas = rfb["ceis_sancoes_ativas"] if "ceis_sancoes_ativas" in rfb.columns \
        else pd.Series(0, index=rfb.index)
    rfb["veto_governanca"] = (
        (rfb["embargado"] == 1) | (ceis_ativas > 0)
    ).astype(int)
    n_veto = rfb["veto_governanca"].sum()
    print(f"\n[INFO] Veto de Governanca (embargado=1 OU ceis_sancoes_ativas>0): "
          f"{n_veto:,} empresas ({n_veto/len(rfb):.2%}) — score final = 0.")

    # ------------------------------------------------------------------
    # 5. Social — RAIS + CAGED  [Pilar S]
    # ------------------------------------------------------------------
    print("\n[5/7] Integrando dimensao Social [Pilar S]...")

    # Features Social setoriais (perfil por CNAE atribuido a cada CNPJ).
    # Fonte: etl_social.py (RAIS Vinculos + NOVO CAGED, janela 36 meses).
    SOCIAL_COLS = ["salario_medio_sm", "tempo_emprego_medio", "pct_vinculo_ativo",
                   "rotatividade", "saldo_setor_taxa", "intensidade_emprego_cnae",
                   # Equidade (S2/S4) — RAIS: gap salarial ajustado, chefia, dispersao
                   "gap_genero_ajustado_cnae", "gap_raca_ajustado_cnae",
                   "pct_chefia_feminina_cnae", "dispersao_salarial_cnae"]

    if social_disponivel:
        social = ler("social")
        social = padronizar_cnpj(social, "CNPJ_LIMPO")
        social = social.rename(columns={"CNPJ_LIMPO": "CNPJ"})

        cols_social = ["CNPJ"]
        for col in SOCIAL_COLS + ["cnae_origem_social"]:
            if col in social.columns:
                if col != "cnae_origem_social":
                    social[col] = pd.to_numeric(social[col], errors="coerce")
                cols_social.append(col)
        social = social[cols_social].drop_duplicates("CNPJ")

        rfb = rfb.merge(social, on="CNPJ", how="left")

        # Imputacao: CNAEs sem cobertura recebem a mediana global da feature
        for col in SOCIAL_COLS:
            if col in rfb.columns:
                mediana = rfb[col].median()
                rfb[col] = rfb[col].fillna(mediana if pd.notna(mediana) else 0)

        n_real = (rfb["cnae_origem_social"] == "subclasse").sum() \
            if "cnae_origem_social" in rfb.columns else 0
        print(f"[OK]  Social setorial integrado: {n_real:,} CNPJs com match por subclasse CNAE.")
    else:
        # Social ausente: cria colunas neutras para nao quebrar o clustering.
        print("[WARN] stg_social.csv ausente. Dimensao S ficara neutra (mediana).")
        print("       Execute etl_social.py e reprocesse o merge para dados completos.")
        for col in SOCIAL_COLS:
            rfb[col] = 0.0

    # NOTA METODOLOGICA — separacao medio/grande:
    # Nenhuma fonte publica do MTE traz numero de funcionarios por CNPJ
    # (RAIS/CAGED sao anonimizados por sigilo). A separacao fina medio/grande
    # por headcount exigiria RAIS identificado (acesso restrito). Mantem-se,
    # portanto, a classificacao por porte da Receita Federal (ME/EPP/Demais).

    # ------------------------------------------------------------------
    # 6. Limpeza final e tipagem
    # ------------------------------------------------------------------
    print("\n[6/7] Tipagem final e limpeza...")

    if "capital_social" in rfb.columns:
        rfb["capital_social"] = pd.to_numeric(rfb["capital_social"], errors="coerce").fillna(0)

    if "porte" in rfb.columns:
        rfb["porte"] = rfb["porte"].fillna("00").str.strip()

    if "cnae_fiscal_principal" in rfb.columns:
        rfb["cnae_divisao"] = rfb["cnae_fiscal_principal"].str.strip().str[:2]

    # ------------------------------------------------------------------
    # 6b. Benchmark ESG setorial B3 (ISE) — features E e S externas
    # ------------------------------------------------------------------
    if os.path.exists(ARQUIVOS["b3_esg"]) and "cnae_divisao" in rfb.columns:
        print("\n[6b]  Integrando benchmark ESG setorial B3 (ISE) por CNAE...")
        b3 = pd.read_csv(ARQUIVOS["b3_esg"], sep=";", dtype={"cnae_divisao": str},
                         encoding="utf-8-sig")
        b3["cnae_divisao"] = b3["cnae_divisao"].astype(str).str.zfill(2)
        cols_b3 = ["cnae_divisao", "esg_ambiental_setor", "esg_social_setor",
                   "esg_geral_setor"]
        cols_b3 = [c for c in cols_b3 if c in b3.columns]
        rfb["cnae_divisao"] = rfb["cnae_divisao"].astype(str).str.zfill(2)
        rfb = rfb.merge(b3[cols_b3], on="cnae_divisao", how="left")
        for c in ["esg_ambiental_setor", "esg_social_setor", "esg_geral_setor"]:
            if c in rfb.columns:
                med = rfb[c].median()
                rfb[c] = pd.to_numeric(rfb[c], errors="coerce").fillna(med if pd.notna(med) else 0)
        print(f"[OK]  Features B3 ESG integradas: esg_ambiental_setor [E], "
              f"esg_social_setor [S] (media E={rfb['esg_ambiental_setor'].mean():.3f}, "
              f"S={rfb['esg_social_setor'].mean():.3f}).")
    else:
        print("[WARN] stg_b3_esg.csv ausente — sem features B3 ESG. Rode etl_b3_esg.py.")

    # anos_desde_ultima_infracao: NaN = nunca autuado -> 30 anos (menor risco)
    rfb["anos_desde_ultima_infracao"] = rfb["anos_desde_ultima_infracao"].fillna(30.0)

    antes = len(rfb)
    rfb = rfb[rfb["CNPJ"].str.len() == 14]
    if antes - len(rfb):
        print(f"[WARN] {antes - len(rfb):,} linhas removidas por CNPJ invalido.")

    print(f"[OK]  Base analitica final: {len(rfb):,} registros | {len(rfb.columns)} colunas.")

    # ------------------------------------------------------------------
    # 7. Exporta
    # ------------------------------------------------------------------
    print(f"\n[7/7] Exportando para {OUTPUT} ...")
    rfb.to_csv(OUTPUT, sep=";", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(" RESUMO DA BASE ANALITICA")
    print("=" * 70)
    print(f"  Registros totais        : {len(rfb):,}")
    print(f"  Colunas                 : {len(rfb.columns)}")
    print(f"  Com autuacao IBAMA [E]  : {rfb['qtd_infracoes'].gt(0).sum():,} "
          f"({rfb['qtd_infracoes'].gt(0).mean():.1%})")
    print(f"  Embargados IBAMA [E]    : {rfb['embargado'].sum():,} "
          f"({rfb['embargado'].mean():.1%})")
    if "ceis_sancoes_ativas" in rfb.columns:
        print(f"  CEIS sancoes ativas [G] : {rfb['ceis_sancoes_ativas'].gt(0).sum():,} "
              f"({rfb['ceis_sancoes_ativas'].gt(0).mean():.1%})")
    print(f"  Veto governanca [G]     : {rfb['veto_governanca'].sum():,} "
          f"({rfb['veto_governanca'].mean():.1%}) — EXCLUIDOS DO MODELO ML")
    if social_disponivel and "salario_medio_sm" in rfb.columns:
        print(f"  Salario setorial medio [S]: {rfb['salario_medio_sm'].mean():.2f} SM")
        print(f"  Rotatividade media [S]    : {rfb['rotatividade'].mean():.3f}")
        if "porte" in rfb.columns:
            print(f"\n  Distribuicao por porte RFB:")
            porte_nome = {"01": "ME (Micro)", "03": "EPP (Pequena)", "05": "Demais (Media)"}
            for p, n in rfb["porte"].value_counts().items():
                print(f"    {porte_nome.get(p, p):16s}: {n:>10,} ({n/len(rfb):.1%})")
    print(f"\n  Salvo em: {OUTPUT}")
    print(f"\n  NOTA METODOLOGICA:")
    print(f"  - Colunas CEIS estao na base para auditoria mas NAO sao features do modelo ML.")
    print(f"  - veto_governanca=1 identifica empresas inelegiveis (score=0).")
    print(f"  - clustering.py e classificacao.py excluem veto_governanca=1 antes de treinar.")
    print(f"  - Pilar S e setorial (perfil CNAE via RAIS/CAGED), atribuido por CNAE da firma.")
    print(f"  - Separacao medio/grande usa porte RFB (headcount por CNPJ exige RAIS restrito).")
    print("\nProximo passo: clustering.py")


if __name__ == "__main__":
    main()
