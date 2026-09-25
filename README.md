# ML Crédito Verde

**Triagem socioambiental de micro e pequenas empresas (PMEs) para crédito sustentável/ESG via Machine Learning e dados públicos.**

## Sobre o Projeto

Trata-se de um TCC de MBA (USP/Esalq — Economia, Investimentos e Banking) que propõe uma arquitetura de **duas fases** para triagem automatizada de PMEs elegíveis para crédito verde/ESG:

1. **Fase 1 — Rubrica de Enquadramento:** ancorada na Taxonomia Verde da FEBRABAN, classifica empresas por atividade (o que faz) e conduta (como se comporta — dados de IBAMA, CEIS, equidade trabalhista).

2. **Fase 2 — Classificação Supervisionada:** modelos de ML (XGBoost, LogReg) preveem o enquadramento usando **apenas dados cadastrais públicos** (porte, capital social, tempo de empresa, CNAE).

**Escopo:** ~2,76 milhões de PMEs da Receita Federal, com modelo de dados ESG abrangendo eixos Ambiental (E) e Social (S).

## Pipeline de Processamento

Os scripts estão em `Códigos/` e devem ser executados na seguinte ordem:

### Recorte temporal (base de tudo)
- `recorte.py` — **fonte única das datas de corte**. Define o recorte normativo (31/05/2026) e o corte dos dados (31/08/2026), além da função de filtro aplicada nos ETLs. As bases do IBAMA e do CEIS são cumulativas: sem esse filtro, entrariam fatos posteriores ao recorte declarado.

### Coleta automatizada
- `atualizar_dados_ibama.py` — baixa autos de infração e termos de embargo resolvendo as URLs pela API CKAN do IBAMA; grava manifesto de proveniência (URL, tamanho, data de publicação)
- `consolidar_ibama.py` — consolida os CSV anuais do IBAMA num arquivo único, descartando anos posteriores ao corte

> **Fontes que exigem download manual:** Receita Federal (repositório migrado para Nextcloud privado, retorna 404) e CEIS/CGU (protegido por WAF com CAPTCHA, retorna 403).

### Etapas de Coleta & Transformação (ETLs)
- `etl_ibama.py` — autuações e embargos (IBAMA)
- `etl_receita.py` — dados do cadastro (RFB)
- `etl_ceis.py` — sanções de integridade (CGU)
- `etl_social.py` — métricas trabalhistas e equidade (RAIS/CAGED)
- `etl_b3_esg.py` — benchmark setorial (ISE/B3)
- `etl_dados_abertos.py` — contexto territorial (IDHM, etc.)

### Integração & Taxonomia
- `preparar_dados_publicos.py` — organiza os downloads de dados públicos
- `merge_bases.py` — integra todas as ETLs em base analítica única
- `construir_base_febraban.py` — liga a taxonomia FEBRABAN (CNAE → atividade verde)

### Modelo (Fase 1 — Rótulo Oficial)
- `rubrica_enquadramento.py` — calcula score E e S; gera faixa (Verde-A/B, Amarelo, Vermelho)

### Análise Exploratória
- `clustering.py` — agrupamento K-Means (descritivo, não alimenta rótulo)

### Modelo (Fase 2 — Predição Cadastral)
- `classificacao.py` — LogReg + XGBoost com features cadastrais (porte, capital, tempo, CNAE)
- `roc_binaria.py` — análise de elegibilidade binária (Verde × não-Verde)

### Fundamentação amostral
- `analise_amostral.py` — apura os parâmetros da população varrendo a base bruta da RFB, testa a aderência da amostra por região/setor/porte (qui-quadrado e V de Cramér) e dimensiona o erro por estrato
- `estimativas_populacionais.py` — aplica o peso amostral e expande os resultados para a população, com IC 95% e correção para população finita

### Análise Descritiva & Resultados
- `analise_descritiva.py` — estatísticas descritivas (inicial e final)
- `analise_regional.py` — distribuição por Região/UF e igualdade de oportunidade (viés geográfico)
- `score_credito_verde.py` — converte rubrica em score 0–1000; simula carteira

### Apêndices & Consolidação
- `apendice_modelos.py` — comparação multi-modelo (Optuna + SHAP)
- `exportar_tabelas_excel.py` — **[Passo obrigatório]** consolida todas as tabelas em `Tabelas_e_Graficos_TCC.xlsx`
- `gerar_figuras_tcc.py` — figuras do documento final em escala de cinza, 300 dpi

### Legado
- `etl_fornecedores_verdes.py` — [OBSOLETO] dicionário CNAE-verde anterior (6 divisões BNDES)
- `diagnostico_ftp.py` — [OBSOLETO] utilitário de diagnóstico

## Requisitos

- **Python 3.8+**
- **pandas, numpy, scikit-learn, xgboost, lightgbm, catboost**
- **openpyxl** (exportação Excel)
- **matplotlib, seaborn** (gráficos)
- **optuna, shap** (apêndice multi-modelo)
- **statsmodels** (validações estatísticas)

Instale com:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost openpyxl matplotlib seaborn optuna shap statsmodels
```

## Estrutura de Dados

### Base Analítica (insumo)
Saída: `base_analitica_febraban.csv` (~2,76M linhas, 42 colunas)
- CNPJ, porte, CNAE, capital social, data de início
- Marcações FEBRABAN (economia verde, exposição)
- IBAMA, CEIS, equidade setorial (por CNAE)

### Base Pontuada (resultado)
Saída: `base_analitica_scored.csv` (~2,76M linhas, 52 colunas)
- Tudo anterior +
- Scores E e S (0–100)
- Faixa (Verde-A/B/Amarelo/Vermelho/Vetado/Fora de escopo)
- Score socioambiental (0–1000)
- Previsões Fase 2 (probabilidade, classe)

### Tabelas de Resultado
Consolidadas em: `Tabelas_e_Graficos_TCC.xlsx` (12 abas)
- T1–T10: tabelas do Resultado Preliminar
- Apoio_UF: distribuição por Unidade da Federação (27 UFs)
- Leia-me: instruções e origem dos dados

## Metodologia-Chave

### Duas Camadas (Anti-Dupla-Contagem)
- **ATIVIDADE** (FEBRABAN, por CNAE) ≠ **CONDUTA** (comportamento firma-a-firma)
- Eixo Social da FEBRABAN = propósito social da atividade (hospital, escola)
- Pilar S de conduta = gap salarial, equidade, CEIS (como a firma se comporta)

### Não-Vazamento e partição (Fase 2)
- Partição estratificada **60/20/20** — treino ajusta parâmetros, validação seleciona o modelo, teste é usado uma única vez
- SMOTE apenas no treino; validação e teste preservam a prevalência real
- O ponto de corte da decisão binária é definido na **validação** e apenas aplicado ao teste
- Features preditoras: só cadastrais (porte, capital, tempo, CNAE)
- Variáveis socioambientais que alimentam a rubrica: **proibidas** como preditoras

### Recortes (dois, distintos)
- **Normativo: 31/05/2026** — taxonomia FEBRABAN (versão Dez/2020), TSB, PRSAC
- **Dados: 31/08/2026** — posição das bases públicas
- A FEBRABAN substituiu a Taxonomia Verde pela Taxonomia de Finanças Sustentáveis em julho/2026, **posteriormente** ao recorte normativo — a versão utilizada era a vigente

## Métricas-Chave (Resultados Preliminares)

| Dimensão | Valor |
|---|---|
| População | 27.647.482 estabelecimentos ativos |
| Amostra | 2.764.563 (fração 9,9993%; erro ±0,06 p.p.) |
| Economia verde | 424.091 (15,3%) |
| Fase 2 AUC (XGBoost) | 0,9193 validação / 0,9188 teste |
| ROC binária (elegível × não) | 0,9555 |
| Aprovados (Verde-A/B) | 49,9% — estimados 13,78 mi na população |
| Equidade regional | Sem penalidade; N/NE aprovam **mais** que Sudeste |

## Como Usar

1. **Preparar dados públicos:**
   ```bash
   python Códigos/preparar_dados_publicos.py
   ```

2. **Rodar ETLs em paralelo** (ou sequencial):
   ```bash
   python Códigos/etl_ibama.py
   python Códigos/etl_receita.py
   # ... demais ETLs
   ```

3. **Integrar e construir base com FEBRABAN:**
   ```bash
   python Códigos/merge_bases.py
   python Códigos/construir_base_febraban.py
   ```

4. **Calcular rubrica (Fase 1):**
   ```bash
   python Códigos/rubrica_enquadramento.py
   ```

5. **Treinar modelos (Fase 2):**
   ```bash
   python Códigos/classificacao.py
   python Códigos/roc_binaria.py
   ```

6. **Análises & Resultados:**
   ```bash
   python Códigos/analise_descritiva.py base_analitica_febraban.csv inicial
   python Códigos/analise_regional.py
   python Códigos/score_credito_verde.py
   python Códigos/apendice_modelos.py
   ```

7. **Consolidar para Excel (obrigatório):**
   ```bash
   python Códigos/exportar_tabelas_excel.py
   ```

## Limitações Declaradas

- **S setorial:** métricas trabalhistas por CNAE×porte (não individuais); dados MTE são anonimizados
- **Calibração:** probabilidades não calibradas (ranqueadas); árvore + SMOTE descalibra
- **Rótulo:** construção do autor; não há ground truth de elegibilidade verde PME
- **Temporal:** recorte é um único instante (31/05/2026); generalização no tempo não testada
- **Amostra:** AAS 10% dos estabelecimentos ativos (RFB)
- **Bônus IDH:** inativo nesta rodada (SIAFI vs IBGE code mismatch)

## Agentes (Claude Code)

O projeto inclui subagentes especializados (para análise/defesa da tese, se usado em Claude Code):
- `mentor-python-ml` — tutor SAS→Python, rigor estatístico
- `regulacao-bacen` — âncora normativa (PRSAC, TSB, FEBRABAN)
- `banca-redteam` — red team (7 vetores de crítica à metodologia)

## Referências

- **FEBRABAN Taxonomia Verde** (Dez/2020): https://www.febraban.org.br
- **TSB — Taxonomia Sustentável Brasileira** (Decreto 12.705/2025): https://www.gov.br
- **PRSAC** (Resolução CMN 4.945/2021): https://www.bcb.gov.br
- **Green Bond Principles** (ICMA): https://www.icmagroup.org
- **Green Loan Principles** (LMA): https://www.lma.eu.com
- **Empreender Clima** (COP30): https://www.gov.br/empreender-clima

## Autor

**Hélio Vinícius Moreira Ribeiro**  
TCC — MBA em Economia, Investimentos e Banking  
Universidade de São Paulo (USP) / Escola Superior de Agricultura Luiz de Queiroz (Esalq)  
Orientadora: Profa. Dra. Rafaela Carvalho Pinheiro

Junho de 2026

---

**Nota:** Este repositório contém **apenas os códigos Python do pipeline**. A documentação metodológica, decisões de projeto e relatórios detalhados estão no trabalho acadêmico (TCC) original.
