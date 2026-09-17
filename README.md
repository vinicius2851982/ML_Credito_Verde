# ML Crédito Verde

**Triagem socioambiental de micro e pequenas empresas (PMEs) para crédito sustentável/ESG via Machine Learning e dados públicos.**

## Sobre o Projeto

Trata-se de um TCC de MBA (USP/Esalq — Economia, Investimentos e Banking) que propõe uma arquitetura de **duas fases** para triagem automatizada de PMEs elegíveis para crédito verde/ESG:

1. **Fase 1 — Rubrica de Enquadramento:** ancorada na Taxonomia Verde da FEBRABAN, classifica empresas por atividade (o que faz) e conduta (como se comporta — dados de IBAMA, CEIS, equidade trabalhista).

2. **Fase 2 — Classificação Supervisionada:** modelos de ML (XGBoost, LogReg) preveem o enquadramento usando **apenas dados cadastrais públicos** (porte, capital social, tempo de empresa, CNAE).

**Escopo:** ~2,76 milhões de PMEs da Receita Federal, com modelo de dados ESG abrangendo eixos Ambiental (E) e Social (S).

## Pipeline de Processamento

Os scripts estão em `Códigos/` e devem ser executados na seguinte ordem:

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

### Análise Descritiva & Resultados
- `analise_descritiva.py` — estatísticas descritivas (inicial e final)
- `analise_regional.py` — distribuição por Região/UF e igualdade de oportunidade (viés geográfico)
- `score_credito_verde.py` — converte rubrica em score 0–1000; simula carteira

### Apêndices & Consolidação
- `apendice_modelos.py` — comparação multi-modelo (Optuna + SHAP)
- `exportar_tabelas_excel.py` — **[Passo 16 obrigatório]** consolida todas as tabelas em `Tabelas_e_Graficos_TCC.xlsx`

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

### Não-Vazamento (Fase 2)
- SMOTE apenas no treino; teste intocado
- Features preditoras: só cadastrais (porte, capital, tempo, CNAE)
- Variáveis socioambientais que alimentam a rubrica: **proibidas** como preditoras

### Recorte & Vigência
- **Corte: 31/05/2026** (FEBRABAN Dez/2020)
- **Monitoramento: 01/06–31/10/2026** (novas edições FEBRABAN/TSB)

## Métricas-Chave (Resultados Preliminares)

| Dimensão | Valor |
|---|---|
| Amostra | 2.764.563 PMEs (AAS 10% RFB) |
| Economia verde | 424.091 (15,3%) |
| Fase 2 AUC (XGBoost, 4 classes) | 0,9188 |
| ROC binária (elegível × não) | 0,9555 |
| Aprovados (Verde-A/B) | 49,9% |
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
