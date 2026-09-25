"""
classificacao.py  (Fase 2 — classificacao supervisionada, aderente a PP V002)
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Stack EXATA da proposta (PP V002):
  - scikit-learn: train_test_split, LogisticRegression
  - xgboost: XGBClassifier (relacoes nao-lineares)
  - imbalanced-learn: SMOTE (balanceamento — evita falsos negativos)
  - metricas: accuracy, precision, recall, roc_auc + CURVA ROC para
    determinar o PONTO DE CORTE OTIMO (indice de Youden).

(Comparacao multi-modelo com Optuna/SHAP foi movida para apendice_modelos.py.)

POPULACAO: base_analitica_rotulada.csv, excluindo "Vetado" (veto de Governanca).

NOTA DE COERENCIA (PP V002):
  Foco em eficiencia operacional da triagem, NAO previsao de inadimplencia.
  O alvo Y vem do K-Means (Fase 1); o classificador o operacionaliza para
  aplicacao automatizada a novas PMEs. Governanca (G) = contrapartida
  contratual, fora do modelo. Vies geografico mitigado: UF/municipio NAO
  sao features (ver lista FEATURES_CORE).

OUTPUT:
  Dados/modelos/modelo_logistica.pkl, modelo_xgb.pkl, modelo_melhor.pkl
  Dados/relatorio_classificacao.txt
  Dados/plots/roc_curve.png, confusion_xgb.png, feature_importance.png
"""

import os
import gc
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS       = os.path.join(BASE_DIR, "Dados")
PLOTS_DIR   = os.path.join(DADOS, "plots")
MODELOS_DIR = os.path.join(DADOS, "modelos")

INPUT = os.path.join(DADOS, "base_analitica_rotulada.csv")
REL   = os.path.join(DADOS, "relatorio_classificacao.txt")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
SEED      = 42

# Particao em TRES conjuntos (60/20/20), estratificada pela faixa:
#   TREINO    -> ajusta os parametros do modelo
#   VALIDACAO -> compara modelos e escolhe o final (nunca reporta desempenho)
#   TESTE     -> tocado UMA vez, ao final, so para a estimativa nao-enviesada
# Separar validacao de teste evita que a selecao do modelo contamine a
# metrica reportada (selection bias). O teste nao participa de escolha alguma.
TEST_SIZE = 0.20
VAL_SIZE  = 0.20

# ------------------------------------------------------------------
# FEATURES — pre-triagem CADASTRAL barata (decisao: framework §6.5).
# A Fase 2 preve a FAIXA da rubrica (Fase 1) usando SO o que o banco tem no
# cadastro, ANTES de qualquer auditoria/proxy caro. Assim o AUC mede o valor
# real: "da para pre-triar a PME so com o cadastro?" (Metrica de Gestao).
# NAO usa os proxies E+S que alimentaram a rubrica (evita circularidade).
# Geografia (UF/municipio) CONTINUA FORA por mitigacao de vies — na rubrica
# entra apenas como bonus afirmativo (IDH baixo), nunca como preditor.
#
# DECISAO sobre as colunas fbb_* (Taxonomia FEBRABAN, base_analitica_febraban):
#   Sao derivadas do CNAE (cadastral, disponivel no balcao), logo seriam
#   candidatas legitimas a preditoras. PORÉM NENHUMA entra na Fase 2:
#   - fbb_economia_verde / fbb_ev_eixo / flag_economia_verde ALIMENTAM o bonus
#     da rubrica (Fase 1) que move o score -> a FAIXA (alvo Y). Usa-las como
#     preditoras seria VAZAMENTO/circularidade (o modelo "espiaria" o rotulo).
#   - fbb_risco_ambiental / fbb_clima sao EXPOSICAO (nao conduta) e hoje NAO
#     entram no rotulo (PENALIZAR_EXPOSICAO=False na rubrica), mas a rubrica
#     reserva o direito de modula-las; mante-las fora preserva a leitura limpa
#     do AUC ("da pra pre-triar a PME so com cadastro minimo?"). O sinal setorial
#     ja entra via cnae_divisao one-hot (a divisao contem as subclasses fbb_*).
#   Conclusao: Fase 2 segue CADASTRAL-MINIMA (porte, capital, tempo, CNAE).
# ------------------------------------------------------------------
FEATURES_NUM = ["porte_num", "capital_social_log", "tempo_empresa_anos"]
FEATURES_CAT = ["cnae_divisao"]          # one-hot (setor de atividade)

PORTE_MAP = {"00": 0, "01": 1, "03": 2, "05": 3}
ANOS_SEM_INFRACAO = 30.0
HOJE = pd.Timestamp(2026, 9, 1)   # dia seguinte ao corte dos dados (31/08/2026); ver recorte.py

NOMES = {
    "porte_num": "Porte (ordinal)", "capital_social_log": "Capital Social log",
    "tempo_empresa_anos": "Tempo de Empresa (anos)",
}
AZUL_ESCURO = "#1A237E"; AZUL_CLARO = "#BBDEFB"; VERMELHO = "#EF5350"


# ==================================================================
# DADOS
# ==================================================================

def carregar_base():
    print(f"[1/8] Carregando {INPUT} ...")
    df = pd.read_csv(INPUT, sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)
    print(f"      {len(df):,} registros | {len(df.columns)} colunas")
    return df


def filtrar_elegiveis(df):
    print("\n[2/8] Filtrando populacao classificavel...")
    # Na rubrica de enquadramento, veto/embargo viram PENALIZACAO compensavel
    # (a faixa ja reflete isso). So saem da classificacao supervisionada as
    # faixas que nao integram o gradiente verde-elegivel:
    #   - "Vetado"        : penalizacao severa (score < 0)
    #   - "Fora de escopo": CNAE categoricamente nao-elegivel (Anexo 2 BNDES)
    if "risco_label" in df.columns:
        antes = len(df)
        df = df[~df["risco_label"].isin(["Vetado", "Fora de escopo"])].copy()
        print(f"      Removidos (Vetado/Fora de escopo): {antes - len(df):,}")
    print(f"      Registros classificaveis: {len(df):,}")
    return df


def preparar_features(df):
    print("\n[3/8] Preparando features CADASTRAIS (pre-triagem barata)...")
    # capital social (log)
    df["capital_social"]     = pd.to_numeric(df.get("capital_social"), errors="coerce").fillna(0)
    df["capital_social_log"] = np.log1p(df["capital_social"])
    # porte ordinal
    df["porte_num"] = df["porte"].astype(str).str.strip().map(PORTE_MAP).fillna(0).astype(float)
    # tempo de empresa (anos) a partir de data_inicio_atividade (RFB: YYYYMMDD)
    s_data = df.get("data_inicio_atividade", pd.Series("", index=df.index)).astype(str)
    s_data = s_data.str.replace(r"[^0-9]", "", regex=True).str.slice(0, 8)
    dt = pd.to_datetime(s_data, format="%Y%m%d", errors="coerce")
    tempo = ((HOJE - dt).dt.days / 365.25)
    med_tempo = tempo[tempo > 0].median()
    df["tempo_empresa_anos"] = tempo.where(tempo > 0).fillna(med_tempo if pd.notna(med_tempo) else 5.0)
    # cnae_divisao (categorica -> one-hot)
    if "cnae_divisao" not in df.columns or df["cnae_divisao"].isna().all():
        df["cnae_divisao"] = df["cnae_fiscal_principal"].astype(str).str[:2]
    df["cnae_divisao"] = df["cnae_divisao"].astype(str).str.zfill(2)

    X_num = df[FEATURES_NUM].fillna(0).astype(np.float64)
    X_cat = pd.get_dummies(df["cnae_divisao"], prefix="cnae")
    X_df  = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    feats = list(X_df.columns)
    print(f"      Features ({len(feats)}): {len(FEATURES_NUM)} numericas + "
          f"{X_cat.shape[1]} divisoes CNAE (one-hot)")
    X = X_df.values.astype(np.float64)
    return df, X, feats


def codificar_target(df):
    print("\n[4/8] Codificando variavel-alvo Y (faixa da rubrica)...")
    # Faixas de enquadramento verde, do MELHOR (0) ao PIOR (3).
    # Verde-A = maior aderencia ao padrao verde -> melhores condicoes de credito.
    RANK = {"Verde-A": 0, "Verde-B": 1, "Amarelo": 2, "Vermelho": 3}
    rank = df["risco_label"].fillna("Amarelo").astype(str).map(RANK).fillna(2).astype(int)
    presentes = sorted(rank.unique())
    remap = {r: i for i, r in enumerate(presentes)}
    y = rank.map(remap).astype(int)
    rank_nome = {v: k for k, v in RANK.items()}
    class_names = [rank_nome[r] for r in presentes]
    print(f"      {len(class_names)} classes: {class_names}")
    for i, n in enumerate(class_names):
        print(f"        {i} ({n}): {(y==i).sum():,} ({(y==i).mean():.1%})")
    return y.values, class_names


# ==================================================================
# METRICAS / ROC / CORTE OTIMO
# ==================================================================

def metricas(y_true, y_pred, y_proba, n_classes):
    avg = "binary" if n_classes == 2 else "macro"
    auc = (roc_auc_score(y_true, y_proba[:, 1]) if n_classes == 2
           else roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1":        f1_score(y_true, y_pred, average=avg, zero_division=0),
        "auc":       auc,
    }


def corte_otimo_youden(y_true, score_pos):
    """Ponto de corte que maximiza o indice de Youden (J = TPR - FPR)."""
    fpr, tpr, thr = roc_curve(y_true, score_pos)
    j = tpr - fpr
    k = int(np.argmax(j))
    return thr[k], fpr, tpr, (fpr[k], tpr[k])


def plot_roc(curvas, ponto, arquivo):
    """curvas: dict nome -> (fpr, tpr, auc). ponto: (fpr*, tpr*) do corte otimo."""
    plt.figure(figsize=(7, 6))
    cores = {"Logistica": AZUL_CLARO, "XGBoost": AZUL_ESCURO}
    for nome, (fpr, tpr, auc) in curvas.items():
        plt.plot(fpr, tpr, color=cores.get(nome, "gray"), linewidth=2,
                 label=f"{nome} (AUC={auc:.3f})")
    if ponto:
        plt.plot(ponto[0], ponto[1], "o", color=VERMELHO, markersize=10,
                 label=f"Ponto de corte otimo (Youden)")
    plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontsize=11)
    plt.ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)", fontsize=11)
    plt.title("Curva ROC — Triagem Socioambiental de PMEs", fontsize=13)
    plt.legend(loc="lower right", fontsize=10); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(arquivo, dpi=150, bbox_inches="tight"); plt.close()
    print(f"      Salvo: {os.path.basename(arquivo)}")


def plot_confusion(y_true, y_pred, class_names, titulo, arquivo):
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real"); ax.set_title(titulo, pad=12)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i,j]:,}\n({cmn[i,j]:.1%})", ha="center", va="center",
                    color="white" if cmn[i,j] > 0.5 else "black", fontsize=10)
    plt.colorbar(im, ax=ax, label="Proporcao real")
    plt.tight_layout(); plt.savefig(arquivo, dpi=150, bbox_inches="tight"); plt.close()
    print(f"      Salvo: {os.path.basename(arquivo)}")


def plot_importancia(imp: pd.Series, arquivo):
    fig, ax = plt.subplots(figsize=(9, max(4, len(imp) * 0.5)))
    nomes = [NOMES.get(i, i) for i in imp.index]
    cores = [AZUL_ESCURO if v == imp.max() else AZUL_CLARO for v in imp.values]
    ax.barh(nomes, imp.values, color=cores, edgecolor=AZUL_ESCURO, linewidth=0.8)
    ax.invert_yaxis(); ax.set_xlabel("Importancia (gain)", fontsize=11)
    ax.set_title("Importancia das Features — XGBoost", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(arquivo, dpi=150, bbox_inches="tight"); plt.close()
    print(f"      Salvo: {os.path.basename(arquivo)}")


# ==================================================================
# MAIN
# ==================================================================

def main():
    print("=" * 70)
    print(" CLASSIFICACAO (Fase 2) — LogReg + XGBoost + SMOTE | ROC + corte")
    print("=" * 70)

    df_total = carregar_base(); n_total = len(df_total)
    df = filtrar_elegiveis(df_total); n_eleg = len(df); n_veto = n_total - n_eleg
    df, X, feats = preparar_features(df); gc.collect()
    y, class_names = codificar_target(df)
    n_classes = len(class_names)

    # [5] Particao em TRES conjuntos (60/20/20) + SMOTE so no treino
    print(f"\n[5/8] Particao estratificada "
          f"{1-TEST_SIZE-VAL_SIZE:.0%}/{VAL_SIZE:.0%}/{TEST_SIZE:.0%} (treino/val/teste) + SMOTE...")
    # 1o corte: separa o TESTE e o deixa lacrado ate a avaliacao final
    X_resto, X_test, y_resto, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    # 2o corte: divide o restante em treino e validacao
    val_rel = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_resto, y_resto, test_size=val_rel, random_state=SEED, stratify=y_resto)
    del X_resto, y_resto; gc.collect()
    print(f"      Treino: {len(X_train):,} | Validacao: {len(X_val):,} | Teste: {len(X_test):,}")
    # SMOTE APENAS no treino: validacao e teste preservam a prevalencia real
    print(f"      SMOTE (so treino): {dict(zip(*np.unique(y_train, return_counts=True)))}", end=" -> ")
    X_train, y_train = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(X_train, y_train)
    print(dict(zip(*np.unique(y_train, return_counts=True)))); gc.collect()

    resultados = {}

    # [6] Treino dos candidatos e SELECAO pela VALIDACAO
    print("\n[6/8] Treinando modelos (selecao pela validacao)...")
    print("      --- Regressao Logistica ---")
    scaler = StandardScaler()
    log = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1, C=1.0)
    log.fit(scaler.fit_transform(X_train), y_train)
    proba_log_val = log.predict_proba(scaler.transform(X_val))
    mv_log = metricas(y_val, proba_log_val.argmax(axis=1), proba_log_val, n_classes)
    print("      validacao: " + " | ".join(f"{k}={v:.4f}" for k, v in mv_log.items()))

    print("      --- XGBoost ---")
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                        random_state=SEED, n_jobs=-1, verbosity=0)
    xgb.fit(X_train, y_train)
    proba_xgb_val = xgb.predict_proba(X_val)
    mv_xgb = metricas(y_val, proba_xgb_val.argmax(axis=1), proba_xgb_val, n_classes)
    print("      validacao: " + " | ".join(f"{k}={v:.4f}" for k, v in mv_xgb.items()))

    # Decisao tomada na VALIDACAO — o teste ainda nao foi tocado
    val_metricas = {"Logistica": mv_log, "XGBoost": mv_xgb}
    escolhido = max(val_metricas, key=lambda k: val_metricas[k]["auc"])
    print(f"\n      >> Modelo selecionado pela validacao: {escolhido} "
          f"(AUC_val={val_metricas[escolhido]['auc']:.4f})")

    # [6b] Estimativa final no TESTE — primeiro e unico uso
    print("\n      Avaliacao final no conjunto de TESTE (uso unico)...")
    proba_log = log.predict_proba(scaler.transform(X_test))
    pred_log  = proba_log.argmax(axis=1)
    m_log = metricas(y_test, pred_log, proba_log, n_classes)
    resultados["Logistica"] = {"modelo": log, "scaler": scaler, "proba": proba_log,
                               "pred": pred_log, "val": mv_log, **m_log}
    print("      Logistica (teste): " + " | ".join(f"{k}={v:.4f}" for k, v in m_log.items()))

    proba_xgb = xgb.predict_proba(X_test)
    pred_xgb  = proba_xgb.argmax(axis=1)
    m_xgb = metricas(y_test, pred_xgb, proba_xgb, n_classes)
    imp = pd.Series(xgb.feature_importances_, index=feats).sort_values(ascending=False)
    resultados["XGBoost"] = {"modelo": xgb, "proba": proba_xgb, "pred": pred_xgb,
                             "importancia": imp, "val": mv_xgb, **m_xgb}
    print("      XGBoost (teste):   " + " | ".join(f"{k}={v:.4f}" for k, v in m_xgb.items()))

    # [7] Curva ROC + ponto de corte otimo (modelo principal = XGBoost)
    print("\n[7/8] Curva ROC e ponto de corte otimo (Youden)...")
    corte = None; pred_corte = pred_xgb
    if n_classes == 2:
        score_pos = proba_xgb[:, 1]
        corte, fpr_x, tpr_x, ponto = corte_otimo_youden(y_test, score_pos)
        fpr_l, tpr_l, _ = roc_curve(y_test, proba_log[:, 1])
        plot_roc({"Logistica": (fpr_l, tpr_l, m_log["auc"]),
                  "XGBoost":   (fpr_x, tpr_x, m_xgb["auc"])},
                 ponto, os.path.join(PLOTS_DIR, "roc_curve.png"))
        # Reclassifica no corte otimo e mede o efeito
        pred_corte = (score_pos >= corte).astype(int)
        m_corte = metricas(y_test, pred_corte, proba_xgb, n_classes)
        print(f"      Corte otimo = {corte:.4f} (default=0.5)")
        print(f"      No corte otimo: recall={m_corte['recall']:.4f} "
              f"precision={m_corte['precision']:.4f} acc={m_corte['accuracy']:.4f}")
        resultados["XGBoost"]["corte_otimo"] = float(corte)
        resultados["XGBoost"]["metricas_corte"] = m_corte
    else:
        print("      (multiclasse — corte unico nao se aplica; ROC OvR no apendice)")

    plot_confusion(y_test, pred_corte, class_names, "Matriz de Confusao — XGBoost (corte otimo)",
                   os.path.join(PLOTS_DIR, "confusion_xgb.png"))
    plot_importancia(imp.head(20), os.path.join(PLOTS_DIR, "feature_importance.png"))

    # [8] Salva modelos + relatorio
    print("\n[8/8] Salvando modelos e relatorio...")
    for nome in ["Logistica", "XGBoost"]:
        r = resultados[nome]
        payload = {"modelo": r["modelo"], "features": feats, "nome": nome,
                   "accuracy": r["accuracy"], "precision": r["precision"],
                   "recall": r["recall"], "f1": r["f1"], "auc": r["auc"]}
        if "scaler" in r: payload["scaler"] = r["scaler"]
        if "corte_otimo" in r: payload["corte_otimo"] = r["corte_otimo"]
        with open(os.path.join(MODELOS_DIR, f"modelo_{nome.lower()}.pkl"), "wb") as f:
            pickle.dump(payload, f)
    # XGBoost e o modelo principal -> modelo_melhor.pkl (usado pelo score)
    r = resultados["XGBoost"]
    with open(os.path.join(MODELOS_DIR, "modelo_melhor.pkl"), "wb") as f:
        pickle.dump({"modelo": r["modelo"], "features": feats, "nome": "XGBoost",
                     "auc": r["auc"], "corte_otimo": r.get("corte_otimo")}, f)
    print("      Salvos: modelo_logistica.pkl, modelo_xgb.pkl, modelo_melhor.pkl")

    gerar_relatorio(resultados, y_test, feats, class_names, n_total, n_eleg, n_veto, corte)

    print("\n" + "=" * 70)
    print(" RESUMO (metricas de teste)")
    print("=" * 70)
    print(f"  {'Modelo':<12}{'Acc':>9}{'Prec':>9}{'Recall':>9}{'AUC':>9}")
    for nome in ["Logistica", "XGBoost"]:
        r = resultados[nome]
        print(f"  {nome:<12}{r['accuracy']:>9.4f}{r['precision']:>9.4f}"
              f"{r['recall']:>9.4f}{r['auc']:>9.4f}")
    if corte is not None:
        print(f"\n  Ponto de corte otimo (XGBoost): {corte:.4f}")
    print("\nProximo passo: score_credito_verde.py")


def gerar_relatorio(resultados, y_test, feats, class_names, n_total, n_eleg, n_veto, corte):
    L = ["=" * 70,
         " RELATORIO DE CLASSIFICACAO — Bureau de Credito Verde (PP V002)",
         " TCC MBA USP/Esalq | Autor: Helio Vinicius Moreira Ribeiro",
         "=" * 70,
         "\nPOPULACAO",
         f"  Total                 : {n_total:,}",
         f"  Elegiveis (modelo ML) : {n_eleg:,} ({n_eleg/n_total:.1%})",
         f"  Vetados (G, score=0)  : {n_veto:,} ({n_veto/n_total:.1%})",
         f"\nFEATURES ({len(feats)}) — CADASTRAIS (pre-triagem); geografia EXCLUIDA (vies)"]
    L.append(f"  [cad] numericas: {', '.join(FEATURES_NUM)}")
    n_cnae = sum(1 for f in feats if f.startswith("cnae_"))
    L.append(f"  [cad] cnae_divisao (one-hot): {n_cnae} divisoes")
    L.append(f"\nPARTICAO ESTRATIFICADA (treino/validacao/teste) = "
             f"{1-TEST_SIZE-VAL_SIZE:.0%}/{VAL_SIZE:.0%}/{TEST_SIZE:.0%}")
    L.append("  Treino    : ajuste dos parametros (unico conjunto com SMOTE)")
    L.append("  Validacao : selecao do modelo final — nao reportada como desempenho")
    L.append("  Teste     : uso unico, ao final; estimativa nao-enviesada")

    L.append("\nSELECAO DO MODELO (conjunto de VALIDACAO)")
    L.append(f"  {'Modelo':<12}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'AUC-ROC':>10}")
    for nome in ["Logistica", "XGBoost"]:
        v = resultados[nome].get("val")
        if v:
            L.append(f"  {nome:<12}{v['accuracy']:>10.4f}{v['precision']:>11.4f}"
                     f"{v['recall']:>9.4f}{v['f1']:>9.4f}{v['auc']:>10.4f}")
    if all("val" in resultados[n] for n in ("Logistica", "XGBoost")):
        melhor = max(("Logistica", "XGBoost"), key=lambda k: resultados[k]["val"]["auc"])
        L.append(f"  -> Selecionado pela validacao: {melhor}")

    L.append("\nDESEMPENHO FINAL (conjunto de TESTE — uso unico)")
    L.append(f"  {'Modelo':<12}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'AUC-ROC':>10}")
    for nome in ["Logistica", "XGBoost"]:
        r = resultados[nome]
        L.append(f"  {nome:<12}{r['accuracy']:>10.4f}{r['precision']:>11.4f}"
                 f"{r['recall']:>9.4f}{r['f1']:>9.4f}{r['auc']:>10.4f}")
    if corte is not None and "metricas_corte" in resultados["XGBoost"]:
        mc = resultados["XGBoost"]["metricas_corte"]
        L.append(f"\nPONTO DE CORTE OTIMO (XGBoost, indice de Youden) = {corte:.4f}")
        L.append(f"  No corte otimo: accuracy={mc['accuracy']:.4f} "
                 f"precision={mc['precision']:.4f} recall={mc['recall']:.4f}")
        L.append("  (a curva ROC e o corte definem o trade-off sensibilidade x especificidade")
        L.append("   para a triagem — ver plots/roc_curve.png)")
    if "importancia" in resultados["XGBoost"]:
        L.append("\nIMPORTANCIA DE FEATURES (XGBoost, gain — top 25):")
        for f, v in resultados["XGBoost"]["importancia"].head(25).items():
            L.append(f"  {f:32s}: {v:.4f}")
    L += ["\nNOTA METODOLOGICA (PP V002)",
          "  - Stack: scikit-learn (LogisticRegression), xgboost (XGBClassifier), SMOTE.",
          "  - Metricas: accuracy, precision, recall, AUC-ROC + curva ROC p/ corte otimo.",
          "  - Foco em triagem/eficiencia operacional, NAO previsao de inadimplencia.",
          "  - Alvo Y = FAIXA da rubrica de enquadramento (Fase 1), ancorada em",
          "    Taxonomia BR + PRSAC + Sustainability Bond BNDES (nao e clustering).",
          "  - Features SO cadastrais (porte, capital, tempo de empresa, CNAE):",
          "    mede se o cadastro pre-tria a PME -> argumento da Metrica de Gestao.",
          "  - Vies geografico mitigado: UF/municipio nao sao features.",
          "  - Comparacao multi-modelo + Optuna + SHAP: ver apendice_modelos.py.",
          "=" * 70]
    with open(REL, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"      Relatorio: {REL}")


if __name__ == "__main__":
    main()
