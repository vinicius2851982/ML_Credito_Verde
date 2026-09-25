"""
roc_binaria.py
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Curva ROC e PONTO DE CORTE OTIMO na leitura BINARIA (item 2 do Resultado
Preliminar): "elegivel a credito verde (Verde-A/Verde-B) x nao-elegivel".

Por que binaria: a curva ROC e o indice de Youden sao naturais no problema
de 2 classes — e essa e a decisao real do banco (conceder ou nao a condicao
verde). Em multiclasse, a ROC fica como OvR no apendice.

Reaproveita as variaveis CADASTRAIS baratas da Fase 2 (porte, capital, tempo
de empresa, CNAE), prevendo se a empresa cai nas faixas verdes da rubrica.

OUTPUT:
  Dados/plots/roc_curve_binaria.png
  Dados/relatorio_roc_binaria.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

import classificacao as C   # reaproveita carregar/filtrar/preparar features cadastrais

SEED = 42
# Particao 60/20/20: o PONTO DE CORTE e um parametro de decisao e, portanto,
# precisa ser escolhido na VALIDACAO. Escolhe-lo no teste otimizaria o corte
# para o proprio conjunto onde o desempenho e reportado (selection bias).
TEST_SIZE = 0.20
VAL_SIZE  = 0.20
ELEGIVEL = ["Verde-A", "Verde-B"]
PLOTS = C.PLOTS_DIR
REL = os.path.join(C.DADOS, "relatorio_roc_binaria.txt")


def main():
    print("=" * 64)
    print(" CURVA ROC BINARIA + PONTO DE CORTE (elegivel x nao-elegivel)")
    print("=" * 64)

    df = C.carregar_base()
    df = C.filtrar_elegiveis(df)               # remove Vetado/Fora de escopo
    df, X, feats = C.preparar_features(df)     # variaveis CADASTRAIS

    y = df["risco_label"].isin(ELEGIVEL).astype(int).values
    print(f"  Elegiveis (Verde-A/B): {y.mean():.1%} | n={len(y):,}")

    # 1o corte: TESTE lacrado; 2o corte: treino x validacao
    X_resto, X_te, y_resto, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_resto, y_resto, test_size=VAL_SIZE / (1.0 - TEST_SIZE),
        random_state=SEED, stratify=y_resto)
    print(f"  Treino: {len(y_tr):,} | Validacao: {len(y_val):,} | Teste: {len(y_te):,}")

    # XGBoost binario com balanceamento por scale_pos_weight (sem SMOTE -> rapido)
    pos = max(int(y_tr.sum()), 1); neg = len(y_tr) - pos
    modelo = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                           scale_pos_weight=neg / pos, random_state=SEED,
                           n_jobs=-1, verbosity=0)
    print("  Treinando XGBoost binario ...")
    modelo.fit(X_tr, y_tr)

    # --- corte escolhido na VALIDACAO (Youden) ---
    score_val = modelo.predict_proba(X_val)[:, 1]
    fpr_v, tpr_v, thr_v = roc_curve(y_val, score_val)
    k = int(np.argmax(tpr_v - fpr_v))
    corte = thr_v[k]
    auc_val = roc_auc_score(y_val, score_val)
    print(f"  Corte otimo definido na VALIDACAO: {corte:.4f} (AUC_val={auc_val:.4f})")

    # --- avaliacao final no TESTE, aplicando o corte da validacao ---
    score_pos = modelo.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, score_pos)
    fpr, tpr, thr = roc_curve(y_te, score_pos)
    # ponto do teste correspondente ao corte escolhido na validacao
    k_te = int(np.argmin(np.abs(thr - corte)))
    fpr_k, tpr_k = fpr[k_te], tpr[k_te]

    pred = (score_pos >= corte).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0     # recall/sensibilidade
    espec = tn / (tn + fp) if (tn + fp) else 0    # especificidade
    prec = tp / (tp + fp) if (tp + fp) else 0
    acc = (tp + tn) / len(y_te)

    # --- grafico ---
    os.makedirs(PLOTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#1A237E", lw=2.5, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="Aleatorio")
    ax.scatter([fpr_k], [tpr_k], color="#C62828", zorder=5, s=90,
               label=f"Corte da validacao (Youden) = {corte:.3f}")
    ax.set_xlabel("1 - Especificidade (FPR)")
    ax.set_ylabel("Sensibilidade (TPR)")
    ax.set_title("Curva ROC — elegivel (Verde) x nao-elegivel")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    cam = os.path.join(PLOTS, "roc_curve_binaria.png")
    plt.tight_layout(); plt.savefig(cam, dpi=150, bbox_inches="tight"); plt.close()

    # --- relatorio ---
    L = ["=" * 64,
         " ROC BINARIA — elegivel (Verde-A/B) x nao-elegivel",
         "=" * 64,
         f"  n teste              : {len(y_te):,}",
         f"  Prevalencia elegivel : {y_te.mean():.1%}",
         f"  AUC-ROC              : {auc:.4f}",
         f"  Ponto de corte (Youden): {corte:.4f}",
         f"  No corte otimo:",
         f"    Sensibilidade (recall) : {sens:.4f}",
         f"    Especificidade         : {espec:.4f}",
         f"    Precisao               : {prec:.4f}",
         f"    Acuracia               : {acc:.4f}",
         f"  Matriz de confusao (corte otimo): TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}",
         "",
         "  Leitura: o corte de Youden maximiza sensibilidade + especificidade,",
         "  equilibrando incluir PMEs verdes (recall) e evitar falsos verdes (especificidade).",
         f"  Grafico: {cam}",
         "=" * 64]
    print("\n".join(L))
    with open(REL, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n  Salvo: {REL}")


if __name__ == "__main__":
    main()
