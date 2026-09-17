"""
apendice_modelos.py  (APENDICE METODOLOGICO — alem do escopo da proposta)
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

APENDICE: comparacao multi-modelo (LogReg + XGBoost + LightGBM + CatBoost)
com Otimizacao Bayesiana (Optuna) e explicabilidade SHAP. Material extra
para o APENDICE do TCC — a proposta (PP V002) define como nucleo apenas
LogReg + XGBoost (ver classificacao.py). Tecnicas inspiradas no curso
"ML de Predicao a Saude" (USP). Nao sobrescreve os artefatos principais.

Fase supervisionada: usa a FAIXA da rubrica (rubrica_enquadramento.py) como
variavel-alvo Y e prediz a partir de variaveis CADASTRAIS baratas (mesmo
conjunto de classificacao.py), comparando varios modelos.

METODOLOGIA (incorporada do curso "ML de Predicao a Saude", USP):
  1. Comparacao sistematica de modelos:
       Regressao Logistica (baseline) + XGBoost + LightGBM + CatBoost
  2. Otimizacao Bayesiana de hiperparametros (Optuna / TPE):
       objetivo = ROC-AUC em validacao cruzada estratificada
  3. Selecao do melhor modelo por AUC de teste
  4. Explicabilidade com SHAP (TreeExplainer) — alem do gain
  Referencias: Akiba et al. (2019, Optuna); Lundberg & Lee (2017, SHAP);
               Chen & Guestrin (2016, XGBoost); Ke et al. (2017, LightGBM);
               Prokhorenkova et al. (2018, CatBoost).

NOTA METODOLOGICA (honestidade cientifica):
  O alvo Y e a FAIXA da rubrica (criterios oficiais; nao e clustering). As
  features sao SO cadastrais (porte, capital, tempo de empresa, CNAE) — as
  mesmas de classificacao.py —, evitando circularidade. O AUC mede a
  capacidade de PRE-TRIAGEM a partir do cadastro. Optuna e SHAP aumentam o
  rigor da SELECAO e da INTERPRETACAO do modelo.

POPULACAO: base_analitica_rotulada.csv, excluindo "Vetado" e "Fora de escopo"
           (penalizacao severa / atividade nao-elegivel — fora do gradiente verde).

OUTPUT:
  Dados/modelos/modelo_<melhor>.pkl, modelo_logistica.pkl
  Dados/relatorio_classificacao.txt
  Dados/comparacao_modelos.csv
  Dados/plots/comparacao_modelos.png
  Dados/plots/confusion_<melhor>.png
  Dados/plots/shap_importancia.png, shap_impacto.png
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.base            import clone
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Dependencias opcionais (o pipeline degrada com elegancia se faltarem)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    TEM_OPTUNA = True
except ImportError:
    TEM_OPTUNA = False

try:
    from lightgbm import LGBMClassifier
    TEM_LGBM = True
except ImportError:
    TEM_LGBM = False

try:
    from catboost import CatBoostClassifier
    TEM_CATBOOST = True
except ImportError:
    TEM_CATBOOST = False

try:
    import shap
    TEM_SHAP = True
except ImportError:
    TEM_SHAP = False

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS       = os.path.join(BASE_DIR, "Dados")
PLOTS_DIR   = os.path.join(DADOS, "plots")
MODELOS_DIR = os.path.join(DADOS, "modelos")

INPUT   = os.path.join(DADOS, "base_analitica_rotulada.csv")
REL     = os.path.join(DADOS, "apendice_relatorio_modelos.txt")
COMPARA = os.path.join(DADOS, "apendice_comparacao_modelos.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------
SEED          = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5            # folds para a metrica final (cross_val do melhor)
OPTUNA_FOLDS  = 3           # folds dentro do Optuna (mais rapido)
N_TRIALS      = 30          # trials de Optuna por modelo
TUNING_SAMPLE = 150_000     # subamostra estratificada p/ tunar (escala 2.7M)
SHAP_SAMPLE   = 5_000       # amostra p/ SHAP (TreeExplainer e custoso)

# Features CADASTRAIS (mesmas de classificacao.py) — pre-triagem barata,
# evita circularidade. Geografia (UF/municipio) FORA por mitigacao de vies.
FEATURES_NUM = ["porte_num", "capital_social_log", "tempo_empresa_anos"]
FEATURES_CAT = ["cnae_divisao"]   # one-hot

PORTE_MAP = {"00": 0, "01": 1, "03": 2, "05": 3}
ANOS_SEM_INFRACAO = 30.0
HOJE = pd.Timestamp(2026, 6, 1)

NOMES_FEATURES = {
    "porte_num": "Porte (ordinal)", "capital_social_log": "Capital Social log",
    "tempo_empresa_anos": "Tempo de Empresa (anos)",
}


# ==================================================================
# DADOS
# ==================================================================

def carregar_base() -> pd.DataFrame:
    print(f"[1/9] Carregando {INPUT} ...")
    df = pd.read_csv(INPUT, sep=";", dtype=str, encoding="utf-8-sig", low_memory=False)
    print(f"      {len(df):,} registros | {len(df.columns)} colunas")
    return df


def filtrar_elegiveis(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/9] Filtrando populacao classificavel...")
    if "risco_label" in df.columns:
        antes = len(df)
        df = df[~df["risco_label"].isin(["Vetado", "Fora de escopo"])].copy()
        print(f"      Removidos (Vetado/Fora de escopo): {antes - len(df):,}")
    print(f"      Registros classificaveis: {len(df):,}")
    return df


def preparar_features(df: pd.DataFrame):
    """Variaveis CADASTRAIS (mesmas de classificacao.py): porte, capital,
    tempo de empresa e divisao da CNAE (one-hot). Retorna df, X (DataFrame), features."""
    print("\n[3/9] Preparando features CADASTRAIS (pre-triagem)...")
    df["capital_social"]     = pd.to_numeric(df.get("capital_social"), errors="coerce").fillna(0)
    df["capital_social_log"] = np.log1p(df["capital_social"])
    df["porte_num"] = df["porte"].astype(str).str.strip().map(PORTE_MAP).fillna(0).astype(float)
    s = df.get("data_inicio_atividade", pd.Series("", index=df.index)).astype(str)
    s = s.str.replace(r"[^0-9]", "", regex=True).str.slice(0, 8)
    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    tempo = (HOJE - dt).dt.days / 365.25
    med = tempo[tempo > 0].median()
    df["tempo_empresa_anos"] = tempo.where(tempo > 0).fillna(med if pd.notna(med) else 5.0)
    if "cnae_divisao" not in df.columns or df["cnae_divisao"].isna().all():
        df["cnae_divisao"] = df["cnae_fiscal_principal"].astype(str).str[:2]
    df["cnae_divisao"] = df["cnae_divisao"].astype(str).str.zfill(2)

    X_num = df[FEATURES_NUM].fillna(0).astype(np.float64)
    X_cat = pd.get_dummies(df["cnae_divisao"], prefix="cnae")
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    features = list(X.columns)
    print(f"      Features ({len(features)}): {len(FEATURES_NUM)} numericas + {X_cat.shape[1]} CNAE (one-hot)")
    return df, X, features


def codificar_target(df: pd.DataFrame) -> tuple:
    """Faixa da rubrica (0=melhor ... 3=pior). Retorna (y, class_names)."""
    print("\n[4/9] Codificando variavel-alvo Y (faixa da rubrica)...")
    if "risco_label" not in df.columns:
        raise ValueError("Coluna 'risco_label' ausente. Execute rubrica_enquadramento.py primeiro.")
    RANK = {"Verde-A": 0, "Verde-B": 1, "Amarelo": 2, "Vermelho": 3}
    rank = df["risco_label"].fillna("Amarelo").astype(str).map(RANK).fillna(2).astype(int)
    presentes = sorted(rank.unique())
    remap = {r: i for i, r in enumerate(presentes)}
    y = rank.map(remap).astype(int)
    rank_nome = {v: k for k, v in RANK.items()}
    class_names = [rank_nome[r] for r in presentes]
    print(f"      {len(class_names)} classes: {class_names}")
    for i, nome in enumerate(class_names):
        n = (y == i).sum()
        print(f"        Classe {i} ({nome}): {n:,} ({n/len(y):.1%})")
    return y.values, class_names


# ==================================================================
# AUC robusto (binario ou multiclasse)
# ==================================================================

def auc_score(y_true, proba, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")


# ==================================================================
# OTIMIZACAO BAYESIANA (Optuna)
# ==================================================================

def espaco_busca(nome, trial):
    """Espacos de busca (adaptados do notebook de ML em Saude)."""
    if nome == "XGBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 2, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_categorical("reg_lambda", [1, 1.5, 2, 5, 10]),
        )
    if nome == "LightGBM":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_categorical("reg_lambda", [1, 1.5, 2, 5, 10]),
        )
    if nome == "CatBoost":
        return dict(
            iterations=trial.suggest_int("iterations", 50, 500, step=50),
            depth=trial.suggest_int("depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            rsm=trial.suggest_float("rsm", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            l2_leaf_reg=trial.suggest_categorical("l2_leaf_reg", [1, 1.5, 2, 5, 10]),
        )
    return {}


def construir_modelo(nome, params=None):
    params = params or {}
    if nome == "XGBoost":
        return XGBClassifier(random_state=SEED, eval_metric="auc", n_jobs=-1,
                             verbosity=0, **params)
    if nome == "LightGBM":
        # subsample_freq>0 e necessario para o subsample (bagging) ter efeito
        return LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1,
                             subsample_freq=1, **params)
    if nome == "CatBoost":
        # bootstrap_type='Bernoulli' e necessario para aceitar 'subsample'
        return CatBoostClassifier(random_state=SEED, verbose=0,
                                 bootstrap_type="Bernoulli", **params)
    raise ValueError(nome)


def tunar_optuna(nome, X_sub, y_sub):
    """Otimizacao Bayesiana: maximiza AUC em CV estratificada na subamostra."""
    if not TEM_OPTUNA:
        print(f"      [WARN] Optuna ausente — {nome} usa defaults.")
        return {}
    cv = StratifiedKFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=SEED)
    # roc_auc simples so vale p/ binario; multiclasse exige OvR ponderado.
    scoring = "roc_auc" if len(np.unique(y_sub)) == 2 else "roc_auc_ovr_weighted"

    def objective(trial):
        params = espaco_busca(nome, trial)
        modelo = construir_modelo(nome, params)
        scores = cross_val_score(modelo, X_sub, y_sub, cv=cv,
                                 scoring=scoring, n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"      {nome}: melhor AUC-CV (tuning) = {study.best_value:.4f}")
    return study.best_params


# ==================================================================
# TREINO E AVALIACAO
# ==================================================================

def avaliar(modelo, X_test, y_test, n_classes) -> dict:
    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    return {
        "acuracia": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "auc":      auc_score(y_test, y_proba, n_classes),
        "y_pred":   y_pred,
        "y_proba":  y_proba,
    }


def treinar_todos(X_train, y_train, X_test, y_test, X_sub, y_sub,
                  features, n_classes) -> dict:
    """Treina e avalia: LogReg (baseline) + arvores tunadas por Optuna."""
    print("\n[6/9] Treinando e comparando modelos...")
    resultados = {}

    # ---- Baseline: Regressao Logistica (escalonada) ----
    print("\n      --- Regressao Logistica (baseline) ---")
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    log = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1,
                             C=1.0, multi_class="auto")
    log.fit(Xtr_s, y_train)
    m = avaliar(log, Xte_s, y_test, n_classes)
    m.update({"modelo": log, "scaler": scaler, "params": {"C": 1.0}})
    resultados["Logistica"] = m
    print(f"      AUC={m['auc']:.4f} | F1={m['f1_macro']:.4f} | Acc={m['acuracia']:.4f}")

    # ---- Arvores com Optuna ----
    arvores = ["XGBoost"]
    if TEM_LGBM:     arvores.append("LightGBM")
    if TEM_CATBOOST: arvores.append("CatBoost")

    for nome in arvores:
        print(f"\n      --- {nome} (Optuna {N_TRIALS} trials) ---")
        best_params = tunar_optuna(nome, X_sub, y_sub)
        modelo = construir_modelo(nome, best_params)
        modelo.fit(X_train, y_train)
        m = avaliar(modelo, X_test, y_test, n_classes)
        m.update({"modelo": modelo, "params": best_params})
        resultados[nome] = m
        print(f"      AUC={m['auc']:.4f} | F1={m['f1_macro']:.4f} | Acc={m['acuracia']:.4f}")
        gc.collect()

    return resultados


# ==================================================================
# SHAP
# ==================================================================

def explicar_shap(modelo, X_test_df, features):
    if not TEM_SHAP:
        print("      [WARN] SHAP ausente — pulando explicabilidade SHAP.")
        return
    print(f"\n[8/9] Explicabilidade SHAP (amostra {min(SHAP_SAMPLE, len(X_test_df)):,})...")
    # Mantem os nomes ORIGINAIS nas colunas (XGBoost proibe [ ] < nos nomes);
    # os nomes amigaveis vao apenas como rotulos de exibicao do grafico.
    Xs = X_test_df.sample(min(SHAP_SAMPLE, len(X_test_df)), random_state=SEED)
    nomes_display = [NOMES_FEATURES.get(c, c) for c in Xs.columns]
    try:
        explainer = shap.TreeExplainer(modelo)
        sv = explainer.shap_values(Xs)
        # binario: shap_values pode vir como lista [classe0, classe1] ou array 3D
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]   # classe positiva

        plt.figure()
        shap.summary_plot(sv, Xs, feature_names=nomes_display,
                          plot_type="bar", max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "apendice_shap_importancia.png"), dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(sv, Xs, feature_names=nomes_display,
                          max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "apendice_shap_impacto.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      Salvos: apendice_shap_importancia.png, apendice_shap_impacto.png")
    except Exception as e:
        print(f"      [WARN] SHAP falhou ({e}). Seguindo sem SHAP.")


# ==================================================================
# PLOTS E RELATORIO
# ==================================================================

AZUL_ESCURO = "#1A237E"; AZUL_CLARO = "#BBDEFB"

def plot_comparacao(resultados: dict):
    nomes = list(resultados.keys())
    aucs  = [resultados[n]["auc"] for n in nomes]
    f1s   = [resultados[n]["f1_macro"] for n in nomes]

    x = np.arange(len(nomes)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w/2, aucs, w, label="AUC-ROC", color=AZUL_ESCURO)
    ax.bar(x + w/2, f1s,  w, label="F1 Macro", color=AZUL_CLARO, edgecolor=AZUL_ESCURO)
    ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=11)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Metrica (teste)", fontsize=11)
    ax.set_title("Comparacao de Modelos — Bureau de Credito Verde", fontsize=13)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (a, f) in enumerate(zip(aucs, f1s)):
        ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "apendice_comparacao_modelos.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("      Salvo: apendice_comparacao_modelos.png")


def plot_confusion(y_test, y_pred, class_names, titulo, arquivo):
    n = len(class_names)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n)))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real"); ax.set_title(titulo, pad=12)
    for i in range(n):
        for j in range(n):
            cor = "white" if cmn[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]:,}\n({cmn[i,j]:.1%})", ha="center", va="center",
                    fontsize=10, color=cor)
    plt.colorbar(im, ax=ax, label="Proporcao real")
    plt.tight_layout(); plt.savefig(arquivo, dpi=150, bbox_inches="tight"); plt.close()
    print(f"      Salvo: {os.path.basename(arquivo)}")


def salvar_modelos(resultados, melhor, features):
    print("\n[9/9] Salvando modelos (apendice — nao sobrescreve os principais)...")
    for nome in ["Logistica", melhor]:
        r = resultados[nome]
        payload = {"modelo": r["modelo"], "features": features, "nome": nome,
                   "acuracia": r["acuracia"], "f1_macro": r["f1_macro"],
                   "auc": r["auc"], "params": r["params"]}
        if "scaler" in r: payload["scaler"] = r["scaler"]
        nome_arq = f"apendice_modelo_{nome.lower()}.pkl"
        with open(os.path.join(MODELOS_DIR, nome_arq), "wb") as f:
            pickle.dump(payload, f)
        print(f"      Salvo: {nome_arq}")


def gerar_relatorio(resultados, melhor, y_test, features, class_names,
                    n_total, n_eleg, n_veto):
    linhas = ["=" * 70,
              " RELATORIO DE CLASSIFICACAO — Bureau de Credito Verde",
              " TCC MBA USP/Esalq | Autor: Helio Vinicius Moreira Ribeiro",
              "=" * 70,
              "\nPOPULACAO",
              f"  Total                 : {n_total:,}",
              f"  Elegiveis (modelo ML) : {n_eleg:,} ({n_eleg/n_total:.1%})",
              f"  Removidos (Vetado/Fora): {n_veto:,} ({n_veto/n_total:.1%})",
              f"\nFEATURES ({len(features)}) — cadastrais; geografia excluida (vies)"]
    n_cnae = sum(1 for f in features if f.startswith("cnae_"))
    linhas.append(f"  numericas: {', '.join(FEATURES_NUM)}")
    linhas.append(f"  cnae_divisao (one-hot): {n_cnae} divisoes")

    linhas.append("\nCOMPARACAO DE MODELOS (teste)")
    linhas.append(f"  {'Modelo':<14}{'AUC':>10}{'F1 Macro':>12}{'Acuracia':>12}")
    linhas.append(f"  {'-'*46}")
    for nome, r in sorted(resultados.items(), key=lambda kv: -kv[1]["auc"]):
        marca = "  <== MELHOR" if nome == melhor else ""
        linhas.append(f"  {nome:<14}{r['auc']:>10.4f}{r['f1_macro']:>12.4f}"
                      f"{r['acuracia']:>12.4f}{marca}")

    linhas.append(f"\nMELHOR MODELO: {melhor}")
    linhas.append(f"  Hiperparametros (Optuna): {resultados[melhor]['params']}")
    linhas.append(f"\n  Relatorio por classe ({melhor}):")
    rpt = classification_report(y_test, resultados[melhor]["y_pred"],
                                labels=list(range(len(class_names))),
                                target_names=class_names, zero_division=0)
    for l in rpt.split("\n"):
        linhas.append(f"    {l}")

    linhas += ["\nNOTA METODOLOGICA",
               "  - Selecao de modelos via comparacao sistematica + Optuna (Akiba et al., 2019).",
               "  - Explicabilidade por SHAP (Lundberg & Lee, 2017) — ver plots/shap_*.png.",
               "  - SMOTE (Chawla et al., 2002) aplicado SO no treino (sem data leakage).",
               "  - Alvo Y = faixa da rubrica (criterios oficiais); features SO cadastrais.",
               "    O AUC mede pre-triagem pelo cadastro, nao predicao de default.",
               "  - Governanca (G) e veto contratual, nao feature ML (objetivo do TCC).",
               "=" * 70]

    with open(REL, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    pd.DataFrame([{"Modelo": n, "AUC": r["auc"], "F1_macro": r["f1_macro"],
                   "Acuracia": r["acuracia"], "Params": r["params"]}
                  for n, r in resultados.items()]).to_csv(COMPARA, sep=";", index=False,
                                                          encoding="utf-8-sig")
    print(f"      Relatorio: {REL}")
    print(f"      Comparacao: {COMPARA}")


# ==================================================================
# MAIN
# ==================================================================

def main():
    print("=" * 70)
    print(" CLASSIFICACAO v2 — Multi-modelo + Optuna + SHAP | Pilares E+S")
    print("=" * 70)
    print(f"  Optuna:{TEM_OPTUNA}  LightGBM:{TEM_LGBM}  CatBoost:{TEM_CATBOOST}  SHAP:{TEM_SHAP}")

    df_total = carregar_base(); n_total = len(df_total)
    df = filtrar_elegiveis(df_total); n_eleg = len(df); n_veto = n_total - n_eleg
    df, X, features = preparar_features(df); gc.collect()
    y, class_names = codificar_target(df)
    n_classes = len(class_names)

    # [5] Split + SMOTE + subamostra de tuning
    print(f"\n[5/9] Split {1-TEST_SIZE:.0%}/{TEST_SIZE:.0%} + SMOTE + subamostra tuning...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print(f"      Treino: {len(X_train):,} | Teste: {len(X_test):,}")

    # Subamostra estratificada para o Optuna (antes do SMOTE; AUC lida com desbalanceio)
    if len(X_train) > TUNING_SAMPLE:
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=TUNING_SAMPLE,
            random_state=SEED, stratify=y_train)
    else:
        X_sub, y_sub = X_train, y_train
    print(f"      Subamostra de tuning: {len(X_sub):,}")

    # SMOTE no treino completo (final fit)
    print(f"      SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}", end=" -> ")
    sm = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    print(dict(zip(*np.unique(y_train_bal, return_counts=True))))
    gc.collect()

    # [6] Treina e compara
    resultados = treinar_todos(X_train_bal, y_train_bal, X_test, y_test,
                               X_sub, y_sub, features, n_classes)

    # [7] Seleciona melhor (por AUC) entre as arvores; Logistica e baseline
    candidatos = {k: v for k, v in resultados.items() if k != "Logistica"}
    melhor = max(candidatos or resultados, key=lambda k: resultados[k]["auc"])
    print(f"\n[7/9] Melhor modelo por AUC: {melhor} (AUC={resultados[melhor]['auc']:.4f})")
    plot_comparacao(resultados)
    plot_confusion(y_test, resultados[melhor]["y_pred"], class_names,
                   f"Matriz de Confusao — {melhor}",
                   os.path.join(PLOTS_DIR, f"apendice_confusion_{melhor.lower()}.png"))

    # [8] SHAP no melhor modelo de arvore
    explicar_shap(resultados[melhor]["modelo"], X_test, features)

    # [9] Salva
    salvar_modelos(resultados, melhor, features)
    gerar_relatorio(resultados, melhor, y_test, features, class_names,
                    n_total, n_eleg, n_veto)

    print("\n" + "=" * 70)
    print(" RESUMO")
    print("=" * 70)
    for nome, r in sorted(resultados.items(), key=lambda kv: -kv[1]["auc"]):
        print(f"  {nome:<14} AUC={r['auc']:.4f} | F1={r['f1_macro']:.4f} | Acc={r['acuracia']:.4f}")
    print(f"\n  Melhor: {melhor}  ->  usado no score_credito_verde.py")
    print("\nProximo passo: score_credito_verde.py")


if __name__ == "__main__":
    main()
