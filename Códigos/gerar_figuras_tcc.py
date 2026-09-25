"""
gerar_figuras_tcc.py  (passo 17 — figuras para o documento final)
------------------------------------------------------------------
TCC MBA USP/Esalq - Credito Verde e IA para PMEs
Autor: Helio Vinicius Moreira Ribeiro

Gera as figuras do TCC em ESCALA DE CINZA, conforme as normas do programa
(tabelas e figuras sem cores). Fonte Arial, tamanhos legiveis em impressao.

Distingue series por TRAMA (hachura) e TOM DE CINZA, nunca por cor — para
que continuem legiveis em impressao preto e branco.

OUTPUT: Dados/plots/tcc/figura_N_*.png (300 dpi)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(BASE_DIR, "Dados")
SAIDA = os.path.join(DADOS, "plots", "tcc")
os.makedirs(SAIDA, exist_ok=True)

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
rcParams["font.size"] = 10
rcParams["axes.edgecolor"] = "black"
rcParams["axes.linewidth"] = 0.8

CINZA = ["#2B2B2B", "#5E5E5E", "#8F8F8F", "#BFBFBF", "#E0E0E0"]
DPI = 300


def br(valor, casas=1):
    """Formata numero no padrao brasileiro: virgula decimal, ponto de milhar."""
    s = f"{valor:,.{casas}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _limpar(ax, ylabel=None, xlabel=None):
    """Remove molduras superiores/direita — padrao editorial sobrio."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def figura_1_faixas():
    """Distribuicao das empresas por faixa de enquadramento."""
    faixas = ["Verde-A", "Verde-B", "Amarelo", "Vermelho"]
    valores = [493799, 884439, 813269, 572630]
    total = 2764563
    pct = [100 * v / total for v in valores]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    barras = ax.bar(faixas, pct, color=["#3A3A3A", "#6E6E6E", "#A5A5A5", "#D0D0D0"],
                    edgecolor="black", linewidth=0.7, width=0.62)
    for b, v, p in zip(barras, valores, pct):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.6, f"{br(p)}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(pct) * 1.18)
    _limpar(ax, ylabel="Participação (%)")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#BBBBBB")
    ax.set_axisbelow(True)
    plt.tight_layout()
    cam = os.path.join(SAIDA, "figura_1_faixas.png")
    plt.savefig(cam, dpi=DPI, bbox_inches="tight"); plt.close()
    return cam


def figura_2_economia_verde():
    """Composicao da economia verde por eixo (FEBRABAN)."""
    eixos = ["Social", "Ambiental", "Social +\nAmbiental"]
    valores = [310980, 95416, 17695]
    total = 2764563
    pct = [100 * v / total for v in valores]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    hachuras = ["", "///", "..."]
    barras = ax.bar(eixos, pct, color=["#4A4A4A", "#9A9A9A", "#D5D5D5"],
                    edgecolor="black", linewidth=0.7, width=0.55)
    for b, h in zip(barras, hachuras):
        b.set_hatch(h)
    for b, v, p in zip(barras, valores, pct):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.25,
                f"{br(p)}%\n({br(v, 0)})",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(pct) * 1.32)
    _limpar(ax, ylabel="Participação no total de empresas (%)")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#BBBBBB")
    ax.set_axisbelow(True)
    plt.tight_layout()
    cam = os.path.join(SAIDA, "figura_2_economia_verde.png")
    plt.savefig(cam, dpi=DPI, bbox_inches="tight"); plt.close()
    return cam


def figura_3_regional():
    """Aprovacao por regiao (rubrica) vs media nacional."""
    regioes = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
    aprov = [52.0, 54.0, 49.5, 48.5, 49.2]
    media = 49.9

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    barras = ax.bar(regioes, aprov, color="#8F8F8F", edgecolor="black",
                    linewidth=0.7, width=0.6)
    for b in barras[:2]:
        b.set_facecolor("#4A4A4A")
    ax.axhline(media, color="black", linestyle="--", linewidth=1.1,
               label=f"Média nacional: {br(media)}%")
    # rotulos DENTRO das barras: evita colisao com a linha da media
    for b, v in zip(barras, aprov):
        escura = b.get_facecolor()[0] < 0.5
        ax.text(b.get_x() + b.get_width() / 2, v - 0.35, f"{br(v)}%",
                ha="center", va="top", fontsize=9,
                color="white" if escura else "black")
    ax.set_ylim(44, 56)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right",
              bbox_to_anchor=(1.0, 1.04))
    _limpar(ax, ylabel="Empresas classificadas como elegíveis (%)")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#BBBBBB")
    ax.set_axisbelow(True)
    plt.tight_layout()
    cam = os.path.join(SAIDA, "figura_3_regional.png")
    plt.savefig(cam, dpi=DPI, bbox_inches="tight"); plt.close()
    return cam


def figura_4_fairness():
    """Sensibilidade e taxa de falsos positivos por regiao (Fase 2, teste).
    Le os valores vigentes do CSV; usa fallback apenas se ele nao existir."""
    regioes = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
    recall = [0.900, 0.907, 0.915, 0.921, 0.921]
    fpr = [0.205, 0.199, 0.163, 0.147, 0.149]
    cam = os.path.join(DADOS, "tabela_regional_fairness.csv")
    if os.path.exists(cam):
        t = pd.read_csv(cam, sep=";", encoding="utf-8-sig").set_index("regiao")
        t = t.reindex([r for r in regioes if r in t.index])
        if len(t):
            regioes = list(t.index)
            recall = t["recall (sensibilidade)"].tolist()
            fpr = t["fpr (falso verde)"].tolist()

    x = np.arange(len(regioes)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    b1 = ax.bar(x - w / 2, recall, w, label="Sensibilidade (recall)",
                color="#4A4A4A", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x + w / 2, fpr, w, label="Taxa de falsos positivos",
                color="#D5D5D5", edgecolor="black", linewidth=0.7, hatch="///")
    for bars, vals in ((b1, recall), (b2, fpr)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, br(v, 3),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(regioes)
    ax.set_ylim(0, 1.06)
    _limpar(ax, ylabel="Proporção")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#BBBBBB")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.16))
    plt.tight_layout()
    cam = os.path.join(SAIDA, "figura_4_fairness.png")
    plt.savefig(cam, dpi=DPI, bbox_inches="tight"); plt.close()
    return cam


def figura_5_arquitetura():
    """Diagrama das duas fases (esquema conceitual)."""
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")

    def caixa(x, y, w, h, titulo, linhas, fill="#F2F2F2"):
        r = plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor="black", linewidth=1.0)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h - 0.32, titulo, ha="center", va="top",
                fontsize=9.5, fontweight="bold")
        for i, ln in enumerate(linhas):
            ax.text(x + w / 2, y + h - 0.72 - i * 0.30, ln, ha="center",
                    va="top", fontsize=8)

    caixa(0.15, 1.5, 2.7, 2.3, "Dados públicos",
          ["Receita Federal", "IBAMA · CEIS", "RAIS · CAGED"])
    caixa(3.35, 1.5, 3.0, 2.3, "Fase 1 — Rubrica",
          ["Taxonomia FEBRABAN", "+ conduta da firma", "→ faixa (rótulo)"], fill="#DCDCDC")
    caixa(6.85, 1.5, 3.0, 2.3, "Fase 2 — Modelo",
          ["Só dados cadastrais", "porte · capital", "tempo · CNAE"], fill="#F2F2F2")

    for x0, x1 in ((2.95, 3.3), (6.45, 6.8)):
        ax.annotate("", xy=(x1, 2.65), xytext=(x0, 2.65),
                    arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.1))
    ax.text(4.85, 1.20, "Critério oficial e auditável", ha="center", fontsize=8.5, style="italic")
    ax.text(8.35, 1.20, "Pré-triagem no balcão", ha="center", fontsize=8.5, style="italic")
    ax.text(5.00, 0.45, "as variáveis da rubrica são PROIBIDAS como preditoras na Fase 2",
            ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    cam = os.path.join(SAIDA, "figura_5_arquitetura.png")
    plt.savefig(cam, dpi=DPI, bbox_inches="tight"); plt.close()
    return cam


def main():
    print("=" * 62)
    print(" FIGURAS DO TCC (escala de cinza, 300 dpi)")
    print("=" * 62)
    for fn in (figura_1_faixas, figura_2_economia_verde, figura_3_regional,
               figura_4_fairness, figura_5_arquitetura):
        print("  gerada:", os.path.basename(fn()))
    print(f"\nDestino: {SAIDA}")


if __name__ == "__main__":
    main()
