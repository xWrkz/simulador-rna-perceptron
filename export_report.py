"""Generador del Informe Técnico (Sesión 14 - UPN) en Word (.docx) y PDF (.pdf).

Enfocado directamente en el desarrollo práctico del ejercicio (sin marco teórico
introductorio extenso), incorporando 4 figuras de alta resolución generadas
desde perceptron.py — incluyendo el diagrama completo "Predicción del Perceptrón
(¿cómo decide la neurona?)" junto con la función escalón hardlim.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from perceptron import PerceptronRNA, binarize_grades, evaluate_manual_parameters


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets_report"
DOCX_PATH = BASE_DIR / "INFORME_TECNICO_RNA_PERCEPTRON.docx"
PDF_PATH = BASE_DIR / "INFORME_TECNICO_RNA_PERCEPTRON.pdf"


# ==========================================
# 1. GENERACIÓN DE FIGURAS (MATPLOTLIB 200 DPI)
# ==========================================
def draw_perceptron_prediction_figure(model: PerceptronRNA, out_path: Path) -> None:
    """Genera la figura 'Predicción del Perceptrón (¿cómo decide la neurona?)'
    combinando el diagrama visual de la neurona (caso U1=14, U2=10, U3=12.5) y
    la curva de activación escalón hardlim con el punto evaluado.
    """
    grades = (14.0, 10.0, 12.5)
    x_bin = binarize_grades(grades)
    y_pred, z_val = model.predict(x_bin)

    fig = plt.figure(figsize=(10.2, 5.6), dpi=200, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], width_ratios=[1.25, 1.0], hspace=0.28, wspace=0.2)

    # --- Panel Superior: Diagrama de la Neurona Artificial ---
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    # Títulos de columnas
    col_headers = [
        (10.5, "ENTRADAS (xᵢ)"),
        (31.0, "PESOS (wᵢ · xᵢ)"),
        (51.5, "SUMA PONDERADA"),
        (71.5, "FUNCIÓN ESCALÓN"),
        (90.0, "PREDICCIÓN (ŷ)"),
    ]
    for cx, title in col_headers:
        ax.text(cx, 39.8, title, ha="center", va="center", fontsize=8.3, fontweight="bold", color="#64748B")

    soma_cx, soma_cy = 51.5, 22.5
    inputs_spec = [
        ("U1", grades[0], int(x_bin[0]), float(model.w[0]), 33.0),
        ("U2", grades[1], int(x_bin[1]), float(model.w[1]), 22.5),
        ("U3", grades[2], int(x_bin[2]), float(model.w[2]), 12.0),
    ]

    for idx, (lbl, g, xv, wv, cy) in enumerate(inputs_spec, start=1):
        active = xv == 1
        bg = "#DCFCE7" if active else "#FEE2E2"
        edge = "#15803D" if active else "#DC2626"
        txt = "#166534" if active else "#991B1B"
        lcol = "#10B981" if active else "#CBD5E1"
        lstyle = "-" if active else "--"
        lwidth = 2.6 if active else 1.6

        # Caja de entrada
        box = patches.FancyBboxPatch(
            (3.0, cy - 3.6), 15.0, 7.2,
            boxstyle="round,pad=0.4,rounding_size=1.5",
            facecolor=bg, edgecolor=edge, linewidth=1.8, zorder=3,
        )
        ax.add_patch(box)
        ax.text(10.5, cy + 1.3, f"{lbl}: {g:.1f} pts", ha="center", va="center", fontsize=8.6, fontweight="bold", color=txt, zorder=4)
        ax.text(10.5, cy - 1.5, f"x{idx} = {xv}", ha="center", va="center", fontsize=9.5, fontweight="bold", color=txt, zorder=4)

        # Flecha sináptica hacia el soma
        ax.annotate(
            "", xy=(soma_cx - 6.8, soma_cy), xytext=(18.5, cy),
            arrowprops=dict(arrowstyle="-|>", color=lcol, lw=lwidth, ls=lstyle, mutation_scale=13),
            zorder=2,
        )

        # Etiqueta del peso en el tramo sináptico
        contrib = xv * wv
        mx = 31.0
        my = (cy + soma_cy) / 2.0 + (1.6 if idx == 1 else (-1.8 if idx == 3 else 1.5))
        pill = patches.FancyBboxPatch(
            (mx - 7.8, my - 1.8), 15.6, 3.6,
            boxstyle="round,pad=0.2,rounding_size=1.4",
            facecolor="#FFFFFF", edgecolor=lcol, linewidth=1.3, zorder=5,
        )
        ax.add_patch(pill)
        ax.text(mx, my, f"w{idx}={wv:+.2f} → {contrib:+.2f}", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1E293B", zorder=6)

    # Bias inferior
    bias_box = patches.FancyBboxPatch(
        (41.5, 2.0), 20.0, 5.8,
        boxstyle="round,pad=0.3,rounding_size=1.4",
        facecolor="#F5F3FF", edgecolor="#7C3AED", linewidth=1.6, zorder=3,
    )
    ax.add_patch(bias_box)
    ax.text(51.5, 5.8, "Sesgo / Bias (x_b = 1)", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#5B21B6", zorder=4)
    ax.text(51.5, 3.4, f"b = {model.wb:+.2f}", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#7C3AED", zorder=4)
    ax.annotate(
        "", xy=(soma_cx, soma_cy - 6.9), xytext=(51.5, 8.2),
        arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=2.2, mutation_scale=13),
        zorder=2,
    )

    # Soma (círculo amarillo dorado)
    soma = patches.Circle((soma_cx, soma_cy), radius=6.6, facecolor="#FEF3C7", edgecolor="#D97706", linewidth=2.4, zorder=4)
    ax.add_patch(soma)
    ax.text(soma_cx, soma_cy + 2.6, "Σ", ha="center", va="center", fontsize=14, fontweight="bold", color="#92400E", zorder=5)
    ax.text(soma_cx, soma_cy - 0.3, "z (Net)", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#78350F", zorder=5)
    ax.text(soma_cx, soma_cy - 3.0, f"{z_val:+.2f}", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#111827", zorder=5)

    # Flecha Soma -> Activación
    ax.annotate(
        "", xy=(64.0, soma_cy), xytext=(soma_cx + 6.8, soma_cy),
        arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2.2, mutation_scale=13),
        zorder=2,
    )

    # Caja función de activación hardlim
    act_box = patches.FancyBboxPatch(
        (64.5, soma_cy - 6.0), 14.0, 12.0,
        boxstyle="round,pad=0.3,rounding_size=1.5",
        facecolor="#F8FAFC", edgecolor="#334155", linewidth=1.8, zorder=3,
    )
    ax.add_patch(act_box)
    ax.text(71.5, soma_cy + 4.1, "f(z) · hardlim", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#334155", zorder=4)
    # Mini escalón dentro de la caja
    ax.plot([67.2, 75.8], [soma_cy - 1.6, soma_cy - 1.6], color="#94A3B8", lw=0.9, zorder=4)
    ax.plot([71.5, 71.5], [soma_cy - 3.3, soma_cy + 2.4], color="#94A3B8", lw=0.9, zorder=4)
    ax.plot([67.5, 71.5, 71.5, 75.5], [soma_cy - 1.6, soma_cy - 1.6, soma_cy + 1.6, soma_cy + 1.6], color="#10B981", lw=2.4, zorder=5)
    ax.text(71.5, soma_cy - 4.5, "z ≥ 0 → 1", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#047857", zorder=4)

    # Flecha Activación -> Salida
    out_col = "#10B981" if y_pred == 1 else "#EF4444"
    out_bg = "#ECFDF5" if y_pred == 1 else "#FEF2F2"
    ax.annotate(
        "", xy=(82.2, soma_cy), xytext=(78.9, soma_cy),
        arrowprops=dict(arrowstyle="-|>", color=out_col, lw=2.5, mutation_scale=13),
        zorder=2,
    )

    # Caja de salida final
    out_box = patches.FancyBboxPatch(
        (82.6, soma_cy - 5.6), 14.8, 11.2,
        boxstyle="round,pad=0.3,rounding_size=1.6",
        facecolor=out_bg, edgecolor=out_col, linewidth=2.4, zorder=3,
    )
    ax.add_patch(out_box)
    ax.text(90.0, soma_cy + 3.1, "Salida", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#334155", zorder=4)
    ax.text(90.0, soma_cy - 0.2, f"ŷ = {y_pred}", ha="center", va="center", fontsize=13.5, fontweight="bold", color=out_col, zorder=4)
    ax.text(90.0, soma_cy - 3.4, "APROBADO (1)" if y_pred == 1 else "DESAPROBADO (0)", ha="center", va="center", fontsize=7.6, fontweight="bold", color=out_col, zorder=4)

    # --- Panel Inferior Izquierdo: Bloques de cálculo + Dictamen ---
    ax_txt = fig.add_subplot(gs[1, 0])
    ax_txt.set_xlim(0, 100)
    ax_txt.set_ylim(0, 42)
    ax_txt.axis("off")

    b1 = patches.FancyBboxPatch(
        (1, 28), 98, 11, boxstyle="round,pad=0.3,rounding_size=1.8",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2,
    )
    ax_txt.add_patch(b1)
    ax_txt.text(50, 33.5, "z = (1)(0.50) + (0)(0.50) + (1)(0.50) − 1.00 = +0.00", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#111827")

    b2 = patches.FancyBboxPatch(
        (1, 15), 98, 10, boxstyle="round,pad=0.3,rounding_size=1.8",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2,
    )
    ax_txt.add_patch(b2)
    ax_txt.text(50, 20.0, "Como z = +0.00 ≥ 0   →   ŷ = 1 (APROBADO)", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#111827")

    b3 = patches.FancyBboxPatch(
        (1, 1.5), 98, 11, boxstyle="round,pad=0.3,rounding_size=1.8",
        facecolor="#ECFDF5", edgecolor="#10B981", linewidth=1.8,
    )
    ax_txt.add_patch(b3)
    ax_txt.text(50, 8.6, "ESTUDIANTE APROBADO (ŷ = 1)", ha="center", va="center", fontsize=9.4, fontweight="bold", color="#047857")
    ax_txt.text(50, 4.3, "Tiene 2 de 3 unidades aprobadas (Patrón P6). Cada unidad aprobada suma +0.50.", ha="center", va="center", fontsize=7.6, color="#065F46")

    # --- Panel Inferior Derecho: Función Escalón hardlim ---
    ax_h = fig.add_subplot(gs[1, 1])
    ax_h.axvspan(-2.2, 0, color="#EF4444", alpha=0.06)
    ax_h.axvspan(0, 2.2, color="#10B981", alpha=0.07)
    ax_h.plot([-2.2, 0], [0, 0], color="#EF4444", linewidth=2.8, label="Desaprobado (z < 0 → ŷ = 0)")
    ax_h.plot([0, 2.2], [1, 1], color="#10B981", linewidth=2.8, label="Aprobado (z ≥ 0 → ŷ = 1)")
    ax_h.scatter([0], [0], s=55, facecolor="white", edgecolor="#EF4444", linewidth=2.0, zorder=5)
    ax_h.scatter([0], [1], s=55, facecolor="#10B981", edgecolor="#10B981", zorder=5)
    ax_h.scatter([z_val], [y_pred], s=110, marker="D", facecolor="#F4B400", edgecolor="#10B981", linewidth=2.0, zorder=6, label="Estudiante actual")
    ax_h.annotate(
        f"z = {z_val:+.2f} → ŷ = {y_pred}",
        xy=(z_val, y_pred), xytext=(0.45, 0.52),
        arrowprops=dict(arrowstyle="->", color="#111827", lw=1.2),
        fontsize=8.2, fontweight="bold", color="#111827",
    )
    ax_h.axvline(0, color="#94A3B8", linestyle=":", linewidth=1.1)
    ax_h.set_title("Función de Activación Escalón (hardlim)", fontsize=9.2, fontweight="bold", color="#111827", pad=6)
    ax_h.set_xlabel("Suma ponderada z (Net)", fontsize=8.2, color="#334155")
    ax_h.set_ylabel("Predicción ŷ", fontsize=8.2, color="#334155")
    ax_h.set_xlim(-2.2, 2.2)
    ax_h.set_ylim(-0.28, 1.32)
    ax_h.set_yticks([0, 1])
    ax_h.tick_params(labelsize=8)
    ax_h.grid(True, linestyle="--", alpha=0.5)
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)

    fig.suptitle("Predicción del Perceptrón (¿cómo decide la neurona?) — Caso U1=14.0, U2=10.0, U3=12.5",
                 fontsize=11.0, fontweight="bold", color="#111827", y=0.98)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.91, bottom=0.10)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def generate_figures(model: PerceptronRNA) -> dict[str, Path]:
    ASSETS_DIR.mkdir(exist_ok=True)
    epoch_df = model.get_epoch_summary_df()
    history_df = model.get_history_df()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.alpha": 0.8,
        }
    )

    # --- Figura 1: Errores acumulados por época ---
    fig1_path = ASSETS_DIR / "fig1_errores_epoca.png"
    fig, ax = plt.subplots(figsize=(7.2, 3.5), dpi=200)
    epochs = epoch_df["Época"].tolist()
    errors = epoch_df["Errores"].tolist()
    bar_colors = ["#10B981" if e == 0 else "#EF4444" for e in errors]

    bars = ax.bar(epochs, errors, width=0.52, color=bar_colors, edgecolor="#1F2937", linewidth=0.8, zorder=3)
    ax.plot(epochs, errors, color="#D97706", linestyle=":", linewidth=2.2, marker="o", markersize=7,
            markerfacecolor="#F4B400", markeredgecolor="#111827", zorder=4)

    for bar, err in zip(bars, errors):
        lbl = "1 error" if err == 1 else f"{err} errores"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            err + 0.25,
            lbl,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#111827",
        )

    ax.set_title("Errores cometidos por la Red Neuronal en cada Época", fontsize=11.5, fontweight="bold", pad=12, color="#111827")
    ax.set_xlabel("Época de Entrenamiento", fontsize=9.5, color="#334155")
    ax.set_ylabel("Número de Errores (de 8 patrones)", fontsize=9.5, color="#334155")
    ax.set_xticks(epochs)
    ax.set_ylim(0, max(errors) + 1.5)
    ax.grid(axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig1_path, dpi=200, facecolor="white")
    plt.close(fig)

    # --- Figura 2: Evolución de pesos y sesgo paso a paso ---
    fig2_path = ASSETS_DIR / "fig2_evolucion_pesos.png"
    pasos = [0] + history_df["Paso"].tolist()
    w1 = [float(model.initial_w[0])] + history_df["w1_nuevo"].tolist()
    w2 = [float(model.initial_w[1])] + history_df["w2_nuevo"].tolist()
    w3 = [float(model.initial_w[2])] + history_df["w3_nuevo"].tolist()
    wb = [float(model.initial_wb)] + history_df["wb_nuevo"].tolist()

    fig, ax = plt.subplots(figsize=(7.6, 3.7), dpi=200)
    ax.plot(pasos, w1, label="w1 (Unidad 1)", color="#2563EB", linewidth=2.1, marker="o", markersize=4.5)
    ax.plot(pasos, w2, label="w2 (Unidad 2)", color="#10B981", linewidth=2.1, linestyle="--", marker="s", markersize=4.2)
    ax.plot(pasos, w3, label="w3 (Unidad 3)", color="#F97316", linewidth=2.1, linestyle=":", marker="^", markersize=4.8)
    ax.plot(pasos, wb, label="b (Sesgo / Bias)", color="#7C3AED", linewidth=2.7, marker="D", markersize=4.5)

    for ep in range(1, model.epochs_run + 1):
        x_end = ep * 8
        ax.axvline(x_end, color="#94A3B8", linestyle="--", linewidth=1.0, zorder=1)
        ax.text(x_end - 0.3, 1.24, f"Fin Época {ep}", ha="right", va="top", fontsize=8.2, color="#475569", fontweight="bold")

    ax.axhline(0, color="#64748B", linewidth=0.9)
    ax.set_title("Evolución Paso a Paso de los Pesos Sinápticos y el Sesgo (b)", fontsize=11.5, fontweight="bold", pad=22, color="#111827")
    ax.set_xlabel("Paso de Entrenamiento (Paso 0 = Valores iniciales UPN)", fontsize=9.5, color="#334155")
    ax.set_ylabel("Valor del Parámetro", fontsize=9.5, color="#334155")
    ax.set_ylim(-1.25, 1.38)
    ax.set_xlim(-0.5, max(pasos) + 0.8)
    ax.grid(True, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=4, frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig2_path, dpi=200, facecolor="white")
    plt.close(fig)

    # --- Figura 3: Predicción del Perceptrón (¿cómo decide la neurona?) ---
    fig3_path = ASSETS_DIR / "fig3_prediccion_perceptron.png"
    draw_perceptron_prediction_figure(model, fig3_path)

    # --- Figura 4: Experimento de sensibilidad del sesgo (3 escenarios) ---
    fig4_path = ASSETS_DIR / "fig4_experimento_sesgo.png"
    scenarios = [
        ("Aprendido: b = -1.0\n(Regla >= 2 u. | 8/8 aciertos)", -1.0),
        ("Alterado: b = -0.5\n(Tipo OR | 5/8 aciertos)", -0.5),
        ("Alterado: b = -1.5\n(Tipo AND | 5/8 aciertos)", -1.5),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), dpi=200, sharey=True)
    for ax, (title, b_val) in zip(axes, scenarios):
        df_s, _, _ = evaluate_manual_parameters(0.5, 0.5, 0.5, b_val)
        pats = df_s["Patrón"].tolist()
        zs = df_s["z (Net)"].tolist()
        cols = ["#10B981" if ok else "#EF4444" for ok in df_s["Acertó"]]
        ax.axhspan(0, 1.6, color="#10B981", alpha=0.06)
        ax.axhspan(-1.8, 0, color="#EF4444", alpha=0.05)
        ax.bar(pats, zs, color=cols, edgecolor="#1F2937", linewidth=0.6, zorder=3)
        for p_idx, (z_v, ok) in enumerate(zip(zs, df_s["Acertó"])):
            if not ok:
                ax.scatter([p_idx], [z_v], color="#DC2626", marker="X", s=65, edgecolor="white", linewidth=0.8, zorder=5)
        ax.axhline(0, color="#D97706", linestyle="--", linewidth=1.6, zorder=4)
        ax.set_title(title, fontsize=9, fontweight="bold", color="#111827", pad=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(-1.75, 1.45)
        ax.grid(axis="y", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Valor de z (Net)", fontsize=9, color="#334155")
    fig.suptitle("Experimento de Sensibilidad: Efecto de Modificar el Sesgo b (Verde = Acierto | Rojo = Error)",
                 fontsize=10.5, fontweight="bold", color="#111827", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig4_path, dpi=200, facecolor="white")
    plt.close(fig)

    return {"fig1": fig1_path, "fig2": fig2_path, "fig3": fig3_path, "fig4": fig4_path}


# ==========================================
# 2. DATOS COMPARTIDOS DEL INFORME
# ==========================================
COVER_META = [
    ("Institución", "Universidad Privada del Norte (UPN)"),
    ("Curso", "Sistemas Inteligentes"),
    ("Tema", "Sesión 14: Redes Neuronales Artificiales (Perceptrón Simple)"),
    ("Docente", "Marcelino Torres Villanueva"),
    ("Estudiante", "Marco Antonio Gamez Monzon"),
    ("Código de Estudiante", "N00324879"),
    ("Ciclo", "10mo ciclo"),
    ("Fecha", "Noviembre de 2026"),
    ("Tecnologías", "Python, Streamlit, NumPy, Pandas, Plotly / Altair"),
]

TRUTH_TRAIN_ROWS = [
    ["Patrón", "x1 (U1)", "x2 (U2)", "x3 (U3)", "Unid. Aprob.", "Yd", "Estado Esperado"],
    ["P1", "0", "0", "0", "0", "0", "Desaprobado"],
    ["P2", "0", "0", "1", "1", "0", "Desaprobado"],
    ["P3", "0", "1", "0", "1", "0", "Desaprobado"],
    ["P4", "0", "1", "1", "2", "1", "Aprobado"],
    ["P5", "1", "0", "0", "1", "0", "Desaprobado"],
    ["P6", "1", "0", "1", "2", "1", "Aprobado"],
    ["P7", "1", "1", "0", "2", "1", "Aprobado"],
    ["P8", "1", "1", "1", "3", "1", "Aprobado"],
]

EPOCH_SUMMARY_ROWS = [
    ["Época", "Errores", "Aciertos", "Precisión", "w1 final", "w2 final", "w3 final", "Sesgo (b)"],
    ["1", "6", "2 / 8", "25.0 %", "0.50", "0.50", "1.00", "-0.50"],
    ["2", "1", "7 / 8", "87.5 %", "0.50", "0.50", "0.50", "-1.00"],
    ["3", "0", "8 / 8", "100.0 %", "0.50", "0.50", "0.50", "-1.00"],
]

VERIFY_ROWS = [
    ["Patrón", "Entrada (x1, x2, x3)", "Suma z (Net)", "Predicción ŷ", "Esperado Yd", "Error", "Dictamen"],
    ["P1", "(0, 0, 0)", "-1.00", "0", "0", "0", "Desaprobado"],
    ["P2", "(0, 0, 1)", "-0.50", "0", "0", "0", "Desaprobado"],
    ["P3", "(0, 1, 0)", "-0.50", "0", "0", "0", "Desaprobado"],
    ["P4", "(0, 1, 1)", "0.00", "1", "1", "0", "Aprobado"],
    ["P5", "(1, 0, 0)", "-0.50", "0", "0", "0", "Desaprobado"],
    ["P6", "(1, 0, 1)", "0.00", "1", "1", "0", "Aprobado"],
    ["P7", "(1, 1, 0)", "0.00", "1", "1", "0", "Aprobado"],
    ["P8", "(1, 1, 1)", "+0.50", "1", "1", "0", "Aprobado"],
]

SENSITIVITY_ROWS = [
    ["Configuración probada", "Comportamiento lógico", "Aciertos", "Errores", "Casos fallidos"],
    ["b = -1.0 (Aprendido)", "Mayoría (>= 2 unidades)", "8 / 8", "0", "Ninguno"],
    ["b = -0.5", "Compuerta OR (>= 1 unidad)", "5 / 8", "3", "P2, P3, P5"],
    ["b = -1.5", "Compuerta AND (3 unidades)", "5 / 8", "3", "P4, P6, P7"],
    ["b = 0.0", "Aprueba incluso con 0 unid.", "4 / 8", "4", "P1, P2, P3, P5"],
]

INDICATORS_ROWS = [
    ["Indicador", "Mecanismo implementado", "Caso de validación"],
    ["a. 3 promedios (3 unidades)", "3 variables de entrada (P1, P2, P3) -> (x1, x2, x3).", "Diagrama de 3 dendritas y 2^3 = 8 patrones."],
    ["b. Mínimo 2 unid. aprobadas", "Salida deseada Yd = 1 únicamente si x1 + x2 + x3 >= 2.", "Tabla final con 8/8 aciertos (Error = 0)."],
    [
        "c. Sin cálculo de promedio final",
        "La binarización previa descarta la nota continua antes de predecir.",
        "Caso 1: (12, 12, 0) -> prom. 8.00 pero APROBADO.\nCaso 2: (20, 11.5, 11.5) -> prom. 14.33 pero DESAPROBADO.",
    ],
    ["d. Rango de 0 a 20 por unidad", "Controles deslizantes limitados estrictamente a [0.0, 20.0].", "Imposible ingresar notas fuera del rango 0-20."],
]


# ==========================================
# 3. EXPORTADOR A WORD (.DOCX)
# ==========================================
def set_cell_shading(cell, fill_hex: str) -> None:
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_margins(cell, top: int = 80, bottom: int = 80, left: int = 120, right: int = 120) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def add_page_number_field(run) -> None:
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls("w"))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls("w"))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls("w"))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls("w"))
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)


def add_docx_table(doc: Document, rows: list[list[str]], col_widths_cm: list[float] | None = None, align_center_from: int = 1) -> None:
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    for r_idx, row_vals in enumerate(rows):
        row = table.rows[r_idx]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=65, bottom=65, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (r_idx == 0 or c_idx >= align_center_from) else WD_ALIGN_PARAGRAPH.LEFT

            run = p.add_run(val)
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            if r_idx == 0:
                set_cell_shading(cell, "111827")
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            else:
                if r_idx % 2 == 0:
                    set_cell_shading(cell, "F8FAFC")
                run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

            if col_widths_cm:
                cell.width = Cm(col_widths_cm[c_idx])

    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)


def add_docx_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
        pPr = p._p.get_or_add_pPr()
        pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="12" w:space="3" w:color="F4B400"/></w:pBdr>')
        pPr.append(pBdr)
    else:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.bold = True
        run.font.name = "Calibri"
        run.font.size = Pt(11.5)
        run.font.color.rgb = RGBColor(0xB4, 0x53, 0x09)


def add_docx_formula_box(doc: Document, formula_lines: list[str]) -> None:
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_shading(cell, "FEFCE8")
    set_cell_margins(cell, top=90, bottom=90, left=160, right=160)
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="4" w:color="F4B400"/>'
        '<w:left w:val="single" w:sz="24" w:color="F4B400"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="F4B400"/>'
        '<w:right w:val="single" w:sz="4" w:color="F4B400"/>'
        "</w:tcBorders>"
    )
    tcPr.append(borders)

    for idx, line in enumerate(formula_lines):
        p = cell.paragraphs[0] if idx == 0 else cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(line)
        run.bold = True
        run.font.name = "Consolas"
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(0x11, 0x18, 0x27)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)


def add_docx_body(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)


def add_docx_bullet(doc: Document, bold_prefix: str, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.12
    r1 = p.add_run(bold_prefix)
    r1.bold = True
    r1.font.name = "Calibri"
    r1.font.size = Pt(10.5)
    r1.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
    r2 = p.add_run(text)
    r2.font.name = "Calibri"
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)


def add_docx_figure(doc: Document, img_path: Path, caption: str, width_inches: float = 6.0) -> None:
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.keep_with_next = True
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.add_run().add_picture(str(img_path), width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(8)
    r = p_cap.add_run(caption)
    r.italic = True
    r.font.name = "Calibri"
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)


def build_docx(figs: dict[str, Path]) -> None:
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.3)
    section.right_margin = Cm(2.3)
    section.different_first_page_header_footer = True

    hdr = section.header.paragraphs[0]
    hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_hdr = hdr.add_run("UPN · Sistemas Inteligentes (Sesión 14) — Informe Técnico: Perceptrón Simple")
    r_hdr.font.name = "Calibri"
    r_hdr.font.size = Pt(8.5)
    r_hdr.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    ftr = section.footer.paragraphs[0]
    ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_f1 = ftr.add_run("Universidad Privada del Norte  |  Página ")
    r_f1.font.name = "Calibri"
    r_f1.font.size = Pt(8.5)
    r_f1.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    r_pg = ftr.add_run()
    r_pg.font.name = "Calibri"
    r_pg.font.size = Pt(8.5)
    r_pg.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    add_page_number_field(r_pg)

    # ---------- PORTADA FORMAL UPN ----------
    banner = doc.add_table(rows=1, cols=1)
    banner.alignment = WD_TABLE_ALIGNMENT.CENTER
    b_cell = banner.cell(0, 0)
    set_cell_shading(b_cell, "111827")
    set_cell_margins(b_cell, top=180, bottom=180, left=200, right=200)
    bp = b_cell.paragraphs[0]
    bp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_u = bp.add_run("UNIVERSIDAD PRIVADA DEL NORTE (UPN)\n")
    r_u.bold = True
    r_u.font.name = "Calibri"
    r_u.font.size = Pt(17)
    r_u.font.color.rgb = RGBColor(0xF4, 0xB4, 0x00)
    r_f = bp.add_run("FACULTAD DE INGENIERÍA\nCarrera Profesional de Ingeniería de Sistemas Computacionales")
    r_f.font.name = "Calibri"
    r_f.font.size = Pt(11.5)
    r_f.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for _ in range(3):
        doc.add_paragraph()

    p_tag = doc.add_paragraph()
    p_tag.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_tag = p_tag.add_run("INFORME TÉCNICO — SESIÓN 14")
    r_tag.bold = True
    r_tag.font.name = "Calibri"
    r_tag.font.size = Pt(12)
    r_tag.font.color.rgb = RGBColor(0xB4, 0x53, 0x09)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(6)
    p_title.paragraph_format.space_after = Pt(10)
    r_title = p_title.add_run(
        "Entrenamiento de Red Neuronal Artificial (Perceptrón)\npara Evaluación de Aprobación Estudiantil"
    )
    r_title.bold = True
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(20)
    r_title.font.color.rgb = RGBColor(0x11, 0x18, 0x27)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(24)
    r_sub = p_sub.add_run(
        "Implementación supervisada con Regla Delta Generalizada, función de activación escalón (hardlim) "
        "y simulador interactivo en Python / Streamlit"
    )
    r_sub.italic = True
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(10.5)
    r_sub.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    meta_rows = [["Campo", "Detalle"]] + [[k, v] for k, v in COVER_META]
    add_docx_table(doc, meta_rows, col_widths_cm=[5.5, 10.9], align_center_from=99)
    doc.add_page_break()

    # ---------- DESARROLLO DIRECTO DEL EJERCICIO ----------
    add_docx_heading(doc, "1. Especificación de Reglas de Negocio y Preprocesamiento", 1)
    add_docx_body(
        doc,
        "El ejercicio de la Sesión 14 exige entrenar una Red Neuronal Artificial (Perceptrón Simple) que determine la "
        "aprobación curricular de un estudiante cumpliendo estrictamente cuatro indicadores:",
    )
    add_docx_bullet(doc, "a. Tres promedios (3 unidades): ", "Se consideran exactamente 3 notas correspondientes a las unidades P1, P2 y P3.")
    add_docx_bullet(doc, "b. Mínimo 2 unidades aprobadas: ", "El estudiante aprueba la experiencia curricular solo si tiene al menos 2 unidades aprobadas.")
    add_docx_bullet(doc, "c. Sin cálculo matemático para el resultado final: ", "No se calcula promedio aritmético ni ponderado; se aplica únicamente la regla del punto b.")
    add_docx_bullet(doc, "d. Rango de 0 a 20 por unidad: ", "Cada promedio de unidad debe estar comprendido entre 0 y 20.")

    add_docx_heading(doc, "1.1 Criterio de Binarización", 2)
    add_docx_body(
        doc,
        "Dado que la nota mínima aprobatoria en la UPN es 12, cada nota continua Pi en [0, 20] se transforma "
        "en una entrada binaria xi antes de ingresar a la red neuronal:",
    )
    add_docx_formula_box(
        doc,
        [
            "xi = 1   si Pi >= 12   (Unidad Aprobada)",
            "xi = 0   si Pi <  12   (Unidad Desaprobada)",
            "Regla lógica aprendida:   Yd = 1   si y solo si   x1 + x2 + x3 >= 2",
        ],
    )

    add_docx_heading(doc, "2. Matriz de Patrones de Entrenamiento (Tabla de Verdad)", 1)
    add_docx_body(doc, "Con 3 entradas binarias existen exactamente 2^3 = 8 patrones posibles de entrenamiento:")
    add_docx_table(doc, TRUTH_TRAIN_ROWS, col_widths_cm=[1.8, 1.9, 1.9, 1.9, 2.5, 1.8, 4.6], align_center_from=0)

    add_docx_heading(doc, "3. Algoritmo de Entrenamiento (Regla Delta Generalizada)", 1)
    add_docx_body(
        doc,
        "Durante cada época, el Perceptrón recorre secuencialmente los 8 patrones aplicando la suma ponderada "
        "(Net o z), la función de activación escalón (hardlim) y la actualización de pesos y sesgo (wb o b) cuando existe error:",
    )
    add_docx_formula_box(
        doc,
        [
            "Suma ponderada:      z (Net) = x1·w1 + x2·w2 + x3·w3 + 1·b",
            "Activación hardlim:  ŷ = 1 si z >= 0,   ŷ = 0 si z < 0",
            "Error:               e = Yd - ŷ",
            "Actualización (e≠0): wi(t+1) = wi(t) + fa·e·xi   |   b(t+1) = b(t) + fa·e",
        ],
    )

    add_docx_heading(doc, "4. Ejecución del Entrenamiento, Convergencia y Predicción", 1)
    add_docx_body(
        doc,
        "Siguiendo la metodología UPN, se fijaron los parámetros iniciales w1 = 0.5, w2 = 0.5, w3 = 0.5, sesgo b = 0.5 "
        "y factor de aprendizaje fa = 0.5.",
    )

    add_docx_heading(doc, "4.1 Resumen de Épocas (3 épocas × 8 patrones = 24 evaluaciones)", 2)
    add_docx_bullet(doc, "Época 1 (Pasos 1 al 8 — 6 errores): ", "Se corrigen los parámetros en P1 (-1), P2 (-1), P3 (-1), P4 (+1), P5 (-1) y P6 (+1); P7 y P8 ya aciertan. Cierra con w = (0.50, 0.50, 1.00) y b = -0.50.")
    add_docx_bullet(doc, "Época 2 (Pasos 9 al 16 — 1 error): ", "Únicamente falla P2 (0, 0, 1) en la evaluación 10 de 24 (z = +0.50 -> ŷ = 1, e = -1). El ajuste fija w3 = 0.50 y b = -1.00; desde P3 hasta P8 todos aciertan.")
    add_docx_bullet(doc, "Época 3 (Pasos 17 al 24 — 0 errores): ", "Se verifican los 8 patrones obteniendo 0 errores y 100 % de precisión, finalizando el entrenamiento por convergencia.")

    add_docx_table(doc, EPOCH_SUMMARY_ROWS, col_widths_cm=[1.7, 1.8, 2.0, 2.3, 2.1, 2.1, 2.1, 2.3], align_center_from=0)
    add_docx_figure(doc, figs["fig1"], "Figura 1. Errores cometidos por la red neuronal en cada época hasta alcanzar convergencia (0 errores).", 5.6)
    add_docx_figure(doc, figs["fig2"], "Figura 2. Evolución paso a paso de los pesos sinápticos (w1, w2, w3) y el sesgo (b) en las 24 evaluaciones.", 6.0)

    add_docx_heading(doc, "4.2 Modelo Matemático Aprendido y Verificación Final", 2)
    add_docx_formula_box(doc, ["z = 0.50·x1 + 0.50·x2 + 0.50·x3 − 1.00"])
    add_docx_body(
        doc,
        "Interpretación práctica: cada unidad aprobada aporta +0.50 puntos y el sesgo (-1.00) establece el requisito "
        "mínimo. Con 0 o 1 unidad aprobada la suma resulta negativa (-1.00 o -0.50 -> Desaprobado); con 2 o 3 "
        "unidades aprobadas alcanza 0.00 o +0.50 -> Aprobado.",
    )
    add_docx_table(doc, VERIFY_ROWS, col_widths_cm=[1.6, 3.0, 2.3, 2.2, 2.2, 1.5, 3.6], align_center_from=0)

    add_docx_heading(doc, "4.3 Predicción del Perceptrón (¿cómo decide la neurona?)", 2)
    add_docx_body(
        doc,
        "Una vez entrenada la red, la Figura 3 ilustra en vivo cómo decide la neurona para un caso práctico "
        "(U1 = 14.0, U2 = 10.0, U3 = 12.5): las notas se binarizan a (1, 0, 1), se multiplican por sus pesos "
        "(0.50 cada uno), se resta el sesgo (-1.00) obteniendo z = 0.00 y la función escalón (hardlim) dispara ŷ = 1 (APROBADO).",
    )
    add_docx_figure(doc, figs["fig3"], "Figura 3. Predicción del Perceptrón (¿cómo decide la neurona?) mostrando entradas binarizadas, pesos, suma ponderada, función escalón hardlim y dictamen final.", 6.4)

    add_docx_heading(doc, "4.4 Experimento de Sensibilidad sobre el Sesgo (b)", 2)
    add_docx_body(
        doc,
        "Para comprobar el papel determinante del sesgo en la clasificación, se varió manualmente b manteniendo "
        "w1 = w2 = w3 = 0.50. Como muestra la tabla y la Figura 4, cambiar b desplaza la frontera z = 0 y rompe la "
        "regla de negocio convirtiendo la neurona en una compuerta OR o AND:",
    )
    add_docx_table(doc, SENSITIVITY_ROWS, col_widths_cm=[3.8, 4.6, 2.0, 1.8, 4.2], align_center_from=2)
    add_docx_figure(doc, figs["fig4"], "Figura 4. Experimento de sensibilidad al modificar el sesgo b frente a la frontera z = 0 (Verde = Acierto | Rojo = Error).", 6.3)

    add_docx_heading(doc, "5. Validación de los 4 Indicadores del Ejercicio", 1)
    add_docx_table(doc, INDICATORS_ROWS, col_widths_cm=[4.2, 5.7, 6.5], align_center_from=99)

    add_docx_heading(doc, "6. Arquitectura del Simulador Interactivo en Streamlit", 1)
    add_docx_body(
        doc,
        "La solución fue implementada en Python bajo una arquitectura modular de tres archivos integrados con "
        "Streamlit, NumPy, Pandas y Plotly (con respaldo automático en Altair):",
    )
    add_docx_bullet(doc, "perceptron.py (Capa de Modelo): ", "Implementa la clase PerceptronRNA, la binarización binarize_grades(), el registro completo de las 24 iteraciones y evaluate_manual_parameters() para el experimento de sensibilidad.")
    add_docx_bullet(doc, "visualizations.py (Capa de Visualización): ", "Genera el diagrama SVG dinámico en vivo de la neurona y los gráficos interactivos de errores por época, evolución de pesos, función escalón hardlim y comparación del experimento con el sesgo.")
    add_docx_bullet(doc, "app.py (Capa de Interfaz Web): ", "Estructura la aplicación en 4 pestañas pedagógicas: (1) Simulador en vivo y verificación de reglas a-d, (2) Entrenamiento paso a paso con selector de Época y Patrón + exportación CSV, (3) Modelo aprendido y experimento interactivo con el sesgo, y (4) Resumen teórico.")

    add_docx_heading(doc, "7. Conclusiones", 1)
    add_docx_bullet(doc, "Convergencia rápida y exacta: ", "El criterio de mayoría (>= 2 de 3 unidades aprobadas) es linealmente separable, permitiendo que el Perceptrón converja al 100 % de precisión en solo 3 épocas (24 evaluaciones) con w = (0.50, 0.50, 0.50) y b = -1.00.")
    add_docx_bullet(doc, "Cumplimiento estricto sin promedio aritmético: ", "La binarización previa (Pi >= 12) asegura que la red evalúe únicamente la cantidad de unidades aprobadas, comprobado con los contraejemplos (12, 12, 0) y (20, 11.5, 11.5).")
    add_docx_bullet(doc, "Importancia crítica del sesgo: ", "Modificar el sesgo aprendido b = -1.00 hacia -0.50 o -1.50 altera la frontera z = 0 y produce 3 errores sobre los 8 casos posibles.")
    add_docx_bullet(doc, "Valor pedagógico del simulador web: ", "El despliegue interactivo en Streamlit con diagrama dinámico, recorrido por época/patrón y experimentos en tiempo real hace comprensible y verificable todo el proceso de decisión de la RNA.")

    doc.save(DOCX_PATH)


# ==========================================
# 4. EXPORTADOR A PDF (.PDF CON REPORTLAB)
# ==========================================
def register_unicode_fonts() -> tuple[str, str, str, str]:
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    reg_path = font_dir / "DejaVuSans.ttf"
    bold_path = font_dir / "DejaVuSans-Bold.ttf"
    obl_path = font_dir / "DejaVuSans-Oblique.ttf"
    mono_path = font_dir / "DejaVuSansMono-Bold.ttf"
    if reg_path.exists() and bold_path.exists() and obl_path.exists() and mono_path.exists():
        pdfmetrics.registerFont(TTFont("DV", str(reg_path)))
        pdfmetrics.registerFont(TTFont("DV-Bold", str(bold_path)))
        pdfmetrics.registerFont(TTFont("DV-Oblique", str(obl_path)))
        pdfmetrics.registerFont(TTFont("DV-MonoBold", str(mono_path)))
        pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-Bold", italic="DV-Oblique", boldItalic="DV-Bold")
        return ("DV", "DV-Bold", "DV-Oblique", "DV-MonoBold")
    return ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier-Bold")


PDF_F_REG, PDF_F_BOLD, PDF_F_ITALIC, PDF_F_MONO = register_unicode_fonts()


def draw_pdf_header_footer(canvas, doc) -> None:
    if doc.page == 1:
        return
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#F4B400"))
    canvas.setLineWidth(1.2)
    canvas.line(2.0 * cm, h - 1.65 * cm, w - 2.0 * cm, h - 1.65 * cm)
    canvas.setFont(PDF_F_REG, 7.8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawRightString(
        w - 2.0 * cm,
        h - 1.45 * cm,
        "UPN · Sistemas Inteligentes (Sesión 14) — Informe Técnico: Perceptrón Simple",
    )
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.setLineWidth(0.7)
    canvas.line(2.0 * cm, 1.65 * cm, w - 2.0 * cm, 1.65 * cm)
    canvas.drawCentredString(w / 2.0, 1.15 * cm, f"Universidad Privada del Norte  |  Página {doc.page}")
    canvas.restoreState()


def build_pdf(figs: dict[str, Path]) -> None:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.1 * cm,
        bottomMargin=2.1 * cm,
        title="Informe Técnico: Perceptrón Simple - Sesión 14 UPN",
        author="Marco Antonio Gamez Monzon",
    )

    styles = getSampleStyleSheet()
    s_body = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName=PDF_F_REG,
        fontSize=9.3,
        leading=13.8,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6,
    )
    s_bullet = ParagraphStyle(
        "BulletCustom",
        parent=s_body,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=4,
    )
    s_h1 = ParagraphStyle(
        "H1Custom",
        parent=styles["Heading1"],
        fontName=PDF_F_BOLD,
        fontSize=12.2,
        leading=15.5,
        textColor=colors.HexColor("#111827"),
        spaceBefore=12,
        spaceAfter=4,
        keepWithNext=True,
    )
    s_h2 = ParagraphStyle(
        "H2Custom",
        parent=styles["Heading2"],
        fontName=PDF_F_BOLD,
        fontSize=10.2,
        leading=13.5,
        textColor=colors.HexColor("#B45309"),
        spaceBefore=9,
        spaceAfter=4,
        keepWithNext=True,
    )
    s_formula = ParagraphStyle(
        "FormulaCustom",
        parent=styles["Normal"],
        fontName=PDF_F_MONO,
        fontSize=8.8,
        leading=13.2,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111827"),
    )
    s_caption = ParagraphStyle(
        "CaptionCustom",
        parent=styles["Normal"],
        fontName=PDF_F_ITALIC,
        fontSize=8.2,
        leading=11.2,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
        spaceBefore=3,
        spaceAfter=9,
    )
    s_cell = ParagraphStyle(
        "CellCustom",
        parent=styles["Normal"],
        fontName=PDF_F_REG,
        fontSize=8.2,
        leading=10.8,
        textColor=colors.HexColor("#1E293B"),
    )
    s_cell_c = ParagraphStyle("CellCenter", parent=s_cell, alignment=TA_CENTER)
    s_th = ParagraphStyle(
        "ThCustom",
        parent=s_cell_c,
        fontName=PDF_F_BOLD,
        textColor=colors.white,
    )

    def make_h1(text: str) -> KeepTogether:
        return KeepTogether(
            [
                Paragraph(text, s_h1),
                HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#F4B400"), spaceAfter=6),
            ]
        )

    def make_formula_box(lines: list[str]) -> Table:
        html = "<br/>".join(lines)
        t = Table([[Paragraph(html, s_formula)]], colWidths=[17.0 * cm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEFCE8")),
                    ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#F4B400")),
                    ("LINEBEFORE", (0, 0), (0, -1), 3.5, colors.HexColor("#F4B400")),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        return t

    def make_table(rows: list[list[str]], col_widths_cm: list[float], center_from: int = 1) -> Table:
        data = []
        for r_idx, row in enumerate(rows):
            formatted_row = []
            for c_idx, val in enumerate(row):
                txt = val.replace("\n", "<br/>")
                if r_idx == 0:
                    formatted_row.append(Paragraph(txt, s_th))
                else:
                    st = s_cell_c if c_idx >= center_from else s_cell
                    formatted_row.append(Paragraph(txt, st))
            data.append(formatted_row)

        tbl = Table(data, colWidths=[w * cm for w in col_widths_cm], repeatRows=1)
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]
        for r_idx in range(1, len(rows)):
            if r_idx % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx), colors.HexColor("#F8FAFC")))
        tbl.setStyle(TableStyle(style_cmds))
        return tbl

    def make_figure(img_path: Path, caption: str, width_cm: float = 15.2) -> KeepTogether:
        from PIL import Image as PILImage

        with PILImage.open(img_path) as im:
            iw, ih = im.size
        aspect = ih / float(iw)
        img = RLImage(str(img_path), width=width_cm * cm, height=width_cm * aspect * cm)
        img.hAlign = "CENTER"
        return KeepTogether([Spacer(1, 4), img, Paragraph(caption, s_caption)])

    story = []

    # ---------- PORTADA FORMAL UPN ----------
    cover_banner_style = ParagraphStyle(
        "CoverBanner",
        fontName=PDF_F_BOLD,
        fontSize=14.5,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#F4B400"),
    )
    cover_banner_sub = ParagraphStyle(
        "CoverBannerSub",
        fontName=PDF_F_REG,
        fontSize=10,
        leading=14.5,
        alignment=TA_CENTER,
        textColor=colors.white,
    )
    banner_tbl = Table(
        [
            [Paragraph("UNIVERSIDAD PRIVADA DEL NORTE (UPN)", cover_banner_style)],
            [Paragraph("FACULTAD DE INGENIERÍA<br/>Carrera Profesional de Ingeniería de Sistemas Computacionales", cover_banner_sub)],
        ],
        colWidths=[17.0 * cm],
    )
    banner_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("LINEBELOW", (0, 1), (-1, 1), 4, colors.HexColor("#F4B400")),
                ("TOPPADDING", (0, 0), (-1, 0), 16),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ("TOPPADDING", (0, 1), (-1, 1), 2),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 16),
            ]
        )
    )
    story.append(banner_tbl)
    story.append(Spacer(1, 2.2 * cm))

    story.append(
        Paragraph(
            "INFORME TÉCNICO — SESIÓN 14",
            ParagraphStyle(
                "TagCover",
                fontName=PDF_F_BOLD,
                fontSize=11,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#B45309"),
                spaceAfter=10,
            ),
        )
    )
    story.append(
        Paragraph(
            "Entrenamiento de Red Neuronal Artificial (Perceptrón)<br/>para Evaluación de Aprobación Estudiantil",
            ParagraphStyle(
                "TitleCover",
                fontName=PDF_F_BOLD,
                fontSize=17.5,
                leading=24,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#111827"),
                spaceAfter=12,
            ),
        )
    )
    story.append(
        Paragraph(
            "Implementación supervisada con Regla Delta Generalizada, función de activación escalón (hardlim) "
            "y simulador interactivo en Python / Streamlit",
            ParagraphStyle(
                "SubCover",
                fontName=PDF_F_ITALIC,
                fontSize=9.6,
                leading=14,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#475569"),
                spaceAfter=28,
            ),
        )
    )

    meta_rows = [["Campo", "Detalle"]] + [[k, v] for k, v in COVER_META]
    story.append(make_table(meta_rows, [5.4, 11.6], center_from=99))
    story.append(PageBreak())

    # ---------- DESARROLLO DIRECTO DEL EJERCICIO ----------
    story.append(make_h1("1. Especificación de Reglas de Negocio y Preprocesamiento"))
    story.append(
        Paragraph(
            "El ejercicio de la Sesión 14 plantea entrenar un Perceptrón Simple que determine si un estudiante aprueba "
            "o no una experiencia curricular cumpliendo estrictamente cuatro indicadores:",
            s_body,
        )
    )
    story.append(Paragraph("• <b>a. Tres promedios (3 unidades):</b> Se consideran exactamente 3 notas (P<sub>1</sub>, P<sub>2</sub>, P<sub>3</sub>).", s_bullet))
    story.append(Paragraph("• <b>b. Mínimo 2 unidades aprobadas:</b> El estudiante aprueba el curso si aprueba al menos 2 de las 3 unidades.", s_bullet))
    story.append(Paragraph("• <b>c. Sin cálculo matemático de promedio final:</b> No se aplica promedio aritmético ni ponderado; solo rige el punto b.", s_bullet))
    story.append(Paragraph("• <b>d. Rango de 0 a 20 por unidad:</b> Cada promedio se restringe estrictamente al intervalo [0, 20].", s_bullet))

    story.append(Paragraph("1.1 Criterio de Binarización", s_h2))
    story.append(
        Paragraph(
            "Con la nota mínima aprobatoria de la UPN de 12, cada promedio continuo P<sub>i</sub> se binariza "
            "antes de enviarse a la neurona:",
            s_body,
        )
    )
    story.append(
        make_formula_box(
            [
                "x_i = 1   si P_i >= 12   (Unidad Aprobada)",
                "x_i = 0   si P_i <  12   (Unidad Desaprobada)",
                "Regla de mayoría:   Yd = 1   si y solo si   x1 + x2 + x3 >= 2",
            ]
        )
    )
    story.append(Spacer(1, 6))

    story.append(
        KeepTogether(
            [
                Paragraph("2. Matriz de Patrones de Entrenamiento (Tabla de Verdad)", s_h1),
                HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#F4B400"), spaceAfter=6),
                Paragraph("Con 3 entradas binarias se generan 2<super>3</super> = 8 patrones posibles:", s_body),
                make_table(TRUTH_TRAIN_ROWS, [1.8, 2.0, 2.0, 2.0, 2.6, 1.8, 4.8], center_from=0),
                Spacer(1, 6),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("3. Algoritmo de Entrenamiento (Regla Delta Generalizada)", s_h1),
                HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#F4B400"), spaceAfter=6),
                Paragraph(
                    "En cada época el Perceptrón recorre los 8 patrones calculando la suma ponderada z (Net), la "
                    "activación escalón hardlim y la corrección de pesos y sesgo b (w<sub>b</sub>) cuando hay error:",
                    s_body,
                ),
                make_formula_box(
                    [
                        "Suma ponderada:      z (Net) = x1·w1 + x2·w2 + x3·w3 + 1·b",
                        "Activación hardlim:  ŷ = 1 si z >= 0,   ŷ = 0 si z < 0",
                        "Error:               e = Yd − ŷ",
                        "Actualización (e≠0): wi(t+1) = wi(t) + fa·e·xi   |   b(t+1) = b(t) + fa·e",
                    ]
                ),
                Spacer(1, 6),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("4. Ejecución del Entrenamiento, Convergencia y Predicción", s_h1),
                HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#F4B400"), spaceAfter=6),
                Paragraph(
                    "Con la inicialización oficial UPN (w<sub>1</sub> = w<sub>2</sub> = w<sub>3</sub> = 0.5, sesgo b = 0.5 y "
                    "fa = 0.5), la red completa <b>3 épocas × 8 patrones = 24 evaluaciones</b> hasta converger con 0 errores:",
                    s_body,
                ),
                Paragraph("• <b>Época 1 (Pasos 1–8 · 6 errores):</b> Corrige pesos en P1, P2, P3, P4, P5 y P6; cierra con w = (0.50, 0.50, 1.00) y b = -0.50.", s_bullet),
                Paragraph("• <b>Época 2 (Pasos 9–16 · 1 error):</b> Solo falla P2 (evaluación 10 de 24), ajustando w<sub>3</sub> = 0.50 y b = -1.00.", s_bullet),
                Paragraph("• <b>Época 3 (Pasos 17–24 · 0 errores):</b> Verifica los 8 patrones con 100 % de precisión y finaliza el entrenamiento.", s_bullet),
                Spacer(1, 4),
                make_table(EPOCH_SUMMARY_ROWS, [1.8, 1.9, 2.1, 2.4, 2.2, 2.2, 2.2, 2.2], center_from=0),
            ]
        )
    )
    story.append(make_figure(figs["fig1"], "Figura 1. Errores acumulados por época hasta alcanzar convergencia (0 errores en Época 3).", 14.2))
    story.append(make_figure(figs["fig2"], "Figura 2. Evolución paso a paso de los pesos sinápticos (w1, w2, w3) y el sesgo (b) en las 24 evaluaciones.", 15.4))

    story.append(
        KeepTogether(
            [
                Paragraph("4.2 Modelo Matemático Aprendido y Verificación Final", s_h2),
                make_formula_box(["z = 0.50·x1 + 0.50·x2 + 0.50·x3 − 1.00"]),
                Spacer(1, 4),
                Paragraph(
                    "<b>Interpretación práctica:</b> cada unidad aprobada suma +0.50 puntos y el sesgo (-1.00) actúa como "
                    "requisito mínimo. Así, 0 o 1 unidad producen z negativo (-1.00 o -0.50 → Desaprobado), mientras que 2 o 3 "
                    "unidades producen z ≥ 0 (0.00 o +0.50 → Aprobado).",
                    s_body,
                ),
                make_table(VERIFY_ROWS, [1.7, 3.1, 2.4, 2.3, 2.3, 1.6, 3.6], center_from=0),
                Spacer(1, 6),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("4.3 Predicción del Perceptrón (¿cómo decide la neurona?)", s_h2),
                Paragraph(
                    "La <b>Figura 3</b> muestra el proceso completo de decisión de la neurona para un estudiante con "
                    "notas U<sub>1</sub> = 14.0, U<sub>2</sub> = 10.0 y U<sub>3</sub> = 12.5: sus entradas se binarizan "
                    "a (1, 0, 1), los pesos aportan +0.50 + 0.00 + 0.50, el sesgo resta −1.00 dando z = 0.00 y la "
                    "función escalón (hardlim) activa ŷ = 1 (<b>APROBADO</b>).",
                    s_body,
                ),
                make_figure(
                    figs["fig3"],
                    "Figura 3. Predicción del Perceptrón (¿cómo decide la neurona?): entradas binarizadas, pesos, suma ponderada, función escalón hardlim y dictamen.",
                    15.6,
                ),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("4.4 Experimento de Sensibilidad sobre el Sesgo (b)", s_h2),
                Paragraph(
                    "Manteniendo w<sub>1</sub> = w<sub>2</sub> = w<sub>3</sub> = 0.50, se probó alterar manualmente el sesgo b "
                    "para verificar su impacto sobre la clasificación de los 8 patrones:",
                    s_body,
                ),
                make_table(SENSITIVITY_ROWS, [4.0, 4.8, 2.1, 1.9, 4.2], center_from=2),
                make_figure(figs["fig4"], "Figura 4. Efecto de modificar el sesgo b frente a la frontera z = 0 (Verde = Acierto | Rojo = Error).", 16.2),
            ]
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("5. Validación de los 4 Indicadores del Ejercicio", s_h1),
                HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#F4B400"), spaceAfter=6),
                make_table(INDICATORS_ROWS, [4.2, 5.8, 7.0], center_from=99),
                Spacer(1, 6),
            ]
        )
    )

    story.append(make_h1("6. Arquitectura del Simulador Interactivo en Streamlit"))
    story.append(
        Paragraph(
            "La aplicación se estructuró de manera modular en tres módulos Python integrados con Streamlit, NumPy, "
            "Pandas y Plotly (con respaldo automático en Altair):",
            s_body,
        )
    )
    story.append(Paragraph("• <b>perceptron.py:</b> Clase PerceptronRNA, binarización, historial completo de 24 evaluaciones y evaluación manual de sensibilidad.", s_bullet))
    story.append(Paragraph("• <b>visualizations.py:</b> Diagrama SVG dinámico de la neurona, gráficos de errores por época, evolución de pesos, función escalón hardlim y gráfico comparativo del experimento de sesgo.", s_bullet))
    story.append(Paragraph("• <b>app.py:</b> Interfaz de 4 pestañas: (1) Simulador en vivo + panel de reglas a–d, (2) Entrenamiento paso a paso por Época y Patrón + descarga CSV, (3) Modelo aprendido y experimento con el sesgo, y (4) Resumen teórico.", s_bullet))

    story.append(make_h1("7. Conclusiones"))
    story.append(Paragraph("• <b>Convergencia exacta:</b> El problema de mayoría (&gt;= 2 de 3 unidades) es linealmente separable y converge al 100 % en 3 épocas (24 evaluaciones) con w = (0.50, 0.50, 0.50) y b = -1.00.", s_bullet))
    story.append(Paragraph("• <b>Cumplimiento riguroso de reglas:</b> La binarización previa (P<sub>i</sub> &gt;= 12) garantiza que no intervenga el promedio aritmético, validado mediante los contraejemplos (12, 12, 0) y (20, 11.5, 11.5).", s_bullet))
    story.append(Paragraph("• <b>Papel clave del sesgo:</b> Alterar b = -1.00 a -0.50 o -1.50 desplaza la frontera z = 0 y produce 3 errores al convertir la neurona en compuertas OR o AND.", s_bullet))
    story.append(Paragraph("• <b>Valor pedagógico:</b> El simulador web en Streamlit permite visualizar, recorrer paso a paso y sustentar de manera clara cada etapa de decisión del Perceptrón.", s_bullet))

    doc.build(story, onFirstPage=draw_pdf_header_footer, onLaterPages=draw_pdf_header_footer)


def main() -> None:
    global DOCX_PATH, PDF_PATH
    model = PerceptronRNA()
    epochs, converged = model.train()
    assert epochs == 3 and converged, "El modelo debe converger en 3 épocas con parámetros UPN."

    figs = generate_figures(model)

    try:
        build_docx(figs)
        print(f"OK DOCX: {DOCX_PATH}")
    except PermissionError:
        alt_docx = BASE_DIR / "INFORME_TECNICO_RNA_PERCEPTRON_ACTUALIZADO.docx"
        DOCX_PATH = alt_docx
        build_docx(figs)
        print(f"OK DOCX (original abierto en Word): {DOCX_PATH}")

    try:
        build_pdf(figs)
        print(f"OK PDF : {PDF_PATH}")
    except PermissionError:
        alt_pdf = BASE_DIR / "INFORME_TECNICO_RNA_PERCEPTRON_ACTUALIZADO.pdf"
        PDF_PATH = alt_pdf
        build_pdf(figs)
        print(f"OK PDF (original abierto): {PDF_PATH}")


if __name__ == "__main__":
    main()
