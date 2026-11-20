"""Visualizaciones limpias y pedagógicas para el Simulador RNA (Perceptrón).

Alineado al estilo didáctico de las diapositivas UPN (ejemplo OR y Sesión 14):
- Diagrama SVG en vivo de la neurona (Entradas → Pesos → Σ z → Escalón → ŷ).
- Gráficos claros de Errores por Época y Evolución de Pesos.
- Gráfico de Función Escalón (estilo Diapositiva 11, con punto abierto/cerrado en z = 0).
- Gráfico 2D intuitivo para el Experimento con Pesos y Sesgo (estilo Diapositiva 23),
  mostrando qué patrones cruzan el umbral z = 0 y cuáles fallan al cambiar b.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    go = None  # type: ignore
    HAS_PLOTLY = False


COLOR_GOLD = "#F4B400"
COLOR_DARK = "#111827"
COLOR_GREEN = "#10B981"
COLOR_RED = "#EF4444"
COLOR_BLUE = "#3B82F6"
COLOR_PURPLE = "#8B5CF6"
COLOR_ORANGE = "#F97316"


def render_perceptron_svg(
    grades: tuple[float, float, float],
    x_bin: np.ndarray,
    weights: np.ndarray,
    bias_w: float,
    net: float,
    y_pred: int,
) -> str:
    """Genera un diagrama SVG limpio del Perceptrón con valores en tiempo real."""
    out_color = COLOR_GREEN if y_pred == 1 else COLOR_RED
    out_bg = "#ECFDF5" if y_pred == 1 else "#FEF2F2"
    out_label = "APROBADO (1)" if y_pred == 1 else "DESAPROBADO (0)"

    def node_style(active: bool) -> tuple[str, str, str]:
        if active:
            return "#DCFCE7", "#15803D", "#166534"
        return "#FEE2E2", "#DC2626", "#991B1B"

    def line_style(active: bool, weight: float) -> tuple[str, str]:
        if not active:
            return "#CBD5E1", "2.2"
        color = COLOR_GREEN if weight >= 0 else COLOR_RED
        width = str(min(5.0, max(2.8, 2.5 + abs(weight) * 1.8)))
        return color, width

    inputs_meta = [
        ("U1", grades[0], int(x_bin[0]), float(weights[0]), 66),
        ("U2", grades[1], int(x_bin[1]), float(weights[1]), 154),
        ("U3", grades[2], int(x_bin[2]), float(weights[2]), 242),
    ]

    input_elements = []
    for idx, (label, grade, x_val, w_val, cy) in enumerate(inputs_meta, start=1):
        fill, stroke, text_col = node_style(x_val == 1)
        l_col, l_width = line_style(x_val == 1, w_val)
        contrib = x_val * w_val
        dash = "" if x_val == 1 else 'stroke-dasharray="6,4"'

        mid_x = 248
        mid_y = (cy + 175) / 2 - (10 if cy < 175 else (-14 if cy > 175 else 10))
        input_elements.append(
            f"""
            <line x1="138" y1="{cy}" x2="358" y2="175"
                  stroke="{l_col}" stroke-width="{l_width}" {dash}
                  marker-end="url(#arrow)"/>
            <rect x="{mid_x - 58}" y="{mid_y - 13}" width="116" height="24" rx="12"
                  fill="#FFFFFF" stroke="{l_col}" stroke-width="1.4"/>
            <text x="{mid_x}" y="{mid_y + 3}" text-anchor="middle"
                  font-size="11.5" font-weight="700" fill="#1E293B">
                w{idx}={w_val:+.2f} → {contrib:+.2f}
            </text>
            <rect x="20" y="{cy - 28}" width="118" height="56" rx="14"
                  fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>
            <text x="79" y="{cy - 6}" text-anchor="middle"
                  font-size="12" font-weight="700" fill="{text_col}">
                {label}: {grade:.1f} pts
            </text>
            <text x="79" y="{cy + 14}" text-anchor="middle"
                  font-size="13.5" font-weight="800" fill="{text_col}">
                x{idx} = {x_val}
            </text>
            """
        )

    bias_col = "#D97706" if bias_w >= 0 else "#7C3AED"

    svg = f"""
    <div style="width:100%; overflow-x:auto; background:#FFFFFF; border:1px solid #E2E8F0;
                border-radius:16px; padding:14px 10px; box-shadow:0 4px 16px rgba(15,23,42,0.05);">
      <svg viewBox="0 0 820 330" width="100%" style="max-height:325px; display:block; margin:auto; font-family:Inter,system-ui,sans-serif;">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
                  markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748B"/>
          </marker>
          <linearGradient id="somaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#FEF3C7"/>
            <stop offset="100%" stop-color="#FDE68A"/>
          </linearGradient>
        </defs>
        <text x="79" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#64748B" letter-spacing="0.8">ENTRADAS (xᵢ)</text>
        <text x="248" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#64748B" letter-spacing="0.8">PESOS (wᵢ · xᵢ)</text>
        <text x="412" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#64748B" letter-spacing="0.8">SUMA PONDERADA</text>
        <text x="572" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#64748B" letter-spacing="0.8">FUNCIÓN ESCALÓN</text>
        <text x="735" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#64748B" letter-spacing="0.8">PREDICCIÓN (ŷ)</text>
        {"".join(input_elements)}
        <line x1="412" y1="274" x2="412" y2="233" stroke="{bias_col}" stroke-width="3" marker-end="url(#arrow)"/>
        <rect x="334" y="274" width="156" height="44" rx="12" fill="#F5F3FF" stroke="{bias_col}" stroke-width="2"/>
        <text x="412" y="293" text-anchor="middle" font-size="11.5" font-weight="700" fill="#5B21B6">Sesgo / Bias</text>
        <text x="412" y="309" text-anchor="middle" font-size="12" font-weight="800" fill="{bias_col}">b = {bias_w:+.2f}</text>
        <circle cx="412" cy="175" r="54" fill="url(#somaGrad)" stroke="#D97706" stroke-width="3"/>
        <text x="412" y="160" text-anchor="middle" font-size="22" font-weight="800" fill="#92400E">Σ</text>
        <text x="412" y="180" text-anchor="middle" font-size="11" font-weight="700" fill="#78350F">z (Net)</text>
        <text x="412" y="203" text-anchor="middle" font-size="16" font-weight="800" fill="#111827">{net:+.2f}</text>
        <line x1="466" y1="175" x2="514" y2="175" stroke="#475569" stroke-width="3" marker-end="url(#arrow)"/>
        <rect x="518" y="130" width="110" height="90" rx="14" fill="#F8FAFC" stroke="#334155" stroke-width="2.2"/>
        <text x="573" y="150" text-anchor="middle" font-size="11.5" font-weight="700" fill="#334155">f(z) · hardlim</text>
        <line x1="538" y1="194" x2="608" y2="194" stroke="#94A3B8" stroke-width="1.2"/>
        <line x1="573" y1="158" x2="573" y2="206" stroke="#94A3B8" stroke-width="1.2"/>
        <polyline points="540,194 573,194 573,166 606,166" fill="none" stroke="{out_color}" stroke-width="3"/>
        <text x="573" y="213" text-anchor="middle" font-size="10" font-weight="600" fill="#475569">{"z ≥ 0 → 1" if net >= 0 else "z < 0 → 0"}</text>
        <line x1="628" y1="175" x2="668" y2="175" stroke="{out_color}" stroke-width="3.2" marker-end="url(#arrow)"/>
        <rect x="672" y="133" width="132" height="84" rx="16" fill="{out_bg}" stroke="{out_color}" stroke-width="3"/>
        <text x="738" y="158" text-anchor="middle" font-size="12" font-weight="700" fill="#334155">Salida</text>
        <text x="738" y="185" text-anchor="middle" font-size="22" font-weight="900" fill="{out_color}">ŷ = {y_pred}</text>
        <text x="738" y="205" text-anchor="middle" font-size="10.5" font-weight="800" fill="{out_color}">{out_label}</text>
      </svg>
    </div>
    """
    return "\n".join(line.strip() for line in svg.splitlines() if line.strip())


def build_epoch_error_chart(epoch_df: pd.DataFrame):
    """Gráfico simple y directo de Errores Totales por Época."""
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=epoch_df["Época"],
                y=epoch_df["Errores"],
                marker_color=[COLOR_GREEN if e == 0 else COLOR_RED for e in epoch_df["Errores"]],
                text=[f"{int(e)} errores" if e != 1 else "1 error" for e in epoch_df["Errores"]],
                textposition="outside",
                hovertemplate="<b>Época %{x}</b><br>Errores cometidos: %{y}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epoch_df["Época"],
                y=epoch_df["Errores"],
                mode="lines+markers",
                line=dict(color=COLOR_GOLD, width=2.4, dash="dot"),
                marker=dict(size=9, color=COLOR_GOLD, line=dict(color=COLOR_DARK, width=1.5)),
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            title=dict(text="¿Cuántos errores cometió la red en cada época?", y=0.95),
            xaxis=dict(title="Época", tickmode="linear", dtick=1),
            yaxis=dict(
                title="Errores (de 8 casos)",
                range=[0, max(7.5, float(epoch_df["Errores"].max()) + 1.8)],
            ),
            height=380,
            margin=dict(l=20, r=20, t=65, b=30),
            showlegend=False,
            template="plotly_white",
        )
        return ("plotly", fig)

    chart = (
        alt.Chart(epoch_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Época:O", title="Época"),
            y=alt.Y("Errores:Q", title="Errores (de 8 casos)"),
            color=alt.condition(
                alt.datum.Errores == 0, alt.value(COLOR_GREEN), alt.value(COLOR_RED)
            ),
            tooltip=["Época", "Errores", "Aciertos", "Precisión (%)"],
        )
        .properties(title="Errores cometidos por época", height=320)
    )
    return ("altair", chart)


def build_weight_evolution_chart(
    history_df: pd.DataFrame, initial_w: np.ndarray, initial_wb: float
):
    """Evolución de los pesos w1, w2, w3 y el sesgo b paso a paso."""
    init_row = pd.DataFrame(
        [
            {
                "Paso": 0,
                "Etiqueta": "Inicio",
                "w1": float(initial_w[0]),
                "w2": float(initial_w[1]),
                "w3": float(initial_w[2]),
                "b": float(initial_wb),
            }
        ]
    )
    steps_df = pd.DataFrame(
        {
            "Paso": history_df["Paso"],
            "Etiqueta": history_df.apply(lambda r: f"Época {r['Época']} · {r['Patrón']}", axis=1),
            "w1": history_df["w1_nuevo"],
            "w2": history_df["w2_nuevo"],
            "w3": history_df["w3_nuevo"],
            "b": history_df["wb_nuevo"],
        }
    )
    full_df = pd.concat([init_row, steps_df], ignore_index=True)

    if HAS_PLOTLY:
        fig = go.Figure()
        series = [
            ("w1", "w₁ (Unidad 1)", COLOR_BLUE, "solid"),
            ("w2", "w₂ (Unidad 2)", COLOR_GREEN, "dash"),
            ("w3", "w₃ (Unidad 3)", COLOR_ORANGE, "dot"),
            ("b", "b (Sesgo / Bias)", COLOR_PURPLE, "solid"),
        ]
        for col, name, color, dash in series:
            fig.add_trace(
                go.Scatter(
                    x=full_df["Paso"],
                    y=full_df[col],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=3.2 if col == "b" else 2.3, dash=dash),
                    marker=dict(size=6),
                    customdata=full_df["Etiqueta"],
                    hovertemplate=f"<b>%{{customdata}}</b> (Paso %{{x}})<br>{name}: <b>%{{y:.2f}}</b><extra></extra>",
                )
            )

        max_epoch = int(history_df["Época"].max())
        for ep in range(1, max_epoch + 1):
            vline_kwargs = dict(x=ep * 8, line_dash="dash", line_color="#CBD5E1")
            if max_epoch <= 10:
                vline_kwargs.update(
                    annotation_text=f"Fin Época {ep}",
                    annotation_position="top left",
                    annotation_font_color="#64748B",
                )
            fig.add_vline(**vline_kwargs)

        y_min = float(full_df[["w1", "w2", "w3", "b"]].min().min())
        y_max = float(full_df[["w1", "w2", "w3", "b"]].max().max())
        fig.add_hline(y=0, line_color="#94A3B8", line_width=1)
        fig.update_layout(
            title=dict(text="¿Cómo se fueron ajustando los pesos y el sesgo?", y=0.96),
            xaxis_title="Paso de entrenamiento (Paso 0 = Valores iniciales)",
            yaxis=dict(title="Valor", range=[y_min - 0.25, y_max + 0.45]),
            height=420,
            margin=dict(l=20, r=20, t=95, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white",
            hovermode="x unified",
        )
        return ("plotly", fig)

    melted = full_df.melt(
        id_vars=["Paso", "Etiqueta"],
        value_vars=["w1", "w2", "w3", "b"],
        var_name="Parámetro",
        value_name="Valor",
    )
    chart = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("Paso:Q", title="Paso de entrenamiento"),
            y=alt.Y("Valor:Q", title="Valor"),
            color=alt.Color("Parámetro:N"),
            tooltip=["Etiqueta", "Paso", "Parámetro", "Valor"],
        )
        .properties(title="Evolución de pesos y sesgo", height=320)
    )
    return ("altair", chart)


def build_hardlim_live_chart(z_val: float, y_pred: int):
    """Función escalón estilo Diapositiva 11 (punto abierto en (0,0), cerrado en (0,1))."""
    span = max(2.2, abs(z_val) + 0.8)
    point_color = COLOR_GREEN if y_pred == 1 else COLOR_RED

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_vrect(x0=-span, x1=0, fillcolor="rgba(239,68,68,0.06)", line_width=0)
        fig.add_vrect(x0=0, x1=span, fillcolor="rgba(16,185,129,0.07)", line_width=0)

        # Rama izquierda (z < 0 -> 0) con círculo abierto en (0, 0)
        fig.add_trace(
            go.Scatter(
                x=[-span, 0],
                y=[0, 0],
                mode="lines",
                line=dict(color=COLOR_RED, width=3.5),
                name="Desaprobado (z < 0 → ŷ = 0)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=10, color="#FFFFFF", line=dict(color=COLOR_RED, width=2.5)),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Rama derecha (z >= 0 -> 1) con círculo relleno en (0, 1)
        fig.add_trace(
            go.Scatter(
                x=[0, span],
                y=[1, 1],
                mode="lines",
                line=dict(color=COLOR_GREEN, width=3.5),
                name="Aprobado (z ≥ 0 → ŷ = 1)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[1],
                mode="markers",
                marker=dict(size=10, color=COLOR_GREEN),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Punto del estudiante actual
        fig.add_trace(
            go.Scatter(
                x=[z_val],
                y=[y_pred],
                mode="markers+text",
                name="Estudiante actual",
                marker=dict(
                    size=15,
                    color=COLOR_GOLD,
                    symbol="diamond",
                    line=dict(color=point_color, width=3),
                ),
                text=[f"z = {z_val:+.2f} → ŷ = {y_pred}"],
                textposition="top center" if y_pred == 0 else "bottom center",
                textfont=dict(size=12, color=COLOR_DARK),
                hovertemplate="<b>Estudiante actual</b><br>z = %{x:.2f}<br>ŷ = %{y}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Función de Activación Escalón (hardlim)",
            xaxis=dict(title="Suma ponderada z (Net)", zeroline=True, zerolinecolor="#94A3B8"),
            yaxis=dict(title="Predicción ŷ", tickvals=[0, 1], range=[-0.3, 1.35]),
            height=300,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.38, xanchor="center", x=0.5),
            template="plotly_white",
        )
        return ("plotly", fig)

    df = pd.DataFrame({"z": [-span, 0, 0, span], "y": [0, 0, 1, 1]})
    pt = pd.DataFrame({"z": [z_val], "y": [y_pred]})
    base = alt.Chart(df).mark_line(color=COLOR_DARK, strokeWidth=2.5).encode(x="z:Q", y="y:Q")
    dot = alt.Chart(pt).mark_point(size=180, filled=True, color=point_color).encode(x="z:Q", y="y:Q")
    return ("altair", (base + dot).properties(title="Función de Activación Escalón", height=270))


def build_experiment_comparison_chart(exp_df: pd.DataFrame):
    """Gráfico 2D súper intuitivo para el Experimento con Pesos y Sesgo (Diap. 23).

    Muestra el valor z (Net) de cada uno de los 8 patrones frente a la línea de
    corte z = 0, coloreando en verde los aciertos y en rojo los errores.
    """
    labels = [f"{r['Patrón']} {r['Entrada (x₁, x₂, x₃)']}" for _, r in exp_df.iterrows()]
    z_vals = exp_df["z (Net)"].tolist()
    ok_flags = exp_df["Acertó"].tolist()
    colors = [COLOR_GREEN if ok else COLOR_RED for ok in ok_flags]
    texts = [
        f"z={z:+.2f} ({'✓' if ok else '✗'})"
        for z, ok in zip(z_vals, ok_flags)
    ]

    if HAS_PLOTLY:
        max_abs = max(1.6, max(abs(v) for v in z_vals) + 0.5)
        fig = go.Figure()
        fig.add_hrect(
            y0=0,
            y1=max_abs,
            fillcolor="rgba(16, 185, 129, 0.06)",
            line_width=0,
            annotation_text="Zona APROBADO (z ≥ 0 → ŷ = 1)",
            annotation_position="top left",
            annotation_font_color="#047857",
        )
        fig.add_hrect(
            y0=-max_abs,
            y1=0,
            fillcolor="rgba(239, 68, 68, 0.05)",
            line_width=0,
            annotation_text="Zona DESAPROBADO (z < 0 → ŷ = 0)",
            annotation_position="bottom left",
            annotation_font_color="#B91C1C",
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=z_vals,
                marker_color=colors,
                text=texts,
                textposition="outside",
                customdata=np.stack(
                    [exp_df["Predicción ŷ"], exp_df["Esperado y"], exp_df["Resultado"]], axis=-1
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "z (Net): %{y:+.2f}<br>"
                    "Predicción ŷ: %{customdata[0]} | Esperado y: %{customdata[1]}<br>"
                    "<b>%{customdata[2]}</b><extra></extra>"
                ),
            )
        )
        fig.add_hline(
            y=0,
            line_color=COLOR_GOLD,
            line_width=3,
            line_dash="dash",
            annotation_text="Frontera de decisión (z = 0)",
            annotation_position="top right",
            annotation_font_color="#92400E",
        )
        fig.update_layout(
            title="Efecto de los pesos y el sesgo en los 8 casos (Verde = Acierto | Rojo = Error)",
            xaxis_title="Patrón (x₁, x₂, x₃)",
            yaxis=dict(title="Valor de z (Net)", range=[-max_abs, max_abs]),
            height=410,
            margin=dict(l=20, r=20, t=55, b=20),
            showlegend=False,
            template="plotly_white",
        )
        return ("plotly", fig)

    chart_df = exp_df.copy()
    chart_df["Etiqueta"] = labels
    bar = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Etiqueta:O", title="Patrón"),
            y=alt.Y("z (Net):Q", title="Valor de z (Net)"),
            color=alt.condition(
                alt.datum.Acertó, alt.value(COLOR_GREEN), alt.value(COLOR_RED)
            ),
            tooltip=["Patrón", "Entrada (x₁, x₂, x₃)", "z (Net)", "Predicción ŷ", "Esperado y", "Resultado"],
        )
        .properties(title="Efecto de pesos y sesgo en los 8 casos", height=360)
    )
    line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=COLOR_GOLD, strokeDash=[5, 5]).encode(y="y:Q")
    return ("altair", bar + line)
