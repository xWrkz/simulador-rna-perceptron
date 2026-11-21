"""Simulador Interactivo de Red Neuronal Artificial (Perceptrón) — Sesión 14 UPN.

Diseñado siguiendo la estructura clara y progresiva de las diapositivas UPN
(ejemplo OR paso a paso + ejercicio de 3 unidades):
1. Simulador en vivo + verificación directa de los 4 indicadores (a, b, c, d).
2. Entrenamiento paso a paso en tarjetas limpias + tabla y gráficos por época.
3. Modelo aprendido explicado con sentido común + Experimento con pesos y sesgo.
4. Resumen teórico esencial y equivalencia de notaciones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from perceptron import (
    PerceptronRNA,
    UPN_DEFAULT_BIAS,
    UPN_DEFAULT_FA,
    UPN_DEFAULT_MAX_EPOCHS,
    UPN_DEFAULT_THRESHOLD,
    UPN_DEFAULT_WEIGHTS,
    X_TRAIN_UPN,
    YD_TRAIN_UPN,
    binarize_grades,
    evaluate_manual_parameters,
    format_linear_equation,
)
from visualizations import (
    build_epoch_error_chart,
    build_experiment_comparison_chart,
    build_hardlim_live_chart,
    build_weight_evolution_chart,
    render_perceptron_svg,
)


# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ==========================================
st.set_page_config(
    page_title="Simulador Perceptrón | Sesión 14 - UPN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.hero-banner {
    background: linear-gradient(135deg, #111827 0%, #1F2937 68%, #2E291A 100%);
    border-left: 6px solid #F4B400;
    border-radius: 16px;
    padding: 1.35rem 1.7rem;
    margin-bottom: 1.2rem;
    color: #F9FAFB;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.16);
}
.hero-badge {
    display: inline-block;
    background: rgba(244, 180, 0, 0.18);
    color: #FCD34D;
    border: 1px solid rgba(244, 180, 0, 0.45);
    padding: 0.2rem 0.72rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 1.72rem;
    font-weight: 800;
    margin: 0;
    color: #FFFFFF;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #D1D5DB;
    margin-top: 0.4rem;
    margin-bottom: 0;
}
.formula-bar {
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 12px;
    padding: 0.85rem 1.2rem;
    text-align: center;
    font-size: 1.22rem;
    font-weight: 800;
    letter-spacing: 0.01em;
    margin-bottom: 0.65rem;
}
.formula-bar-gold {
    background: rgba(244, 180, 0, 0.12);
    border: 1.5px solid rgba(244, 180, 0, 0.65);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    text-align: center;
    font-size: 1.38rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
}
.unit-card {
    border-radius: 14px;
    padding: 0.85rem 1rem;
    margin-top: 0.35rem;
    border: 1px solid;
}
.unit-approved {
    background: rgba(16, 185, 129, 0.10);
    border-color: rgba(16, 185, 129, 0.45);
}
.unit-failed {
    background: rgba(239, 68, 68, 0.09);
    border-color: rgba(239, 68, 68, 0.45);
}
.pill {
    display: inline-block;
    padding: 0.18rem 0.62rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
}
.pill-ok { background: #10B981; color: #FFFFFF; }
.pill-fail { background: #EF4444; color: #FFFFFF; }
.verdict-card {
    border-radius: 16px;
    padding: 1.15rem 1.35rem;
    border-left: 6px solid;
    margin-top: 0.4rem;
    margin-bottom: 0.75rem;
}
.verdict-ok {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(16, 185, 129, 0.05));
    border-color: #10B981;
}
.verdict-fail {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.14), rgba(239, 68, 68, 0.05));
    border-color: #EF4444;
}
.verdict-title {
    font-size: 1.18rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.check-card {
    border-radius: 13px;
    padding: 0.85rem 1rem;
    border: 1px solid rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.07);
    height: 100%;
}
.info-box {
    border-radius: 14px;
    padding: 1rem 1.15rem;
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: rgba(148, 163, 184, 0.07);
    margin-bottom: 0.75rem;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    background: rgba(148, 163, 184, 0.05);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# 2. ESTADO INICIAL Y CALLBACKS
# ==========================================
def init_session_state() -> None:
    defaults = {
        "nota_u1": 14.0,
        "nota_u2": 10.0,
        "nota_u3": 12.5,
        "hp_w1": UPN_DEFAULT_WEIGHTS[0],
        "hp_w2": UPN_DEFAULT_WEIGHTS[1],
        "hp_w3": UPN_DEFAULT_WEIGHTS[2],
        "hp_wb": UPN_DEFAULT_BIAS,
        "hp_fa": UPN_DEFAULT_FA,
        "hp_max_epochs": UPN_DEFAULT_MAX_EPOCHS,
        "hp_threshold": UPN_DEFAULT_THRESHOLD,
        "sel_epoch_num": 1,
        "sel_pattern_num": 1,
        "exp_w1": 0.5,
        "exp_w2": 0.5,
        "exp_w3": 0.5,
        "exp_b": -1.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_upn_hyperparameters() -> None:
    st.session_state["hp_w1"] = UPN_DEFAULT_WEIGHTS[0]
    st.session_state["hp_w2"] = UPN_DEFAULT_WEIGHTS[1]
    st.session_state["hp_w3"] = UPN_DEFAULT_WEIGHTS[2]
    st.session_state["hp_wb"] = UPN_DEFAULT_BIAS
    st.session_state["hp_fa"] = UPN_DEFAULT_FA
    st.session_state["hp_max_epochs"] = UPN_DEFAULT_MAX_EPOCHS
    st.session_state["hp_threshold"] = UPN_DEFAULT_THRESHOLD
    st.session_state["sel_epoch_num"] = 1
    st.session_state["sel_pattern_num"] = 1


def apply_student_preset(u1: float, u2: float, u3: float) -> None:
    st.session_state["nota_u1"] = u1
    st.session_state["nota_u2"] = u2
    st.session_state["nota_u3"] = u3


def apply_experiment_preset(w1: float, w2: float, w3: float, b: float) -> None:
    st.session_state["exp_w1"] = w1
    st.session_state["exp_w2"] = w2
    st.session_state["exp_w3"] = w3
    st.session_state["exp_b"] = b


def render_chart(engine: str, fig) -> None:
    if engine == "plotly":
        st.plotly_chart(fig, width="stretch")
    else:
        st.altair_chart(fig, width="stretch")


@st.cache_resource(show_spinner=False)
def train_cached_perceptron(
    w1: float, w2: float, w3: float, wb: float, fa: float, max_epochs: int
) -> PerceptronRNA:
    model = PerceptronRNA(
        num_inputs=3,
        fa=fa,
        initial_weights=(w1, w2, w3),
        initial_bias=wb,
    )
    model.train(X_TRAIN_UPN, YD_TRAIN_UPN, max_epochs=max_epochs)
    return model


init_session_state()


# ==========================================
# 3. BARRA LATERAL LIMPIA Y PRÁCTICA
# ==========================================
with st.sidebar:
    st.markdown("### 🧪 Casos de Prueba Rápidos")
    st.caption("Haz clic para demostrar el comportamiento del Perceptrón y los 4 indicadores:")

    st.button(
        "✅ Aprobado normal (14, 10, 12.5)",
        on_click=apply_student_preset,
        args=(14.0, 10.0, 12.5),
        width="stretch",
    )
    st.button(
        "🔥 Regla c: Prom. 8.0 pero APRUEBA (12, 12, 0)",
        on_click=apply_student_preset,
        args=(12.0, 12.0, 0.0),
        width="stretch",
        help="Tiene promedio aritmético 8.00, pero aprueba por tener 2 unidades ≥ 12.",
    )
    st.button(
        "🔥 Regla c: Prom. 14.3 pero DESAPRUEBA (20, 11.5, 11.5)",
        on_click=apply_student_preset,
        args=(20.0, 11.5, 11.5),
        width="stretch",
        help="Tiene promedio aritmético 14.33, pero desaprueba por tener solo 1 unidad ≥ 12.",
    )
    st.button(
        "⚖️ Caso límite exacto (12, 11.5, 12)",
        on_click=apply_student_preset,
        args=(12.0, 11.5, 12.0),
        width="stretch",
    )
    st.button(
        "🌟 Sobresaliente (18, 16, 19)",
        on_click=apply_student_preset,
        args=(18.0, 16.0, 19.0),
        width="stretch",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parámetros de Entrenamiento")
    st.caption("Por defecto usa la configuración oficial UPN (`w = 0.5`, `b = 0.5`, `fa = 0.5`).")

    with st.expander("Ajustar valores iniciales (opcional)", expanded=False):
        st.slider("Tasa de aprendizaje (fa / η)", 0.05, 1.0, step=0.05, key="hp_fa")
        st.slider("Máximo de épocas", 1, 40, step=1, key="hp_max_epochs")
        st.number_input("w₁ inicial", -5.0, 5.0, step=0.1, key="hp_w1")
        st.number_input("w₂ inicial", -5.0, 5.0, step=0.1, key="hp_w2")
        st.number_input("w₃ inicial", -5.0, 5.0, step=0.1, key="hp_w3")
        st.number_input("Sesgo inicial (b / w_b)", -5.0, 5.0, step=0.1, key="hp_wb")
        st.number_input("Nota aprobatoria por unidad", 1.0, 20.0, step=0.5, key="hp_threshold")

    st.button(
        "↺ Restaurar valores UPN",
        on_click=reset_upn_hyperparameters,
        width="stretch",
    )


# ==========================================
# 4. ENTRENAMIENTO DEL MODELO
# ==========================================
model = train_cached_perceptron(
    st.session_state["hp_w1"],
    st.session_state["hp_w2"],
    st.session_state["hp_w3"],
    st.session_state["hp_wb"],
    st.session_state["hp_fa"],
    st.session_state["hp_max_epochs"],
)
history_df = model.get_history_df()
epoch_df = model.get_epoch_summary_df()
training_df = model.get_training_table_df()
truth_df = model.get_truth_table_df()
threshold = float(st.session_state["hp_threshold"])

if st.session_state["sel_epoch_num"] > model.epochs_run:
    st.session_state["sel_epoch_num"] = 1

is_default_upn = (
    np.isclose(st.session_state["hp_w1"], UPN_DEFAULT_WEIGHTS[0])
    and np.isclose(st.session_state["hp_w2"], UPN_DEFAULT_WEIGHTS[1])
    and np.isclose(st.session_state["hp_w3"], UPN_DEFAULT_WEIGHTS[2])
    and np.isclose(st.session_state["hp_wb"], UPN_DEFAULT_BIAS)
    and np.isclose(st.session_state["hp_fa"], UPN_DEFAULT_FA)
    and np.isclose(threshold, UPN_DEFAULT_THRESHOLD)
)


# ==========================================
# 5. ENCABEZADO HERO Y RESUMEN EJECUTIVO
# ==========================================
mode_tag = "Configuración Oficial UPN" if is_default_upn else "Modo Personalizado"
st.markdown(
    f"""
    <div class="hero-banner">
        <span class="hero-badge">UPN · SISTEMAS INTELIGENTES · SESIÓN 14 · {mode_tag.upper()}</span>
        <h1 class="hero-title">🧠 Perceptrón Simple: Evaluación de Aprobación Estudiantil</h1>
        <p class="hero-subtitle">
            Explicación paso a paso de cómo una neurona artificial aprende la regla de
            <b>aprobar con al menos 2 de 3 unidades</b>, desde el entrenamiento hasta la verificación final.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

final_acc = float(epoch_df["Precisión (%)"].iloc[-1])
final_err = int(epoch_df["Errores"].iloc[-1])
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Estado de la Red",
    "Aprendió ✓" if model.converged else "Sin converger",
    delta="0 errores en la última época" if model.converged else f"{final_err} errores pendientes",
    delta_color="normal" if model.converged else "inverse",
)
k2.metric(
    "Épocas necesarias",
    f"{model.epochs_run}",
    delta=f"{len(history_df)} evaluaciones totales",
    delta_color="off",
)
k3.metric(
    "Casos acertados",
    f"{int(epoch_df['Aciertos'].iloc[-1])} / 8",
    delta=f"Precisión: {final_acc:.0f}%",
    delta_color="normal" if final_acc == 100 else "inverse",
)
k4.metric(
    "Parámetros aprendidos",
    f"w=({model.w[0]:.1f}, {model.w[1]:.1f}, {model.w[2]:.1f})",
    delta=f"Sesgo b = {model.wb:.1f}",
    delta_color="off",
)

if not model.converged:
    st.warning(
        f"Con estos parámetros la red no llegó a 0 errores en {model.epochs_run} épocas. "
        "Haz clic en **↺ Restaurar valores UPN** en la barra lateral."
    )


# ==========================================
# 6. PESTAÑAS GUIADAS (ESTILO DIAPOSITIVAS UPN)
# ==========================================
tab_sim, tab_train, tab_exp, tab_theory = st.tabs(
    [
        "🎯 1. Simulador y Reglas (a–d)",
        "📘 2. Entrenamiento Paso a Paso",
        "🧪 3. Modelo Aprendido y Experimento",
        "📚 4. Resumen Teórico",
    ]
)

# ------------------------------------------
# TAB 1: SIMULADOR EN VIVO + 4 INDICADORES
# ------------------------------------------
with tab_sim:
    st.subheader("1. Ingresa los 3 promedios por unidad (rango 0 a 20)")
    st.caption(
        f"Cada nota se convierte en **1 (Aprobada)** si es ≥ {threshold:.1f}, o en **0 (Desaprobada)** si es < {threshold:.1f}."
    )

    u_cols = st.columns(3)
    unit_defs = [
        ("Unidad 1 (P₁)", "nota_u1", "x₁"),
        ("Unidad 2 (P₂)", "nota_u2", "x₂"),
        ("Unidad 3 (P₃)", "nota_u3", "x₃"),
    ]
    grades = []
    for col, (label, key, x_sym) in zip(u_cols, unit_defs):
        with col:
            nota = st.slider(label, min_value=0.0, max_value=20.0, step=0.5, key=key)
            grades.append(float(nota))
            is_ok = nota >= threshold
            x_val = 1 if is_ok else 0
            card_cls = "unit-approved" if is_ok else "unit-failed"
            pill_cls = "pill-ok" if is_ok else "pill-fail"
            status_txt = "Aprobada ✓" if is_ok else "Desaprobada ✗"
            st.progress(min(1.0, nota / 20.0))
            st.markdown(
                f"""
                <div class="unit-card {card_cls}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700;">Entrada {x_sym} = {x_val}</span>
                        <span class="pill {pill_cls}">{status_txt}</span>
                    </div>
                    <div style="font-size:0.83rem; opacity:0.82; margin-top:0.25rem;">
                        Promedio: <b>{nota:.1f} / 20.0</b> (mínimo {threshold:.1f})
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    x_input = binarize_grades(grades, threshold=threshold)
    prediccion, z_val = model.predict(x_input)
    unidades_aprobadas = int(np.sum(x_input))
    pattern_idx = int(x_input[0] * 4 + x_input[1] * 2 + x_input[2] + 1)
    prom_aritmetico = float(np.mean(grades))

    st.markdown("---")
    st.subheader("2. Predicción del Perceptrón (¿cómo decide la neurona?)")

    svg_html = render_perceptron_svg(
        grades=(grades[0], grades[1], grades[2]),
        x_bin=x_input,
        weights=model.w,
        bias_w=model.wb,
        net=z_val,
        y_pred=prediccion,
    )
    st.markdown(svg_html, unsafe_allow_html=True)

    st.markdown("")
    # Bloques de cálculo estilo Diapositiva 12
    c_calc1, c_calc2 = st.columns([1.2, 1.0])
    with c_calc1:
        b_sign = "+" if model.wb >= 0 else "−"
        st.markdown(
            f"""
            <div class="formula-bar">
                z = ({x_input[0]})({model.w[0]:.2f}) + ({x_input[1]})({model.w[1]:.2f}) + ({x_input[2]})({model.w[2]:.2f}) {b_sign} {abs(model.wb):.2f} = {z_val:+.2f}
            </div>
            <div class="formula-bar">
                Como z = {z_val:+.2f} {"≥ 0" if z_val >= 0 else "< 0"} &nbsp;→&nbsp; ŷ = {prediccion} ({"APROBADO" if prediccion == 1 else "DESAPROBADO"})
            </div>
            """,
            unsafe_allow_html=True,
        )

        if prediccion == 1:
            st.markdown(
                f"""
                <div class="verdict-card verdict-ok">
                    <div class="verdict-title">🎉 ESTUDIANTE APROBADO (ŷ = 1)</div>
                    <div>
                        Tiene <b>{unidades_aprobadas} de 3 unidades aprobadas</b> (patrón <b>P{pattern_idx}</b>).
                        Cada unidad aprobada sumó <code>+{model.w[0]:.2f}</code>, superando el sesgo <code>{model.wb:+.2f}</code>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-card verdict-fail">
                    <div class="verdict-title">❌ ESTUDIANTE DESAPROBADO (ŷ = 0)</div>
                    <div>
                        Solo tiene <b>{unidades_aprobadas} de 3 unidades aprobadas</b> (patrón <b>P{pattern_idx}</b>).
                        Al necesitar al menos 2 unidades, la suma ponderada queda negativa (<code>z = {z_val:+.2f} &lt; 0</code>).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c_calc2:
        eng_h, fig_h = build_hardlim_live_chart(z_val, prediccion)
        render_chart(eng_h, fig_h)

    st.markdown("---")
    st.subheader("3. Verificación en vivo de los 4 indicadores del ejercicio")
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.markdown(
            f"""
            <div class="check-card">
                <b>✅ a. 3 promedios (3 unidades)</b><br>
                <span style="font-size:0.88rem;">
                La neurona recibe exactamente 3 entradas: <code>U1={grades[0]:.1f}</code>,
                <code>U2={grades[1]:.1f}</code>, <code>U3={grades[2]:.1f}</code>.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ic2:
        st.markdown(
            f"""
            <div class="check-card">
                <b>✅ b. Mínimo 2 unidades aprobadas</b><br>
                <span style="font-size:0.88rem;">
                Actual: <b>{unidades_aprobadas} aprobada(s)</b> → Dictamen:
                <b>{"Aprobado" if prediccion == 1 else "Desaprobado"}</b>.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ic3:
        prom_dir = "Aprobaría" if prom_aritmetico >= threshold else "Desaprobaría"
        contraste = (
            "¡Nota la diferencia!"
            if (prom_aritmetico >= threshold) != (prediccion == 1)
            else "Usa los botones 🔥 de la barra lateral para ver el contraste."
        )
        st.markdown(
            f"""
            <div class="check-card">
                <b>✅ c. Sin promedio matemático final</b><br>
                <span style="font-size:0.88rem;">
                Promedio aritmético sería <b>{prom_aritmetico:.2f}</b> ({prom_dir}),
                pero la RNA solo usa <code>({x_input[0]},{x_input[1]},{x_input[2]})</code>. {contraste}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ic4:
        st.markdown(
            f"""
            <div class="check-card">
                <b>✅ d. Rango de 0 a 20 por unidad</b><br>
                <span style="font-size:0.88rem;">
                Los 3 controles restringen estrictamente cada nota entre <code>0.0</code> y <code>20.0</code>.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------
# TAB 2: ENTRENAMIENTO PASO A PASO (DIAP. 15–20)
# ------------------------------------------
with tab_train:
    st.subheader("1. Tabla de Entrenamiento")
    st.caption(
        f"Parámetros iniciales: **w₁ = {model.initial_w[0]:.1f}**, **w₂ = {model.initial_w[1]:.1f}**, "
        f"**w₃ = {model.initial_w[2]:.1f}**, **sesgo b = {model.initial_wb:.1f}** y **tasa de aprendizaje fa = {model.fa:.1f}**."
    )
    st.dataframe(training_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("2. Cálculo Paso a Paso")
    st.caption(
        f"Cada época revisa los **8 patrones (P1 a P8)**. En total se realizaron "
        f"**{model.epochs_run} épocas × 8 patrones = {len(history_df)} evaluaciones**. "
        "Cuando `e = 0` no cambia nada; cuando `e ≠ 0`, corrige los pesos y el sesgo."
    )

    epoch_numbers = epoch_df["Época"].tolist()
    epoch_err_map = dict(zip(epoch_df["Época"], epoch_df["Errores"]))

    def fmt_epoch(ep_num: int) -> str:
        errs = int(epoch_err_map[ep_num])
        err_txt = "1 error" if errs == 1 else f"{errs} errores"
        icon = "✅" if errs == 0 else "🔄"
        return f"{icon} Época {ep_num} ({err_txt})"

    sel_c1, sel_c2 = st.columns([1.25, 1.0])
    with sel_c1:
        if len(epoch_numbers) <= 5:
            sel_epoch = st.radio(
                "1) Elige la época:",
                options=epoch_numbers,
                format_func=fmt_epoch,
                horizontal=True,
                key="sel_epoch_num",
            )
        else:
            sel_epoch = st.selectbox(
                "1) Elige la época:",
                options=epoch_numbers,
                format_func=fmt_epoch,
                key="sel_epoch_num",
            )

    epoch_rows = history_df[history_df["Época"] == sel_epoch].reset_index(drop=True)
    pattern_map = {
        int(r["Pattern_Idx"]): (
            f"{r['Patrón']} {r['Entradas']} — "
            + ("✓ correcto (e = 0)" if int(r["Error"]) == 0 else f"✗ ajuste (e = {int(r['Error']):+d})")
        )
        for _, r in epoch_rows.iterrows()
    }

    with sel_c2:
        sel_pattern = st.selectbox(
            "2) Elige el patrón dentro de esta época (P1 a P8):",
            options=list(pattern_map.keys()),
            format_func=lambda idx: pattern_map[idx],
            key="sel_pattern_num",
        )

    row = epoch_rows[epoch_rows["Pattern_Idx"] == sel_pattern].iloc[0]
    err_val = int(row["Error"])
    wb_sign = "+" if row["wb"] >= 0 else "−"

    st.markdown(
        f"#### Época {int(row['Época'])} · Patrón {row['Patrón']}: X = {row['Entradas']} "
        f"&nbsp;<span style='font-size:0.88rem; opacity:0.7;'>(Evaluación {int(row['Paso'])} de {len(history_df)})</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Valores al iniciar este ejemplo: w₁ = {row['w1']:.2f}, w₂ = {row['w2']:.2f}, "
        f"w₃ = {row['w3']:.2f}, b = {row['wb']:.2f}, fa = {model.fa:.2f}."
    )

    # Tarjetas grandes de ecuaciones idénticas al estilo del PDF del OR
    st.markdown(
        f"""
        <div class="formula-bar">
            z = ({int(row['x1'])})({row['w1']:.2f}) + ({int(row['x2'])})({row['w2']:.2f}) + ({int(row['x3'])})({row['w3']:.2f}) {wb_sign} {abs(row['wb']):.2f} = {row['Net']:+.2f}
        </div>
        <div class="formula-bar">
            ŷ = {int(row['y'])} &nbsp;&nbsp;|&nbsp;&nbsp; y (deseada) = {int(row['Yd'])} &nbsp;&nbsp;|&nbsp;&nbsp; e = y − ŷ = {err_val:+d}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if err_val != 0:
        st.warning(
            f"**Predicción incorrecta (e = {err_val:+d}):** El perceptrón predijo `{int(row['y'])}`, "
            f"pero la respuesta correcta era `{int(row['Yd'])}`. Por eso se corrigen los pesos y el sesgo:"
        )
        u1_c, u2_c = st.columns(2)
        with u1_c:
            st.markdown(
                f"""
                <div class="formula-bar">w₁ = {row['w1']:.2f} + ({model.fa:.2f})({err_val:+d})({int(row['x1'])}) = {row['w1_nuevo']:.2f}</div>
                <div class="formula-bar">w₂ = {row['w2']:.2f} + ({model.fa:.2f})({err_val:+d})({int(row['x2'])}) = {row['w2_nuevo']:.2f}</div>
                """,
                unsafe_allow_html=True,
            )
        with u2_c:
            st.markdown(
                f"""
                <div class="formula-bar">w₃ = {row['w3']:.2f} + ({model.fa:.2f})({err_val:+d})({int(row['x3'])}) = {row['w3_nuevo']:.2f}</div>
                <div class="formula-bar">b = {row['wb']:.2f} + ({model.fa:.2f})({err_val:+d})(1) = {row['wb_nuevo']:.2f}</div>
                """,
                unsafe_allow_html=True,
            )
        st.caption(
            f"Nuevos parámetros tras este paso: **w₁ = {row['w1_nuevo']:.2f}**, **w₂ = {row['w2_nuevo']:.2f}**, "
            f"**w₃ = {row['w3_nuevo']:.2f}**, **b = {row['wb_nuevo']:.2f}**."
        )
    else:
        st.success(
            f"**Predicción correcta (e = 0):** El perceptrón predijo `{int(row['y'])}` y la respuesta esperada "
            "también es `{0}`. Como no hay error, **no cambian los pesos ni el sesgo**.".format(int(row["Yd"]))
        )

    st.markdown("---")
    st.subheader("3. Evolución del Entrenamiento por Época")
    st.dataframe(
        epoch_df[["Época", "Errores", "w₁ final", "w₂ final", "w₃ final", "Sesgo final (b)"]],
        width="stretch",
        hide_index=True,
    )
    if model.converged:
        st.info(
            f"📌 **Idea clave:** Para esta inicialización y orden de muestras, en la **época {model.epochs_run}** "
            "ya no se cometen errores (`0 errores`) y el entrenamiento termina."
        )

    eng_e, fig_e = build_epoch_error_chart(epoch_df)
    if eng_e == "plotly":
        fig_e.update_traces(line=dict(color="#F4B400", width=2.4, dash="dot"), selector=dict(type="scatter"))
        fig_e.update_layout(height=380, margin=dict(l=20, r=20, t=65, b=30))
    render_chart(eng_e, fig_e)

    eng_w, fig_w = build_weight_evolution_chart(history_df, model.initial_w, model.initial_wb)
    if eng_w == "plotly":
        w_vals = history_df[["w1_nuevo", "w2_nuevo", "w3_nuevo", "wb_nuevo"]]
        y_min = min(float(w_vals.min().min()), float(model.initial_w.min()), float(model.initial_wb))
        y_max = max(float(w_vals.max().max()), float(model.initial_w.max()), float(model.initial_wb))
        fig_w.update_layout(
            title=dict(text="¿Cómo se fueron ajustando los pesos y el sesgo?", y=0.96),
            yaxis=dict(title="Valor", range=[y_min - 0.25, y_max + 0.45]),
            height=420,
            margin=dict(l=20, r=20, t=95, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
    render_chart(eng_w, fig_w)

    with st.expander("📋 Ver historial completo de las iteraciones y descargar CSV", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            ep_opts = ["Todas"] + [f"Época {e}" for e in epoch_df["Época"].tolist()]
            sel_ep = st.selectbox("Filtrar por época:", ep_opts)
        with f2:
            only_err = st.checkbox("Solo pasos donde hubo ajuste (e ≠ 0)", value=False)
        with f3:
            st.download_button(
                "⬇️ Descargar historial CSV",
                data=history_df.to_csv(index=False).encode("utf-8"),
                file_name="historial_perceptron_sesion14.csv",
                mime="text/csv",
                width="stretch",
            )

        filt = history_df.copy()
        if sel_ep != "Todas":
            filt = filt[filt["Época"] == int(sel_ep.split()[-1])]
        if only_err:
            filt = filt[filt["Error"] != 0]

        cols_show = [
            "Paso", "Época", "Patrón", "Entradas", "w1", "w2", "w3", "wb",
            "Net", "y", "Yd", "Error", "Ajuste", "w1_nuevo", "w2_nuevo", "w3_nuevo", "wb_nuevo",
        ]
        st.dataframe(filt[cols_show], width="stretch", hide_index=True)


# ------------------------------------------
# TAB 3: MODELO APRENDIDO Y EXPERIMENTO (DIAP. 21–23)
# ------------------------------------------
with tab_exp:
    st.subheader("1. Modelo Finalmente Aprendido")
    st.markdown(
        f'<div class="formula-bar-gold">{model.equation_text()}</div>',
        unsafe_allow_html=True,
    )

    if (
        np.isclose(model.w[0], 0.5)
        and np.isclose(model.w[1], 0.5)
        and np.isclose(model.w[2], 0.5)
        and np.isclose(model.wb, -1.0)
    ):
        st.markdown(
            """
            <div class="info-box">
                <b>¿Cómo entender esta fórmula en lenguaje simple (sin matemáticas complicadas)?</b>
                <ul style="margin-top:0.4rem; margin-bottom:0;">
                    <li>Cada unidad aprobada suma <b>+0.5 puntos</b> (porque <code>w₁ = w₂ = w₃ = 0.5</code>).</li>
                    <li>El sesgo resta <b>−1.0 punto</b> (es la "valla" o requisito mínimo que debes superar para llegar a 0).</li>
                    <li><b>0 unidades aprobadas:</b> 0.0 − 1.0 = <b>−1.0</b> (&lt; 0) → ❌ Desaprobado.</li>
                    <li><b>1 unidad aprobada:</b> +0.5 − 1.0 = <b>−0.5</b> (&lt; 0) → ❌ Desaprobado.</li>
                    <li><b>2 unidades aprobadas:</b> +1.0 − 1.0 = <b>0.0</b> (≥ 0) → ✅ Aprobado.</li>
                    <li><b>3 unidades aprobadas:</b> +1.5 − 1.0 = <b>+0.5</b> (≥ 0) → ✅ Aprobado.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("2. Verificación de los 8 Casos")
    st.caption(
        f"El patrón del estudiante actual en la Pestaña 1 es **P{pattern_idx} "
        f"({x_input[0]}, {x_input[1]}, {x_input[2]})** (resaltado en dorado)."
    )

    def highlight_student_row(r: pd.Series) -> list[str]:
        if r["Patrón"] == f"P{pattern_idx}":
            return ["background-color: rgba(244, 180, 0, 0.25); font-weight: 700"] * len(r)
        return [""] * len(r)

    st.dataframe(
        truth_df[
            ["Patrón", "Entrada (x₁, x₂, x₃)", "Unidades aprobadas", "z (Net)", "Predicción ŷ", "Esperado y", "Dictamen ŷ", "Resultado"]
        ].style.apply(highlight_student_row, axis=1),
        width="stretch",
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("3. Experimento con los Pesos y el Sesgo")
    st.caption(
        "¿Por qué el sesgo `b = −1.0` es tan importante como los pesos? Prueba cambiarlo con un clic "
        "y observa en el gráfico qué casos se rompen al cruzar la línea dorada `z = 0`."
    )

    eb1, eb2, eb3, eb4 = st.columns(4)
    with eb1:
        st.button(
            "✅ Aprendido (b = -1.0 · ≥ 2 u.)",
            on_click=apply_experiment_preset,
            args=(float(model.w[0]), float(model.w[1]), float(model.w[2]), float(model.wb)),
            width="stretch",
        )
    with eb2:
        st.button(
            "⚠️ Sesgo b = -0.5 (Tipo OR · ≥ 1 u.)",
            on_click=apply_experiment_preset,
            args=(0.5, 0.5, 0.5, -0.5),
            width="stretch",
        )
    with eb3:
        st.button(
            "⚠️ Sesgo b = -1.5 (Tipo AND · 3 u.)",
            on_click=apply_experiment_preset,
            args=(0.5, 0.5, 0.5, -1.5),
            width="stretch",
        )
    with eb4:
        st.button(
            "⚠️ Sesgo b = 0.0 (Aprueba todo)",
            on_click=apply_experiment_preset,
            args=(0.5, 0.5, 0.5, 0.0),
            width="stretch",
        )

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        exp_w1 = st.slider("Probar w₁", -1.5, 2.0, step=0.1, key="exp_w1")
    with sc2:
        exp_w2 = st.slider("Probar w₂", -1.5, 2.0, step=0.1, key="exp_w2")
    with sc3:
        exp_w3 = st.slider("Probar w₃", -1.5, 2.0, step=0.1, key="exp_w3")
    with sc4:
        exp_b = st.slider("Probar Sesgo (b)", -2.5, 1.5, step=0.1, key="exp_b")

    exp_df, exp_ok, exp_err = evaluate_manual_parameters(exp_w1, exp_w2, exp_w3, exp_b)
    exp_eq = format_linear_equation((exp_w1, exp_w2, exp_w3), exp_b)

    em1, em2, em3 = st.columns([1.8, 1.0, 1.0])
    em1.markdown(f'<div class="formula-bar" style="margin-top:0.2rem;">{exp_eq}</div>', unsafe_allow_html=True)
    em2.metric("Aciertos del experimento", f"{exp_ok} / 8", f"{(exp_ok/8)*100:.0f}%")
    em3.metric(
        "Errores cometidos",
        f"{exp_err}",
        "Regla exacta ✓" if exp_err == 0 else "Regla rota ✗",
        delta_color="normal" if exp_err == 0 else "inverse",
    )

    if exp_err == 0:
        st.success(
            "Estos parámetros clasifican correctamente los **8 casos**. Todos los desaprobados quedan "
            "debajo de `z = 0` y todos los aprobados quedan en `z ≥ 0`."
        )
    else:
        failed_list = ", ".join(exp_df.loc[~exp_df["Acertó"], "Patrón"].tolist())
        st.error(
            f"Con estos valores la neurona **falla en {exp_err} caso(s): {failed_list}**. "
            "Esto demuestra que modificar el sesgo o un peso desplaza el umbral y empeora la clasificación."
        )

    eng_exp, fig_exp = build_experiment_comparison_chart(exp_df)
    render_chart(eng_exp, fig_exp)


# ------------------------------------------
# TAB 4: RESUMEN TEÓRICO ESENCIAL (PARA EXPONER)
# ------------------------------------------
with tab_theory:
    st.subheader("1. ¿Cómo funciona el Perceptrón en 4 pasos?")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(
            f"""
            <div class="info-box">
                <h4>1️⃣ Entradas (xᵢ)</h4>
                <p>Recibe las 3 unidades binarizadas: <code>1</code> si la nota es ≥ {threshold:g}, o <code>0</code> si es menor.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """
            <div class="info-box">
                <h4>2️⃣ Suma ponderada</h4>
                <p>Multiplica cada entrada por su peso y suma el sesgo: <code>z = w₁x₁ + w₂x₂ + w₃x₃ + b</code>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            """
            <div class="info-box">
                <h4>3️⃣ Escalón (hardlim)</h4>
                <p>Si <code>z ≥ 0</code> predice <b>1 (Aprobado)</b>; si <code>z &lt; 0</code> predice <b>0 (Desaprobado)</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t4:
        st.markdown(
            """
            <div class="info-box">
                <h4>4️⃣ Corrección</h4>
                <p>Calcula <code>e = y − ŷ</code>. Si <code>e ≠ 0</code>, ajusta los pesos y el sesgo usando la tasa <code>fa</code>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    c_th1, c_th2 = st.columns(2)
    with c_th1:
        st.subheader("2. Regla de Aprendizaje")
        st.markdown(
            """
            <div class="formula-bar">e = y − ŷ</div>
            <div class="formula-bar">wᵢ(nuevo) = wᵢ(actual) + fa · e · xᵢ</div>
            <div class="formula-bar">b(nuevo) = b(actual) + fa · e</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Si el error es `0`, las multiplicaciones se anulan y los parámetros se mantienen iguales."
        )

    with c_th2:
        st.subheader("3. Equivalencia de Notaciones")
        st.caption("Ambas notaciones expresan exactamente la misma matemática:")
        eq_df = pd.DataFrame(
            {
                "Concepto": [
                    "Suma ponderada",
                    "Peso del sesgo / umbral",
                    "Tasa de aprendizaje",
                    "Salida esperada (real)",
                    "Salida predicha",
                    "Error",
                ],
                "Notación Sesión 14": ["Net", "wb (con entrada 1)", "fa", "Yd", "y", "error = Yd − y"],
                "Notación Estándar": ["z", "b (sesgo)", "η", "y", "ŷ", "e = y − ŷ"],
            }
        )
        st.dataframe(eq_df, width="stretch", hide_index=True)
