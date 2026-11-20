# 🧠 Simulador de Perceptrón Simple — Sesión 14 UPN

Aplicación interactiva en **Streamlit** basada en la **Sesión 14 de Sistemas Inteligentes (UPN)** y estructurada con la misma claridad paso a paso del ejemplo **OR** de clase.

## ✨ Contenido de las 4 pestañas

1. **`🎯 1. Simulador y Reglas (a–d)`**: Ingreso de 3 promedios (`0–20`), binarización automática (`≥ 11 → 1`), diagrama SVG en vivo, cálculo estilo diapositivas, función escalón (`hardlim`) y verificación en tiempo real de los 4 indicadores del ejercicio (incluyendo casos de prueba en la barra lateral para demostrar el indicador *c*).
2. **`📘 2. Entrenamiento Paso a Paso`**: Tabla de entrenamiento de 8 casos, recorrido paso a paso con tarjetas de ecuaciones grandes, tabla resumen por época (3 épocas hasta `0 errores`), gráfico de errores por época y evolución de los pesos (`w₁, w₂, w₃, b`).
3. **`🧪 3. Modelo Aprendido y Experimento`**: Explicación intuitiva de `z = 0.5x₁ + 0.5x₂ + 0.5x₃ − 1.0`, tabla de verificación final y **experimento interactivo con los pesos y el sesgo** (probando `b = -1.0`, `b = -0.5` tipo OR y `b = -1.5` tipo AND en un gráfico 2D claro frente a `z = 0`).
4. **`📚 4. Resumen Teórico`**: Los 4 pasos del Perceptrón, la regla de aprendizaje y la tabla de equivalencia de símbolos entre ambos materiales UPN (`Net ↔ z`, `Yd ↔ y`, `y ↔ ŷ`, `wb ↔ b`, `fa ↔ η`).

## 🚀 Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```
