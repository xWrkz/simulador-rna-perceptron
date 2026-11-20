<div align="center">

# UNIVERSIDAD PRIVADA DEL NORTE (UPN)
### FACULTAD DE INGENIERÍA
**Carrera Profesional de Ingeniería de Sistemas Computacionales**

---

## INFORME TÉCNICO – SESIÓN 14
# Entrenamiento de Red Neuronal Artificial (Perceptrón) para Evaluación de Aprobación Estudiantil

---

| Campo | Detalle |
| :--- | :--- |
| **Institución:** | Universidad Privada del Norte (UPN) |
| **Curso:** | Sistemas Inteligentes |
| **Docente:** | Marcelino Torres Villanueva |
| **Estudiante:** | Marco Antonio Gamez Monzon |
| **Código de Estudiante:** | N00324879 |
| **Ciclo:** | 10mo ciclo |
| **Fecha:** | Noviembre de 2026 |
| **Tecnologías:** | Python, Streamlit, NumPy, Pandas, Plotly / Altair |

</div>

---

## 1. Marco Teórico y Fundamentos de Redes Neuronales Artificiales

Las Redes Neuronales Artificiales (RNA) son modelos computacionales inspirados en la estructura y funcionamiento del sistema nervioso biológico. De acuerdo con la documentación académica de la Universidad Privada del Norte (UPN), una RNA se define como una red interconectada masivamente en paralelo de elementos simples (usualmente adaptativos) con organización jerárquica, diseñada para interactuar con problemas reales emulando el aprendizaje biológico.

### 1.1 Analogía Biológica y Componentes de la Neurona Artificial

El modelo matemático del **Perceptrón Simple** abstrae los tres componentes principales de la neurona biológica:

- **Dendritas (Entradas y Pesos Sinápticos):** Reciben las señales de entrada $x_j$, cada una ponderada por un peso sináptico $w_j$ que cuantifica cuánto influye dicha variable en la decisión final.
- **Cuerpo Celular o Soma (Suma Ponderada y Activación):** Integra las entradas ponderadas junto con un sesgo o *bias* ($w_b$ o $b$) que desplaza el umbral de activación:
  $$Net = z = \sum_{j=1}^{n} w_j x_j + b$$
  Luego aplica una función de activación $f(z)$ para determinar la respuesta.
- **Axón (Salida):** Emite la salida final predicha $\hat{y} = f(z)$ del sistema.

### 1.2 Funciones de Activación y Aprendizaje Supervisado

En el material de la Sesión 14 se presentan funciones de activación típicas como la **escalón o límite duro (`hardlim`)**, escalón simétrico (`hardlims`), sigmoidal logarítmica (`logsig`), tangente hiperbólica (`tansig`), lineal pura (`purelin`), lineal saturada (`satlins`) y base radial (`radbas`).

Para el problema de clasificación binaria del Perceptrón clásico se emplea la función **escalón (`hardlim`)**, que produce $1$ si la suma ponderada es mayor o igual a $0$, y $0$ en caso contrario.

En el paradigma de **Aprendizaje Supervisado**, cada patrón de entrada se empareja con su salida deseada $Y_d$. Al comparar la salida esperada con la predicha se obtiene el error ($e = Y_d - \hat{y}$), que realimenta al Perceptrón para reajustar los pesos sinápticos y el sesgo de manera iterativa hasta eliminar los errores.

### 1.3 Equivalencia de Notaciones

Para mantener consistencia entre el material de la Sesión 14 y la notación estándar de Machine Learning, la tabla siguiente resume las equivalencias utilizadas a lo largo del informe y el simulador:

| Concepto | Notación Sesión 14 (UPN) | Notación Estándar |
| :--- | :---: | :---: |
| Suma ponderada | $Net$ | $z$ |
| Peso del sesgo / umbral | $w_b$ (con entrada fija $1$) | $b$ |
| Tasa o factor de aprendizaje | $fa$ | $\eta$ |
| Salida esperada (real) | $Y_d$ | $y$ |
| Salida predicha | $y$ | $\hat{y}$ |
| Error de clasificación | $error = Y_d - y$ | $e = y - \hat{y}$ |

---

## 2. Especificación de Reglas de Negocio y Preprocesamiento

El ejercicio propuesto en la Sesión 14 plantea entrenar una RNA para determinar si un estudiante aprueba o no una experiencia curricular cumpliendo estrictamente **4 indicadores**:

- **a.** Se deben considerar **3 promedios** (correspondientes a 3 unidades académicas: $P_1, P_2, P_3$).
- **b.** El estudiante debe tener **por lo menos 2 unidades aprobadas** para aprobar la experiencia curricular.
- **c.** Para el resultado final **no se considera ningún cálculo matemático** (no se promedian aritméticamente las notas); solo se aplica la regla del punto *b*.
- **d.** Los promedios por cada unidad deben estar comprendidos en el rango de **0 a 20**.

### 2.1 Criterio de Binarización

Considerando la nota mínima aprobatoria universitaria estándar de $11$, se aplica una transformación discreta previa a la entrada del Perceptrón para mapear cada promedio continuo $P_i \in [0, 20]$ a una variable binaria $x_i \in \{0, 1\}$:

$$x_i = \begin{cases} 1 & \text{si } P_i \ge 11 \quad (\text{Unidad Aprobada}) \\ 0 & \text{si } P_i < 11 \quad (\text{Unidad Desaprobada}) \end{cases}$$

De esta forma, la red neuronal nunca opera sobre el promedio aritmético de las notas; únicamente evalúa el vector binario $(x_1, x_2, x_3)$ y aprende la regla de mayoría:

$$Y_d = 1 \iff x_1 + x_2 + x_3 \ge 2$$

---

## 3. Matriz de Patrones de Entrenamiento (Tabla de Verdad)

Con $n = 3$ entradas binarias, el espacio completo de estados consta de $2^3 = 8$ patrones de entrenamiento:

| Patrón | $x_1$ (U1) | $x_2$ (U2) | $x_3$ (U3) | Unidades Aprobadas | Salida Deseada ($Y_d$) | Estado Esperado |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P1** | 0 | 0 | 0 | 0 | 0 | Desaprobado |
| **P2** | 0 | 0 | 1 | 1 | 0 | Desaprobado |
| **P3** | 0 | 1 | 0 | 1 | 0 | Desaprobado |
| **P4** | 0 | 1 | 1 | 2 | 1 | Aprobado |
| **P5** | 1 | 0 | 0 | 1 | 0 | Desaprobado |
| **P6** | 1 | 0 | 1 | 2 | 1 | Aprobado |
| **P7** | 1 | 1 | 0 | 2 | 1 | Aprobado |
| **P8** | 1 | 1 | 1 | 3 | 1 | Aprobado |

---

## 4. Algoritmo de Entrenamiento (Regla Delta Generalizada)

El Perceptrón monocapa se entrena recorriendo secuencialmente los 8 patrones por época mediante las siguientes ecuaciones:

1. **Suma ponderada ($Net$ / $z$):**
   $$Net = x_1 w_1 + x_2 w_2 + x_3 w_3 + 1 \cdot w_b$$
2. **Función de activación escalón (`hardlim`):**
   $$\hat{y} = f(Net) = \begin{cases} 1 & \text{si } Net \ge 0 \\ 0 & \text{si } Net < 0 \end{cases}$$
3. **Cálculo del error:**
   $$e = Y_d - \hat{y}$$
4. **Actualización de pesos sinápticos y sesgo (solo si $e \neq 0$):**
   $$w_i(t+1) = w_i(t) + fa \cdot e \cdot x_i$$
   $$w_b(t+1) = w_b(t) + fa \cdot e \cdot 1$$

Cuando una época completa recorre los 8 patrones obteniendo $e = 0$ en todos los casos, el algoritmo alcanza la **convergencia** y finaliza.

---

## 5. Ejecución del Entrenamiento y Convergencia

Siguiendo la metodología de la Sesión 14 (UPN), se inicializan los parámetros con:

- Pesos iniciales: $w_1 = 0.5$, $w_2 = 0.5$, $w_3 = 0.5$
- Sesgo inicial (*bias*): $w_b = 0.5$
- Factor de aprendizaje: $fa = 0.5$

### 5.1 Detalle de la Época 1 (6 errores)

- **P1 $(0, 0, 0 \to Y_d = 0)$:** $Net = 0.50 \implies \hat{y} = 1$, $e = -1$.  
  Ajuste: $w_1 = 0.50,\; w_2 = 0.50,\; w_3 = 0.50,\; w_b = 0.00$.
- **P2 $(0, 0, 1 \to Y_d = 0)$:** $Net = 0.50 \implies \hat{y} = 1$, $e = -1$.  
  Ajuste: $w_1 = 0.50,\; w_2 = 0.50,\; w_3 = 0.00,\; w_b = -0.50$.
- **P3 $(0, 1, 0 \to Y_d = 0)$:** $Net = 0.00 \implies \hat{y} = 1$, $e = -1$.  
  Ajuste: $w_1 = 0.50,\; w_2 = 0.00,\; w_3 = 0.00,\; w_b = -1.00$.
- **P4 $(0, 1, 1 \to Y_d = 1)$:** $Net = -1.00 \implies \hat{y} = 0$, $e = +1$.  
  Ajuste: $w_1 = 0.50,\; w_2 = 0.50,\; w_3 = 0.50,\; w_b = -0.50$.
- **P5 $(1, 0, 0 \to Y_d = 0)$:** $Net = 0.00 \implies \hat{y} = 1$, $e = -1$.  
  Ajuste: $w_1 = 0.00,\; w_2 = 0.50,\; w_3 = 0.50,\; w_b = -1.00$.
- **P6 $(1, 0, 1 \to Y_d = 1)$:** $Net = -0.50 \implies \hat{y} = 0$, $e = +1$.  
  Ajuste: $w_1 = 0.50,\; w_2 = 0.50,\; w_3 = 1.00,\; w_b = -0.50$.
- **P7 $(1, 1, 0 \to Y_d = 1)$ y P8 $(1, 1, 1 \to Y_d = 1)$:** Predicción correcta ($e = 0$), sin cambios.

### 5.2 Épocas 2 y 3 (Convergencia en 24 evaluaciones)

- **Época 2 (1 error):** Al evaluar **P2 $(0, 0, 1 \to Y_d = 0)$** (evaluación 10 de 24), se obtiene $Net = 0.50 \implies \hat{y} = 1$ ($e = -1$). El ajuste corrige $w_3$ de $1.00$ a $0.50$ y $w_b$ de $-0.50$ a $-1.00$. Los demás patrones de la época ya no presentan errores.
- **Época 3 (0 errores):** Se evalúan nuevamente los 8 patrones (`P1` a `P8`) para validar el aprendizaje; todos producen $e = 0$, completando **$3\text{ épocas} \times 8\text{ patrones} = 24\text{ evaluaciones}$** y alcanzando 100 % de precisión.

#### Tabla Resumen de Evolución por Época

| Época | Errores | Aciertos | Precisión (%) | $w_1$ final | $w_2$ final | $w_3$ final | Sesgo final ($b$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 6 | 2 / 8 | 25.0 % | 0.50 | 0.50 | 1.00 | -0.50 |
| **2** | 1 | 7 / 8 | 87.5 % | 0.50 | 0.50 | 0.50 | -1.00 |
| **3** | **0** | **8 / 8** | **100.0 %** | **0.50** | **0.50** | **0.50** | **-1.00** |

### 5.3 Modelo Matemático Aprendido e Interpretación Intuitiva

Los parámetros convergidos son $w_1 = 0.50$, $w_2 = 0.50$, $w_3 = 0.50$ y $b = -1.00$, definiendo el modelo final:

$$z = 0.50x_1 + 0.50x_2 + 0.50x_3 - 1.00$$

Su interpretación práctica es directa:
- Cada unidad aprobada aporta **$+0.50$ puntos** a la suma ponderada.
- El sesgo **$-1.00$** actúa como requisito mínimo a superar para llegar a $z \ge 0$.
- Por ello, con 0 o 1 unidad aprobada la suma resulta negativa ($-1.00$ o $-0.50 \implies \hat{y} = 0$), mientras que con 2 o 3 unidades aprobadas alcanza $0.00$ o $+0.50 \implies \hat{y} = 1$.

#### Tabla de Verificación Final (Época 3)

| Patrón | Entrada $(x_1, x_2, x_3)$ | Suma Ponderada ($z$) | Predicción $\hat{y}$ | Esperado $Y_d$ | Error | Dictamen |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P1** | $(0, 0, 0)$ | $-1.00$ | 0 | 0 | 0 | Desaprobado |
| **P2** | $(0, 0, 1)$ | $-0.50$ | 0 | 0 | 0 | Desaprobado |
| **P3** | $(0, 1, 0)$ | $-0.50$ | 0 | 0 | 0 | Desaprobado |
| **P4** | $(0, 1, 1)$ | $0.00$ | 1 | 1 | 0 | Aprobado |
| **P5** | $(1, 0, 0)$ | $-0.50$ | 0 | 0 | 0 | Desaprobado |
| **P6** | $(1, 0, 1)$ | $0.00$ | 1 | 1 | 0 | Aprobado |
| **P7** | $(1, 1, 0)$ | $0.00$ | 1 | 1 | 0 | Aprobado |
| **P8** | $(1, 1, 1)$ | $+0.50$ | 1 | 1 | 0 | Aprobado |

### 5.4 Experimento de Sensibilidad sobre el Sesgo ($b$)

Para comprobar la relevancia del sesgo en la clasificación final, se evaluaron variaciones controladas manteniendo fijos $w_1 = w_2 = w_3 = 0.50$:

| Configuración probada | Comportamiento lógico resultante | Aciertos | Errores | Casos fallidos |
| :--- | :--- | :---: | :---: | :--- |
| **$b = -1.0$ (Aprendido)** | **Mayoría ($\ge 2$ unidades aprobadas)** | **8 / 8** | **0** | Ninguno |
| $b = -0.5$ | Compuerta OR ($\ge 1$ unidad aprobada) | 5 / 8 | 3 | P2, P3, P5 |
| $b = -1.5$ | Compuerta AND (exige las 3 unidades) | 5 / 8 | 3 | P4, P6, P7 |
| $b = 0.0$ | Aprueba incluso con 0 unidades | 4 / 8 | 4 | P1, P2, P3, P5 |

Este experimento demuestra que alterar el sesgo desplaza la frontera $z = 0$ y rompe de inmediato la regla de negocio del ejercicio.

---

## 6. Validación de los 4 Indicadores y Pruebas de Contraste

Para verificar el cumplimiento estricto de los 4 requisitos del enunciado, se definieron pruebas funcionales directas dentro del simulador:

| Indicador | Mecanismo de cumplimiento | Caso de validación |
| :--- | :--- | :--- |
| **a. 3 promedios (3 unidades)** | El modelo recibe exactamente 3 variables de entrada $(P_1, P_2, P_3) \to (x_1, x_2, x_3)$. | Diagrama de 3 entradas y tabla de $2^3 = 8$ patrones. |
| **b. Mínimo 2 unidades aprobadas** | Salida deseada $Y_d = 1$ únicamente cuando $\sum x_i \ge 2$. | Verificación 8/8 con error $0$ en la tabla final. |
| **c. Sin cálculo matemático de promedio final** | La binarización previa descarta la magnitud numérica exacta antes de la inferencia. | **Caso 1:** Notas `(11, 11, 0)` tienen promedio aritmético `7.33`, pero la RNA dictamina **APROBADO** (`2` unidades $\ge 11$).<br>**Caso 2:** Notas `(20, 10.5, 10.5)` tienen promedio aritmético `13.67`, pero la RNA dictamina **DESAPROBADO** (`1` unidad $\ge 11$). |
| **d. Rango de 0 a 20 por unidad** | Controles deslizantes limitados estrictamente a $[0.0,\; 20.0]$. | Imposibilidad de ingresar notas fuera del rango permitido. |

---

## 7. Arquitectura del Simulador Interactivo en Streamlit

Para transformar el desarrollo teórico en una herramienta educativa interactiva, se implementó una aplicación modular en Python con **Streamlit**, **NumPy**, **Pandas** y **Plotly** (con respaldo automático en **Altair**):

### 7.1 Estructura Modular del Proyecto

- **`perceptron.py` (Capa de Modelo):** Encapsula la clase `PerceptronRNA`, la binarización `binarize_grades()`, el registro completo de las 24 iteraciones y del resumen por época, y la función `evaluate_manual_parameters()` para los experimentos de sensibilidad.
- **`visualizations.py` (Capa de Visualización):** Genera el diagrama SVG dinámico de la neurona en tiempo real y los gráficos interactivos (errores por época, evolución paso a paso de pesos y sesgo, función escalón `hardlim` con discontinuidad en $z = 0$, y gráfico comparativo 2D del experimento de sensibilidad).
- **`app.py` (Capa de Interfaz de Usuario):** Organiza la experiencia en una barra lateral de casos de prueba rápidos e hiperparámetros configurables, un encabezado con KPIs de convergencia y **4 pestañas progresivas**:
  1. **🎯 1. Simulador y Reglas (a–d):** Deslizadores de notas (`0–20`), binarización visual por unidad, diagrama SVG en vivo de la neurona, cálculo en tarjetas limpias, curva `hardlim` interactiva y panel en vivo de cumplimiento de los 4 indicadores.
  2. **📘 2. Entrenamiento Paso a Paso:** Tabla de entrenamiento de 8 patrones, inspector interactivo por **Época (1, 2, 3)** y **Patrón (`P1` a `P8`)** mostrando el cálculo exacto de la Regla Delta, tabla resumen por época, gráficos apilados a ancho completo de errores y evolución de pesos, y exportación del historial completo a **CSV**.
  3. **🧪 3. Modelo Aprendido y Experimento:** Ecuación convergida explicada intuitivamente, tabla de verificación de los 8 casos resaltando el patrón del estudiante actual, y laboratorio interactivo para modificar pesos y sesgo visualizando en verde/rojo el efecto sobre la frontera $z = 0$.
  4. **📚 4. Resumen Teórico:** Síntesis de los 4 pasos del Perceptrón, ecuaciones de la Regla Delta y tabla de equivalencia de notaciones.

---

## 8. Conclusiones

1. **Convergencia exacta del Perceptrón Monocapa:** La regla de aprobación por mayoría de 2 de 3 unidades es linealmente separable, lo que permitió que el Perceptrón alcanzara el **100 % de precisión en 3 épocas (24 evaluaciones)** con pesos finales $w_1 = w_2 = w_3 = 0.50$ y $b = -1.00$.
2. **Cumplimiento riguroso de las reglas de negocio:** La binarización previa ($P_i \ge 11$) garantiza que el sistema evalúe exclusivamente la cantidad de unidades aprobadas y no el promedio aritmético, validado mediante los contraejemplos `(11, 11, 0)` y `(20, 10.5, 10.5)`.
3. **Rol determinante del sesgo (*bias*):** El experimento interactivo demostró que modificar el sesgo $b = -1.00$ hacia $-0.50$ o $-1.50$ transforma el comportamiento de la neurona hacia compuertas `OR` o `AND`, generando errores de clasificación en 3 de los 8 patrones.
4. **Claridad pedagógica mediante visualización interactiva:** La separación modular del simulador en Streamlit —combinando el diagrama SVG en vivo, el recorrido por época y patrón, los gráficos de convergencia y el experimento de sensibilidad— facilita tanto la comprensión del algoritmo como su sustentación académica.
