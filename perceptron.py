"""Módulo del modelo Perceptrón Monocapa (Sesión 14 - Sistemas Inteligentes UPN).

Implementa la lógica exacta del Perceptrón con función de activación escalón
(hardlim) y entrenamiento supervisado mediante la Regla Delta Generalizada,
registrando métricas paso a paso, resumen por época (estilo Diap. 20) y
evaluación manual para experimentos con pesos y sesgo (estilo Diap. 23).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Dataset oficial de los 8 patrones binarios (3 unidades académicas)
X_TRAIN_UPN = np.array(
    [
        [0, 0, 0],  # P1: 0 aprobadas -> Desaprobado
        [0, 0, 1],  # P2: 1 aprobada  -> Desaprobado
        [0, 1, 0],  # P3: 1 aprobada  -> Desaprobado
        [0, 1, 1],  # P4: 2 aprobadas -> Aprobado
        [1, 0, 0],  # P5: 1 aprobada  -> Desaprobado
        [1, 0, 1],  # P6: 2 aprobadas -> Aprobado
        [1, 1, 0],  # P7: 2 aprobadas -> Aprobado
        [1, 1, 1],  # P8: 3 aprobadas -> Aprobado
    ],
    dtype=float,
)

YD_TRAIN_UPN = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=int)

# Hiperparámetros base definidos en la metodología UPN (Informe Técnico, Sección 5)
UPN_DEFAULT_WEIGHTS = (0.5, 0.5, 0.5)
UPN_DEFAULT_BIAS = 0.5
UPN_DEFAULT_FA = 0.5
UPN_DEFAULT_THRESHOLD = 11.0
UPN_DEFAULT_MAX_EPOCHS = 20


def binarize_grades(
    grades: list[float] | tuple[float, ...], threshold: float = UPN_DEFAULT_THRESHOLD
) -> np.ndarray:
    """Transforma notas continuas [0, 20] a entradas binarias {0, 1}."""
    return np.array([1 if g >= threshold else 0 for g in grades], dtype=int)


def format_linear_equation(w: np.ndarray | list[float] | tuple[float, ...], b: float) -> str:
    """Formatea la ecuación z = w1*x1 + w2*x2 + w3*x3 + b de forma legible."""

    def fmt_signed(val: float, var: str, first: bool = False) -> str:
        if first:
            return f"{val:.2f}{var}"
        sign = "+" if val >= 0 else "-"
        return f" {sign} {abs(val):.2f}{var}"

    return (
        "z = "
        + fmt_signed(float(w[0]), "x₁", first=True)
        + fmt_signed(float(w[1]), "x₂")
        + fmt_signed(float(w[2]), "x₃")
        + fmt_signed(float(b), "")
    )


def evaluate_manual_parameters(
    w1: float,
    w2: float,
    w3: float,
    b: float,
    X: np.ndarray = X_TRAIN_UPN,
    Yd: np.ndarray = YD_TRAIN_UPN,
) -> tuple[pd.DataFrame, int, int]:
    """Evalúa los 8 patrones con pesos y sesgo elegidos manualmente (Experimento Diap. 23)."""
    w_vec = np.array([w1, w2, w3], dtype=float)
    rows = []
    correct = 0

    for i in range(len(X)):
        x_i = X[i]
        target = int(Yd[i])
        z_val = float(np.dot(x_i, w_vec) + float(b))
        y_hat = 1 if round(z_val, 10) >= 0.0 else 0
        ok = y_hat == target
        if ok:
            correct += 1

        rows.append(
            {
                "Patrón": f"P{i + 1}",
                "Entrada (x₁, x₂, x₃)": f"({int(x_i[0])}, {int(x_i[1])}, {int(x_i[2])})",
                "Unidades aprobadas": int(np.sum(x_i)),
                "z (Net)": round(z_val, 2),
                "Predicción ŷ": y_hat,
                "Esperado y": target,
                "Dictamen ŷ": "Aprobado" if y_hat == 1 else "Desaprobado",
                "Resultado": "✓ Acierto" if ok else "✗ Error",
                "Acertó": ok,
            }
        )

    df = pd.DataFrame(rows)
    errors = len(X) - correct
    return df, correct, errors


class PerceptronRNA:
    """Perceptrón Monocapa con Regla Delta Generalizada y registro completo."""

    def __init__(
        self,
        num_inputs: int = 3,
        fa: float = UPN_DEFAULT_FA,
        initial_weights: list[float] | tuple[float, ...] | None = None,
        initial_bias: float = UPN_DEFAULT_BIAS,
    ) -> None:
        self.num_inputs = num_inputs
        self.fa = float(fa)

        if initial_weights is None:
            self.initial_w = np.full(num_inputs, 0.5, dtype=float)
        else:
            self.initial_w = np.array(initial_weights, dtype=float)

        self.initial_wb = float(initial_bias)
        self.w = self.initial_w.copy()
        self.wb = self.initial_wb

        self.history: list[dict] = []
        self.epoch_summary: list[dict] = []
        self.epochs_run: int = 0
        self.converged: bool = False

    @staticmethod
    def activation_hardlim(net: float) -> int:
        """Función de activación escalón (hardlim): 1 si z >= 0, 0 si z < 0."""
        return 1 if round(float(net), 10) >= 0.0 else 0

    def train(
        self,
        X: np.ndarray = X_TRAIN_UPN,
        Yd: np.ndarray = YD_TRAIN_UPN,
        max_epochs: int = UPN_DEFAULT_MAX_EPOCHS,
    ) -> tuple[int, bool]:
        """Entrena el Perceptrón usando la Regla Delta Generalizada."""
        self.w = self.initial_w.copy()
        self.wb = self.initial_wb
        self.history = []
        self.epoch_summary = []

        epoch = 0
        converged = False
        step = 0
        n_patterns = len(X)

        while not converged and epoch < max_epochs:
            epoch += 1
            total_abs_error = 0

            for i in range(n_patterns):
                step += 1
                x_i = np.asarray(X[i], dtype=float)
                target = int(Yd[i])

                w_before = self.w.copy()
                wb_before = float(self.wb)

                # Suma ponderada: z (Net) = x1*w1 + x2*w2 + x3*w3 + 1*b
                net = float(np.dot(x_i, self.w) + 1.0 * self.wb)
                y_hat = self.activation_hardlim(net)
                error = int(target - y_hat)

                total_abs_error += abs(error)
                updated = error != 0

                # Regla de aprendizaje: Wi(nuevo) = Wi(actual) + fa * error * xi
                if updated:
                    self.w = self.w + self.fa * error * x_i
                    self.wb = self.wb + self.fa * error * 1.0

                self.history.append(
                    {
                        "Paso": step,
                        "Época": epoch,
                        "Patrón": f"P{i + 1}",
                        "Pattern_Idx": i + 1,
                        "x1": int(x_i[0]),
                        "x2": int(x_i[1]),
                        "x3": int(x_i[2]),
                        "Entradas": f"({int(x_i[0])}, {int(x_i[1])}, {int(x_i[2])})",
                        "Yd": target,
                        "Net": round(net, 4),
                        "y": y_hat,
                        "Error": error,
                        "Ajuste": "Sí" if updated else "No",
                        "w1": round(float(w_before[0]), 4),
                        "w2": round(float(w_before[1]), 4),
                        "w3": round(float(w_before[2]), 4),
                        "wb": round(float(wb_before), 4),
                        "w1_nuevo": round(float(self.w[0]), 4),
                        "w2_nuevo": round(float(self.w[1]), 4),
                        "w3_nuevo": round(float(self.w[2]), 4),
                        "wb_nuevo": round(float(self.wb), 4),
                    }
                )

            correct = n_patterns - total_abs_error
            accuracy = (correct / n_patterns) * 100.0

            self.epoch_summary.append(
                {
                    "Época": epoch,
                    "Errores": int(total_abs_error),
                    "Aciertos": int(correct),
                    "Precisión (%)": round(accuracy, 1),
                    "w₁ final": round(float(self.w[0]), 2),
                    "w₂ final": round(float(self.w[1]), 2),
                    "w₃ final": round(float(self.w[2]), 2),
                    "Sesgo final (b)": round(float(self.wb), 2),
                }
            )

            if total_abs_error == 0:
                converged = True

        self.epochs_run = epoch
        self.converged = converged
        return epoch, converged

    def predict(self, x_i: np.ndarray | list[int]) -> tuple[int, float]:
        """Realiza inferencia para un vector binarizado [x1, x2, x3]."""
        x_arr = np.asarray(x_i, dtype=float)
        net = float(np.dot(x_arr, self.w) + 1.0 * self.wb)
        y_hat = self.activation_hardlim(net)
        return y_hat, net

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def get_epoch_summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.epoch_summary)

    def get_training_table_df(
        self,
        X: np.ndarray = X_TRAIN_UPN,
        Yd: np.ndarray = YD_TRAIN_UPN,
    ) -> pd.DataFrame:
        """Tabla limpia de entrenamiento (estilo Diapositiva 16)."""
        return pd.DataFrame(
            {
                "Patrón": [f"P{i + 1}" for i in range(len(X))],
                "Unidad 1 (x₁)": X[:, 0].astype(int),
                "Unidad 2 (x₂)": X[:, 1].astype(int),
                "Unidad 3 (x₃)": X[:, 2].astype(int),
                "Unidades aprobadas": X.sum(axis=1).astype(int),
                "Salida deseada (y)": Yd.astype(int),
                "Estado esperado": ["Aprobado" if v == 1 else "Desaprobado" for v in Yd],
            }
        )

    def get_truth_table_df(
        self,
        X: np.ndarray = X_TRAIN_UPN,
        Yd: np.ndarray = YD_TRAIN_UPN,
    ) -> pd.DataFrame:
        """Tabla de verificación del modelo aprendido (estilo Diapositiva 22)."""
        df, _, _ = evaluate_manual_parameters(
            float(self.w[0]), float(self.w[1]), float(self.w[2]), float(self.wb), X=X, Yd=Yd
        )
        return df

    def equation_text(self) -> str:
        return format_linear_equation(self.w, self.wb)
