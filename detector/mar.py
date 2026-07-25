"""
detector.mar — Cálculo vetorizado do MAR (Mouth Aspect Ratio)
==============================================================
Fórmula análoga ao EAR, aplicada aos 6 landmarks bucais:

    MAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 · ‖p1 − p4‖)

Onde:
    p1 = canto esquerdo (idx 61)
    p2 = lábio superior centro (idx 13)
    p3 = lábio superior direito (idx 312)
    p4 = canto direito (idx 291)
    p5 = lábio inferior direito (idx 317)
    p6 = lábio inferior centro (idx 14)

Pós-processamento: filtro EMA idêntico ao módulo EAR.
"""

import numpy as np

from config import MOUTH_IDX, EMA_ALPHA


def _compute_mar_raw(landmarks: np.ndarray) -> float:
    """
    Calcula MAR bruto (sem filtro).

    Parameters
    ----------
    landmarks : np.ndarray
        Array (478, 3) de landmarks normalizados.

    Returns
    -------
    float
        Valor MAR bruto. Retorna 0.0 em caso de degeneração.
    """
    pts = landmarks[list(MOUTH_IDX), :2]  # (6, 2)

    # Distâncias verticais (abertura da boca)
    A = np.linalg.norm(pts[1] - pts[5])  # ‖p2 − p6‖ (centro)
    B = np.linalg.norm(pts[2] - pts[4])  # ‖p3 − p5‖ (lateral)

    # Distância horizontal (largura da boca)
    C = np.linalg.norm(pts[0] - pts[3])  # ‖p1 − p4‖

    if C < 1e-6:
        return 0.0

    return (A + B) / (2.0 * C)


class MARCalculator:
    """
    Calculadora de MAR com filtro EMA integrado.

    Attributes
    ----------
    _alpha : float
        Fator de suavização EMA.
    _ema_value : float | None
        Estado EMA atual (None até primeiro frame).
    """

    __slots__ = ("_alpha", "_ema_value")

    def __init__(self, alpha: float = EMA_ALPHA) -> None:
        self._alpha = alpha
        self._ema_value: float | None = None

    def compute(self, landmarks: np.ndarray) -> float:
        """
        Calcula MAR filtrado (EMA).

        Parameters
        ----------
        landmarks : np.ndarray
            Array (478, 3) de landmarks normalizados.

        Returns
        -------
        float
            Valor MAR após filtragem EMA.
        """
        raw = _compute_mar_raw(landmarks)

        if self._ema_value is None:
            self._ema_value = raw
        else:
            self._ema_value = (
                self._alpha * raw + (1.0 - self._alpha) * self._ema_value
            )

        return self._ema_value

    def reset(self) -> None:
        """Reinicia estado EMA."""
        self._ema_value = None
