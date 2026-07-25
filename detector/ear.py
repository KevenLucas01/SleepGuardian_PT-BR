"""
detector.ear — Cálculo vetorizado do EAR (Eye Aspect Ratio)
============================================================
Fórmula (Soukupová & Čech, 2016):

    EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 · ‖p1 − p4‖)

Onde p1..p6 são os 6 landmarks palpebrais definidos em config.py.

Pós-processamento:
    Filtro EMA (Exponential Moving Average) aplicado ao output para
    suavizar micro-oscilações estocásticas do MediaPipe antes de
    repassar o sinal à Máquina de Estados.

    S_t = α · X_t + (1 − α) · S_{t−1}
"""

import numpy as np

from config import LEFT_EYE_IDX, RIGHT_EYE_IDX, EMA_ALPHA


def _compute_ear_raw(landmarks: np.ndarray, eye_indices: tuple) -> float:
    """
    Calcula EAR bruto (sem filtro) para um olho.

    Parameters
    ----------
    landmarks : np.ndarray
        Array (478, 3) de landmarks normalizados.
    eye_indices : tuple
        Tupla de 6 índices [p1, p2, p3, p4, p5, p6].

    Returns
    -------
    float
        Valor EAR bruto. Retorna 0.0 em caso de degeneração geométrica.
    """
    # Slicing vetorizado: extrai (6, 2) descartando coordenada z
    pts = landmarks[list(eye_indices), :2]

    # Distâncias verticais (pálpebra superior ↔ inferior)
    A = np.linalg.norm(pts[1] - pts[5])  # ‖p2 − p6‖
    B = np.linalg.norm(pts[2] - pts[4])  # ‖p3 − p5‖

    # Distância horizontal (canto externo ↔ canto interno)
    C = np.linalg.norm(pts[0] - pts[3])  # ‖p1 − p4‖

    # Guard: degeneração (pontos colineares ou sobrepostos)
    if C < 1e-6:
        return 0.0

    return (A + B) / (2.0 * C)


class EARCalculator:
    """
    Calculadora de EAR com filtro EMA integrado.

    O filtro EMA suaviza o sinal frame-a-frame, prevenindo que
    micro-oscilações do MediaPipe resetem prematuramente o contador
    de frames consecutivos na Máquina de Estados.

    Attributes
    ----------
    _alpha : float
        Fator de suavização EMA (0 < α ≤ 1).
    _ema_left : float | None
        Estado EMA do olho esquerdo (None até primeiro frame).
    _ema_right : float | None
        Estado EMA do olho direito (None até primeiro frame).
    """

    __slots__ = ("_alpha", "_ema_left", "_ema_right")

    def __init__(self, alpha: float = EMA_ALPHA) -> None:
        self._alpha = alpha
        self._ema_left: float | None = None
        self._ema_right: float | None = None

    def compute(
        self, landmarks: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Calcula EAR filtrado (EMA) para ambos os olhos.

        Parameters
        ----------
        landmarks : np.ndarray
            Array (478, 3) de landmarks normalizados.

        Returns
        -------
        tuple[float, float, float]
            (ear_left, ear_right, ear_avg) após filtragem EMA.
        """
        raw_left = _compute_ear_raw(landmarks, LEFT_EYE_IDX)
        raw_right = _compute_ear_raw(landmarks, RIGHT_EYE_IDX)

        # Inicialização: primeiro frame define estado EMA
        if self._ema_left is None:
            self._ema_left = raw_left
            self._ema_right = raw_right
        else:
            # S_t = α · X_t + (1 − α) · S_{t−1}
            self._ema_left = (
                self._alpha * raw_left + (1.0 - self._alpha) * self._ema_left
            )
            self._ema_right = (
                self._alpha * raw_right + (1.0 - self._alpha) * self._ema_right
            )

        avg = (self._ema_left + self._ema_right) / 2.0
        return self._ema_left, self._ema_right, avg

    def reset(self) -> None:
        """Reinicia estado EMA (ex: após reconexão de câmera)."""
        self._ema_left = None
        self._ema_right = None
