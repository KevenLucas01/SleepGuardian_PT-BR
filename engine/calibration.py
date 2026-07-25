"""
engine.calibration — Calibração Dinâmica do Threshold EAR
=========================================================
Coleta o EAR médio dos primeiros N frames válidos para estabelecer
o baseline pessoal do usuário. O threshold de fadiga é derivado
dinamicamente:

    threshold = EAR_baseline × multiplicador (default: 0.80)

Isso elimina falsos positivos causados por variação anatômica
(ex: olhos naturalmente mais estreitos, ptose palpebral).

Fallback: se a calibração não completar (ex: face não detectada
por tempo suficiente), o threshold estático (0.20) é mantido.
"""

import numpy as np

from config import (
    CALIBRATION_FRAMES,
    EAR_BASELINE_MULTIPLIER,
    EAR_THRESHOLD_DEFAULT,
    CALIBRATION_EAR_MIN,
    CALIBRATION_EAR_MAX,
)


class DynamicCalibrator:
    """
    Calibrador de baseline EAR com rejeição de outliers.

    O calibrador aceita apenas valores EAR dentro da faixa
    [CALIBRATION_EAR_MIN, CALIBRATION_EAR_MAX] para evitar
    contaminação por frames com oclusão parcial ou detecção
    ruidosa.

    Attributes
    ----------
    _n_frames : int
        Número de frames necessários para completar calibração.
    _multiplier : float
        Fator multiplicativo sobre o baseline para definir threshold.
    _buffer : list[float]
        Buffer de valores EAR coletados durante calibração.
    _baseline : float | None
        EAR médio em estado alerta (None até calibração completa).
    _threshold : float
        Threshold ativo (inicia com fallback estático).
    _calibrated : bool
        Flag de calibração completa.
    """

    __slots__ = (
        "_n_frames",
        "_multiplier",
        "_buffer",
        "_baseline",
        "_threshold",
        "_calibrated",
    )

    def __init__(
        self,
        n_frames: int = CALIBRATION_FRAMES,
        multiplier: float = EAR_BASELINE_MULTIPLIER,
    ) -> None:
        self._n_frames = n_frames
        self._multiplier = multiplier
        self._buffer: list[float] = []
        self._baseline: float | None = None
        self._threshold: float = EAR_THRESHOLD_DEFAULT
        self._calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        """Retorna True se a calibração foi concluída com sucesso."""
        return self._calibrated

    @property
    def threshold(self) -> float:
        """
        Threshold EAR ativo.
        Antes da calibração: EAR_THRESHOLD_DEFAULT (0.20).
        Após calibração: baseline × multiplier.
        """
        return self._threshold

    @property
    def baseline(self) -> float | None:
        """EAR médio do baseline (None se não calibrado)."""
        return self._baseline

    @property
    def progress(self) -> int:
        """Número de frames válidos coletados até o momento."""
        return len(self._buffer)

    @property
    def total_frames(self) -> int:
        """Número total de frames necessários para calibração."""
        return self._n_frames

    def feed(self, ear_avg: float) -> bool:
        """
        Alimenta um valor EAR médio ao buffer de calibração.

        Valores fora da faixa [CALIBRATION_EAR_MIN, CALIBRATION_EAR_MAX]
        são rejeitados silenciosamente (outliers).

        Parameters
        ----------
        ear_avg : float
            EAR médio (average de olho esquerdo e direito).

        Returns
        -------
        bool
            True se a calibração está completa (neste frame ou em anterior).
        """
        if self._calibrated:
            return True

        # Rejeição de outliers: evita contaminar baseline com
        # frames onde os olhos estão fechados ou a detecção falhou
        if ear_avg < CALIBRATION_EAR_MIN or ear_avg > CALIBRATION_EAR_MAX:
            return False

        self._buffer.append(ear_avg)

        if len(self._buffer) >= self._n_frames:
            self._baseline = float(np.mean(self._buffer))
            self._threshold = self._baseline * self._multiplier
            self._calibrated = True
            return True

        return False
