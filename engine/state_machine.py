"""
engine.state_machine — Máquina de Estados Finitos para Detecção de Fadiga
==========================================================================
Estados:
    AWAKE       → EAR ≥ threshold, operação normal.
    DROWSY      → EAR < threshold, contagem de frames iniciada.
    MICROSLEEP  → EAR < threshold por ≥ EAR_CONSEC_FRAMES → alerta crítico.

Canal paralelo:
    YAWNING     → MAR > MAR_THRESHOLD por ≥ MAR_CONSEC_FRAMES.
    Flag booleana independente dos estados EAR.

Transições:
    AWAKE ──(EAR < thresh)──→ DROWSY ──(frames ≥ N)──→ MICROSLEEP
      ↑                          │                          │
      └──(EAR ≥ thresh)─────────┴──────────────────────────┘
"""

from enum import Enum, auto

from config import EAR_CONSEC_FRAMES, MAR_THRESHOLD, MAR_CONSEC_FRAMES


class FatigueState(Enum):
    """Estados da máquina de fadiga."""

    AWAKE = auto()
    DROWSY = auto()
    MICROSLEEP = auto()


class FatigueStateMachine:
    """
    FSM de fadiga com contadores de frames consecutivos.

    O threshold EAR é injetado externamente (via set_threshold)
    após a calibração dinâmica. Até lá, usa o fallback de config.py.

    Attributes
    ----------
    _state : FatigueState
        Estado atual da FSM.
    _ear_counter : int
        Frames consecutivos com EAR < threshold.
    _mar_counter : int
        Frames consecutivos com MAR > MAR_THRESHOLD.
    _yawn_total : int
        Total acumulado de bocejos na sessão.
    _is_yawning : bool
        Flag de bocejo ativo (para evitar contagem dupla).
    _ear_threshold : float
        Threshold EAR ativo (atualizado pela calibração).
    """

    __slots__ = (
        "_state",
        "_ear_counter",
        "_mar_counter",
        "_yawn_total",
        "_is_yawning",
        "_ear_threshold",
    )

    def __init__(self) -> None:
        self._state = FatigueState.AWAKE
        self._ear_counter: int = 0
        self._mar_counter: int = 0
        self._yawn_total: int = 0
        self._is_yawning: bool = False
        self._ear_threshold: float = 0.20  # Override via set_threshold()

    # ──────── Properties ────────

    @property
    def state(self) -> FatigueState:
        """Estado atual da FSM."""
        return self._state

    @property
    def ear_counter(self) -> int:
        """Frames consecutivos com EAR abaixo do threshold."""
        return self._ear_counter

    @property
    def mar_counter(self) -> int:
        """Frames consecutivos com MAR acima do threshold."""
        return self._mar_counter

    @property
    def yawn_total(self) -> int:
        """Total de bocejos detectados na sessão."""
        return self._yawn_total

    @property
    def is_yawning(self) -> bool:
        """True se um bocejo está em progresso."""
        return self._is_yawning

    @property
    def ear_threshold(self) -> float:
        """Threshold EAR ativo."""
        return self._ear_threshold

    # ──────── Controle ────────

    def set_threshold(self, threshold: float) -> None:
        """
        Atualiza o threshold EAR (chamado após calibração dinâmica).

        Parameters
        ----------
        threshold : float
            Novo threshold derivado de baseline × multiplicador.
        """
        self._ear_threshold = threshold

    def update(self, ear: float, mar: float) -> FatigueState:
        """
        Atualiza a FSM com novos valores EAR e MAR.

        Parameters
        ----------
        ear : float
            EAR médio filtrado (pós-EMA).
        mar : float
            MAR filtrado (pós-EMA).

        Returns
        -------
        FatigueState
            Estado resultante após transição.
        """
        # ── Canal EAR: AWAKE → DROWSY → MICROSLEEP ──
        if ear < self._ear_threshold:
            self._ear_counter += 1
            if self._ear_counter >= EAR_CONSEC_FRAMES:
                self._state = FatigueState.MICROSLEEP
            else:
                self._state = FatigueState.DROWSY
        else:
            self._ear_counter = 0
            self._state = FatigueState.AWAKE

        # ── Canal MAR: detecção de bocejo (paralelo) ──
        if mar > MAR_THRESHOLD:
            self._mar_counter += 1
            if self._mar_counter >= MAR_CONSEC_FRAMES:
                if not self._is_yawning:
                    self._yawn_total += 1
                    self._is_yawning = True
        else:
            self._mar_counter = 0
            self._is_yawning = False

        return self._state
