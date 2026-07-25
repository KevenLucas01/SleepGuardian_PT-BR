"""
engine.alert — Sistema de Alertas Sonoros Contínuos (Non-Blocking)
==================================================================
Plataforma primária: Windows (winsound.PlaySound com SND_ASYNC | SND_LOOP).
Fallback: terminal bell (\a) em loop via thread daemon para Linux/macOS.

Comportamento:
    - trigger() inicia reprodução em LOOP CONTÍNUO (irritante por design).
    - O som repete ininterruptamente até stop() ser chamado.
    - stop() é chamado pelo main.py quando o estado SAI de MICROSLEEP.
    - Idempotente: chamadas repetidas a trigger() enquanto já ativo são no-op.
"""

import time
import platform
import threading

from config import ALERT_WAV_PATH


class AlertSystem:
    """
    Sistema de alerta sonoro com repetição contínua.

    O alerta reproduz em loop infinito enquanto o estado MICROSLEEP
    estiver ativo. O objetivo é forçar o motorista a acordar —
    o som só para quando os olhos abrirem (EAR ≥ threshold).

    Windows: winsound.PlaySound com SND_LOOP | SND_ASYNC.
    Fallback: thread daemon com beeps em loop rápido.

    Attributes
    ----------
    _is_windows : bool
        Flag de detecção de plataforma.
    _active : bool
        True se o alerta está reproduzindo (evita re-trigger).
    _stop_event : threading.Event
        Sinal para interromper loop do fallback multiplataforma.
    _fallback_thread : threading.Thread | None
        Thread do fallback (Linux/macOS).
    """

    __slots__ = ("_is_windows", "_active", "_stop_event", "_fallback_thread")

    def __init__(self) -> None:
        self._is_windows = platform.system() == "Windows"
        self._active: bool = False
        self._stop_event = threading.Event()
        self._fallback_thread: threading.Thread | None = None

    @property
    def is_active(self) -> bool:
        """True se o alerta está reproduzindo atualmente."""
        return self._active

    def trigger(self) -> None:
        """
        Inicia alerta sonoro em loop contínuo.

        Idempotente: se o alerta já está ativo, não faz nada.
        O som repete até stop() ser chamado explicitamente.
        """
        if self._active:
            return  # Já tocando — no-op

        self._active = True

        if self._is_windows:
            self._play_windows_loop()
        else:
            self._play_fallback_loop()

    def _play_windows_loop(self) -> None:
        """Reproduz WAV em loop infinito via winsound (Windows-native)."""
        import winsound

        try:
            # SND_ASYNC: non-blocking (retorna imediatamente)
            # SND_LOOP: repete infinitamente até PlaySound(None, SND_PURGE)
            # SND_FILENAME: lê de arquivo WAV
            winsound.PlaySound(
                ALERT_WAV_PATH,
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
            )
        except Exception:
            # Fallback: beep do sistema em loop via thread
            self._play_fallback_loop()

    def _play_fallback_loop(self) -> None:
        """Fallback multiplataforma: beeps rápidos em thread daemon."""
        self._stop_event.clear()

        def _beep_loop() -> None:
            while not self._stop_event.is_set():
                print("\a", end="", flush=True)
                # Intervalo curto entre beeps para ser irritante
                self._stop_event.wait(0.3)

        self._fallback_thread = threading.Thread(
            target=_beep_loop, daemon=True
        )
        self._fallback_thread.start()

    def stop(self) -> None:
        """
        Interrompe alerta sonoro imediatamente.

        Chamado quando o estado da FSM sai de MICROSLEEP
        (motorista abriu os olhos).
        """
        if not self._active:
            return

        self._active = False

        # Parar winsound (Windows)
        if self._is_windows:
            try:
                import winsound

                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

        # Parar thread de fallback
        self._stop_event.set()
        if self._fallback_thread is not None:
            self._fallback_thread.join(timeout=1.0)
            self._fallback_thread = None
