"""
Sleep Guardian — Entry Point
=============================
Loop principal de captura, inferência e orquestração.

Pipeline por frame:
    1. Captura via cv2.VideoCapture (DirectShow no Windows)
    2. Inferência MediaPipe Face Mesh → array (478, 3)
    3. Cálculo EAR/MAR com filtro EMA
    4. Alimentação do calibrador dinâmico (primeiros N frames)
    5. Atualização da FSM de fadiga
    6. Disparo de alertas (se MICROSLEEP)
    7. Renderização HUD (texto: todo frame / contornos: throttled)

Tolerância a falha:
    - Face não detectada: mantém último estado válido por até
      MAX_NO_FACE_FRAMES frames, depois reseta para AWAKE.
    - Câmera desconectada: tenta reconexão com delay configurável.
    - Exceções no loop: captura genérica com logging stderr.
"""

import sys
import time

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    HUD_FRAME_SKIP,
    MAX_NO_FACE_FRAMES,
    CAMERA_RECONNECT_DELAY,
)
from detector import FaceMeshDetector, EARCalculator, MARCalculator
from engine import FatigueState, FatigueStateMachine, DynamicCalibrator, AlertSystem
from utils import draw_hud, draw_calibration_overlay


def _init_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Inicializa câmera com resolução configurada.

    No Windows, força backend DirectShow (cv2.CAP_DSHOW) para
    menor latência de inicialização.

    Parameters
    ----------
    index : int
        Índice da câmera.

    Returns
    -------
    cv2.VideoCapture
        Objeto de captura configurado.

    Raises
    ------
    RuntimeError
        Se a câmera não puder ser aberta.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback sem backend explícito
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Falha ao abrir camera index={index}. "
            "Verifique conexao e permissoes."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimizar latência de buffer

    return cap


def main() -> None:
    """Loop principal do Sleep Guardian."""

    # ── Inicialização dos componentes ──
    print("[Sleep Guardian] Inicializando...", file=sys.stderr)

    cap = _init_camera()
    face_mesh = FaceMeshDetector()
    ear_calc = EARCalculator()
    mar_calc = MARCalculator()
    calibrator = DynamicCalibrator()
    state_machine = FatigueStateMachine()
    alert_system = AlertSystem()

    print("[Sleep Guardian] Sistema ativo. Pressione 'q' para encerrar.", file=sys.stderr)

    # ── Estado do loop ──
    frame_count: int = 0
    fps: float = 0.0
    fps_alpha: float = 0.1  # EMA para suavização do FPS display
    prev_time: float = time.monotonic()
    no_face_counter: int = 0

    # Valores do HUD (persistem entre frames sem face)
    hud_ear: float = 0.0
    hud_mar: float = 0.0
    hud_state: FatigueState = FatigueState.AWAKE
    hud_yawn_count: int = 0
    hud_is_yawning: bool = False
    hud_ear_counter: int = 0
    hud_landmarks: np.ndarray | None = None

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                # ── Reconexão da câmera ──
                print(
                    "[Sleep Guardian] Frame perdido. Tentando reconexao...",
                    file=sys.stderr,
                )
                cap.release()
                time.sleep(CAMERA_RECONNECT_DELAY)
                try:
                    cap = _init_camera()
                except RuntimeError:
                    continue
                # Reset dos filtros EMA após reconexão
                ear_calc.reset()
                mar_calc.reset()
                continue

            # Espelhar frame para feedback intuitivo
            frame = cv2.flip(frame, 1)
            frame_count += 1

            # ── FPS (EMA-smoothed) ──
            now = time.monotonic()
            dt = now - prev_time
            prev_time = now
            instant_fps = 1.0 / dt if dt > 0 else 0.0
            fps = fps_alpha * instant_fps + (1.0 - fps_alpha) * fps

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            #  INFERÊNCIA (todo frame, sem skip)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            landmarks = face_mesh.process(frame)

            if landmarks is not None:
                no_face_counter = 0

                # Métricas biométricas (pós-EMA)
                _, _, ear_avg = ear_calc.compute(landmarks)
                mar_val = mar_calc.compute(landmarks)

                # ── Calibração dinâmica ──
                if not calibrator.is_calibrated:
                    calibrator.feed(ear_avg)
                    if calibrator.is_calibrated:
                        state_machine.set_threshold(calibrator.threshold)
                        print(
                            f"[Sleep Guardian] Calibracao completa. "
                            f"Baseline EAR={calibrator.baseline:.4f}, "
                            f"Threshold={calibrator.threshold:.4f}",
                            file=sys.stderr,
                        )
                    # Durante calibração, manter estado AWAKE
                    current_state = FatigueState.AWAKE
                else:
                    # ── Atualização da FSM ──
                    current_state = state_machine.update(ear_avg, mar_val)

                    # ── Alerta de microssono ──
                    if current_state == FatigueState.MICROSLEEP:
                        alert_system.trigger()
                    else:
                        # Motorista acordou → cortar alarme imediatamente
                        alert_system.stop()

                # Atualizar dados do HUD
                hud_ear = ear_avg
                hud_mar = mar_val
                hud_state = current_state
                hud_yawn_count = state_machine.yawn_total
                hud_is_yawning = state_machine.is_yawning
                hud_ear_counter = state_machine.ear_counter
                hud_landmarks = landmarks

            else:
                # ── Face não detectada ──
                no_face_counter += 1
                hud_landmarks = None

                if no_face_counter > MAX_NO_FACE_FRAMES:
                    # Reset para AWAKE após perda prolongada de face
                    hud_state = FatigueState.AWAKE
                    hud_ear_counter = 0
                # Caso contrário, mantém último estado válido (oclusão breve)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            #  RENDERIZAÇÃO HUD (throttled)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            should_draw_contours = (frame_count % HUD_FRAME_SKIP == 0)

            if not calibrator.is_calibrated:
                # Overlay de calibração
                draw_calibration_overlay(
                    frame,
                    progress=calibrator.progress,
                    total=calibrator.total_frames,
                )
                # FPS no canto durante calibração
                cv2.putText(
                    frame,
                    f"FPS: {fps:.0f}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
            else:
                draw_hud(
                    frame=frame,
                    ear_avg=hud_ear,
                    mar=hud_mar,
                    state=hud_state,
                    fps=fps,
                    yawn_count=hud_yawn_count,
                    is_yawning=hud_is_yawning,
                    ear_counter=hud_ear_counter,
                    threshold=calibrator.threshold,
                    landmarks=hud_landmarks,
                    draw_contours=should_draw_contours,
                )

            # ── Display ──
            cv2.imshow("Sleep Guardian", frame)

            # ── Input: 'q' para encerrar ──
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[Sleep Guardian] Interrompido pelo usuario.", file=sys.stderr)
    finally:
        # ── Cleanup ──
        print("[Sleep Guardian] Encerrando...", file=sys.stderr)
        alert_system.stop()
        face_mesh.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
