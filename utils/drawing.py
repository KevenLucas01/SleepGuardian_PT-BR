"""
utils.drawing — HUD (Head-Up Display) de Monitoramento
=======================================================
Renderiza overlay visual sobre o frame da câmera com:
    - Valores EAR/MAR em tempo real com barras de progresso
    - Indicador de estado (AWAKE/DROWSY/MICROSLEEP) com cor codificada
    - Contornos dos olhos e boca (throttled via HUD_FRAME_SKIP)
    - FPS counter (EMA-smoothed)
    - Contador de bocejos acumulados
    - Barra de progresso de sonolência (frames até MICROSLEEP)
    - Overlay de calibração (durante fase inicial)
    - Borda pulsante vermelha durante MICROSLEEP

Throttling de renderização:
    - Texto (putText): renderizado SEMPRE (custo negligível, ~0.1ms)
    - Polylines (contornos): renderizados a cada HUD_FRAME_SKIP frames
    - Borda de alerta: renderizada SEMPRE (indicador de segurança crítico)
"""

import math
import time

import cv2
import numpy as np

from config import (
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_CONTOUR,
    MOUTH_CONTOUR,
    MAR_THRESHOLD,
    EAR_CONSEC_FRAMES,
)
from engine.state_machine import FatigueState


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSTANTES VISUAIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# Cores BGR
_COLOR_GREEN = (0, 200, 0)
_COLOR_YELLOW = (0, 220, 255)
_COLOR_RED = (0, 0, 255)
_COLOR_ORANGE = (0, 165, 255)
_COLOR_WHITE = (255, 255, 255)
_COLOR_GRAY = (180, 180, 180)
_COLOR_DARK_GRAY = (80, 80, 80)
_COLOR_BORDER = (150, 150, 150)
_COLOR_CYAN = (255, 255, 0)
_COLOR_BG_OVERLAY = (20, 20, 20)

# Mapeamento estado → cor
_STATE_COLORS = {
    FatigueState.AWAKE: _COLOR_GREEN,
    FatigueState.DROWSY: _COLOR_YELLOW,
    FatigueState.MICROSLEEP: _COLOR_RED,
}

# Mapeamento estado → label
_STATE_LABELS = {
    FatigueState.AWAKE: "AWAKE",
    FatigueState.DROWSY: "DROWSY",
    FatigueState.MICROSLEEP: "MICROSLEEP",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FUNÇÕES AUXILIARES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _landmarks_to_pixels(
    landmarks: np.ndarray, indices: tuple, w: int, h: int
) -> np.ndarray:
    """
    Converte landmarks normalizados para coordenadas de pixel.

    Parameters
    ----------
    landmarks : np.ndarray
        Array (478, 3) de landmarks normalizados.
    indices : tuple
        Índices dos landmarks a converter.
    w, h : int
        Dimensões do frame.

    Returns
    -------
    np.ndarray
        Array (N, 1, 2) int32 para uso com cv2.polylines.
    """
    pts = landmarks[list(indices), :2]  # (N, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(np.int32).reshape(-1, 1, 2)


def _draw_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    fill_ratio: float,
    bar_color: tuple,
    threshold_ratio: float | None = None,
) -> None:
    """
    Desenha barra de progresso horizontal com threshold marker opcional.

    Parameters
    ----------
    frame : np.ndarray
        Frame de destino.
    x, y : int
        Canto superior esquerdo da barra.
    width, height : int
        Dimensões da barra.
    fill_ratio : float
        Preenchimento (0.0 a 1.0).
    bar_color : tuple
        Cor BGR do preenchimento.
    threshold_ratio : float | None
        Se fornecido, desenha marcador vertical na posição relativa.
    """
    fill = int(np.clip(fill_ratio, 0.0, 1.0) * width)

    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), _COLOR_DARK_GRAY, -1)
    # Fill
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + height), bar_color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), _COLOR_BORDER, 1)

    # Threshold marker
    if threshold_ratio is not None:
        tx = x + int(np.clip(threshold_ratio, 0.0, 1.0) * width)
        cv2.line(frame, (tx, y - 2), (tx, y + height + 2), _COLOR_CYAN, 2)


def _draw_contours(
    frame: np.ndarray, landmarks: np.ndarray, w: int, h: int
) -> None:
    """Desenha contornos dos olhos e boca como polylines."""
    # Olho esquerdo
    pts_le = _landmarks_to_pixels(landmarks.copy(), LEFT_EYE_CONTOUR, w, h)
    cv2.polylines(frame, [pts_le], isClosed=True, color=_COLOR_GREEN, thickness=1)

    # Olho direito
    pts_re = _landmarks_to_pixels(landmarks.copy(), RIGHT_EYE_CONTOUR, w, h)
    cv2.polylines(frame, [pts_re], isClosed=True, color=_COLOR_GREEN, thickness=1)

    # Boca
    pts_m = _landmarks_to_pixels(landmarks.copy(), MOUTH_CONTOUR, w, h)
    cv2.polylines(frame, [pts_m], isClosed=True, color=_COLOR_CYAN, thickness=1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FUNÇÕES PÚBLICAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def draw_hud(
    frame: np.ndarray,
    ear_avg: float,
    mar: float,
    state: FatigueState,
    fps: float,
    yawn_count: int,
    is_yawning: bool,
    ear_counter: int,
    threshold: float,
    landmarks: np.ndarray | None,
    draw_contours: bool,
) -> None:
    """
    Renderiza HUD completo sobre o frame.

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR para desenho in-place.
    ear_avg : float
        EAR médio filtrado.
    mar : float
        MAR filtrado.
    state : FatigueState
        Estado atual da FSM.
    fps : float
        FPS atual (smoothed).
    yawn_count : int
        Total de bocejos na sessão.
    is_yawning : bool
        Bocejo em andamento.
    ear_counter : int
        Frames consecutivos com EAR abaixo do threshold.
    threshold : float
        Threshold EAR ativo (calibrado ou default).
    landmarks : np.ndarray | None
        Landmarks para renderização de contornos (None = sem face).
    draw_contours : bool
        Se True, renderiza polylines dos contornos (throttled).
    """
    h, w = frame.shape[:2]
    state_color = _STATE_COLORS.get(state, _COLOR_GRAY)
    state_label = _STATE_LABELS.get(state, "UNKNOWN")

    # ── TOP-LEFT: FPS ──
    cv2.putText(
        frame, f"FPS: {fps:.0f}", (10, 28),
        _FONT, 0.65, _COLOR_GRAY, 1, cv2.LINE_AA,
    )

    # ── TOP-LEFT: Estado com badge colorido ──
    (tw, th), _ = cv2.getTextSize(state_label, _FONT_BOLD, 0.7, 2)
    badge_x1, badge_y1 = 10, 38
    badge_x2, badge_y2 = 20 + tw, 42 + th + 8
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), state_color, -1)
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), _COLOR_WHITE, 1)
    cv2.putText(
        frame, state_label, (15, badge_y2 - 6),
        _FONT_BOLD, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
    )

    # ── TOP-RIGHT: EAR com barra ──
    bar_w = 140
    bar_h = 14
    bar_x = w - bar_w - 20
    ear_color = _COLOR_GREEN if ear_avg >= threshold else _COLOR_RED

    cv2.putText(
        frame, f"EAR: {ear_avg:.3f}", (bar_x, 25),
        _FONT, 0.55, _COLOR_GRAY, 1, cv2.LINE_AA,
    )
    _draw_bar(
        frame, bar_x, 32, bar_w, bar_h,
        fill_ratio=ear_avg / 0.40,
        bar_color=ear_color,
        threshold_ratio=threshold / 0.40,
    )

    # ── TOP-RIGHT: MAR com barra ──
    mar_color = _COLOR_GREEN if mar <= MAR_THRESHOLD else _COLOR_ORANGE

    cv2.putText(
        frame, f"MAR: {mar:.3f}", (bar_x, 70),
        _FONT, 0.55, _COLOR_GRAY, 1, cv2.LINE_AA,
    )
    _draw_bar(
        frame, bar_x, 77, bar_w, bar_h,
        fill_ratio=mar / 1.0,
        bar_color=mar_color,
        threshold_ratio=MAR_THRESHOLD / 1.0,
    )

    # ── BOTTOM-LEFT: Bocejos ──
    cv2.putText(
        frame, f"Bocejos: {yawn_count}", (10, h - 15),
        _FONT, 0.55, _COLOR_GRAY, 1, cv2.LINE_AA,
    )

    # ── BOTTOM-RIGHT: Threshold info ──
    cv2.putText(
        frame, f"Limiar EAR: {threshold:.3f}", (w - 190, h - 15),
        _FONT, 0.5, _COLOR_GRAY, 1, cv2.LINE_AA,
    )

    # ── CONTORNOS (THROTTLED) ──
    if draw_contours and landmarks is not None:
        _draw_contours(frame, landmarks, w, h)

    # ── DROWSY: barra de progresso até MICROSLEEP ──
    if state == FatigueState.DROWSY and EAR_CONSEC_FRAMES > 0:
        progress = ear_counter / EAR_CONSEC_FRAMES
        prog_w = 200
        prog_h = 6
        prog_x = (w - prog_w) // 2
        prog_y = h - 40
        _draw_bar(
            frame, prog_x, prog_y, prog_w, prog_h,
            fill_ratio=progress,
            bar_color=_COLOR_YELLOW,
        )
        cv2.putText(
            frame,
            f"Sonolencia: {ear_counter}/{EAR_CONSEC_FRAMES}",
            (prog_x, prog_y - 5),
            _FONT, 0.45, _COLOR_YELLOW, 1, cv2.LINE_AA,
        )

    # ── MICROSLEEP: borda pulsante vermelha ──
    if state == FatigueState.MICROSLEEP:
        pulse = abs(math.sin(time.time() * 4.0))  # 4Hz oscillation
        thickness = int(4 + pulse * 10)
        border_color = (0, 0, int(180 + pulse * 75))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, thickness)

        # Texto de alerta centralizado
        warn_text = "ALERTA: MICROSSONO!"
        (tw, th), _ = cv2.getTextSize(warn_text, _FONT_BOLD, 1.1, 3)
        tx = (w - tw) // 2
        ty = h // 2

        # Background do texto
        cv2.rectangle(
            frame,
            (tx - 12, ty - th - 12),
            (tx + tw + 12, ty + 12),
            (0, 0, 160),
            -1,
        )
        cv2.rectangle(
            frame,
            (tx - 12, ty - th - 12),
            (tx + tw + 12, ty + 12),
            _COLOR_WHITE,
            2,
        )
        cv2.putText(
            frame, warn_text, (tx, ty),
            _FONT_BOLD, 1.1, _COLOR_WHITE, 3, cv2.LINE_AA,
        )

    # ── YAWNING: indicador de bocejo ativo ──
    if is_yawning:
        yawn_text = "BOCEJO DETECTADO"
        (tw, _), _ = cv2.getTextSize(yawn_text, _FONT, 0.7, 2)
        yx = (w - tw) // 2
        cv2.putText(
            frame, yawn_text, (yx, h - 50),
            _FONT, 0.7, _COLOR_ORANGE, 2, cv2.LINE_AA,
        )

    # ── SEM FACE: indicador ──
    if landmarks is None:
        no_face_text = "FACE NAO DETECTADA"
        (tw, th), _ = cv2.getTextSize(no_face_text, _FONT, 0.8, 2)
        nx = (w - tw) // 2
        ny = (h + th) // 2
        cv2.putText(
            frame, no_face_text, (nx, ny),
            _FONT, 0.8, _COLOR_YELLOW, 2, cv2.LINE_AA,
        )


def draw_calibration_overlay(
    frame: np.ndarray,
    progress: int,
    total: int,
) -> None:
    """
    Renderiza overlay de calibração com barra de progresso.

    Exibido durante os primeiros N frames enquanto o sistema
    coleta o baseline EAR do usuário.

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR para desenho in-place.
    progress : int
        Frames válidos coletados até o momento.
    total : int
        Total de frames necessários para calibração.
    """
    h, w = frame.shape[:2]

    # Overlay semi-transparente
    overlay = frame.copy()
    box_w, box_h = 340, 110
    box_x = (w - box_w) // 2
    box_y = (h - box_h) // 2

    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        _COLOR_BG_OVERLAY,
        -1,
    )
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Borda
    cv2.rectangle(
        frame,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        _COLOR_CYAN,
        2,
    )

    # Título
    cv2.putText(
        frame, "CALIBRANDO...", (box_x + 20, box_y + 35),
        _FONT_BOLD, 0.8, _COLOR_CYAN, 2, cv2.LINE_AA,
    )

    # Barra de progresso
    bar_x = box_x + 20
    bar_y = box_y + 50
    bar_w = box_w - 40
    bar_h = 16
    ratio = progress / max(total, 1)
    _draw_bar(frame, bar_x, bar_y, bar_w, bar_h, ratio, _COLOR_CYAN)

    # Texto de progresso
    pct_text = f"{progress}/{total} frames ({ratio * 100:.0f}%)"
    cv2.putText(
        frame, pct_text, (bar_x, bar_y + bar_h + 20),
        _FONT, 0.5, _COLOR_GRAY, 1, cv2.LINE_AA,
    )

    # Instrução
    cv2.putText(
        frame, "Mantenha os olhos abertos", (bar_x, bar_y + bar_h + 40),
        _FONT, 0.5, _COLOR_WHITE, 1, cv2.LINE_AA,
    )
