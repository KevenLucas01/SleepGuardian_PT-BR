"""
Sleep Guardian — Configuração Central
=====================================
Constantes imutáveis, thresholds paramétricos e índices topológicos
do MediaPipe Face Mesh (478 landmarks, refine_landmarks=True).

Convenção de índices (EAR/MAR):
    [p1, p2, p3, p4, p5, p6]
    p1 = canto externo/esquerdo
    p2 = pálpebra superior medial / lábio superior central
    p3 = pálpebra superior lateral / lábio superior lateral
    p4 = canto interno/direito
    p5 = pálpebra inferior lateral / lábio inferior lateral
    p6 = pálpebra inferior medial / lábio inferior central
"""

from typing import Tuple

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LANDMARKS — OLHOS (EAR)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ref: Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks" (2016)
#  Mapeamento para MediaPipe Face Mesh 478-point topology.

LEFT_EYE_IDX: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
#  p1=33  (canto externo)      p4=133 (canto interno)
#  p2=160 (sup. medial)        p5=153 (inf. lateral)
#  p3=158 (sup. lateral)       p6=144 (inf. medial)

RIGHT_EYE_IDX: Tuple[int, ...] = (263, 387, 385, 362, 380, 373)
#  p1=263 (canto externo)      p4=362 (canto interno)
#  p2=387 (sup. medial)        p5=380 (inf. lateral)
#  p3=385 (sup. lateral)       p6=373 (inf. medial)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LANDMARKS — BOCA (MAR)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MOUTH_IDX: Tuple[int, ...] = (61, 13, 312, 291, 317, 14)
#  p1=61  (canto esquerdo)     p4=291 (canto direito)
#  p2=13  (lábio sup. centro)  p5=317 (lábio inf. direito)
#  p3=312 (lábio sup. direito) p6=14  (lábio inf. centro)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONTORNOS COMPLETOS (RENDERIZAÇÃO / POLYLINES)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEFT_EYE_CONTOUR: Tuple[int, ...] = (
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7,
)
RIGHT_EYE_CONTOUR: Tuple[int, ...] = (
    263, 466, 388, 387, 386, 385, 384, 398,
    362, 382, 381, 380, 374, 373, 390, 249,
)
MOUTH_CONTOUR: Tuple[int, ...] = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  THRESHOLDS E CONTADORES DE FRAMES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# EAR — fallback estático (usado APENAS se calibração falhar)
EAR_THRESHOLD_DEFAULT: float = 0.20
# Frames consecutivos com EAR < threshold → MICROSLEEP
EAR_CONSEC_FRAMES: int = 20

# Calibração dinâmica: threshold = baseline_ear × multiplicador
EAR_BASELINE_MULTIPLIER: float = 0.80

# MAR — limiar estático (boca tem menos variação anatômica que olhos)
MAR_THRESHOLD: float = 0.75
# Frames consecutivos com MAR > threshold → YAWNING confirmado
MAR_CONSEC_FRAMES: int = 15

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CALIBRAÇÃO DINÂMICA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Número de frames válidos para calcular baseline EAR
CALIBRATION_FRAMES: int = 30
# Faixa de EAR aceitável durante calibração (rejeita outliers)
CALIBRATION_EAR_MIN: float = 0.10
CALIBRATION_EAR_MAX: float = 0.50

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FILTRO EMA (Exponential Moving Average)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  S_t = α · X_t + (1 − α) · S_{t-1}
#  α = 0.3 → τ ≈ 93ms a 30fps (3 frames para 63% de step response)

EMA_ALPHA: float = 0.3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CÂMERA E PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480

# Throttling de renderização: contornos desenhados a cada N frames.
# Inferência e lógica de estado rodam a cada frame (30fps).
# HUD textual (EAR/MAR/FPS) renderiza a cada frame (custo negligível).
# Polylines (contornos olho/boca) renderizam a cada HUD_FRAME_SKIP frames.
HUD_FRAME_SKIP: int = 2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MEDIAPIPE FACE MESH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_NUM_FACES: int = 1
MIN_DETECTION_CONFIDENCE: float = 0.5
MIN_TRACKING_CONFIDENCE: float = 0.5
# 478 landmarks (inclui íris) — necessário para precisão periorbital
REFINE_LANDMARKS: bool = True

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ALERTAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
import sys

def _get_asset_path(filename: str) -> str:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, 'assets', filename)
    return os.path.join('assets', filename)

ALERT_COOLDOWN_SEC: float = 3.0
ALERT_WAV_PATH: str = _get_asset_path("alert.wav")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TOLERÂNCIA A FALHA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Frames sem face detectada antes de resetar estado para AWAKE
MAX_NO_FACE_FRAMES: int = 10
# Delay de reconexão da câmera (segundos)
CAMERA_RECONNECT_DELAY: float = 0.5
