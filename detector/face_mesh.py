"""
detector.face_mesh — Wrapper sobre MediaPipe Face Mesh
======================================================
Responsabilidade única: converter frame BGR em array NumPy (478, 3)
de landmarks normalizados (x, y, z ∈ [0, 1]).

Performance:
    - cv2.cvtColor para conversão BGR→RGB (otimizado em C++)
    - flags.writeable=False hint para evitar cópia interna no MediaPipe
    - Extração vetorizada via list comprehension → np.array (< 1ms para 478 pts)
"""

import numpy as np
import cv2
import mediapipe as mp

from config import (
    MAX_NUM_FACES,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    REFINE_LANDMARKS,
)


class FaceMeshDetector:
    """Encapsula o pipeline MediaPipe FaceMesh com interface NumPy."""

    __slots__ = ("_mesh",)

    def __init__(self) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=REFINE_LANDMARKS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """
        Processa um frame BGR e retorna landmarks como array (478, 3).

        Retorna None se nenhuma face for detectada no frame.
        As coordenadas são normalizadas no intervalo [0, 1]:
            - x: posição horizontal (0=esquerda, 1=direita)
            - y: posição vertical (0=topo, 1=base)
            - z: profundidade relativa (magnitude similar a x)

        Parameters
        ----------
        frame_bgr : np.ndarray
            Frame capturado pela câmera em formato BGR, shape (H, W, 3).

        Returns
        -------
        np.ndarray | None
            Array (478, 3) com coordenadas (x, y, z), ou None.
        """
        # BGR → RGB (cvtColor faz cópia otimizada em C++)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Hint: array não será modificado → MediaPipe evita cópia interna
        frame_rgb.flags.writeable = False

        results = self._mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        # Primeiro rosto detectado (max_num_faces=1)
        face = results.multi_face_landmarks[0]

        # Extração vetorizada: (478, 3) em uma operação
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face.landmark],
            dtype=np.float64,
        )

        return landmarks

    def release(self) -> None:
        """Libera recursos do MediaPipe."""
        self._mesh.close()
