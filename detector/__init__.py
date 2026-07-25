"""
detector — Módulos de aquisição de landmarks e cálculo de métricas biométricas.
"""

from .face_mesh import FaceMeshDetector
from .ear import EARCalculator
from .mar import MARCalculator

__all__ = ["FaceMeshDetector", "EARCalculator", "MARCalculator"]
