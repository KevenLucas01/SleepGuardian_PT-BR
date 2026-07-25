"""
engine — Máquina de estados, calibração dinâmica e sistema de alertas.
"""

from .state_machine import FatigueState, FatigueStateMachine
from .calibration import DynamicCalibrator
from .alert import AlertSystem

__all__ = [
    "FatigueState",
    "FatigueStateMachine",
    "DynamicCalibrator",
    "AlertSystem",
]
