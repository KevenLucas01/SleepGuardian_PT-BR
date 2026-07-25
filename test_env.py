import sys
import pytest

def test_mediapipe_cpp_binding_integrity():
    """Valida se o interpretador atual suporta as extensões nativas do MediaPipe."""
    # Hard Gate 1: Validação de versão do interpretador (MediaPipe homologado até 3.12)
    assert sys.version_info < (3, 13), f"[ERRO FATAL] Python {sys.version_info.major}.{sys.version_info.minor} detectado. Requer Python <= 3.12."
    
    try:
        import mediapipe as mp
        # Hard Gate 2: Validação de Namespace
        _ = mp.solutions.face_mesh.FaceMesh
    except AttributeError as e:
        pytest.fail(f"[FALHA DE BINDING] O módulo C++ falhou ao carregar em memória: {e}")
    except ModuleNotFoundError:
        pytest.fail("[FALHA DE AMBIENTE] MediaPipe não está instalado no VENV ativo.")
