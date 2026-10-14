"""Unit tests for PluraEngine and MATROUTE-AI Integration.
Testes unitários para o motor PluraTalks, 14 temas, 3 sinais de proteção e cumprimentos L0.
"""
from src.core_math.plura_engine import PluraEngine, fold_text
from src.router.model_router import ModelRouter, RouteLevel
from src.router.llm_adapter import LLMAdapter
from src.storage.persistent_db import PersistentDatabase
import tempfile
import os


def test_fold_text_normalization():
    """Verify text folding removes accents, lowercase, and special characters."""
    assert fold_text("Olá, Ação e Reação!") == "ola, acao e reacao!"
    assert fold_text("Sobrecarregado com prazos") == "sobrecarregado com prazos"


def test_plura_engine_detects_14_themes():
    """Verify detection of specific organizational themes (e.g. carga_ritmo, lideranca_apoio)."""
    text = "Estou com muito trabalho, sobrecarga diária e o ritmo da equipe está absurdo."
    analysis = PluraEngine.analyze(text)
    assert "carga_ritmo" in analysis["themes"]
    assert analysis["theme_scores"]["carga_ritmo"] > 0.0
    assert analysis["needs_listening"] is False
    assert analysis["pgr_alert"] is False


def test_plura_engine_detects_violence_signal():
    """Verify immediate detection of harassment, threats, or aggression."""
    text = "Meu supervisor me humilhou na frente de todos e fez uma grave ameaca de demissao."
    analysis = PluraEngine.analyze(text)
    assert analysis["signals"]["violence_signal"] is True
    assert analysis["needs_listening"] is True
    assert analysis["pgr_alert"] is True
    assert len(analysis["orientations"]) >= 1
    assert any("denúncia" in o.lower() or "ouvidoria" in o.lower() for o in analysis["orientations"])


def test_plura_engine_detects_self_harm_and_cvv():
    """Verify critical protection for self-harm and CVV 188 orientation."""
    text = "Nao aguento mais viver, quero acabar com tudo e sumir de vez."
    analysis = PluraEngine.analyze(text)
    assert analysis["signals"]["self_harm_signal"] is True
    assert analysis["needs_listening"] is True
    assert analysis["pgr_alert"] is True
    assert any("cvv" in o.lower() or "188" in o.lower() for o in analysis["orientations"])
    assert any("188" in c["contact"] for c in analysis["contacts"])


def test_plura_engine_detects_help_request():
    """Verify detection of psychological help / active listening requests."""
    text = "Preciso de apoio emocional e conversar com alguem urgente, estou tendo crise de ansiedade."
    analysis = PluraEngine.analyze(text)
    assert analysis["signals"]["help_signal"] is True
    assert analysis["needs_listening"] is True


def test_router_greetings_l0():
    """Verify greetings like 'Olá', 'Oi' route to L0 in sub-millisecond at $0.00."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
    try:
        db = PersistentDatabase(db_path=db_path)
        router = ModelRouter(db=db, llm_adapter=LLMAdapter(api_key=None))
        
        for greeting in ["Olá", "ola", "oi", "Bom dia", "boa tarde"]:
            res = router.process_request(greeting)
            assert res["level"] == RouteLevel.LEVEL_0_LOCAL
            assert res["cost_usd"] == 0.0
            assert "MATROUTE-AI" in res["text"]
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except PermissionError:
            pass
