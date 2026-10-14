"""Unit tests for Mathematical Gate and Model Router.
Testes unitários para o Gate Matemático e Roteador Seletivo de Modelos.
"""
from src.router.gate import MathematicalGate, GateMetrics
from src.router.cache import MultiLevelCache
from src.router.model_router import ModelRouter, RouteLevel
from src.router.llm_adapter import LLMAdapter
from src.storage.persistent_db import PersistentDatabase
import tempfile
import os


def test_gate_mathematical_formula():
    """Verify Gate formula S = wN*N + wR*R + wU*U + wQ*Q - wH*H."""
    gate = MathematicalGate(
        weights={"wN": 0.25, "wR": 0.25, "wU": 0.20, "wQ": 0.20, "wH": 0.20},
        tau=0.50
    )
    
    # Case 1: Simple low-risk question with high cache hit probability
    # N=0.1, R=0.1, U=0.2, Q=0.1, H=0.9
    # S = 0.25*0.1 + 0.25*0.1 + 0.20*0.2 + 0.20*0.1 - 0.20*0.9
    # S = 0.025 + 0.025 + 0.040 + 0.020 - 0.180 = -0.070 < 0.50 -> Local
    metrics_simple = GateMetrics(novelty=0.1, risk=0.1, uncertainty=0.2, complexity=0.1, cache_hit_prob=0.9)
    result_simple = gate.evaluate(metrics_simple)
    assert result_simple["score"] < 0.50
    assert result_simple["call_llm"] is False

    # Case 2: Complex high-risk novel contract analysis with 0 cache hit
    # N=0.9, R=0.8, U=0.8, Q=0.9, H=0.0
    # S = 0.25*0.9 + 0.25*0.8 + 0.20*0.8 + 0.20*0.9 - 0.0 = 0.225 + 0.200 + 0.160 + 0.180 = 0.765 >= 0.50 -> LLM
    metrics_complex = GateMetrics(novelty=0.9, risk=0.8, uncertainty=0.8, complexity=0.9, cache_hit_prob=0.0)
    result_complex = gate.evaluate(metrics_complex)
    assert result_complex["score"] >= 0.50
    assert result_complex["call_llm"] is True


def test_multi_level_cache_hit_and_eviction():
    """Verify L1 RAM LRU cache and L2 persistence."""
    cache = MultiLevelCache(max_l1_items=2)
    cache.set("key1", "response1")
    cache.set("key2", "response2")
    
    assert cache.get("key1") == "response1"
    
    # Insert 3rd item -> evicts least recently used (key2)
    cache.set("key3", "response3")
    assert cache.get("key3") == "response3"
    assert cache.get_stats()["l1_hits"] >= 1


def test_model_router_hierarchy():
    """Verify routing decisions across L0, L2, L4 and cascade escalation."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
    
    try:
        db = PersistentDatabase(db_path=db_path)
        adapter = LLMAdapter(api_key=None)  # Safe mock/fallback mode
        router = ModelRouter(db=db, llm_adapter=adapter)
        
        # 1. Simple FAQ question -> Route L0 (Local, Cost 0)
        resp_l0 = router.process_request("Como faco para redefinir minha senha?")
        assert resp_l0["level"] == RouteLevel.LEVEL_0_LOCAL
        assert resp_l0["cost_usd"] == 0.0
        assert "senha" in resp_l0["text"].lower() or "configurações" in resp_l0["text"].lower()
        
        # 2. Complex contract inquiry -> Route L3, L4, or L5 with Double Check
        resp_l4 = router.process_request("Analise este contrato e identifique possiveis riscos financeiros e juridicos graves.")
        assert resp_l4["level"] in (RouteLevel.LEVEL_3_INTERMEDIATE, RouteLevel.LEVEL_4_ADVANCED, RouteLevel.LEVEL_5_DOUBLE_CHECK)
        assert resp_l4["double_check_performed"] is True
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except PermissionError:
            pass
