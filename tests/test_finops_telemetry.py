"""Unit tests for FinOps Telemetry and Cost Optimization.
Testes unitários para Telemetria FinOps e Otimização de Custos.
"""
from src.router.finops_telemetry import FinOpsTelemetry


def test_finops_dashboard_calculation():
    """Verify FinOps dashboard metrics and savings calculation."""
    telemetry = FinOpsTelemetry()
    
    # 10 requests: 7 resolved locally (cost 0), 3 via cheap LLM ($0.0003 each)
    mock_logs = [
        {"route": "LEVEL_0_LOCAL", "intent": "FAQ", "estimated_cost": 0.0, "latency_ms": 2.5, "pgr_alert": 0},
        {"route": "LEVEL_0_LOCAL", "intent": "CACHE_HIT", "estimated_cost": 0.0, "latency_ms": 0.8, "pgr_alert": 0},
        {"route": "LEVEL_0_LOCAL", "intent": "FAQ", "estimated_cost": 0.0, "latency_ms": 3.1, "pgr_alert": 0},
        {"route": "LEVEL_0_LOCAL", "intent": "CACHE_HIT", "estimated_cost": 0.0, "latency_ms": 0.9, "pgr_alert": 0},
        {"route": "LEVEL_0_LOCAL", "intent": "FAQ", "estimated_cost": 0.0, "latency_ms": 1.9, "pgr_alert": 0},
        {"route": "LEVEL_0_LOCAL", "intent": "INCIDENT", "estimated_cost": 0.0, "latency_ms": 2.0, "pgr_alert": 1},
        {"route": "LEVEL_0_LOCAL", "intent": "FAQ", "estimated_cost": 0.0, "latency_ms": 2.2, "pgr_alert": 0},
        {"route": "LEVEL_2_CHEAP", "intent": "GENERAL", "estimated_cost": 0.0003, "latency_ms": 250.0, "pgr_alert": 0},
        {"route": "LEVEL_2_CHEAP", "intent": "GENERAL", "estimated_cost": 0.0003, "latency_ms": 280.0, "pgr_alert": 0},
        {"route": "LEVEL_4_ADVANCED", "intent": "CONTRACT", "estimated_cost": 0.0015, "latency_ms": 850.0, "pgr_alert": 0},
    ]
    
    dashboard = telemetry.generate_dashboard_metrics(mock_logs)
    
    assert dashboard["total_requests"] == 10
    assert dashboard["local_resolved_pct"] == 70.0
    assert dashboard["cache_hit_pct"] == 20.0
    assert dashboard["llm_call_pct"] == 30.0
    assert dashboard["pgr_alerts_count"] == 1
    # Savings should be significant (> 80%)
    assert dashboard["savings_pct"] >= 85.0
    assert dashboard["j_optimization_index"] > 0.0
