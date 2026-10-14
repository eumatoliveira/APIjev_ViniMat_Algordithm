"""Unit tests for MATROUTE-AI Web Application and REST Endpoints.
Testes unitários para o servidor web FastAPI e endpoints REST.
"""
from fastapi.testclient import TestClient
from src.web.app import app


def test_web_index_serves_html():
    """Verify GET / returns HTTP 200 and loads HTML dashboard."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "MATROUTE-AI" in response.text
    assert "Playground de Roteamento" in response.text


def test_web_api_status():
    """Verify GET /api/status returns system status and gemma info."""
    client = TestClient(app)
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "gemma_server" in data
    assert "tau" in data


def test_web_api_chat_routing_l0():
    """Verify POST /api/chat executes selective routing and handles greetings at L0."""
    client = TestClient(app)
    response = client.post("/api/chat", json={"query": "Olá, tudo bem?"})
    assert response.status_code == 200
    data = response.json()
    assert data["level"] == "LEVEL_0_LOCAL"
    assert data["cost_usd"] == 0.0
    assert "MATROUTE-AI" in data["text"]


def test_web_api_chat_plura_protection():
    """Verify POST /api/chat triggers protection signals and CVV 188 for threats."""
    client = TestClient(app)
    response = client.post("/api/chat", json={"query": "Meu supervisor me ameacou e me humilhou"})
    assert response.status_code == 200
    data = response.json()
    assert data["pgr_alert"] is True
    assert "188" in data["text"] or "ouvidoria" in data["text"].lower()


def test_web_api_calculate_math_entropy():
    """Verify POST /api/calculate computes Shannon entropy and ASCII stats."""
    client = TestClient(app)
    response = client.post("/api/calculate", json={"text": "LINGUAGEM COMO CALCULO"})
    assert response.status_code == 200
    data = response.json()
    assert "entropy" in data
    assert data["entropy"]["char_entropy_bits"] > 0
    assert "ascii_density_grid" in data


def test_web_api_feedback_tau_recalibration():
    """Verify POST /api/feedback updates tau."""
    client = TestClient(app)
    response = client.post("/api/feedback", json={
        "request_id": "test_req_web",
        "thumbs_up": True,
        "route": "LEVEL_0_LOCAL"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "new_tau" in data
