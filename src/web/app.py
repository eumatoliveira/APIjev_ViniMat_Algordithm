"""FastAPI Web Application for MATROUTE-AI Testing Environment.
Servidor Web FastAPI para Ambiente Interativo de Teste do MATROUTE-AI.
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os
import time

from src.storage.persistent_db import PersistentDatabase
from src.router.gate import MathematicalGate
from src.router.cache import MultiLevelCache
from src.router.llm_adapter import LLMAdapter
from src.router.model_router import ModelRouter
from src.router.finops_telemetry import FinOpsTelemetry
from src.core_math.jev_engine import FeedbackLearner
from src.core_math.plura_engine import PluraEngine
from src.core_math.entropy_shannon import calculate_character_entropy, calculate_word_entropy, calculate_text_compressibility_estimate
from src.core_math.ascii_tensor import calculate_ascii_stats, text_to_matrix_2d, matrix_to_ascii_grid_string

app = FastAPI(title="MATROUTE-AI Web Playground", version="1.0.0")

# Shared singletons
db = PersistentDatabase()
adapter = LLMAdapter()
gate = MathematicalGate(tau=0.50)
cache = MultiLevelCache(db=db)
router = ModelRouter(db=db, llm_adapter=adapter, gate=gate, cache=cache)
learner = FeedbackLearner()
telemetry = FinOpsTelemetry()


class ChatRequest(BaseModel):
    query: str
    force_gemma: bool = False
    max_tokens: int = 1500


class FeedbackRequest(BaseModel):
    request_id: str
    thumbs_up: bool
    route: str = "LEVEL_0_LOCAL"


class CalculateRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
async def get_system_status():
    gemma_online = adapter.check_local_gemma_available()
    cache_stats = cache.get_stats()
    return {
        "status": "online",
        "gemma_server": {
            "endpoint": "http://127.0.0.1:13305",
            "online": gemma_online,
            "model": "Gemma-4-E2B-it-GGUF"
        },
        "database": {
            "type": "SQLite FTS5",
            "path": db.db_path,
            "connected": True
        },
        "cache": cache_stats,
        "tau": round(learner.tau, 3)
    }


@app.post("/api/chat")
async def handle_chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        return JSONResponse({"error": "Mensagem vazia."}, status_code=400)

    if req.force_gemma:
        start = time.perf_counter()
        res = adapter.generate_local_gemma(prompt=query, max_tokens=req.max_tokens)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "request_id": f"gemma_{int(time.time())}",
            "text": res["text"],
            "level": "LEVEL_1_SMALL_LOCAL",
            "cost_usd": 0.0,
            "latency_ms": round(elapsed_ms, 2),
            "cache_hit": False,
            "double_check_performed": False,
            "gate_score": 0.55,
            "pgr_alert": False,
            "themes": [],
            "signals": {},
            "model_used": res.get("model_used", "local:Gemma-4-E2B-it-GGUF")
        }

    decision = router.process_request(query)
    # Format level string if Enum
    level_str = decision["level"].value if hasattr(decision["level"], "value") else str(decision["level"])
    decision["level"] = level_str
    return decision


@app.post("/api/feedback")
async def handle_feedback(req: FeedbackRequest):
    learner.record_feedback(req.request_id, req.thumbs_up, req.route)
    db.record_feedback(req.request_id, req.thumbs_up, req.route, learner.tau)
    stats = learner.get_learning_stats()
    return {
        "status": "ok",
        "thumbs_up": req.thumbs_up,
        "new_tau": round(stats["current_tau"], 4),
        "positive_ratio": round(stats["positive_ratio"] * 100, 1),
        "total_feedbacks": stats["total_feedbacks"]
    }


@app.get("/api/finops")
async def handle_finops():
    logs = db.get_recent_audit_logs(limit=200)
    dash = telemetry.generate_dashboard_metrics(logs)
    return dash


@app.post("/api/calculate")
async def handle_calculate(req: CalculateRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "Texto vazio para cálculo."}, status_code=400)

    ascii_stats = calculate_ascii_stats(text)
    char_ent = calculate_character_entropy(text)
    word_ent = calculate_word_entropy(text)
    comp = calculate_text_compressibility_estimate(text)
    plura = PluraEngine.analyze(text)
    matrix_2d = text_to_matrix_2d(text, grid_width=16)
    ascii_grid = matrix_to_ascii_grid_string(matrix_2d)

    return {
        "text_length": len(text),
        "ascii_stats": ascii_stats,
        "entropy": {
            "char_entropy_bits": char_ent["entropy_bits"],
            "word_entropy_bits": word_ent["word_entropy_bits"],
            "compressibility_ratio": comp["estimated_compression_ratio"],
            "normalized_entropy": char_ent["normalized_entropy"]
        },
        "plura": {
            "themes": plura["themes"],
            "theme_scores": plura["theme_scores"],
            "signals": plura["signals"],
            "needs_listening": plura["needs_listening"],
            "orientations": plura["orientations"],
            "contacts": plura["contacts"]
        },
        "ascii_density_grid": ascii_grid
    }
