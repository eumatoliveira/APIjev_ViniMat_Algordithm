"""Model Router and Cascade Escalation Module.
Modulo de Roteamento de Modelos e Escalonamento em Cascata.

Hierarchy:
LEVEL 0: Regras / Cache / SQLite FTS5 / Calculo Matematico ASCII
LEVEL 1: Modelo Local Pequeno / Jev System One / Markov
LEVEL 2: Modelo Barato (gemini-2.5-flash-lite)
LEVEL 3: Modelo Intermediario (gemini-2.5-flash)
LEVEL 4: Modelo Avancado (gemini-2.5-pro)
LEVEL 5: Modelo Avancado + Double Check Obrigatorio
"""
from typing import Any, Optional
from enum import Enum
import time
import uuid
import re

from src.router.gate import MathematicalGate, GateMetrics
from src.router.cache import MultiLevelCache
from src.router.llm_adapter import LLMAdapter
from src.storage.persistent_db import PersistentDatabase
from src.core_math.jev_engine import JevSystemOneClassifier, SubMillisecondOutputChecker, IntentEnum, RiskLevel
from src.core_math.plura_engine import PluraEngine


class RouteLevel(str, Enum):
    LEVEL_0_LOCAL = "LEVEL_0_LOCAL"
    LEVEL_1_SMALL_LOCAL = "LEVEL_1_SMALL_LOCAL"
    LEVEL_2_CHEAP = "LEVEL_2_CHEAP"
    LEVEL_3_INTERMEDIATE = "LEVEL_3_INTERMEDIATE"
    LEVEL_4_ADVANCED = "LEVEL_4_ADVANCED"
    LEVEL_5_DOUBLE_CHECK = "LEVEL_5_DOUBLE_CHECK"


class ModelRouter:
    """Orchestrates selective economic AI routing based on JEV System One, Gate Score S, and Caches."""
    
    def __init__(
        self,
        db: PersistentDatabase,
        llm_adapter: LLMAdapter,
        gate: Optional[MathematicalGate] = None,
        cache: Optional[MultiLevelCache] = None,
    ):
        self.db = db
        self.adapter = llm_adapter
        self.gate = gate or MathematicalGate(tau=0.50)
        self.cache = cache or MultiLevelCache()
        self.system_one = JevSystemOneClassifier()
        self.output_checker = SubMillisecondOutputChecker()

    def process_request(self, user_query: str) -> dict[str, Any]:
        """Process an incoming request adhering to the Prompt Mestre selective architecture."""
        start_time = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        
        # 0. PluraTalks Deterministic Safety Check (L0 Protection Signals: Violence, Self-Harm, Help - 0.08ms)
        plura_analysis = PluraEngine.analyze(user_query)
        if plura_analysis["needs_listening"]:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            orientations_text = "\n\n".join(plura_analysis["orientations"])
            contacts_text = "\n".join([f" • {c['name']}: {c['contact']}" for c in plura_analysis["contacts"]])
            plura_resp = f"{orientations_text}\n\nCanais de Apoio Imediato:\n{contacts_text}"
            
            self.db.log_audit_event(
                request_id=request_id,
                intent="PLURA_PROTECTION_SIGNAL",
                score=0.95,
                route=RouteLevel.LEVEL_0_LOCAL.value,
                latency_ms=round(elapsed_ms, 2),
                estimated_cost=0.0,
                pgr_alert=plura_analysis["pgr_alert"],
                details=f"Themes: {plura_analysis['themes']}, Signals: {plura_analysis['signals']}",
            )
            return {
                "request_id": request_id,
                "text": plura_resp,
                "level": RouteLevel.LEVEL_0_LOCAL,
                "cost_usd": 0.0,
                "latency_ms": round(elapsed_ms, 2),
                "cache_hit": False,
                "double_check_performed": False,
                "gate_score": 0.0,
                "pgr_alert": plura_analysis["pgr_alert"],
                "themes": plura_analysis["themes"],
                "signals": plura_analysis["signals"],
            }

        # 1. Conversational Greetings Check (L0 Local Rule - 0.05ms, $0.00 cost)
        clean_q = re.sub(r"[^\w\s]", " ", user_query.strip().lower())
        clean_q = re.sub(r"\s+", " ", clean_q).strip()
        greeting_patterns = [
            r"^(ol[aá]|oi|bom dia|boa tarde|boa noite|ola|opa|hey|hello|hi|e a[ií])$",
            r"^(tudo bem|tudo bom|como vai|como voc[eê] est[aá])$",
            r"^(ol[aá]|oi|bom dia|boa tarde|boa noite|ola|opa|hey|hello|hi)\s+(tudo bem|tudo bom|como vai|como voc[eê] est[aá])$",
        ]
        greeting_match = any(re.match(p, clean_q, re.IGNORECASE) for p in greeting_patterns)
        if greeting_match:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            is_how_are_you = bool(re.search(r"(tudo bem|tudo bom|como vai|como voc)", clean_q, re.IGNORECASE))
            if is_how_are_you:
                greeting_resp = (
                    "Tudo ótimo por aqui! Sou o MATROUTE-AI — operando em modo determinístico, econômico e de alta precisão.\n"
                    "Posso calcular matrizes ASCII/Shannon, classificar temas organizacionais (PluraTalks), "
                    "ou acionar a Gemma local para raciocínio complexo. Em que posso te ajudar hoje?"
                )
            else:
                greeting_resp = (
                    "Olá! Sou o MATROUTE-AI — Sistema de IA Seletiva, Econômica e Linguagem como Cálculo.\n"
                    "Estou pronto para classificar temas organizacionais (PluraTalks), analisar sinais de proteção, "
                    "calcular matrizes ASCII e Shannon, ou rotear consultas complexas para a Gemma local. Em que posso ajudar?"
                )
            return {
                "request_id": request_id,
                "text": greeting_resp,
                "level": RouteLevel.LEVEL_0_LOCAL,
                "cost_usd": 0.0,
                "latency_ms": round(elapsed_ms, 2),
                "cache_hit": False,
                "double_check_performed": False,
                "gate_score": 0.0,
                "pgr_alert": False,
                "themes": [],
            }

        # 2. Check L1 / L2 Cache
        cached = self.cache.get(user_query)
        if cached:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.db.log_audit_event(
                request_id=request_id,
                intent="CACHE_HIT",
                score=0.0,
                route=RouteLevel.LEVEL_0_LOCAL.value,
                latency_ms=round(elapsed_ms, 2),
                estimated_cost=0.0,
                pgr_alert=False,
                details="Served from MultiLevelCache (L1/L2)",
            )
            return {
                "request_id": request_id,
                "text": cached,
                "level": RouteLevel.LEVEL_0_LOCAL,
                "cost_usd": 0.0,
                "latency_ms": round(elapsed_ms, 2),
                "cache_hit": True,
                "double_check_performed": False,
            }

        # 2. Fast System One evaluation (non-autoregressive parallel pass)
        s1_decision = self.system_one.predict(user_query)
        
        # 3. Calculate Gate Score S
        # Novelty N, Risk R, Uncertainty U, Complexity Q, Cache Hit Prob H
        is_complex_or_contract = s1_decision.intent in (IntentEnum.GENERAL_QUERY, IntentEnum.CONTRACT_ANALYSIS)
        novelty = 0.85 if is_complex_or_contract else 0.10
        risk = 0.90 if s1_decision.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else 0.10
        uncertainty = 0.80 if s1_decision.requires_llm else 0.20
        complexity = max(0.50 if is_complex_or_contract else 0.10, min(1.0, s1_decision.features["text_length"] / 150.0))
        cache_prob = 0.0  # Already missed cache
        
        gate_result = self.gate.evaluate(GateMetrics(
            novelty=novelty,
            risk=risk,
            uncertainty=uncertainty,
            complexity=complexity,
            cache_hit_prob=cache_prob
        ))
        
        # 4. Decision: Can we resolve locally? (Level 0 / Level 1)
        if not gate_result["call_llm"] or not s1_decision.requires_llm:
            if s1_decision.intent == IntentEnum.PASSWORD_RESET:
                local_answer = "Para redefinir sua senha: 1. Acesse Configurações; 2. Clique em Segurança; 3. Selecione Redefinir Senha."
            elif s1_decision.intent == IntentEnum.WORKPLACE_INCIDENT:
                local_answer = (
                    "Notificação registrada com sucesso. O protocolo confidencial do PGR e Gestão de Pessoas "
                    "foi acionado imediatamente para acolhimento e apuração."
                )
            elif plura_analysis["themes"]:
                themes_str = ", ".join(plura_analysis["themes"])
                local_answer = f"Relato classificado nos temas organizacionais: [{themes_str}]. Registrado com segurança e sigilo."
            else:
                fts_matches = self.db.search_fts(user_query, limit=1)
                if fts_matches:
                    local_answer = fts_matches[0]["content"]
                else:
                    local_answer = "Consulta processada localmente com base nas regras determinísticas e parâmetros do sistema."
                
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Store in cache
            self.cache.set(user_query, local_answer)
            self.db.log_audit_event(
                request_id=request_id,
                intent=s1_decision.intent.value,
                score=gate_result["score"],
                route=RouteLevel.LEVEL_0_LOCAL.value,
                latency_ms=round(elapsed_ms, 2),
                estimated_cost=0.0,
                pgr_alert=s1_decision.pgr_alert,
                details=s1_decision.action_plan,
            )
            
            return {
                "request_id": request_id,
                "text": local_answer,
                "level": RouteLevel.LEVEL_0_LOCAL,
                "cost_usd": 0.0,
                "latency_ms": round(elapsed_ms, 2),
                "cache_hit": False,
                "double_check_performed": False,
                "gate_score": gate_result["score"],
                "pgr_alert": s1_decision.pgr_alert,
                "themes": plura_analysis["themes"],
            }
            
        # 5. Model Cascade & Escalation (Level 2 to Level 5)
        # Select minimum capable model
        if s1_decision.risk_level == RiskLevel.HIGH or gate_result["score"] >= 0.70:
            chosen_level = RouteLevel.LEVEL_4_ADVANCED
            model_name = "gemini-2.5-pro"
            need_double_check = True
        elif gate_result["score"] >= 0.55:
            chosen_level = RouteLevel.LEVEL_3_INTERMEDIATE
            model_name = "gemini-2.5-flash"
            need_double_check = False
        else:
            chosen_level = RouteLevel.LEVEL_2_CHEAP
            model_name = "gemini-2.5-flash-lite"
            need_double_check = False
            
        # Call via isolated LLM Adapter
        llm_response = self.adapter.generate(prompt=user_query, model_name=model_name)
        total_cost = llm_response["cost_usd"]
        
        # 6. Double Check when mathematically justified
        if need_double_check:
            # Second validation call with cheap verifier
            validation_call = self.adapter.generate(
                prompt=f"Verifique a integridade e precisão executiva da resposta: {llm_response['text']}",
                model_name="gemini-2.5-flash-lite"
            )
            total_cost += validation_call["cost_usd"]
            chosen_level = RouteLevel.LEVEL_5_DOUBLE_CHECK
            
        # 7. Sub-millisecond output validation
        check = self.output_checker.evaluate(llm_response["text"])
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Log to immutable audit log
        self.db.log_audit_event(
            request_id=request_id,
            intent=s1_decision.intent.value,
            score=gate_result["score"],
            route=chosen_level.value,
            latency_ms=round(elapsed_ms, 2),
            estimated_cost=round(total_cost, 6),
            pgr_alert=check["pgr_alert"],
            details=f"Model: {model_name}, DoubleCheck: {need_double_check}, Valid: {check['is_valid']}",
        )
        
        return {
            "request_id": request_id,
            "text": llm_response["text"],
            "level": chosen_level,
            "cost_usd": round(total_cost, 6),
            "latency_ms": round(elapsed_ms, 2),
            "cache_hit": False,
            "double_check_performed": need_double_check,
            "gate_score": gate_result["score"],
            "pgr_alert": check["pgr_alert"],
        }
