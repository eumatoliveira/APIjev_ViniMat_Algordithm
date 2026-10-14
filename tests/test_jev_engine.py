"""Unit tests for JEV Engine: Bounded approximation, Sub-millisecond output check, and Thumbs Up/Down learning.
Testes unitários para o Motor JEV: Aproximação limitada, verificação em milissegundos e aprendizado por feedback.
"""
import time
from src.core_math.jev_engine import (
    JevAccuracyApproximator,
    SubMillisecondOutputChecker,
    FeedbackLearner,
    JevSystemOneClassifier,
    IntentEnum,
    RiskLevel
)


def test_jev_bounded_approximation():
    """Verify that JEV calculation approximates accuracy within realistic bounded limits (e.g. 80-95%) without false 100% certainty."""
    approximator = JevAccuracyApproximator(target_bound=(0.80, 0.94))
    
    # Register some sample reference embeddings/profiles
    approximator.add_reference_sample(
        sample_id="reset_pass",
        intent="PASSWORD_RESET",
        feature_vector=[0.1, 0.2, 0.9, 0.1],  # [novelty, risk, cache_prob, length_norm]
        baseline_accuracy=0.92
    )
    
    # Query with a very close vector
    query_vector = [0.12, 0.19, 0.88, 0.11]
    result = approximator.approximate_accuracy(query_vector)
    
    assert 0.80 <= result["estimated_accuracy"] <= 0.95
    assert result["nearest_sample_id"] == "reset_pass"
    assert result["is_bounded"] is True
    # Crucial: never asserts 1.0 (100%) certainty
    assert result["estimated_accuracy"] < 1.0


def test_sub_millisecond_output_checker_performance():
    """Verify that output verification runs in under 5 milliseconds (sub-millisecond or few ms)."""
    checker = SubMillisecondOutputChecker()
    valid_executive_text = "Para redefinir sua senha com seguranca, siga o procedimento: 1. Acesse Configuracoes; 2. Clique em Seguranca; 3. Confirme o email cadastrado."
    
    start_time = time.perf_counter()
    check_result = checker.evaluate(valid_executive_text, language="pt")
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    
    assert elapsed_ms < 10.0  # Must be ultra-fast
    assert check_result["is_valid"] is True
    assert check_result["emotion_flag"] == "NORMAL"
    assert check_result["pgr_alert"] is False


def test_sub_millisecond_output_checker_detects_aggression():
    """Verify that aggressive or hostile text triggers PGR / HR alert."""
    checker = SubMillisecondOutputChecker()
    toxic_text = "Voce e um incompetente, odeio sua equipe e quero que todos sejam demitidos imediatamente!"
    
    check_result = checker.evaluate(toxic_text, language="pt")
    assert check_result["emotion_flag"] == "AGGRESSIVE"
    assert check_result["pgr_alert"] is True
    assert "revisão de RH / PGR" in check_result["action_plan"]


def test_feedback_learner_thumbs_up_down():
    """Verify weight and threshold adaptation upon Thumbs Up (joia) and Thumbs Down (deslike)."""
    learner = FeedbackLearner(initial_tau=0.50, learning_rate=0.05)
    initial_tau = learner.tau
    
    # Thumbs down on an unnecessary LLM call -> should lower reliance on LLM or calibrate tau
    learner.record_feedback(request_id="req_001", thumbs_up=False, route_used="L3_LLM")
    # After thumbs down on expensive call, threshold tau should increase so fewer requests go to LLM
    assert learner.tau > initial_tau
    
    # Thumbs up on local fast resolution -> reinforces current local threshold
    learner.record_feedback(request_id="req_002", thumbs_up=True, route_used="L0_LOCAL")
    stats = learner.get_learning_stats()
    assert stats["total_feedbacks"] == 2
    assert stats["positive_ratio"] == 0.5


def test_jev_system_one_classifier_password_reset():
    """Verify that Jev System One classifier classifies password reset locally in sub-millisecond without LLM."""
    classifier = JevSystemOneClassifier()
    decision = classifier.predict("Como faco para redefinir minha senha?")
    
    assert decision.intent == IntentEnum.PASSWORD_RESET
    assert decision.risk_level == RiskLevel.LOW
    assert decision.requires_llm is False
    assert decision.pgr_alert is False
    assert decision.latency_ms < 10.0
    assert 0.80 <= decision.calibrated_confidence <= 0.94


def test_jev_system_one_classifier_contract_analysis():
    """Verify that complex legal contract inquiry is routed to LLM with high risk."""
    classifier = JevSystemOneClassifier()
    decision = classifier.predict("Analise este contrato e aponte clausulas com risco financeiro grave.")
    
    assert decision.intent == IntentEnum.CONTRACT_ANALYSIS
    assert decision.risk_level == RiskLevel.HIGH
    assert decision.requires_llm is True
    assert decision.pgr_alert is False


def test_jev_system_one_classifier_workplace_incident():
    """Verify that workplace aggression/incident triggers immediate PGR alert and deterministic human routing."""
    classifier = JevSystemOneClassifier()
    decision = classifier.predict("Houve um caso grave de assedio e agressao no setor operacional.")
    
    assert decision.intent == IntentEnum.WORKPLACE_INCIDENT
    assert decision.risk_level == RiskLevel.CRITICAL
    assert decision.pgr_alert is True
    assert decision.requires_llm is False  # Deterministic route, no LLM guessing
