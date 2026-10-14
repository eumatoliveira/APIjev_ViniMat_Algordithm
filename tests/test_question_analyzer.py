"""Unit tests for deterministic Question Analyzer.
Testes unitários para o Analisador Determinístico de Perguntas.
"""
from src.core_math.question_analyzer import analyze_question, ExpectedAnswerType


def test_analyze_who_question():
    """Verify that a 'Quem' question maps to PERSON answer type and extracts predicate."""
    q = "Quem descobriu o Brasil em 1500?"
    analysis = analyze_question(q)
    
    assert analysis["interrogative"] == "QUEM"
    assert analysis["expected_answer_type"] == ExpectedAnswerType.PERSON.value
    assert analysis["is_question"] is True
    assert 0.0 <= analysis["complexity_score"] <= 1.0


def test_analyze_how_question():
    """Verify that a 'Como' question maps to PROCEDURE answer type."""
    q = "Como faco para redefinir minha senha de acesso?"
    analysis = analyze_question(q)
    
    assert analysis["interrogative"] == "COMO"
    assert analysis["expected_answer_type"] == ExpectedAnswerType.PROCEDURE.value
    assert analysis["is_question"] is True


def test_analyze_where_question_en():
    """Verify English language interrogative detection."""
    q = "Where is the configuration file stored?"
    analysis = analyze_question(q)
    
    assert analysis["interrogative"] == "WHERE"
    assert analysis["expected_answer_type"] == ExpectedAnswerType.LOCATION.value
    assert analysis["is_question"] is True


def test_non_question_statement():
    """Verify that declarative statement is flagged as non-question."""
    stmt = "O servidor reiniciou com sucesso."
    analysis = analyze_question(stmt)
    assert analysis["is_question"] is False
    assert analysis["interrogative"] is None
