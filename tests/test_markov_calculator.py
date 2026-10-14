"""Unit tests for Markov Transition Matrices and Stochastic Calculation.
Testes unitários para Matrizes de Transição de Markov e Cálculo Estocástico.
"""
from src.core_math.markov_calculator import MarkovTextCalculator


def test_markov_fit_and_row_stochastic():
    """Verify that every row in the computed transition matrix sums to 1.0."""
    calculator = MarkovTextCalculator(order=1)
    corpus = "abacabadabacaba"
    calculator.fit(corpus)
    
    matrix = calculator.get_transition_matrix()
    assert "a" in matrix
    assert "b" in matrix
    
    # Check that probabilities for 'a' sum to ~1.0
    sum_probs_a = sum(matrix["a"].values())
    assert round(sum_probs_a, 4) == 1.0


def test_markov_deterministic_generation():
    """Verify deterministic generation using argmax transition."""
    calculator = MarkovTextCalculator(order=2)
    # Clear pattern: 'ca' is always followed by 't'
    corpus = "cat cat cat cat"
    calculator.fit(corpus)
    
    generated = calculator.generate(seed="ca", max_length=5, deterministic=True)
    assert generated.startswith("cat")


def test_sequence_log_likelihood():
    """Verify that familiar in-distribution sequences have higher likelihood than out-of-distribution ones."""
    calculator = MarkovTextCalculator(order=1)
    corpus = "banana bandada bacana"
    calculator.fit(corpus)
    
    ll_familiar = calculator.calculate_sequence_log_likelihood("bana")
    ll_unfamiliar = calculator.calculate_sequence_log_likelihood("zzzz")
    
    # Familiar sequence has higher log-likelihood (less negative)
    assert ll_familiar > ll_unfamiliar
