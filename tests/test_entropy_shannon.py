"""Unit tests for Shannon Entropy and Information Theory calculations.
Testes unitários para Entropia de Shannon e Teoria da Informação.
"""
import math
from src.core_math.entropy_shannon import (
    calculate_character_entropy,
    calculate_word_entropy,
    calculate_surprisal_profile,
    calculate_text_compressibility_estimate
)


def test_entropy_uniform_distribution():
    """Verify that a uniform distribution of 4 distinct characters achieves exactly log2(4) = 2.0 bits."""
    # "ABCD" repeated uniformly
    sample = "ABCD" * 100
    entropy = calculate_character_entropy(sample)
    assert round(entropy["entropy_bits"], 4) == 2.0
    assert entropy["alphabet_size"] == 4
    assert round(entropy["normalized_entropy"], 4) == 1.0


def test_entropy_zero_variation():
    """Verify that text with a single repeated character has 0 entropy."""
    sample = "AAAAAAA"
    entropy = calculate_character_entropy(sample)
    assert entropy["entropy_bits"] == 0.0
    assert entropy["normalized_entropy"] == 0.0


def test_word_entropy():
    """Verify word-level entropy calculation."""
    sample = "gato cachorro gato passaro"  # gato: 2, cachorro: 1, passaro: 1 -> total 4 words
    # P(gato)=0.5, P(cachorro)=0.25, P(passaro)=0.25
    # H = - (0.5 * -1 + 0.25 * -2 + 0.25 * -2) = 0.5 + 0.5 + 0.5 = 1.5 bits
    result = calculate_word_entropy(sample)
    assert round(result["word_entropy_bits"], 4) == 1.5
    assert result["total_words"] == 4
    assert result["unique_words"] == 3


def test_surprisal_profile():
    """Verify that rare words generate higher surprisal values than common words."""
    sample = "o o o o elefante"
    profile = calculate_surprisal_profile(sample)
    
    # 'o' has high probability -> low surprisal
    # 'elefante' has low probability -> high surprisal
    word_surprisals = {item["token"]: item["surprisal_bits"] for item in profile}
    assert word_surprisals["elefante"] > word_surprisals["o"]


def test_compressibility_estimate():
    """Verify compression ratio estimation based on Shannon's Source Coding Theorem."""
    sample = "abababababababab"
    comp = calculate_text_compressibility_estimate(sample)
    assert 0.0 <= comp["theoretical_min_bits_per_char"] <= 8.0
    assert comp["estimated_compression_ratio"] > 1.0
