"""Shannon Entropy and Information Theory Module.
Modulo de Entropia de Shannon e Teoria da Informacao.

Mathematical foundation:
Claude E. Shannon, "A Mathematical Theory of Communication", Bell System Technical Journal, 1948.
Fundamento matematico formal para quantificar a informacao em bits.
"""
from typing import Any
import math
from collections import Counter


def calculate_character_entropy(text: str) -> dict[str, Any]:
    """Calculate the Shannon Entropy of characters in a string.
    
    H(X) = - sum(P(x) * log2(P(x))) for x in alphabet
    
    Calcula a Entropia de Shannon dos caracteres em bits.
    """
    if not text:
        return {
            "entropy_bits": 0.0,
            "alphabet_size": 0,
            "max_possible_entropy": 0.0,
            "normalized_entropy": 0.0,
            "length": 0,
        }
        
    n = len(text)
    counts = Counter(text)
    entropy = 0.0
    
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
        
    alphabet_size = len(counts)
    max_entropy = math.log2(alphabet_size) if alphabet_size > 1 else 0.0
    norm_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0
    
    return {
        "entropy_bits": round(entropy, 4),
        "alphabet_size": alphabet_size,
        "max_possible_entropy": round(max_entropy, 4),
        "normalized_entropy": round(norm_entropy, 4),
        "length": n,
    }


def calculate_word_entropy(text: str) -> dict[str, Any]:
    """Calculate the Shannon Entropy at word level.
    
    Calcula a Entropia de Shannon no nivel de palavras em bits.
    """
    words = [w.lower() for w in text.split() if w.strip()]
    if not words:
        return {
            "word_entropy_bits": 0.0,
            "total_words": 0,
            "unique_words": 0,
            "max_possible_entropy": 0.0,
            "normalized_entropy": 0.0,
        }
        
    n = len(words)
    counts = Counter(words)
    entropy = 0.0
    
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
        
    unique_words = len(counts)
    max_entropy = math.log2(unique_words) if unique_words > 1 else 0.0
    norm_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0
    
    return {
        "word_entropy_bits": round(entropy, 4),
        "total_words": n,
        "unique_words": unique_words,
        "max_possible_entropy": round(max_entropy, 4),
        "normalized_entropy": round(norm_entropy, 4),
    }


def calculate_surprisal_profile(text: str) -> list[dict[str, Any]]:
    """Compute the self-information (surprisal) for each word token: I(w) = -log2(P(w)).
    
    Calcula a surpresa (auto-informacao) em bits para cada token de palavra.
    """
    words = [w.lower() for w in text.split() if w.strip()]
    if not words:
        return []
        
    n = len(words)
    counts = Counter(words)
    profile: list[dict[str, Any]] = []
    
    for idx, w in enumerate(words):
        p = counts[w] / n
        surprisal = -math.log2(p)
        profile.append({
            "index": idx,
            "token": w,
            "frequency": counts[w],
            "probability": round(p, 4),
            "surprisal_bits": round(surprisal, 4),
        })
        
    return profile


def calculate_text_compressibility_estimate(text: str) -> dict[str, float]:
    """Estimate compressibility ratio according to Shannon's Source Coding Theorem.
    
    Standard raw ASCII uses 8 bits per character. Theoretical limit is H(X) bits per character.
    Calcula taxa teorica de compressibilidade segundo o Teorema de Codificacao de Fonte de Shannon.
    """
    char_metrics = calculate_character_entropy(text)
    h = char_metrics["entropy_bits"]
    
    raw_bits_per_char = 8.0
    theoretical_min_bits = max(0.1, h)
    compression_ratio = raw_bits_per_char / theoretical_min_bits
    
    return {
        "raw_bits_per_char": raw_bits_per_char,
        "theoretical_min_bits_per_char": round(theoretical_min_bits, 4),
        "estimated_compression_ratio": round(compression_ratio, 2),
    }
