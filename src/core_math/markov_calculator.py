"""Markov Text Calculator Module.
Modulo de Calculo de Texto via Cadeias de Markov.

Mathematical modeling of language as a discrete-time Markov chain.
Modelagem matematica da linguagem como cadeia estocastica de transicao de estados.
"""
from typing import Any
import math
from collections import defaultdict, Counter


class MarkovTextCalculator:
    """Computes exact transition matrices and likelihood scores for character n-grams.
    
    Calcula matrizes exatas de transicao e pontuacoes de verossimilhanca.
    """
    
    def __init__(self, order: int = 2):
        self.order = max(1, order)
        # prefix -> Counter of next characters
        self.transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self.transition_matrix: dict[str, dict[str, float]] = {}
        self.vocabulary: set[str] = set()
        self.is_fitted = False
        
    def fit(self, corpus: str):
        """Fit the transition matrix on a given text corpus.
        
        Ajusta a matriz de transicao a partir de um corpus de texto.
        """
        self.transitions.clear()
        self.transition_matrix.clear()
        self.vocabulary.clear()
        
        if len(corpus) <= self.order:
            return
            
        self.vocabulary = set(corpus)
        n = len(corpus)
        
        for i in range(n - self.order):
            prefix = corpus[i:i + self.order]
            next_char = corpus[i + self.order]
            self.transitions[prefix][next_char] += 1
            
        # Normalize counts into row-stochastic probabilities (sum == 1.0)
        for prefix, counts in self.transitions.items():
            total = sum(counts.values())
            self.transition_matrix[prefix] = {
                char: round(cnt / total, 6) for char, cnt in counts.items()
            }
            
        self.is_fitted = True
        
    def get_transition_matrix(self) -> dict[str, dict[str, float]]:
        """Return the normalized transition matrix P_ij."""
        return self.transition_matrix
        
    def transition_probabilities(self, prefix: str) -> dict[str, float]:
        """Get probability distribution P(X_t = c | prefix)."""
        key = prefix[-self.order:] if len(prefix) >= self.order else prefix
        return self.transition_matrix.get(key, {})
        
    def generate(self, seed: str, max_length: int = 50, deterministic: bool = True) -> str:
        """Generate subsequent text based on the transition matrix.
        
        Gera caracteres subsequentes com base na matriz de transicao.
        """
        if not self.is_fitted or not seed:
            return seed
            
        result = list(seed)
        
        for _ in range(max_length):
            current_prefix = "".join(result[-self.order:])
            probs = self.transition_matrix.get(current_prefix)
            if not probs:
                break
                
            if deterministic:
                # Argmax transition: choose character with strictly highest probability
                best_char = max(probs.items(), key=lambda kv: kv[1])[0]
                result.append(best_char)
            else:
                # First available candidate if not deterministic
                result.append(list(probs.keys())[0])
                
        return "".join(result)
        
    def calculate_sequence_log_likelihood(self, sequence: str, epsilon: float = 1e-6) -> float:
        """Compute the cumulative log-likelihood sum(log(P(x_i | x_{i-1}))) for a sequence.
        
        Calcula o logaritmo da verossimilhanca cumulativa de uma sequencia.
        """
        if not self.is_fitted or len(sequence) <= self.order:
            return -100.0
            
        log_likelihood = 0.0
        
        for i in range(len(sequence) - self.order):
            prefix = sequence[i:i + self.order]
            next_char = sequence[i + self.order]
            probs = self.transition_matrix.get(prefix, {})
            p = probs.get(next_char, epsilon)
            log_likelihood += math.log(p)
            
        return round(log_likelihood, 4)
