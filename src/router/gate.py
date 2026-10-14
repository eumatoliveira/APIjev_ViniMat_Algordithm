"""Mathematical Gate Module.
Modulo do Gate Matematico para Decisao de Invocacao de IA.

Mathematical Formula:
S = wN·N + wR·R + wU·U + wQ·Q - wH·H

Rule:
If S < tau: DO NOT call LLM -> Resolve locally (Cost: $0.00).
If S >= tau: Consider calling LLM via Model Router.
"""
from typing import Any
from dataclasses import dataclass


@dataclass
class GateMetrics:
    novelty: float          # N: Novelty (0.0 to 1.0)
    risk: float             # R: Risk / Criticality (0.0 to 1.0)
    uncertainty: float      # U: Uncertainty / Missing information (0.0 to 1.0)
    complexity: float       # Q: Structural/Computational Complexity (0.0 to 1.0)
    cache_hit_prob: float   # H: Probability of finding known answer in cache (0.0 to 1.0)


class MathematicalGate:
    """Calculates gate score S and decides whether an external LLM invocation is mathematically justified."""
    
    def __init__(
        self,
        weights: dict[str, float] = None,
        tau: float = 0.50
    ):
        self.weights = weights or {
            "wN": 0.25,
            "wR": 0.25,
            "wU": 0.20,
            "wQ": 0.20,
            "wH": 0.20,
        }
        self.tau = tau

    def calculate_score(self, m: GateMetrics) -> float:
        """Compute score S = wN*N + wR*R + wU*U + wQ*Q - wH*H."""
        w = self.weights
        s = (
            w["wN"] * m.novelty +
            w["wR"] * m.risk +
            w["wU"] * m.uncertainty +
            w["wQ"] * m.complexity -
            w["wH"] * m.cache_hit_prob
        )
        return round(s, 4)

    def evaluate(self, metrics: GateMetrics) -> dict[str, Any]:
        """Evaluate gate metrics against threshold tau."""
        score = self.calculate_score(metrics)
        call_llm = score >= self.tau
        
        return {
            "score": score,
            "tau": self.tau,
            "call_llm": call_llm,
            "recommendation": "CALL_LLM" if call_llm else "RESOLVE_LOCALLY",
            "weights": self.weights,
            "metrics": {
                "novelty": metrics.novelty,
                "risk": metrics.risk,
                "uncertainty": metrics.uncertainty,
                "complexity": metrics.complexity,
                "cache_hit_prob": metrics.cache_hit_prob,
            }
        }
