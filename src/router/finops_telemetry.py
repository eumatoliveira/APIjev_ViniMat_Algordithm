"""FinOps and Operational Telemetry Module.
Modulo de Telemetria FinOps e Otimizacao Matematica de Custos.

Cost objective function:
J = alpha*C + beta*E + gamma*L + delta*R

Compares Naive Baseline (100% LLM) vs Optimized Architecture (Selective Gate + Cache + Local).
"""
from typing import Any
import math


class FinOpsTelemetry:
    """Computes FinOps savings, latency percentiles, and optimization index J."""
    
    def __init__(
        self,
        alpha: float = 0.40,  # Cost weight
        beta: float = 0.25,   # Error weight
        gamma: float = 0.20,  # Latency weight
        delta: float = 0.15,  # Risk weight
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Naive baseline assumes average 2000 tokens @ $0.002 per 1k tokens = $0.004 per call
        self.naive_cost_per_call_usd = 0.004
        self.naive_latency_ms = 1800.0  # ~1.8 seconds for cloud LLM roundtrip

    def calculate_j_cost(self, cost: float, error_rate: float, latency_sec: float, risk: float) -> float:
        """Compute J = alpha*C + beta*E + gamma*L + delta*R."""
        j = (
            self.alpha * cost +
            self.beta * error_rate +
            self.gamma * latency_sec +
            self.delta * risk
        )
        return round(j, 4)

    def generate_dashboard_metrics(self, audit_logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate audit events into executive FinOps dashboard report."""
        total = len(audit_logs)
        if total == 0:
            return {
                "total_requests": 0,
                "local_resolved_count": 0,
                "local_resolved_pct": 0.0,
                "cache_hits_count": 0,
                "cache_hit_pct": 0.0,
                "llm_calls_count": 0,
                "llm_call_pct": 0.0,
                "optimized_cost_usd": 0.0,
                "naive_cost_usd": 0.0,
                "savings_usd": 0.0,
                "savings_pct": 0.0,
                "mean_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "pgr_alerts_count": 0,
                "j_optimization_index": 0.0,
            }

        local_count = 0
        cache_count = 0
        llm_count = 0
        pgr_alerts = 0
        total_optimized_cost = 0.0
        latencies: list[float] = []

        for log in audit_logs:
            route = log.get("route", "")
            cost = float(log.get("estimated_cost", 0.0))
            lat = float(log.get("latency_ms", 0.0))
            alert = int(log.get("pgr_alert", 0))
            
            total_optimized_cost += cost
            latencies.append(lat)
            if alert:
                pgr_alerts += 1
                
            if "LOCAL" in route:
                local_count += 1
                if log.get("intent") == "CACHE_HIT":
                    cache_count += 1
            else:
                llm_count += 1

        naive_total_cost = total * self.naive_cost_per_call_usd
        savings_usd = max(0.0, naive_total_cost - total_optimized_cost)
        savings_pct = (savings_usd / naive_total_cost * 100.0) if naive_total_cost > 0 else 0.0

        latencies.sort()
        mean_lat = sum(latencies) / total
        p95_index = min(total - 1, int(math.ceil(0.95 * total)) - 1)
        p95_lat = latencies[p95_index]

        # Calculate J optimization index
        avg_cost = total_optimized_cost / total
        j_index = self.calculate_j_cost(
            cost=avg_cost,
            error_rate=0.01,  # 1% residual error
            latency_sec=mean_lat / 1000.0,
            risk=0.05
        )

        return {
            "total_requests": total,
            "local_resolved_count": local_count,
            "local_resolved_pct": round((local_count / total) * 100.0, 1),
            "cache_hits_count": cache_count,
            "cache_hit_pct": round((cache_count / total) * 100.0, 1),
            "llm_calls_count": llm_count,
            "llm_call_pct": round((llm_count / total) * 100.0, 1),
            "optimized_cost_usd": round(total_optimized_cost, 4),
            "naive_cost_usd": round(naive_total_cost, 4),
            "savings_usd": round(savings_usd, 4),
            "savings_pct": round(savings_pct, 1),
            "mean_latency_ms": round(mean_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "pgr_alerts_count": pgr_alerts,
            "j_optimization_index": j_index,
        }
