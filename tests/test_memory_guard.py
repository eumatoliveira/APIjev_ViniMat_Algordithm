"""Unit tests for Memory Guard and Backpressure control.
Testes unitários para o Guardião de Memória e Controle de Backpressure.
"""
import pytest
from src.router.memory_guard import MemoryGuard, BackpressureRejectionError


def test_memory_guard_within_concurrency_limit():
    """Verify that requests within MAX_CONCURRENCY are processed smoothly."""
    guard = MemoryGuard(max_concurrency=2, max_queue=2)
    
    with guard.acquire():
        assert guard.active_requests == 1
        with guard.acquire():
            assert guard.active_requests == 2
            
    assert guard.active_requests == 0


def test_memory_guard_rejects_excess_requests():
    """Verify graceful rejection when concurrency and queue limits are exceeded."""
    guard = MemoryGuard(max_concurrency=1, max_queue=0)
    
    with guard.acquire():
        # Concurrency full and queue full -> next acquire must raise BackpressureRejectionError
        with pytest.raises(BackpressureRejectionError):
            with guard.acquire():
                pass


def test_memory_model_simulation():
    """Verify RAM simulation formula: RAM ≈ B + k*R + Q + C."""
    guard = MemoryGuard(base_memory_mb=200.0, avg_request_memory_mb=20.0, cache_resident_mb=100.0, buffer_mb=50.0)
    # 10 concurrent requests -> 200 + 10*20 + 50 + 100 = 550 MB
    estimated_ram = guard.estimate_ram_usage_mb(concurrent_requests=10)
    assert round(estimated_ram, 1) == 550.0
