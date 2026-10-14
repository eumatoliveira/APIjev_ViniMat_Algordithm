"""Memory Guard and Backpressure Control Module.
Modulo de Controle de Concorrencia, Backpressure e Modelagem de RAM.

Prevents unbounded memory growth and Out-Of-Memory (OOM) failures:
RAM ≈ B + kR + Q + C
"""
import threading
from contextlib import contextmanager


class BackpressureRejectionError(Exception):
    """Raised when request is rejected due to concurrency/queue exhaustion."""
    pass


class MemoryGuard:
    """Enforces strict concurrency limits, queue depth bounds, and RAM modeling."""
    
    def __init__(
        self,
        max_concurrency: int = 10,
        max_queue: int = 20,
        base_memory_mb: float = 200.0,
        avg_request_memory_mb: float = 20.0,
        cache_resident_mb: float = 100.0,
        buffer_mb: float = 50.0
    ):
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.base_memory_mb = base_memory_mb
        self.avg_request_memory_mb = avg_request_memory_mb
        self.cache_resident_mb = cache_resident_mb
        self.buffer_mb = buffer_mb
        
        self.active_requests = 0
        self.queued_requests = 0
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self):
        """Acquire a concurrency slot or raise BackpressureRejectionError if limits are reached."""
        with self._lock:
            if self.active_requests >= self.max_concurrency:
                if self.queued_requests >= self.max_queue:
                    raise BackpressureRejectionError(
                        f"Backpressure limit reached! Active: {self.active_requests}/{self.max_concurrency}, "
                        f"Queue: {self.queued_requests}/{self.max_queue}. Request rejected to prevent OOM."
                    )
                self.queued_requests += 1
                should_wait = True
            else:
                self.active_requests += 1
                should_wait = False
                
        try:
            if should_wait:
                # In synchronous guard, if queue isn't full but concurrency is, reject with controlled fallback
                with self._lock:
                    self.queued_requests -= 1
                raise BackpressureRejectionError("Concurrency capacity saturated; graceful fallback triggered.")
            yield
        finally:
            with self._lock:
                if not should_wait and self.active_requests > 0:
                    self.active_requests -= 1

    def estimate_ram_usage_mb(self, concurrent_requests: int) -> float:
        """Simulate RAM usage according to formula: RAM ≈ B + k*R + Q + C."""
        k = max(0, concurrent_requests)
        ram = self.base_memory_mb + (k * self.avg_request_memory_mb) + self.buffer_mb + self.cache_resident_mb
        return round(ram, 2)
