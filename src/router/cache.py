"""Multi-Level Cache Module (L1 RAM LRU & L2 Persistent).
Modulo de Cache Multinivel para Reducao de Chamadas Desnecessarias.

Calculates: cache_hit_rate = cache_hits / total_requests
"""
from typing import Any, Optional
from collections import OrderedDict
import hashlib
import time


class MultiLevelCache:
    """Manages L1 (In-Memory LRU) and L2 (Persistent SQLite Hash) cache levels with eviction."""
    
    def __init__(self, max_l1_items: int = 500, db: Optional[Any] = None):
        self.max_l1_items = max_l1_items
        self.db = db
        self.l1_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.l2_cache: dict[str, dict[str, Any]] = {}
        self.total_requests = 0
        self.l1_hits = 0
        self.l2_hits = 0

    def _normalize_key(self, text: str) -> str:
        """Create a deterministic hash key from normalized input."""
        cleaned = " ".join(text.lower().strip().split())
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        """Lookup cached response across L1 and L2 levels."""
        self.total_requests += 1
        key = self._normalize_key(query)
        
        # Check L1 (RAM)
        if key in self.l1_cache:
            self.l1_hits += 1
            # Move to end for LRU refresh
            self.l1_cache.move_to_end(key)
            return self.l1_cache[key]["value"]
            
        # Check L2 (Persistent SQLite Database)
        if self.db:
            db_cached = self.db.get_cache(key)
            if db_cached is not None:
                self.l2_hits += 1
                # Promote to L1
                self.set(query, db_cached)
                return db_cached

        # Check L2 (In-Process Fallback)
        if key in self.l2_cache:
            self.l2_hits += 1
            val = self.l2_cache[key]["value"]
            # Promote to L1
            self.set(query, val)
            return val
            
        return None

    def set(self, query: str, value: Any, ttl_seconds: Optional[float] = None):
        """Store response in L1 and L2 cache."""
        key = self._normalize_key(query)
        item = {
            "value": value,
            "created_at": time.time(),
            "ttl": ttl_seconds,
        }
        
        # L1 Insertion with LRU eviction
        if key in self.l1_cache:
            self.l1_cache.move_to_end(key)
        self.l1_cache[key] = item
        
        if len(self.l1_cache) > self.max_l1_items:
            # Evict oldest
            self.l1_cache.popitem(last=False)
            
        # Also store in L2 (SQLite and memory)
        self.l2_cache[key] = item
        if self.db:
            self.db.set_cache(key, value)

    def get_stats(self) -> dict[str, Any]:
        """Compute hit rates and operational metrics."""
        total_hits = self.l1_hits + self.l2_hits
        hit_rate = (total_hits / self.total_requests) if self.total_requests > 0 else 0.0
        
        return {
            "total_requests": self.total_requests,
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "total_hits": total_hits,
            "cache_hit_rate": round(hit_rate, 4),
            "l1_resident_items": len(self.l1_cache),
        }
