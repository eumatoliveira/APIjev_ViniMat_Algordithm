"""Persistent Storage and FTS5 Indexing Module.
Modulo de Armazenamento Persistente e Indexacao FTS5 em SQLite.

Ensures zero RAM explosion by streaming queries and persisting matrices in local SQLite database.
Garante controle de RAM consultando apenas trechos necessarios via SQL e streaming.
"""
from typing import Any, Generator, Optional
import sqlite3
import json
import time
import uuid


class PersistentDatabase:
    """Manages SQLite storage with FTS5 search, matrix blobs, and append-only audit telemetry."""
    
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self._init_db()
        
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
        
    def _init_db(self):
        """Create necessary tables and FTS5 virtual tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Documents table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                language TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                entropy REAL NOT NULL,
                created_at REAL NOT NULL
            );
            """)
            
            # 2. Virtual FTS5 table for fast search
            cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title,
                content,
                content='documents',
                content_rowid='rowid'
            );
            """)
            
            # Triggers to keep documents_fts synchronized with documents table
            cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, title, content) VALUES (new.rowid, new.title, new.content);
            END;
            """)
            cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, content) VALUES('delete', old.rowid, old.title, old.content);
            END;
            """)
            
            # 3. Transition matrices table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS transition_matrices (
                corpus_id TEXT PRIMARY KEY,
                order_n INTEGER NOT NULL,
                matrix_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """)
            
            # 4. Append-only Audit log table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                request_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                intent TEXT NOT NULL,
                score REAL NOT NULL,
                route TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                estimated_cost REAL NOT NULL,
                pgr_alert INTEGER NOT NULL,
                details TEXT
            );
            """)
            
            # 5. Feedback log table (thumbs up/down)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                thumbs_up INTEGER NOT NULL,
                route TEXT NOT NULL,
                updated_tau REAL NOT NULL,
                timestamp REAL NOT NULL
            );
            """)

            # 6. Persistent L2 Cache table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS persistent_cache (
                cache_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """)
            
            conn.commit()

    def set_cache(self, key: str, value: Any):
        """Store item in L2 SQLite persistent cache."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO persistent_cache (cache_key, value_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, json.dumps(value), time.time())
            )
            conn.commit()

    def get_cache(self, key: str) -> Optional[Any]:
        """Retrieve item from L2 SQLite persistent cache."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value_json FROM persistent_cache WHERE cache_key = ?",
                (key,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row["value_json"])
            return None

    def insert_document(
        self,
        title: str,
        language: str,
        category: str,
        content: str,
        entropy: float = 0.0,
        doc_id: Optional[str] = None
    ) -> str:
        """Insert a document and index into FTS5."""
        actual_id = doc_id or str(uuid.uuid4())
        word_count = len(content.split())
        created_at = time.time()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (actual_id,))
            cursor.execute(
                """
                INSERT INTO documents (id, title, language, category, content, word_count, entropy, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (actual_id, title, language, category, content, word_count, entropy, created_at)
            )
            conn.commit()
            
        return actual_id

    def search_fts(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Perform Full-Text Search using FTS5 with sanitized tokens."""
        import re
        # Clean special FTS5 syntax characters like ? " * - ^
        clean_tokens = [w for w in re.findall(r"\w+", query) if len(w) > 1]
        if not clean_tokens:
            return []
            
        # Match any or all tokens with prefix wildcard
        fts_query = " OR ".join(f'"{token}"' for token in clean_tokens)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT d.id, d.title, d.content, d.language, d.category, d.word_count, d.entropy, snippet(documents_fts, 1, '<b>', '</b>', '...', 15) as snippet
                FROM documents_fts f
                JOIN documents d ON f.rowid = d.rowid
                WHERE documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def stream_documents(self, category: Optional[str] = None, chunk_size: int = 20) -> Generator[list[dict[str, Any]], None, None]:
        """Stream documents in small chunks to prevent RAM bloat."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute("SELECT id, title, content, entropy FROM documents WHERE category = ?", (category,))
            else:
                cursor.execute("SELECT id, title, content, entropy FROM documents")
                
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                yield [dict(r) for r in rows]

    def save_transition_matrix(self, corpus_id: str, order_n: int, matrix_data: dict[str, Any]):
        """Persist a Markov transition matrix as JSON blob."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO transition_matrices (corpus_id, order_n, matrix_json, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (corpus_id, order_n, json.dumps(matrix_data), time.time())
            )
            conn.commit()

    def get_transition_matrix(self, corpus_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a stored Markov transition matrix."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT matrix_json FROM transition_matrices WHERE corpus_id = ?", (corpus_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["matrix_json"])
            return None

    def log_audit_event(
        self,
        request_id: str,
        intent: str,
        score: float,
        route: str,
        latency_ms: float,
        estimated_cost: float,
        pgr_alert: bool,
        details: str = ""
    ):
        """Append an audit record to the immutable audit log."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_log (request_id, timestamp, intent, score, route, latency_ms, estimated_cost, pgr_alert, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (request_id, time.time(), intent, score, route, latency_ms, estimated_cost, 1 if pgr_alert else 0, details)
            )
            conn.commit()

    def get_recent_audit_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve recent audit events for dashboard and telemetry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def record_feedback(self, request_id: str, thumbs_up: bool, route: str, updated_tau: float):
        """Save a thumbs up / down feedback event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO feedback_log (request_id, thumbs_up, route, updated_tau, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (request_id, 1 if thumbs_up else 0, route, updated_tau, time.time())
            )
            conn.commit()
