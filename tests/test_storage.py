import os
import tempfile
from src.storage.persistent_db import PersistentDatabase


def test_database_init_and_document_insert():
    """Verify document storage and FTS5 search."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
        
    try:
        db = PersistentDatabase(db_path=db_path)
        doc_id = db.insert_document(
            title="Memorias Postumas",
            language="pt",
            category="literature",
            content="Ao verme que primeiro roeu as frias carnes do meu cadaver dedico como saudosa lembranca estas memorias postumas.",
            entropy=4.25
        )
        assert doc_id is not None
        
        # Test exact FTS search
        results = db.search_fts("cadaver")
        assert len(results) == 1
        assert results[0]["title"] == "Memorias Postumas"
        
        # Test non-matching FTS
        assert len(db.search_fts("computador")) == 0
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except PermissionError:
            pass


def test_transition_matrix_persistence():
    """Verify storing and retrieving Markov transition matrix JSON."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
        
    try:
        db = PersistentDatabase(db_path=db_path)
        sample_matrix = {"ca": {"t": 1.0}}
        db.save_transition_matrix("corpus_cat", order_n=2, matrix_data=sample_matrix)
        
        retrieved = db.get_transition_matrix("corpus_cat")
        assert retrieved == sample_matrix
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except PermissionError:
            pass


def test_audit_and_feedback_logging():
    """Verify append-only audit logging and feedback storage."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
        
    try:
        db = PersistentDatabase(db_path=db_path)
        db.log_audit_event(
            request_id="req_100",
            intent="PASSWORD_RESET",
            score=0.15,
            route="L0_LOCAL",
            latency_ms=1.2,
            estimated_cost=0.0,
            pgr_alert=False,
            details="Resolved via FAQ"
        )
        
        recent = db.get_recent_audit_logs(limit=5)
        assert len(recent) == 1
        assert recent[0]["request_id"] == "req_100"
        assert recent[0]["pgr_alert"] == 0
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except PermissionError:
            pass
