"""Unit tests for Public Data Hub Catalog and Downloader.
Testes unitários para o Catálogo e Downloader de Dados Públicos.
"""
import os
import tempfile
from src.data_hub.catalog import get_catalog_categories, get_resource_by_id, list_all_resources
from src.data_hub.downloader import PublicDataDownloader
from src.storage.persistent_db import PersistentDatabase


def test_catalog_structure():
    """Verify that catalog contains PT-BR, EN, and ES public domain works."""
    categories = get_catalog_categories()
    assert "DOMINIO_PUBLICO_PT_BR" in categories
    assert "PUBLIC_DOMAIN_EN" in categories
    assert "LEXICONS_AND_GRAMMARS" in categories
    
    resources = list_all_resources()
    assert len(resources) >= 6
    
    bras_cubas = get_resource_by_id("gutenberg_pt_bras_cubas")
    assert bras_cubas is not None
    assert bras_cubas["language"] == "pt"


def test_downloader_seed_corpus_offline_resilience():
    """Verify that downloader can seed persistent memory offline without external network."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
    
    try:
        db = PersistentDatabase(db_path=db_path)
        downloader = PublicDataDownloader(db=db, download_dir=tempfile.gettempdir())
        
        # Ingest built-in offline seed corpus
        result = downloader.ingest_offline_seed_corpus()
        assert result["ingested_count"] >= 3
        
        # Verify that FTS can now query the seeded persistent memory
        search_pt = db.search_fts("senha")
        assert len(search_pt) >= 1
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except PermissionError:
            pass
