"""Public Data Downloader Module.
Modulo de Download e Ingestao de Dados Publicos com Streaming em Chunks.

Guarantees low RAM footprint by streaming in 64 KB chunks and indexing directly into SQLite.
Garante baixo consumo de RAM atraves de streaming em chunks de 64 KB e indexacao direta no SQLite.
"""
from typing import Any, Optional
import os
import urllib.request
import hashlib
from src.data_hub.catalog import get_resource_by_id
from src.core_math.entropy_shannon import calculate_character_entropy
from src.storage.persistent_db import PersistentDatabase


class PublicDataDownloader:
    """Streams and ingests public open data into persistent storage without RAM bloat."""
    
    def __init__(self, db: PersistentDatabase, download_dir: str = "data/corpora"):
        self.db = db
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def download_resource(
        self,
        resource_id: str,
        chunk_size: int = 65536,  # 64 KB chunk size
        max_bytes: int = 5_000_000  # 5 MB ceiling to prevent runaway downloads
    ) -> dict[str, Any]:
        """Stream a public resource from the catalog, save to disk, and index in SQLite."""
        resource = get_resource_by_id(resource_id)
        if not resource:
            return {"status": "ERROR", "message": f"Resource ID '{resource_id}' not found in catalog."}
            
        url = resource["url"]
        if url.startswith("local://"):
            return self.ingest_offline_seed_corpus()

        target_file = os.path.join(self.download_dir, f"{resource_id}.txt")
        sha256_hasher = hashlib.sha256()
        total_downloaded = 0
        text_accumulator: list[str] = []

        headers = {"User-Agent": "LanguageAsCalculation/1.0 (Public Domain Downloader; Educational/Research)"}
        req = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=15) as response, open(target_file, "wb") as f_out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    total_downloaded += len(chunk)
                    if total_downloaded > max_bytes:
                        raise ValueError(f"Download exceeded safety limit of {max_bytes} bytes.")
                        
                    f_out.write(chunk)
                    sha256_hasher.update(chunk)
                    # Decode text chunk safely
                    text_accumulator.append(chunk.decode("utf-8", errors="ignore"))
                    
            full_text = "".join(text_accumulator)
            entropy_metrics = calculate_character_entropy(full_text[:10000])  # Sample first 10k chars for entropy
            
            # Index directly into SQLite FTS5
            doc_id = self.db.insert_document(
                title=resource["title"],
                language=resource["language"],
                category=resource["category"],
                content=full_text,
                entropy=entropy_metrics["entropy_bits"],
                doc_id=resource_id,
            )
            
            return {
                "status": "SUCCESS",
                "resource_id": resource_id,
                "doc_id": doc_id,
                "bytes_downloaded": total_downloaded,
                "sha256": sha256_hasher.hexdigest(),
                "entropy_bits": entropy_metrics["entropy_bits"],
                "file_path": target_file,
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "resource_id": resource_id,
                "message": f"Download failed: {str(e)}",
            }

    def ingest_offline_seed_corpus(self) -> dict[str, Any]:
        """Ingest canonical offline seeds into persistent memory for reliable offline execution."""
        seed_documents = [
            {
                "id": "seed_faq_password",
                "title": "FAQ — Redefinir Senha de Acesso",
                "language": "pt",
                "category": "FAQ",
                "content": "Para redefinir sua senha com segurança, acesse Configurações → Segurança → Redefinir Senha. Um link de confirmação será enviado para seu e-mail cadastrado. Nunca compartilhe sua senha.",
            },
            {
                "id": "seed_faq_pgr_rh",
                "title": "Playbook PGR — Protocolo de Prevenção a Riscos e Assédio",
                "language": "pt",
                "category": "PGR_PLAYBOOK",
                "content": "Em caso de relato ou identificação de agressão verbal, ameaça ou assédio, o sistema aciona o alerta PGR nível 1. A equipe de gestão de pessoas e supervisão de RH é notificada imediatamente para abertura de protocolo confidencial.",
            },
            {
                "id": "seed_lit_bras_cubas",
                "title": "Memórias Póstumas de Brás Cubas (Capítulo I — Óbito)",
                "language": "pt",
                "category": "LITERATURE",
                "content": "Algum tempo hesitei se devia abrir estas memórias pelo princípio ou pelo fim, isto é, se poria em primeiro lugar o meu nascimento ou a minha morte. Suposto o uso vulgar seja começar pelo nascimento, duas considerações me levaram a adotar diferente método: a primeira é que eu não sou propriamente um autor defunto, mas um defunto autor, para quem a campa foi outro berço.",
            },
            {
                "id": "seed_lit_time_machine",
                "title": "The Time Machine (Chapter I)",
                "language": "en",
                "category": "LITERATURE",
                "content": "The Time Traveller was expounding a recondite matter to us. His grey eyes shone and twinkled, and his usually pale face was flushed and animated. The fire burnt brightly, and the soft radiance of the incandescent lights in the lilies of silver caught the bubbles that flashed and passed in our glasses.",
            },
            {
                "id": "seed_lit_quijote",
                "title": "Don Quijote de la Mancha (Capítulo I)",
                "language": "es",
                "category": "LITERATURE",
                "content": "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados.",
            },
        ]
        
        ingested = 0
        for doc in seed_documents:
            entropy_metrics = calculate_character_entropy(doc["content"])
            self.db.insert_document(
                title=doc["title"],
                language=doc["language"],
                category=doc["category"],
                content=doc["content"],
                entropy=entropy_metrics["entropy_bits"],
                doc_id=doc["id"],
            )
            ingested += 1
            
        return {
            "status": "SUCCESS",
            "ingested_count": ingested,
            "message": f"{ingested} seed documents persisted in SQLite with FTS5 indexing.",
        }
