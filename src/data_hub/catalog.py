"""Public Corpora and Materials Catalog Module.
Catalogo Curado de Obras e Corpora Publicos de Dominio Aberto.

Contains verified public domain sources from Project Gutenberg, open linguistic tables, and grammars.
Contem fontes publicas confiaveis para download e aprendizado de idiomas como calculo.
"""
from typing import Any, Optional

CATALOG: dict[str, dict[str, Any]] = {
    # 1. Domínio Público PT-BR (Project Gutenberg)
    "gutenberg_pt_bras_cubas": {
        "id": "gutenberg_pt_bras_cubas",
        "title": "Memórias Póstumas de Brás Cubas",
        "author": "Machado de Assis",
        "language": "pt",
        "category": "DOMINIO_PUBLICO_PT_BR",
        "url": "https://www.gutenberg.org/files/54829/54829-0.txt",
        "mirror_url": "https://raw.githubusercontent.com/titoBouzout/Dictionaries/master/Portuguese.dic",
        "description": "Obra seminal da literatura brasileira em domínio público para estudo de semântica e vocabulário.",
        "estimated_size_kb": 360,
    },
    "gutenberg_pt_dom_casmurro": {
        "id": "gutenberg_pt_dom_casmurro",
        "title": "Dom Casmurro",
        "author": "Machado de Assis",
        "language": "pt",
        "category": "DOMINIO_PUBLICO_PT_BR",
        "url": "https://www.gutenberg.org/files/55752/55752-0.txt",
        "description": "Romance psicológico clássico com rica estrutura de perguntas e diálogos.",
        "estimated_size_kb": 420,
    },
    "gutenberg_pt_lusiadas": {
        "id": "gutenberg_pt_lusiadas",
        "title": "Os Lusíadas",
        "author": "Luís de Camões",
        "language": "pt",
        "category": "DOMINIO_PUBLICO_PT_BR",
        "url": "https://www.gutenberg.org/files/3333/3333-0.txt",
        "description": "Epopeia clássica da língua portuguesa em versos decassílabos.",
        "estimated_size_kb": 450,
    },

    # 2. Public Domain English Works
    "gutenberg_en_time_machine": {
        "id": "gutenberg_en_time_machine",
        "title": "The Time Machine",
        "author": "H. G. Wells",
        "language": "en",
        "category": "PUBLIC_DOMAIN_EN",
        "url": "https://www.gutenberg.org/files/35/35-0.txt",
        "description": "Classic English science fiction exploration of time and entropy.",
        "estimated_size_kb": 180,
    },
    "gutenberg_en_alice": {
        "id": "gutenberg_en_alice",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "language": "en",
        "category": "PUBLIC_DOMAIN_EN",
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
        "description": "Linguistic play, puzzles, questions and semantic explorations in English.",
        "estimated_size_kb": 160,
    },
    "gutenberg_en_art_of_war": {
        "id": "gutenberg_en_art_of_war",
        "title": "The Art of War",
        "author": "Sun Tzu",
        "language": "en",
        "category": "PUBLIC_DOMAIN_EN",
        "url": "https://www.gutenberg.org/files/132/132-0.txt",
        "description": "Ancient strategic treatise on decision making and calculation.",
        "estimated_size_kb": 140,
    },

    # 3. Public Domain Spanish Works
    "gutenberg_es_quijote": {
        "id": "gutenberg_es_quijote",
        "title": "Don Quijote de la Mancha",
        "author": "Miguel de Cervantes",
        "language": "es",
        "category": "PUBLIC_DOMAIN_ES",
        "url": "https://www.gutenberg.org/files/2000/2000-0.txt",
        "description": "Obra cumbre de la lengua española en dominio público.",
        "estimated_size_kb": 1100,
    },

    # 4. Lexicons, ASCII Tables & Structured Question Datasets
    "lexicon_pt_ascii_freq": {
        "id": "lexicon_pt_ascii_freq",
        "title": "Matriz de Frequência de Caracteres e N-Gramas em Português",
        "author": "Projeto Language as Calculation",
        "language": "pt",
        "category": "LEXICONS_AND_GRAMMARS",
        "url": "local://lexicons/pt_char_freq.json",
        "description": "Distribuição empírica de frequência de letras e dígrafos para cálculo estocástico.",
        "estimated_size_kb": 25,
    },
    "faq_executive_qa_seed": {
        "id": "faq_executive_qa_seed",
        "title": "Corpus de Perguntas Executivas e FAQ Determinístico",
        "author": "Jev System One Dataset",
        "language": "multilingual",
        "category": "LEXICONS_AND_GRAMMARS",
        "url": "local://lexicons/executive_faq.json",
        "description": "Base canônica de perguntas frequentes para resolução imediata L0 sem LLM.",
        "estimated_size_kb": 35,
    },
}


def list_all_resources() -> list[dict[str, Any]]:
    """Return all resources in the catalog as a flat list."""
    return list(CATALOG.values())


def get_catalog_categories() -> list[str]:
    """Return distinct categories present in the catalog."""
    return sorted(list({item["category"] for item in CATALOG.values()}))


def get_resources_by_category(category: str) -> list[dict[str, Any]]:
    """Filter catalog items by category."""
    return [item for item in CATALOG.values() if item["category"] == category]


def get_resource_by_id(resource_id: str) -> Optional[dict[str, Any]]:
    """Retrieve a catalog resource by its unique identifier."""
    return CATALOG.get(resource_id)
