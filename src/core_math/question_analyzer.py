"""Question Analyzer Module.
Modulo de Analise Determinística de Perguntas.

Decomposes natural language questions into structural predicate tuples without LLMs.
Decompoe perguntas em tuplas estruturais predicativas sem modelos de linguagem.
"""
from typing import Any
import re
from enum import Enum


class ExpectedAnswerType(str, Enum):
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    DATETIME = "DATETIME"
    PROCEDURE = "PROCEDURE"
    REASON = "REASON"
    QUANTITY = "QUANTITY"
    OBJECT_OR_DEFINITION = "OBJECT_OR_DEFINITION"
    BOOLEAN_CONFIRMATION = "BOOLEAN_CONFIRMATION"
    UNKNOWN = "UNKNOWN"


INTERROGATIVE_MAP: dict[str, tuple[str, ExpectedAnswerType]] = {
    # Portuguese
    "quem": ("QUEM", ExpectedAnswerType.PERSON),
    "onde": ("ONDE", ExpectedAnswerType.LOCATION),
    "aonde": ("ONDE", ExpectedAnswerType.LOCATION),
    "quando": ("QUANDO", ExpectedAnswerType.DATETIME),
    "como": ("COMO", ExpectedAnswerType.PROCEDURE),
    "por que": ("POR_QUE", ExpectedAnswerType.REASON),
    "porque": ("POR_QUE", ExpectedAnswerType.REASON),
    "por quê": ("POR_QUE", ExpectedAnswerType.REASON),
    "qual": ("QUAL", ExpectedAnswerType.OBJECT_OR_DEFINITION),
    "quais": ("QUAIS", ExpectedAnswerType.OBJECT_OR_DEFINITION),
    "o que": ("O_QUE", ExpectedAnswerType.OBJECT_OR_DEFINITION),
    "quanto": ("QUANTO", ExpectedAnswerType.QUANTITY),
    "quantos": ("QUANTOS", ExpectedAnswerType.QUANTITY),
    "quantas": ("QUANTAS", ExpectedAnswerType.QUANTITY),
    # English
    "who": ("WHO", ExpectedAnswerType.PERSON),
    "whom": ("WHOM", ExpectedAnswerType.PERSON),
    "where": ("WHERE", ExpectedAnswerType.LOCATION),
    "when": ("WHEN", ExpectedAnswerType.DATETIME),
    "how": ("HOW", ExpectedAnswerType.PROCEDURE),
    "why": ("WHY", ExpectedAnswerType.REASON),
    "what": ("WHAT", ExpectedAnswerType.OBJECT_OR_DEFINITION),
    "which": ("WHICH", ExpectedAnswerType.OBJECT_OR_DEFINITION),
    # Spanish
    "quien": ("QUIEN", ExpectedAnswerType.PERSON),
    "donde": ("DONDE", ExpectedAnswerType.LOCATION),
    "cuando": ("CUANDO", ExpectedAnswerType.DATETIME),
    "como ": ("COMO", ExpectedAnswerType.PROCEDURE),
    "por que ": ("POR_QUE", ExpectedAnswerType.REASON),
}


def analyze_question(text: str) -> dict[str, Any]:
    """Decompose a question into its operator, expected answer type, focus, and complexity.
    
    Decompoe a pergunta em operador interrogativo, tipo de resposta, foco e complexidade.
    """
    cleaned = text.strip()
    is_q = cleaned.endswith("?") or any(cleaned.lower().startswith(k) for k in INTERROGATIVE_MAP)
    
    lower_text = cleaned.lower()
    # Normalize accents for prefix matching
    normalized = (
        lower_text.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )
    
    detected_op = None
    expected_type = ExpectedAnswerType.UNKNOWN
    
    for prefix, (op_name, ans_type) in INTERROGATIVE_MAP.items():
        if re.search(rf"\b{re.escape(prefix)}\b", normalized):
            detected_op = op_name
            expected_type = ans_type
            break
            
    # Calculate structural complexity Q: length + clause markers + punctuation
    words = cleaned.split()
    length_penalty = min(0.5, len(words) / 40.0)
    clauses = len(re.findall(r"[,;—\(\)]", cleaned))
    clause_penalty = min(0.3, clauses * 0.1)
    conjunctions = len(re.findall(r"\b(que|se|e|mas|porque|embora|contudo|and|or|because|which|that)\b", lower_text))
    conj_penalty = min(0.2, conjunctions * 0.05)
    
    complexity_q = round(length_penalty + clause_penalty + conj_penalty, 4)
    
    return {
        "text": cleaned,
        "is_question": is_q,
        "interrogative": detected_op,
        "expected_answer_type": expected_type.value if expected_type else None,
        "word_count": len(words),
        "complexity_score": complexity_q,
    }
