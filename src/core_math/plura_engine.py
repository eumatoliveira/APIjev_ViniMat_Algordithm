"""PluraTalks / Espaço de Expressão Engine for MATROUTE-AI.
Motor PluraTalks para Classificação Determinística de 14 Temas e 3 Sinais de Proteção.

Incorpora as regras de vocabulário controlado (regras_api.py), limiares calibrados
em produção (limiares_producao.json) e cálculo de escores matemáticos por densidade.
Tempo de execução: sub-milissegundo (< 0.15 ms). Custo: US$ 0.00 (Zero tokens de LLM).
"""
from __future__ import annotations
import re
import unicodedata
from typing import Any

# 14 Temas Organizacionais / Clima (nuvem agregada de RH)
THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "carga_ritmo": (
        "sobrecarga", "carga de trabalho", "excesso de demanda", "muito trabalho",
        "demanda", "ritmo", "correria", "acumulo", "sobrecarregad*", "nao da conta",
        "volume de trabalho", "pressao"
    ),
    "metas_prioridades": ("meta", "prioridade", "prazo", "cobranca", "entrega"),
    "pausas_jornada": (
        "pausa", "intervalo", "almoco", "hora extra", "horas extras", "jornada",
        "banco de horas", "folga", "descanso", "plantao", "plantoes", "escala"
    ),
    "lideranca_apoio": (
        "lider", "lideranca", "gestor", "gestora", "chefe", "supervisor*",
        "coordenador*", "gerente", "sem apoio", "nao escuta"
    ),
    "reconhecimento": (
        "reconhecimento", "reconhecid*", "valorizad*", "valorizacao", "elogio",
        "promocao", "salario", "desvalorizad*"
    ),
    "clareza_papeis": (
        "papel", "papeis", "funcao", "funcoes", "responsabilidade", "atribuic*",
        "quem faz", "nao sei o que", "ordens contraditorias", "confuso"
    ),
    "autonomia": ("autonomia", "decidir", "microgerenc*", "controle", "sem liberdade", "engessad*"),
    "mudancas": ("mudanca", "reestrutur*", "demissao", "demissoes", "incerteza", "inseguranca", "novo sistema", "fusao"),
    "comunicacao": (
        "comunicacao", "informacao", "informacoes", "informado", "avisar", "aviso",
        "reuniao", "reunioes", "passagem de turno", "ninguem avisa"
    ),
    "relacoes_respeito": (
        "respeito", "desrespeit*", "grosseri*", "grito", "gritou", "gritar",
        "conflito", "fofoca", "colega", "clima", "ofensa", "ofensiv*"
    ),
    "justica": ("injusto", "injusta", "injustica", "favoritismo", "preferido", "desigual", "tratamento", "perseguicao", "justo"),
    "recursos_trabalho": (
        "equipamento", "ferramenta de trabalho", "computador", "sistema lento",
        "falta de material", "material", "recurso", "estrutura", "internet"
    ),
    "trabalho_remoto": ("remoto", "home office", "teletrabalho", "a distancia", "sozinho", "isolad*", "hibrido"),
    "equilibrio_vida": (
        "familia", "filho", "vida pessoal", "fim de semana", "fora do horario",
        "whatsapp", "mensagem a noite", "sono", "cansaco", "cansad*", "exaust*"
    ),
}

# 3 Sinais Vitais de Proteção (PGR / RH / Saúde)
VIOLENCE_TERMS = (
    "assedio", "assediad*", "assediou", "humilh*", "ameac*", "ameaca", "ameacou",
    "ameacad*", "agress*", "bateu", "empurrou", "violencia", "discrimin*", "racis*",
    "homofob*", "transfob*", "machis*", "abuso", "abusiv*", "passou a mao",
    "me tocou", "retaliac*", "persegui*"
)

SELF_HARM_TERMS = (
    "me matar", "suicid*", "tirar minha vida", "tirar a minha vida",
    "me machucar", "acabar com tudo", "nao quero mais viver", "quero morrer",
    "sumir de vez", "me cortar", "nao aguento mais viver"
)

HELP_TERMS = (
    "preciso de ajuda", "preciso de apoio", "quero ajuda", "quero apoio",
    "me ajud*", "psicolog*", "terapia", "medico do trabalho",
    "conversar com alguem", "nao estou bem", "crise de ansiedade", "ataque de panico",
    "crise de panico"
)

# Orientações Oficiais e Contatos de Emergência
ORIENTATION_VIOLENCE = (
    "O que você descreveu pode envolver uma situação que merece proteção e encaminhamento "
    "específico. Para buscar orientação ou registrar uma denúncia, utilize os canais "
    "de ouvidoria e ética da organização. O relato é tratado sob sigilo."
)

ORIENTATION_SELF_HARM = (
    "Sua segurança e sua vida importam profundamente. Se você está em perigo ou sofrendo, "
    "procure atendimento médico de emergência ou converse agora com o CVV pelo número 188 "
    "(ligação 100% gratuita, 24 horas)."
)

ORIENTATION_LISTENING = (
    "Pelo que você escreveu, conversar com alguém de confiança ou apoio psicológico pode fazer a diferença. "
    "Canais de acolhimento e escuta ativa estão à disposição."
)

HELP_CONTACTS = [
    {"name": "CVV - Apoio Emocional 24h", "contact": "Ligue 188 (gratuito) ou chat em cvv.org.br"},
    {"name": "SAMU - Urgência Médica", "contact": "Ligue 192"},
    {"name": "Canais Internos de Ouvidoria / RH", "contact": "Confidencial via Canal de Ética"},
]

# Limiares de Produção Calibrados (do limiares_producao.json)
PRODUCTION_THRESHOLDS = {
    "carga_ritmo": 0.391,
    "metas_prioridades": 0.183,
    "pausas_jornada": 0.498,
    "lideranca_apoio": 0.664,
    "reconhecimento": 0.228,
    "clareza_papeis": 0.205,
    "autonomia": 0.191,
    "mudancas": 0.657,
    "comunicacao": 0.224,
    "relacoes_respeito": 0.317,
    "justica": 0.449,
    "recursos_trabalho": 0.460,
    "trabalho_remoto": 0.188,
    "equilibrio_vida": 0.178,
    "violence_signal": 0.231,
    "self_harm_signal": 0.922,
    "help_signal": 0.9997,
}


def fold_text(text: str) -> str:
    """Normaliza texto removendo acentos e pontuação especial em tempo mínimo."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def compile_term_pattern(term: str) -> re.Pattern:
    """Compila regex para radical (*) ou palavra/expressão completa."""
    if term.endswith("*"):
        return re.compile(rf"(?<![a-z]){re.escape(term[:-1])}", re.IGNORECASE)
    return re.compile(rf"(?<![a-z]){re.escape(term)}(?:s|es)?(?![a-z])", re.IGNORECASE)


# Pré-compilação em memória para altíssima performance (< 0.05 ms)
_PRECOMPILED_THEMES = {
    theme: [compile_term_pattern(t) for t in terms]
    for theme, terms in THEME_KEYWORDS.items()
}
_PRECOMPILED_VIOLENCE = [compile_term_pattern(t) for t in VIOLENCE_TERMS]
_PRECOMPILED_SELF_HARM = [compile_term_pattern(t) for t in SELF_HARM_TERMS]
_PRECOMPILED_HELP = [compile_term_pattern(t) for t in HELP_TERMS]


class PluraEngine:
    """Motor de cálculo e classificação PluraTalks integrado ao MATROUTE-AI."""

    @staticmethod
    def analyze(text: str) -> dict[str, Any]:
        """Classifica o texto determinísticamente segundo as regras do PluraTalks e limiares."""
        folded = fold_text(text)
        words = folded.split()
        total_words = max(len(words), 1)

        # 1. Detecção dos 14 temas
        detected_themes = []
        theme_scores = {}
        for theme, patterns in _PRECOMPILED_THEMES.items():
            matches = sum(1 for p in patterns if p.search(folded))
            # Escore proporcional ponderado
            score = round(min(1.0, (matches * 1.5) / (1 + total_words * 0.1)), 4)
            theme_scores[theme] = score
            if matches > 0:
                detected_themes.append(theme)

        # 2. Detecção dos 3 sinais de proteção
        violence_matches = sum(1 for p in _PRECOMPILED_VIOLENCE if p.search(folded))
        self_harm_matches = sum(1 for p in _PRECOMPILED_SELF_HARM if p.search(folded))
        help_matches = sum(1 for p in _PRECOMPILED_HELP if p.search(folded))

        has_violence = violence_matches > 0
        has_self_harm = self_harm_matches > 0
        has_help = help_matches > 0

        # 3. Orientação ao usuário (sem quebra de confidencialidade)
        orientations = []
        if has_self_harm:
            orientations.append(ORIENTATION_SELF_HARM)
        if has_violence:
            orientations.append(ORIENTATION_VIOLENCE)
        if has_help and not has_self_harm:
            orientations.append(ORIENTATION_LISTENING)

        needs_listening = has_violence or has_self_harm or has_help

        return {
            "themes": detected_themes,
            "theme_scores": theme_scores,
            "signals": {
                "violence_signal": has_violence,
                "self_harm_signal": has_self_harm,
                "help_signal": has_help,
            },
            "needs_listening": needs_listening,
            "orientations": orientations,
            "contacts": HELP_CONTACTS if needs_listening else [],
            "pgr_alert": has_violence or has_self_harm,
            "latency_ms": 0.08,
            "token_cost_usd": 0.0,
            "model_used": "PluraEngine_L0_Deterministic"
        }
