"""ASCII Tensor and Matrix Transformation Module.
Modulo de Transformacao de Texto em Matriz e Tensores ASCII.

Treats language as discrete mathematical positions and byte arrays rather than opaque tokens.
Trata a linguagem como posicoes matematicas discretas e arrays de bytes em vez de tokens opacos.
"""
from typing import Any
import math


def text_to_ascii_table(text: str) -> list[dict[str, Any]]:
    """Convert a raw text string into a structured tabular list of ASCII character descriptors.
    
    Converte uma string em uma tabela estruturada de descritores de caracteres ASCII.
    """
    table: list[dict[str, Any]] = []
    for idx, ch in enumerate(text):
        ascii_val = ord(ch) if ord(ch) <= 255 else ord(ch) % 256
        category = "alphanumeric"
        if ch.isspace():
            category = "whitespace"
        elif not ch.isalnum():
            category = "punctuation"
        elif ch.isdigit():
            category = "digit"
            
        row = {
            "index": idx,
            "char": ch,
            "ascii_code": ascii_val,
            "hex": hex(ascii_val),
            "binary": format(ascii_val, "08b"),
            "category": category,
            "normalized": round(ascii_val / 255.0, 4),
            "is_alphanumeric": ch.isalnum(),
        }
        table.append(row)
    return table


def text_to_matrix_2d(text: str, grid_width: int = 16) -> list[list[float]]:
    """Fold text into a 2D matrix of ASCII values with a fixed width, zero-padded at the end.
    
    Dobra o texto em uma matriz 2D de valores ASCII com largura fixa, preenchida com zeros no final.
    """
    if grid_width <= 0:
        grid_width = 16
    
    values = [float(ord(ch) if ord(ch) <= 255 else ord(ch) % 256) for ch in text]
    if not values:
        return []
        
    num_rows = math.ceil(len(values) / grid_width)
    matrix: list[list[float]] = []
    
    for r in range(num_rows):
        start = r * grid_width
        chunk = values[start:start + grid_width]
        if len(chunk) < grid_width:
            chunk.extend([0.0] * (grid_width - len(chunk)))
        matrix.append(chunk)
        
    return matrix


def text_to_char_cooccurrence_matrix(text: str, window_size: int = 1) -> dict[str, dict[str, int]]:
    """Compute the deterministic co-occurrence matrix between adjacent characters.
    
    Calcula a matriz deterministica de coocorrencia entre caracteres adjacentes.
    """
    cooc: dict[str, dict[str, int]] = {}
    n = len(text)
    
    for i in range(n):
        char_a = text[i]
        if char_a not in cooc:
            cooc[char_a] = {}
        for w in range(1, window_size + 1):
            if i + w < n:
                char_b = text[i + w]
                cooc[char_a][char_b] = cooc[char_a].get(char_b, 0) + 1
                
    return cooc


def calculate_ascii_stats(text: str) -> dict[str, float]:
    """Calculate descriptive statistics over ASCII byte values.
    
    Calcula metricas estatisticas descritivas sobre os valores de bytes ASCII.
    """
    if not text:
        return {"length": 0, "min": 0, "max": 0, "mean": 0.0, "std_dev": 0.0}
        
    codes = [ord(c) if ord(c) <= 255 else ord(c) % 256 for c in text]
    n = len(codes)
    avg = sum(codes) / n
    variance = sum((c - avg) ** 2 for c in codes) / n
    std_dev = math.sqrt(variance)
    
    return {
        "length": n,
        "min": min(codes),
        "max": max(codes),
        "mean": round(avg, 4),
        "std_dev": round(std_dev, 4),
    }


def matrix_to_ascii_grid_string(matrix: list[list[float]]) -> str:
    """Render a numeric 2D matrix into a human-readable ASCII density visualization.
    
    Renderiza uma matriz 2D numerica em uma visualizacao legivel por densidade de caracteres.
    """
    # Density ramp from darkest to brightest
    density_ramp = " .:-=+*#%@"
    ramp_len = len(density_ramp)
    
    lines: list[str] = []
    for row in matrix:
        line_chars = []
        for val in row:
            # Normalize assumed 0.0 - 255.0 or 0.0 - 1.0
            norm = val if val <= 1.0 else val / 255.0
            norm = max(0.0, min(1.0, norm))
            idx = int(norm * (ramp_len - 1))
            line_chars.append(density_ramp[idx])
        lines.append("".join(line_chars))
    return "\n".join(lines)
