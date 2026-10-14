"""Unit tests for ASCII tensor and table transformations.
Testes unitários para transformações de texto em tabela/tensor ASCII.
"""
from src.core_math.ascii_tensor import (
    text_to_ascii_table,
    text_to_matrix_2d,
    text_to_char_cooccurrence_matrix,
    matrix_to_ascii_grid_string,
    calculate_ascii_stats
)


def test_text_to_ascii_table_basic():
    """Verify that simple text is converted to tabular ASCII representation with correct metadata."""
    sample = "Ola Mundo!"
    table = text_to_ascii_table(sample)
    
    assert len(table) == len(sample)
    # Check first char 'O'
    assert table[0]["index"] == 0
    assert table[0]["char"] == "O"
    assert table[0]["ascii_code"] == ord("O")
    assert table[0]["normalized"] == round(ord("O") / 255.0, 4)
    assert table[0]["is_alphanumeric"] is True
    
    # Check space
    assert table[3]["char"] == " "
    assert table[3]["ascii_code"] == 32
    assert table[3]["category"] == "whitespace"
    
    # Check exclamation
    assert table[-1]["char"] == "!"
    assert table[-1]["category"] == "punctuation"


def test_text_to_matrix_2d():
    """Verify that text generates a fixed-dimension or folded 2D matrix."""
    sample = "CALCULO"
    matrix = text_to_matrix_2d(sample, grid_width=4)
    # 7 characters with width 4 -> 2 rows (4 cols and 3 cols padded with 0.0)
    assert len(matrix) == 2
    assert len(matrix[0]) == 4
    assert len(matrix[1]) == 4
    assert matrix[0][0] == ord("C")
    assert matrix[1][3] == 0.0  # Padded element


def test_char_cooccurrence_matrix():
    """Verify deterministic character co-occurrence counting."""
    sample = "AB AB"
    cooc = text_to_char_cooccurrence_matrix(sample, window_size=1)
    
    # In "AB AB":
    # A followed by B (index 0->1, 3->4): occurs 2 times
    assert cooc["A"]["B"] == 2
    # B followed by space (index 1->2): occurs 1 time
    assert cooc["B"][" "] == 1


def test_calculate_ascii_stats():
    """Verify statistical metrics over ASCII values (mean, std, min, max)."""
    sample = "ABC"  # codes 65, 66, 67
    stats = calculate_ascii_stats(sample)
    assert stats["min"] == 65
    assert stats["max"] == 67
    assert stats["mean"] == 66.0
    assert stats["length"] == 3


def test_matrix_to_ascii_grid_string():
    """Verify conversion of numeric matrix to readable visual grid."""
    matrix = [[0.1, 0.9], [0.5, 0.0]]
    grid_str = matrix_to_ascii_grid_string(matrix)
    assert isinstance(grid_str, str)
    assert len(grid_str.splitlines()) == 2
