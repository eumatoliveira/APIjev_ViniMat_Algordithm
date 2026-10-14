"""Unit tests for Image to ASCII matrix and spatial convolution.
Testes unitários para conversão de imagem em matriz ASCII e convolução espacial.
"""
from src.vision.image_to_ascii import (
    generate_synthetic_image_matrix,
    luminance_matrix_to_ascii,
    apply_sobel_edge_filter,
    calculate_spatial_gradients
)


def test_synthetic_image_generation():
    """Verify generation of deterministic test image matrices."""
    mat = generate_synthetic_image_matrix(width=10, height=8, pattern="horizontal_gradient")
    assert len(mat) == 8
    assert len(mat[0]) == 10
    # Left edge should be darker than right edge
    assert mat[0][0] < mat[0][-1]


def test_luminance_to_ascii_rendering():
    """Verify luminance matrix is transformed into multi-line ASCII art."""
    mat = [[0, 50, 100], [150, 200, 255]]
    ascii_art = luminance_matrix_to_ascii(mat)
    lines = ascii_art.splitlines()
    assert len(lines) == 2
    assert len(lines[0]) == 3
    # Check that minimum luminance produces blank/dot and max produces dense char
    assert lines[0][0] in " ."
    assert lines[1][2] in "%@"


def test_sobel_convolution():
    """Verify that a sharp step edge produces a positive gradient spike in Sobel convolution."""
    # Create step edge in 5x5 matrix
    mat = [
        [0, 0, 255, 255, 255],
        [0, 0, 255, 255, 255],
        [0, 0, 255, 255, 255],
        [0, 0, 255, 255, 255],
        [0, 0, 255, 255, 255],
    ]
    edges = apply_sobel_edge_filter(mat)
    assert len(edges) == 5
    assert len(edges[0]) == 5
    # The vertical edge at column index 1/2 must have high magnitude
    assert edges[2][1] > 0 or edges[2][2] > 0
    # Homogeneous area at column 0 or 4 should have low gradient
    assert edges[2][4] == 0.0


def test_calculate_spatial_gradients():
    """Verify gradient statistics over an image matrix."""
    mat = generate_synthetic_image_matrix(width=6, height=6, pattern="checkerboard")
    stats = calculate_spatial_gradients(mat)
    assert stats["max_gradient"] > 0
    assert 0.0 <= stats["mean_gradient"] <= 255.0 * 4
