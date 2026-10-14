"""Image to ASCII Matrix and Spatial Convolution Module.
Modulo de Conversao de Imagens em Matriz ASCII e Convolucao Espacial.

Converts image rasters into mathematical luminance matrices and ASCII character grids.
Converte rasters de imagens em matrizes numericas de luminancia e grades de caracteres ASCII.
"""
from typing import Any
import math


def generate_synthetic_image_matrix(width: int = 20, height: int = 10, pattern: str = "gradient") -> list[list[int]]:
    """Generate a deterministic synthetic 2D luminance pixel matrix (0-255).
    
    Gera uma matriz sintetica deterministica de pixels de luminancia.
    """
    matrix: list[list[int]] = []
    for y in range(height):
        row: list[int] = []
        for x in range(width):
            if pattern == "horizontal_gradient":
                val = int((x / max(1, width - 1)) * 255)
            elif pattern == "vertical_gradient":
                val = int((y / max(1, height - 1)) * 255)
            elif pattern == "checkerboard":
                val = 255 if ((x // 2) + (y // 2)) % 2 == 0 else 0
            elif pattern == "circle":
                cx, cy = width / 2.0, height / 2.0
                dist = math.sqrt((x - cx) ** 2 + ((y - cy) * 2) ** 2)
                radius = min(width, height) / 2.5
                val = 255 if dist <= radius else 0
            else:
                val = int(((x + y) / max(1, width + height - 2)) * 255)
            row.append(max(0, min(255, val)))
        matrix.append(row)
    return matrix


def luminance_matrix_to_ascii(matrix: list[list[int]], density_ramp: str = " .:-=+*#%@") -> str:
    """Render a 2D luminance matrix into ASCII text art.
    
    Renderiza uma matriz 2D de luminancia em texto ASCII.
    """
    ramp_len = len(density_ramp)
    lines: list[str] = []
    
    for row in matrix:
        line_chars = []
        for val in row:
            norm = max(0.0, min(1.0, val / 255.0))
            idx = int(norm * (ramp_len - 1))
            line_chars.append(density_ramp[idx])
        lines.append("".join(line_chars))
        
    return "\n".join(lines)


def apply_sobel_edge_filter(matrix: list[list[int]]) -> list[list[float]]:
    """Apply a 3x3 Sobel spatial convolution filter to detect gradients and edges.
    
    Aplica filtro de convolucao espacial 3x3 de Sobel para detectar gradientes e bordas.
    """
    height = len(matrix)
    if height == 0:
        return []
    width = len(matrix[0])
    
    # Standard Sobel kernels
    # Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    output: list[list[float]] = [[0.0 for _ in range(width)] for _ in range(height)]
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx = (
                -1 * matrix[y - 1][x - 1] + 1 * matrix[y - 1][x + 1] +
                -2 * matrix[y][x - 1]     + 2 * matrix[y][x + 1] +
                -1 * matrix[y + 1][x - 1] + 1 * matrix[y + 1][x + 1]
            )
            gy = (
                -1 * matrix[y - 1][x - 1] - 2 * matrix[y - 1][x] - 1 * matrix[y - 1][x + 1] +
                 1 * matrix[y + 1][x - 1] + 2 * matrix[y + 1][x] + 1 * matrix[y + 1][x + 1]
            )
            magnitude = math.sqrt(gx * gx + gy * gy)
            output[y][x] = round(magnitude, 2)
            
    return output


def calculate_spatial_gradients(matrix: list[list[int]]) -> dict[str, float]:
    """Compute gradient statistical metrics over an image matrix."""
    edges = apply_sobel_edge_filter(matrix)
    flat_edges = [val for row in edges for val in row]
    if not flat_edges:
        return {"max_gradient": 0.0, "mean_gradient": 0.0}
        
    return {
        "max_gradient": round(max(flat_edges), 2),
        "mean_gradient": round(sum(flat_edges) / len(flat_edges), 2),
        "edge_pixel_ratio": round(sum(1 for v in flat_edges if v > 50.0) / len(flat_edges), 4),
    }
