"""Geometry operations: discretization, smoothing, and affine transforms."""

from aeroforge.geometry.operations.discretize import (
    cosine_spacing,
    linear_spacing,
    repanel,
)
from aeroforge.geometry.operations.smoothing import smooth_savgol
from aeroforge.geometry.operations.transforms import rotate, scale, translate

__all__ = [
    "cosine_spacing",
    "linear_spacing",
    "repanel",
    "smooth_savgol",
    "translate",
    "scale",
    "rotate",
]
