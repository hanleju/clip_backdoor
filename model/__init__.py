"""
CLIP Model Module

This module provides CLIP (Contrastive Language-Image Pre-training) models
with various backbone architectures for image encoding.
"""

from .clip import CLIPImageEncoder, CLIP, get_backbone_info, print_backbone_info

__all__ = [
    'CLIPImageEncoder',
    'CLIP',
    'get_backbone_info',
    'print_backbone_info',
]

__version__ = '1.0.0'
