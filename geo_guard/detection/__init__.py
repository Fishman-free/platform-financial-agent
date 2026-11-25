# -*- coding: utf-8 -*-
"""判定层：六通道特征提取 + 三层融合。"""

from .claims import ClaimVerifier
from .comments import CommentAnalyzer
from .features import FeatureExtractor
from .fusion import AnchorCalibrator, FusionEngine
from .patterns import PatternMatcher

__all__ = [
    "ClaimVerifier",
    "CommentAnalyzer",
    "FeatureExtractor",
    "FusionEngine",
    "AnchorCalibrator",
    "PatternMatcher",
]
