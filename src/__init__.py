"""
LitBench: Core library for LitBench reward model training.

This package provides data loading utilities and Reddit API integration
for training reward models on LitBench datasets.
"""

from .dataloader import LitBenchDataLoader, SFTDataLoaderCOT, SFTDataLoaderDirect
from .reddit_utils import RedditUtils

__version__ = "1.0.0"
__all__ = [
    "LitBenchDataLoader",
    "SFTDataLoaderCOT", 
    "SFTDataLoaderDirect",
    "RedditUtils"
] 