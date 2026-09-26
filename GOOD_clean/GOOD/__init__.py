"""Lightweight GOOD package init for local benchmark use.

The upstream package eagerly imports logging and training utilities at import
time. In this workspace we only need dataset/config access first, so keep the
top-level package minimal and let subpackages import lazily.
"""

from .utils.register import register
from .utils import config_summoner, args_parser

__all__ = ["register", "config_summoner", "args_parser"]
