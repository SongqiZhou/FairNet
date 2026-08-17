"""Backward-compatible alias for the historical, misspelled ``scr`` package."""

from __future__ import annotations

import sys
import warnings

import fairnet as _fairnet
from fairnet import *  # noqa: F403
from fairnet import config, datasets, models, modules, text_datasets, trainers, utils

warnings.warn(
    "The 'scr' package name is deprecated; import from 'fairnet' instead.",
    DeprecationWarning,
    stacklevel=2,
)

for _module in (config, datasets, models, modules, text_datasets, trainers, utils):
    sys.modules[f"{__name__}.{_module.__name__.rsplit('.', 1)[-1]}"] = _module

__all__ = _fairnet.__all__
__version__ = _fairnet.__version__
