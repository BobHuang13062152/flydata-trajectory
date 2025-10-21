#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared styling utilities for figures to make visuals more distinct even when
numbers are similar, and to keep labels concise without parentheses.

Provided helpers:
- get_palette(n): deterministic distinct colors for n items
- get_hatches(n): cycling hatch patterns for n items
- bar_jitter(n, mag): small symmetric jitter to avoid perfectly aligned bars
- annotate_bars(ax, bars, fmt): simple value labels on top of bars
"""
from __future__ import annotations

import numpy as np
from typing import List


# A colorblind-friendly palette (okabe-ito) extended with a few soft tones
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # grey
]

EXTRA = [
    "#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
    "#b3de69", "#fccde5", "#bc80bd", "#ccebc5", "#ffed6f",
]


def get_palette(n: int) -> List[str]:
    base = OKABE_ITO + EXTRA
    if n <= len(base):
        return base[:n]
    # repeat if needed
    k = (n + len(base) - 1) // len(base)
    return (base * k)[:n]


HATCHES = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


def get_hatches(n: int) -> List[str]:
    if n <= len(HATCHES):
        return HATCHES[:n]
    k = (n + len(HATCHES) - 1) // len(HATCHES)
    return (HATCHES * k)[:n]


def bar_jitter(n: int, mag: float = 0.06) -> np.ndarray:
    # symmetric jitter: e.g., for n=4 -> [-0.09, -0.03, 0.03, 0.09]
    idx = np.arange(n) - (n - 1) / 2.0
    if n > 1:
        idx = idx / idx.max()
    return idx * mag


def annotate_bars(ax, bars, fmt: str = "{:.0f}") -> None:
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h,
            fmt.format(h if np.isfinite(h) else 0.0),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333",
        )
