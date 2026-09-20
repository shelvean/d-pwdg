from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def elegant_diverging_cmap(name: str = "elegant_rdbu_strong"):
    """High-contrast red-white-blue map with a narrow neutral band.

    Saturated enough for manuscript fields while avoiding nearly-black tails.
    """
    anchors = [
        (0.00, "#155A9C"),
        (0.20, "#4D9BC3"),
        (0.42, "#C3E1EC"),
        (0.485, "#F7F7F7"),
        (0.515, "#F7F7F7"),
        (0.58, "#F5C5B0"),
        (0.80, "#E56B4A"),
        (1.00, "#A51D34"),
    ]
    return colors.LinearSegmentedColormap.from_list(name, anchors, N=256)


ELEGANT_CMAP = elegant_diverging_cmap()
LINE_BLUE = "#276FAE"
LINE_RED = "#D1543F"
LINE_GOLD = "#D9A441"
MESH_GRAY = "#4A4A4A"
LIGHT_GRID = 0.18
