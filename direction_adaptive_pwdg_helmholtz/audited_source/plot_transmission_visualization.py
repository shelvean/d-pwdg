"""Generate the high-frequency 2D transmission visualizations used in the manuscript.

The quantitative transmission experiments and the visualizations both use
omega=12, so the plotted wave field corresponds to the same frequency reported
in the tables.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from exact import Transmission

OMEGA_VIS = 12.0
N1, N2 = 2.0, 1.0
OUT = Path(__file__).resolve().parent.parent / "generated"
OUT.mkdir(exist_ok=True)

for angle in (69, 29):
    ex = Transmission(OMEGA_VIS, angle, n1=N1, n2=N2)
    x = np.linspace(-1.0, 1.0, 800)
    y = np.linspace(-1.0, 1.0, 800)
    X, Y = np.meshgrid(x, y)
    U = np.real(ex(X, Y))

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.imshow(U, extent=[-1, 1, -1, 1], origin="lower", cmap="coolwarm",
              interpolation="bilinear", aspect="equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.3)
    target = OUT / f"transmission_solution_{angle}deg_2d.png"
    fig.savefig(target, dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(target)
