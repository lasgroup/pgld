import matplotlib.pyplot as plt

DEFAULT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
}

PAPER_STYLE = {
    "font.size": 18,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "legend.title_fontsize": 20,
    "figure.titlesize": 32,
}

# Figure size for paper-ready plots (scale down in LaTeX with \includegraphics[width=...])
PAPER_FIGSIZE = (6, 5)  # (width, height) in inches - fonts sized for this scale


def apply_default_style() -> None:
    """Apply default matplotlib style for regular visualization."""
    plt.rcParams.update(DEFAULT_STYLE)


def apply_paper_style() -> None:
    """Apply paper-ready matplotlib style with larger fonts."""
    plt.rcParams.update(PAPER_STYLE)
