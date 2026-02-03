from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import matplotlib.pyplot as plt
import pandas as pd


PALETTE = [
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#EECA3B",
    "#B279A2",
    "#FF9DA7",
    "#9D755D",
    "#BAB0AC",
]


def _short(sig: str, max_len: int = 14) -> str:
    return sig if len(sig) <= max_len else sig[:max_len] + "…"


def plot_motif_distribution(
    freq: Dict[str, int],
    total: int,
    title: str,
    out_path: Path,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    if total <= 0 or not freq:
        return
    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    labels = [_short(sig) for sig, _ in items]
    concentrations: List[float] = [cnt / total for _, cnt in items]

    plt.figure(figsize=figsize)
    plt.bar(range(len(labels)), concentrations, color=PALETTE[0])
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Concentration")
    plt.title(title)
    for i, c in enumerate(concentrations):
        plt.text(i, c, f"{c:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_motif_distribution_horizontal(
    freq: Dict[str, int],
    total: int,
    title: str,
    out_path: Path,
    top_n: int = 15,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    if total <= 0 or not freq:
        return
    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    labels = [_short(sig) for sig, _ in items]
    concentrations: List[float] = [cnt / total for _, cnt in items]
    y_pos = list(range(len(labels)))[::-1]
    plt.figure(figsize=figsize)
    plt.barh(y_pos, concentrations, color=PALETTE[1])
    plt.yticks(y_pos, labels)
    plt.xlabel("Concentration")
    plt.title(title)
    for y, c in zip(y_pos, concentrations):
        plt.text(c, y, f" {c:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_seed_boxplot(
    concentration_rows: Iterable[Dict[str, float]],
    out_path: Path,
    title: str,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    df = pd.DataFrame(concentration_rows)
    if df.empty:
        return
    # Expect columns: signature, seed, concentration
    pivot = df.pivot(index="signature", columns="seed", values="concentration")
    plt.figure(figsize=figsize)
    pivot.T.boxplot(rot=45)
    plt.ylabel("Concentration")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_significance_scatter(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    if df.empty:
        return
    plt.figure(figsize=figsize)
    plt.scatter(
        df["rand_mean_concentration"],
        df["orig_concentration"],
        c=df["z_score"],
        cmap="coolwarm",
        alpha=0.8,
    )
    plt.colorbar(label="Z-score")
    plt.xlabel("Random ensemble mean concentration")
    plt.ylabel("Original concentration")
    plt.title(title)
    # Annotate top deviations
    top = df.sort_values("z_score", ascending=False).head(5)
    for _, row in top.iterrows():
        plt.text(
            row["rand_mean_concentration"],
            row["orig_concentration"],
            _short(row["signature"], 10),
            fontsize=8,
            ha="left",
        )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_motif_overlay(
    freq_a: Dict[str, int],
    total_a: int,
    label_a: str,
    freq_b: Dict[str, int],
    total_b: int,
    label_b: str,
    out_path: Path,
    title: str,
    top_n: int = 20,
    figsize: Tuple[int, int] = (11, 6),
) -> None:
    """Overlay two concentration distributions for the same dataset/k.

    Selects top_n signatures by max concentration across A and B and plots grouped bars.
    """
    if total_a <= 0 or total_b <= 0 or (not freq_a and not freq_b):
        return
    conc_a = {s: (c / total_a) for s, c in (freq_a or {}).items()}
    conc_b = {s: (c / total_b) for s, c in (freq_b or {}).items()}
    all_sigs = set(conc_a) | set(conc_b)
    scored = []
    for s in all_sigs:
        scored.append((s, max(conc_a.get(s, 0.0), conc_b.get(s, 0.0))))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in scored[:top_n]]

    labels = [_short(s) for s in top]
    vals_a = [conc_a.get(s, 0.0) for s in top]
    vals_b = [conc_b.get(s, 0.0) for s in top]

    import numpy as np
    x = np.arange(len(labels))
    width = 0.42
    plt.figure(figsize=figsize)
    plt.bar(x - width/2, vals_a, width, label=label_a, color=PALETTE[0])
    plt.bar(x + width/2, vals_b, width, label=label_b, color=PALETTE[2])
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Concentration")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_scatter_xy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None,
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Generic scatter for comparing two estimates (e.g., ensemble vs direct)."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    plt.figure(figsize=figsize)
    if color_col and color_col in df.columns:
        sc = plt.scatter(df[x_col], df[y_col], c=df[color_col], cmap="coolwarm", alpha=0.8)
        plt.colorbar(sc, label=color_col)
    else:
        plt.scatter(df[x_col], df[y_col], alpha=0.8, color=PALETTE[0])
    # Diagonal for reference
    try:
        import numpy as np
        lim = [min(df[[x_col, y_col]].min()), max(df[[x_col, y_col]].max())]
        plt.plot(lim, lim, linestyle="--", color="#888888", linewidth=1)
    except Exception:
        pass
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


__all__ = [
    "plot_motif_distribution",
    "plot_motif_distribution_horizontal",
    "plot_seed_boxplot",
    "plot_significance_scatter",
    "plot_motif_overlay",
    "plot_scatter_xy",
]
