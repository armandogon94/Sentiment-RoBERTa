"""One rcParams block, one save helper, and never a blocking ``plt.show()``.

The source notebook ended five figures with ``plt.show()``. Under a non-interactive backend
that is a no-op, but under any interactive one it blocks until a window is closed, which
makes the figure stage physically unable to run unattended and is a hard CI blocker.

Every plotting function in this repo therefore returns the paths it wrote. Interactive
display exists but is gated behind an explicit ``--show`` flag on the figure script, and
``matplotlib`` is forced to ``Agg`` unless that flag is passed.

Palette: Okabe–Ito, which is colourblind-safe and stays legible at README width.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import matplotlib

# Import-time backend choice: safe by default. scripts/export_figures.py --show swaps it.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

#: Okabe-Ito, ordered so the first two are the ones used for the two classes.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",
]

NEGATIVE_COLOR = OKABE_ITO[1]
POSITIVE_COLOR = OKABE_ITO[0]

RC_PARAMS = {
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
}


def apply_style() -> None:
    """Apply the repo-wide figure style. Called once by the figure entrypoints."""
    # rcParams is keyed by a ~300-member Literal union; a plain dict cannot satisfy it.
    plt.rcParams.update(RC_PARAMS)  # type: ignore[arg-type]


def enable_interactive() -> None:
    """Switch to an interactive backend. Only ever called from an explicit ``--show``."""
    matplotlib.use(
        "MacOSX" if matplotlib.get_backend() == "Agg" else matplotlib.get_backend(), force=True
    )


def caption(ax: plt.Axes, text: str) -> None:
    """Attach a provenance caption naming the model and config that produced the figure."""
    ax.figure.text(0.0, -0.045, text, ha="left", va="top", fontsize=8, color="#444444", wrap=True)


def set_provenance(fig: plt.Figure, payload: dict[str, Any]) -> None:
    """Attach a text-free, machine-readable payload that is embedded in the PNG."""
    cast(Any, fig)._sentiment_roberta_provenance = payload


def save_figure(
    fig: plt.Figure,
    name: str,
    out_dirs: Iterable[Path],
    *,
    show: bool = False,
) -> list[Path]:
    """Save ``fig`` as ``<name>.png`` into every directory in ``out_dirs``.

    Returns every path written, so callers can assert on them. ``show`` is honoured only
    when the caller has already switched to an interactive backend.
    """
    written: list[Path] = []
    provenance = dict(getattr(fig, "_sentiment_roberta_provenance", {}))
    provenance["figure"] = name
    metadata = {
        "Title": name,
        "Description": json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ),
    }
    for d in out_dirs:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.png"
        fig.savefig(path, metadata=metadata)
        written.append(path)
    if show:  # pragma: no cover - interactive only
        plt.show()  # reachable only under `if show` — the explicit --show flag, never in CI
    plt.close(fig)
    return written
