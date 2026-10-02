"""Shared figure setup: fonts, light/dark theme colours, output paths and saving."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures" / "out"
CACHE = REPO / "figures" / "cache"
STK_ISL_4C = REPO / "datasets" / "landsat" / "10s_4c_s_landsat_star_inter.csv"

# Foreground colours per theme. The light values are the ones used in the paper figures.
THEMES = {
    "light": dict(fg="#2C2C2A", muted="#888780", rail="#B4B2A9", marker="#444",
                  shade="#F1EFE8", box="white"),
    "dark": dict(fg="#E8E6DF", muted="#B4B2A9", rail="#888780", marker="#CCCCCC",
                 shade="#3A3936", box="#1E1E1C"),
}

USETEX = shutil.which("latex") is not None


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the `--theme`, `--transparent` and `--out` options every figure script takes."""
    ap.add_argument("--theme", choices=THEMES, default="light",
                    help="light (paper) or dark (slides) foreground colours")
    ap.add_argument("--transparent", action="store_true", help="save with a transparent background")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default figures/out)")
    return ap


def setup(rc: dict, theme: str) -> dict:
    """
        setup(rc, theme) -> dict

    Apply Computer Modern fonts and the theme's foreground colours to matplotlib.

      - `rc` — the script's own rcParams (sizes), applied on top
      - `theme` — "light" or "dark"

    Returns the theme colour dict. Uses LaTeX when `latex` is on PATH, otherwise the mathtext
    Computer Modern fallback.
    """
    base = {"text.usetex": USETEX, "font.family": "serif"}
    if USETEX:
        base["font.serif"] = ["Computer Modern Roman"]
    else:
        base.update({"font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True})
    matplotlib.rcParams.update({**base, **rc})
    colours = THEMES[theme]
    if theme == "dark":
        fg = colours["fg"]
        matplotlib.rcParams.update({
            "text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": fg,
            "xtick.color": fg, "ytick.color": fg, "legend.labelcolor": fg,
            "legend.facecolor": colours["box"], "legend.edgecolor": colours["muted"],
            "figure.facecolor": "#1E1E1C", "axes.facecolor": "#1E1E1C",
            "savefig.facecolor": "#1E1E1C",
        })
    return colours


def italic(text: str) -> str:
    """Italic text in either LaTeX or the mathtext fallback."""
    return rf"\textit{{{text}}}" if USETEX else rf"$\mathit{{{text}}}$"


def save(fig, name: str, args, dpi: int = 150, png_dpi: int | None = None) -> None:
    """
        save(fig, name, args, dpi=150, png_dpi=None)

    Save `fig` as `{name}.pdf`, `.svg` and `.png` in `args.out`, with a `_dark` suffix for the
    dark theme.

      - `dpi` — resolution for the PDF and SVG raster elements (dots per inch)
      - `png_dpi` — PNG resolution if it differs from `dpi` (dots per inch)
    """
    args.out.mkdir(parents=True, exist_ok=True)
    stem = name + ("_dark" if args.theme == "dark" else "")
    kw = dict(bbox_inches="tight", transparent=args.transparent)
    for ext in ("pdf", "svg"):
        fig.savefig(args.out / f"{stem}.{ext}", dpi=dpi, **kw)
    fig.savefig(args.out / f"{stem}.png", dpi=png_dpi or dpi, **kw)
    plt.close(fig)
    print(f"saved {args.out / stem}.{{pdf,svg,png}}")
