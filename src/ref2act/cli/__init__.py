from __future__ import annotations

__all__ = ["build_plot_anchor_diagnostics_parser"]


def __getattr__(name: str):
    if name == "build_plot_anchor_diagnostics_parser":
        from .plot_anchor_diagnostics import build_parser

        return build_parser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
