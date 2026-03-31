from __future__ import annotations

__all__ = ["build_convert_batch_parser", "build_convert_parser"]


def __getattr__(name: str):
    if name == "build_convert_parser":
        from .convert import build_parser

        return build_parser
    if name == "build_convert_batch_parser":
        from .convert_batch import build_parser

        return build_parser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
