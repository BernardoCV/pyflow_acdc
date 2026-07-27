# -*- coding: utf-8 -*-
"""Optional PySide6 desktop shell for pyflow_acdc."""

from __future__ import annotations

import importlib.util

# Package present — runtime import verified in launch() only.
HAS_GUI = importlib.util.find_spec("PySide6") is not None


def launch():
    """Start the desktop GUI (requires ``pip install pyflow_acdc[GUI]``)."""
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PySide6 is not installed or failed to load. "
            "Install with: pip install 'pyflow_acdc[GUI]'"
        ) from exc
    from pyflow_acdc.gui.app import run_app

    run_app()
