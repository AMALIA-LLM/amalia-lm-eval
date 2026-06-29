"""Expose stdlib :mod:`inspect` symbols while shipping the AMALIA UI package."""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path

_stdlib_path = Path(sysconfig.get_path("stdlib")) / "inspect.py"
_spec = importlib.util.spec_from_file_location("inspect", _stdlib_path)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to locate the standard library inspect module.")

_stdlib_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stdlib_module)  # type: ignore[arg-type]

for name, value in _stdlib_module.__dict__.items():  # type: ignore[attr-defined]
    if name in {"__name__", "__package__", "__loader__", "__spec__"}:
        continue
    globals()[name] = value

__all__ = list(getattr(_stdlib_module, "__all__", []))  # type: ignore[attr-defined]
__doc__ = getattr(_stdlib_module, "__doc__", None)  # type: ignore[attr-defined]
__file__ = __file__
