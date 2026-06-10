from __future__ import annotations

import contextvars
from typing import List

_pipeline_warnings: contextvars.ContextVar[List[str]] = contextvars.ContextVar(
    "pipeline_warnings",
    default=[],
)


def reset_pipeline_warnings() -> None:
    _pipeline_warnings.set([])


def add_pipeline_warning(message: str) -> None:
    message = message.strip()
    if not message:
        return
    warnings = list(_pipeline_warnings.get([]))
    if message not in warnings:
        warnings.append(message)
    _pipeline_warnings.set(warnings)


def get_pipeline_warnings() -> List[str]:
    return list(_pipeline_warnings.get([]))
