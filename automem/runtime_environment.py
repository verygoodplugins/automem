from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_STANDARD_LOG_FIELDS = frozenset(logging.makeLogRecord({}).__dict__) | {"asctime", "message"}


class StructuredFormatter(logging.Formatter):
    """Retain human-readable logs and append JSON for application record fields."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        extras = {}
        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_FIELDS:
                continue
            try:
                # Sanitize each value independently so one cyclic or unsupported
                # object cannot hide the other metrics or break application logs.
                encoded = json.dumps(value, default=lambda item: f"<{type(item).__name__}>")
                extras[str(key)] = json.loads(encoded)
            except (TypeError, ValueError, OverflowError, RecursionError):
                extras[str(key)] = "<unserializable>"
        if extras:
            message += " | " + json.dumps(extras, sort_keys=True)
        return message


def configure_logging(*, level: int = logging.INFO) -> Any:
    formatter = StructuredFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    logger = logging.getLogger("automem.api")

    for logger_name in ["werkzeug", "flask.app"]:
        framework_logger = logging.getLogger(logger_name)
        framework_logger.handlers.clear()
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        framework_logger.addHandler(stdout_handler)
        framework_logger.setLevel(level)
        framework_logger.propagate = False

    return logger


def ensure_local_package_importable(*, file_path: str) -> None:
    try:
        import automem  # type: ignore  # noqa: F401
    except Exception:
        root = Path(file_path).resolve().parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
