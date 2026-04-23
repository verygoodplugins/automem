from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def configure_logging(*, level: int = logging.INFO) -> Any:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger("automem.api")

    for logger_name in ["werkzeug", "flask.app"]:
        framework_logger = logging.getLogger(logger_name)
        framework_logger.handlers.clear()
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        framework_logger.addHandler(stdout_handler)
        framework_logger.setLevel(level)

    return logger


def ensure_local_package_importable(*, file_path: str) -> None:
    try:
        import automem  # type: ignore  # noqa: F401
    except Exception:
        root = Path(file_path).resolve().parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
