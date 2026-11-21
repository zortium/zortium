from __future__ import annotations

import logging

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"zortium.{name}")
