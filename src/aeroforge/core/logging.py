"""Centralized logging utilities.

The library follows the standard practice of attaching a :class:`NullHandler`
to its root logger so that importing aeroforge never emits output on its own.
Applications opt in to logging by calling :func:`configure_logging`.

Example:
    >>> from aeroforge.core.logging import configure_logging, get_logger
    >>> configure_logging(level="DEBUG")
    >>> log = get_logger(__name__)
    >>> log.info("solver started")
"""

from __future__ import annotations

import logging

_ROOT_LOGGER_NAME = "aeroforge"

# Attach a NullHandler once so library imports are silent by default.
logging.getLogger(_ROOT_LOGGER_NAME).addHandler(logging.NullHandler())

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger namespaced under the ``aeroforge`` root.

    Args:
        name: Usually ``__name__`` of the calling module. When ``None`` the
            library root logger is returned.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    if name is None or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str = logging.INFO,
    *,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> None:
    """Configure a stream handler on the aeroforge root logger.

    Idempotent: repeated calls update the level rather than stacking handlers.

    Args:
        level: Logging level as an int (e.g. ``logging.DEBUG``) or name
            (e.g. ``"DEBUG"``).
        fmt: ``logging`` format string for emitted records.
        datefmt: Date format string for the ``asctime`` field.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)

    # Reuse an existing stream handler if configure_logging was already called.
    stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
    if stream_handlers:
        handler = stream_handlers[0]
    else:
        handler = logging.StreamHandler()
        root.addHandler(handler)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
