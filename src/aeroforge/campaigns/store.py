"""Result persistence for resumable computational campaigns.

A :class:`ResultStore` accumulates :class:`PolarPoint` records and indexes them
by ``(airfoil_hash, operating_point)`` so an interrupted campaign can resume
without recomputing converged points.

Two concrete backends are planned:

* :class:`ParquetResultStore` -- columnar storage, best for analytics-heavy
  post-processing.
* :class:`SqliteResultStore` -- single-file relational store, best for
  small/medium campaigns and easy ``WHERE``-style filtering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from aeroforge.core.types import OperatingPoint

if TYPE_CHECKING:
    from aeroforge.solver.xfoil.results import PolarPoint

PathLike = str | Path


class ResultStore(ABC):
    """Interface for a persistent campaign result store."""

    @abstractmethod
    def write(self, airfoil_hash: str, result: PolarPoint) -> None:
        """Persist a single :class:`PolarPoint`."""
        raise NotImplementedError

    @abstractmethod
    def has(self, airfoil_hash: str, point: OperatingPoint) -> bool:
        """Return ``True`` if a result for this key is already stored."""
        raise NotImplementedError

    @abstractmethod
    def load_all(self) -> list[PolarPoint]:
        """Return every stored :class:`PolarPoint`."""
        raise NotImplementedError


class ParquetResultStore(ResultStore):
    """Parquet-backed result store (planned, milestone M4).

    Args:
        path: Destination ``.parquet`` file.
    """

    def __init__(self, path: PathLike) -> None:
        """Store the destination path."""
        self.path = Path(path)

    def write(self, airfoil_hash: str, result: PolarPoint) -> None:
        """Append a result row (planned)."""
        raise NotImplementedError("ParquetResultStore.write (planned, M4).")

    def has(self, airfoil_hash: str, point: OperatingPoint) -> bool:
        """Check whether a result exists (planned)."""
        raise NotImplementedError("ParquetResultStore.has (planned, M4).")

    def load_all(self) -> list[PolarPoint]:
        """Load every stored result (planned)."""
        raise NotImplementedError("ParquetResultStore.load_all (planned, M4).")
