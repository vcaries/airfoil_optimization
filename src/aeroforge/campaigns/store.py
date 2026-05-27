"""Result persistence for resumable computational campaigns.

A :class:`ResultStore` accumulates :class:`PolarPoint` records and indexes them
by ``(airfoil_hash, operating_point)`` so an interrupted campaign can resume
without recomputing already-converged points.

Two concrete backends are provided:

* :class:`SqliteResultStore` -- single-file SQLite database. Recommended for
  resumable campaigns: writes are O(log N), ``has()`` is a single SELECT,
  schema is forwards-compatible.
* :class:`ParquetResultStore` -- in-memory buffer that flushes to a single
  ``.parquet`` file on :meth:`flush` or :meth:`close`. Best for "run to
  completion + analyse downstream" workflows; less suited for mid-run
  resume.
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from aeroforge.core.exceptions import AeroforgeError
from aeroforge.core.logging import get_logger
from aeroforge.core.types import OperatingPoint

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.xfoil.results import PolarPoint

PathLike = str | Path
_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Airfoil hashing
# --------------------------------------------------------------------------- #
def hash_airfoil(airfoil: Airfoil) -> str:
    """Compute a stable content-based hash of an airfoil.

    Uses SHA-1 over the raw coordinate bytes, truncated to 16 hex characters
    so the keys remain human-readable in dumps. The hash is deterministic
    across processes and platforms as long as the float layout doesn't
    change (NumPy guarantees that for float64).

    Args:
        airfoil: The :class:`Airfoil` to hash.

    Returns:
        A 16-character hexadecimal string.
    """
    h = hashlib.sha1(usedforsecurity=False)
    h.update(airfoil.x.tobytes())
    h.update(airfoil.y.tobytes())
    return h.hexdigest()[:16]


def _op_key(point: OperatingPoint) -> tuple[float, float, float, float]:
    """Return the four-tuple used as the operating-point part of the store key."""
    return (
        float(point.alpha),
        float(point.reynolds),
        float(point.mach),
        float(point.n_crit),
    )


# --------------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------------- #
class ResultStore(ABC):
    """Interface for a persistent campaign result store."""

    @abstractmethod
    def write(self, airfoil_hash: str, result: PolarPoint) -> None:
        """Persist a single :class:`PolarPoint`. Overwrites any existing row."""
        raise NotImplementedError

    @abstractmethod
    def has(self, airfoil_hash: str, point: OperatingPoint) -> bool:
        """Return ``True`` if a result for this key is already stored."""
        raise NotImplementedError

    @abstractmethod
    def load(self, airfoil_hash: str, point: OperatingPoint) -> PolarPoint | None:
        """Return the stored result for this key, or ``None`` if absent."""
        raise NotImplementedError

    @abstractmethod
    def load_all(self) -> list[PolarPoint]:
        """Return every stored :class:`PolarPoint`."""
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 - intentional no-op default
        """Release any resources held by the store. Default is a no-op."""

    def __enter__(self) -> ResultStore:
        """Support ``with store: ...`` usage."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Release resources on context-manager exit."""
        self.close()


# --------------------------------------------------------------------------- #
# SQLite backend
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    airfoil_hash    TEXT NOT NULL,
    alpha           REAL NOT NULL,
    reynolds        REAL NOT NULL,
    mach            REAL NOT NULL,
    n_crit          REAL NOT NULL,
    cl              REAL NOT NULL,
    cd              REAL NOT NULL,
    cdp             REAL NOT NULL,
    cm              REAL NOT NULL,
    x_trans_upper   REAL NOT NULL,
    x_trans_lower   REAL NOT NULL,
    converged       INTEGER NOT NULL,
    PRIMARY KEY (airfoil_hash, alpha, reynolds, mach, n_crit)
);
"""


class SqliteResultStore(ResultStore):
    """SQLite-backed result store. The recommended choice for resumable runs.

    Idempotent: writing the same ``(airfoil_hash, OperatingPoint)`` twice
    replaces the existing row, so a re-run of a partially completed campaign
    just overwrites whatever was already written.

    Args:
        path: Path to the SQLite database file. Created on demand.
    """

    def __init__(self, path: PathLike) -> None:
        """Open the database and ensure the schema is in place."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    def write(self, airfoil_hash: str, result: PolarPoint) -> None:
        """Insert or replace a single result row."""
        op = result.operating_point
        self._conn.execute(
            "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                airfoil_hash,
                op.alpha,
                op.reynolds,
                op.mach,
                op.n_crit,
                result.cl,
                result.cd,
                result.cdp,
                result.cm,
                result.x_trans_upper,
                result.x_trans_lower,
                int(result.converged),
            ),
        )

    def has(self, airfoil_hash: str, point: OperatingPoint) -> bool:
        """Check whether a row matching the key exists."""
        alpha, re_, mach, n_crit = _op_key(point)
        cur = self._conn.execute(
            "SELECT 1 FROM results WHERE airfoil_hash=? AND alpha=? AND "
            "reynolds=? AND mach=? AND n_crit=? LIMIT 1",
            (airfoil_hash, alpha, re_, mach, n_crit),
        )
        return cur.fetchone() is not None

    def load(self, airfoil_hash: str, point: OperatingPoint) -> PolarPoint | None:
        """Return the matching row, or ``None`` if absent."""
        alpha, re_, mach, n_crit = _op_key(point)
        cur = self._conn.execute(
            "SELECT cl, cd, cdp, cm, x_trans_upper, x_trans_lower, converged "
            "FROM results WHERE airfoil_hash=? AND alpha=? AND reynolds=? "
            "AND mach=? AND n_crit=? LIMIT 1",
            (airfoil_hash, alpha, re_, mach, n_crit),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_polar_point(point, row)

    def load_all(self) -> list[PolarPoint]:
        """Return every row as :class:`PolarPoint` instances."""
        cur = self._conn.execute(
            "SELECT alpha, reynolds, mach, n_crit, cl, cd, cdp, cm, "
            "x_trans_upper, x_trans_lower, converged FROM results"
        )
        out: list[PolarPoint] = []
        for row in cur:
            op = OperatingPoint(alpha=row[0], reynolds=row[1], mach=row[2], n_crit=row[3])
            out.append(_row_to_polar_point(op, row[4:]))
        return out

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


# --------------------------------------------------------------------------- #
# Parquet backend
# --------------------------------------------------------------------------- #
class ParquetResultStore(ResultStore):
    """In-memory buffer that flushes to a single ``.parquet`` file.

    Requires the ``io`` extra (``pip install aeroforge[io]``). Imports are
    deferred so the module stays importable in environments without pandas.

    Args:
        path: Path to the destination ``.parquet`` file.
        load_existing: When ``True`` (default), the file is read into memory
            on construction so previously stored results survive a restart.
    """

    def __init__(self, path: PathLike, *, load_existing: bool = True) -> None:
        """Initialize the buffer and optionally load the existing file."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: dict[tuple[str, tuple[float, float, float, float]], PolarPoint] = {}
        if load_existing and self.path.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------ #
    def write(self, airfoil_hash: str, result: PolarPoint) -> None:
        """Add (or replace) a result in the in-memory buffer."""
        self._buffer[(airfoil_hash, _op_key(result.operating_point))] = result

    def has(self, airfoil_hash: str, point: OperatingPoint) -> bool:
        """Check whether the key is present in the buffer."""
        return (airfoil_hash, _op_key(point)) in self._buffer

    def load(self, airfoil_hash: str, point: OperatingPoint) -> PolarPoint | None:
        """Return the buffered result, or ``None`` if absent."""
        return self._buffer.get((airfoil_hash, _op_key(point)))

    def load_all(self) -> list[PolarPoint]:
        """Return every buffered :class:`PolarPoint`."""
        return list(self._buffer.values())

    def flush(self) -> Path:
        """Write the entire buffer to the parquet file at :attr:`path`."""
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise AeroforgeError(
                "ParquetResultStore.flush requires the 'io' extra (pip install aeroforge[io])."
            ) from exc

        rows = [_polar_point_to_row(h, r) for (h, _), r in self._buffer.items()]
        df = pd.DataFrame(rows)
        df.to_parquet(self.path, engine="pyarrow", index=False)
        _log.info("Wrote %d rows to %s", len(rows), self.path)
        return self.path

    def close(self) -> None:
        """Flush the buffer to disk and clear it."""
        if self._buffer:
            self.flush()

    # ------------------------------------------------------------------ #
    def _load_from_disk(self) -> None:
        """Read the existing parquet file into the in-memory buffer."""
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise AeroforgeError(
                "ParquetResultStore needs the 'io' extra to read existing files."
            ) from exc

        df = pd.read_parquet(self.path, engine="pyarrow")
        for _, row in df.iterrows():
            op = OperatingPoint(
                alpha=float(row["alpha"]),
                reynolds=float(row["reynolds"]),
                mach=float(row["mach"]),
                n_crit=float(row["n_crit"]),
            )
            from aeroforge.solver.xfoil.results import PolarPoint as _PolarPoint

            pt = _PolarPoint(
                operating_point=op,
                cl=float(row["cl"]),
                cd=float(row["cd"]),
                cdp=float(row["cdp"]),
                cm=float(row["cm"]),
                x_trans_upper=float(row["x_trans_upper"]),
                x_trans_lower=float(row["x_trans_lower"]),
                converged=bool(row["converged"]),
            )
            self._buffer[(str(row["airfoil_hash"]), _op_key(op))] = pt


# --------------------------------------------------------------------------- #
# Row helpers (private)
# --------------------------------------------------------------------------- #
def _row_to_polar_point(op: OperatingPoint, row: Iterable[float]) -> PolarPoint:
    """Reconstruct a :class:`PolarPoint` from a database/Parquet row."""
    from aeroforge.solver.xfoil.results import PolarPoint as _PolarPoint

    cl, cd, cdp, cm, x_top, x_bot, converged = tuple(row)
    return _PolarPoint(
        operating_point=op,
        cl=float(cl),
        cd=float(cd),
        cdp=float(cdp),
        cm=float(cm),
        x_trans_upper=float(x_top),
        x_trans_lower=float(x_bot),
        converged=bool(converged),
    )


def _polar_point_to_row(airfoil_hash: str, result: PolarPoint) -> dict[str, float | str | int]:
    """Flatten a :class:`PolarPoint` into a row dict for Parquet storage."""
    op = result.operating_point
    return {
        "airfoil_hash": airfoil_hash,
        "alpha": op.alpha,
        "reynolds": op.reynolds,
        "mach": op.mach,
        "n_crit": op.n_crit,
        "cl": result.cl,
        "cd": result.cd,
        "cdp": result.cdp,
        "cm": result.cm,
        "x_trans_upper": result.x_trans_upper,
        "x_trans_lower": result.x_trans_lower,
        "converged": int(result.converged),
    }
