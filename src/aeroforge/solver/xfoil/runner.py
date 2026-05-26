"""Process-level management of the XFOIL binary.

Handles binary discovery, scratch-directory management, subprocess invocation
with a hard timeout, and translation of binary failures into typed exceptions.
This is the only place in the library that owns ``subprocess`` calls; every
other layer drives XFOIL through :class:`XfoilSession`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from aeroforge.core.exceptions import XfoilNotFoundError
from aeroforge.core.types import OperatingPoint
from aeroforge.solver.base import AbstractSolver

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.xfoil.results import PolarPoint

PathLike = str | Path


class XfoilRunner(AbstractSolver):
    """Concrete :class:`AbstractSolver` backed by the XFOIL executable.

    Args:
        binary: Path to or PATH-resolvable name of the XFOIL executable.
            Defaults to ``"xfoil"``.
        work_dir: Directory for scratch ``.dat`` / ``.pol`` / ``.cp`` files.
            One is created per run if ``None``.
        timeout_s: Hard timeout for a single XFOIL invocation.

    Raises:
        XfoilNotFoundError: If ``binary`` cannot be resolved to an executable.

    Note:
        Both the cross-platform process plumbing and the analysis logic are
        planned for milestone M2 (see ``ARCHITECTURE.md``). The constructor
        already performs binary discovery so calling code fails fast.
    """

    def __init__(
        self,
        binary: str = "xfoil",
        *,
        work_dir: PathLike | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        """Validate the binary location and store run parameters."""
        resolved = shutil.which(binary) or (binary if Path(binary).is_file() else None)
        if resolved is None:
            raise XfoilNotFoundError(
                f"XFOIL executable {binary!r} not found on PATH. "
                "Install XFOIL and ensure it is reachable, or pass an explicit path."
            )
        self.binary = resolved
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.timeout_s = float(timeout_s)

    def analyze(self, airfoil: Airfoil, point: OperatingPoint) -> PolarPoint:
        """Run XFOIL on ``airfoil`` at the requested ``point``.

        Args:
            airfoil: Geometry to load.
            point: Aerodynamic operating point to evaluate.

        Returns:
            A :class:`PolarPoint` parsed from the XFOIL run.

        Raises:
            ConvergenceError: If XFOIL exits without a converged solution.
            XfoilExecutionError: For process-level failures (crash, timeout).
            NotImplementedError: Implementation planned for milestone M2.
        """
        raise NotImplementedError("XfoilRunner.analyze (planned, M2).")
