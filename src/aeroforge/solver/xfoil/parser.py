"""Parsers for XFOIL output files (polar dumps and Cp distributions).

Kept separate from the runner so they can be unit-tested on canned fixtures
without ever invoking the binary.
"""

from __future__ import annotations

from pathlib import Path

from aeroforge.core.types import OperatingPoint
from aeroforge.solver.xfoil.results import CpDistribution, Polar

PathLike = str | Path


class XfoilOutputParser:
    """Stateless parser for XFOIL textual output.

    Methods are static for now; the class wrapper exists so that, when XFOIL's
    output format inevitably needs a small fix-up, we can localise state (e.g.
    a parser-version dispatch) inside a single object.
    """

    @staticmethod
    def parse_polar(path: PathLike) -> Polar:
        """Parse an XFOIL ``PACC`` polar dump.

        Args:
            path: Path to the polar text file written by XFOIL.

        Returns:
            A :class:`Polar` with one :class:`PolarPoint` per converged row.

        Raises:
            ParsingError: If the file structure is not recognised.
            NotImplementedError: Implementation planned for milestone M2.
        """
        raise NotImplementedError("XfoilOutputParser.parse_polar (planned, M2).")

    @staticmethod
    def parse_cp(path: PathLike, point: OperatingPoint) -> CpDistribution:
        """Parse an XFOIL ``CPWR`` pressure-coefficient dump.

        Args:
            path: Path to the Cp text file written by XFOIL.
            point: The operating point at which the Cp was sampled.

        Returns:
            A :class:`CpDistribution` with chordwise stations and Cp values.

        Raises:
            ParsingError: If the file structure is not recognised.
            NotImplementedError: Implementation planned for milestone M2.
        """
        raise NotImplementedError("XfoilOutputParser.parse_cp (planned, M2).")
