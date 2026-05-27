"""Parsers for XFOIL output files (polar dumps and Cp distributions).

Kept separate from the runner so they can be unit-tested on canned fixtures
without ever invoking the binary.
"""

from __future__ import annotations

import re
from pathlib import Path

from aeroforge.core.exceptions import ParsingError
from aeroforge.core.logging import get_logger
from aeroforge.core.types import FloatArray, OperatingPoint
from aeroforge.solver.xfoil.results import CpDistribution, Polar, PolarPoint

PathLike = str | Path

_log = get_logger(__name__)

# Matches the table separator line under the column headers, e.g.
# "  ------ -------- --------- --------- -------- -------- --------"
_SEPARATOR_RE = re.compile(r"^\s*-{3,}(\s+-{3,})+\s*$")

# Matches "Mach =   0.000     Re =     1.000 e 6     Ncrit =   9.000".
# Re may be in scientific notation with a stray space ("1.000 e 6") or
# the modern compact form ("1.000e6").
_HEADER_RE = re.compile(
    r"Mach\s*=\s*(?P<mach>[-+]?\d+(?:\.\d+)?)"
    r"\s+Re\s*=\s*(?P<re>[-+]?\d+(?:\.\d+)?(?:\s*e\s*[-+]?\d+)?)"
    r"\s+Ncrit\s*=\s*(?P<ncrit>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_header(text: str) -> tuple[float, float, float]:
    """Extract ``(Re, Mach, Ncrit)`` from the polar-file preamble.

    Args:
        text: The full polar-file content (or just the preamble).

    Returns:
        A ``(reynolds, mach, n_crit)`` tuple. Defaults are returned for any
        field that cannot be located, so a partial header still produces a
        usable :class:`OperatingPoint`.
    """
    match = _HEADER_RE.search(text)
    if match is None:
        return 0.0, 0.0, 9.0
    re_str = match.group("re").replace(" ", "")
    try:
        reynolds = float(re_str)
        mach = float(match.group("mach"))
        n_crit = float(match.group("ncrit"))
    except ValueError:
        return 0.0, 0.0, 9.0
    return reynolds, mach, n_crit


def _find_table_start(lines: list[str]) -> int:
    """Return the index of the first data row in ``lines``.

    Args:
        lines: All lines of the polar file.

    Returns:
        The index immediately after the ``"------ --------"`` separator.

    Raises:
        ParsingError: If no table separator is found.
    """
    for i, line in enumerate(lines):
        if _SEPARATOR_RE.match(line):
            return i + 1
    raise ParsingError("Polar file does not contain a column-separator line.")


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
            If no rows converged the returned polar's ``points`` list is empty.

        Raises:
            ParsingError: If the file structure is not recognised (missing
                table separator or a data row with the wrong number of fields).
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = _find_table_start(lines)
        reynolds, mach, n_crit = _parse_header(text)

        points: list[PolarPoint] = []
        for raw in lines[start:]:
            stripped = raw.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) < 7:
                raise ParsingError(f"Polar row has {len(fields)} fields, expected 7: {raw!r}")
            try:
                alpha, cl, cd, cdp, cm, x_top, x_bot = (float(f) for f in fields[:7])
            except ValueError as exc:
                raise ParsingError(f"Cannot parse polar row {raw!r}: {exc}") from exc

            op = OperatingPoint(alpha=alpha, reynolds=reynolds, mach=mach, n_crit=n_crit)
            points.append(
                PolarPoint(
                    operating_point=op,
                    cl=cl,
                    cd=cd,
                    cdp=cdp,
                    cm=cm,
                    x_trans_upper=x_top,
                    x_trans_lower=x_bot,
                    converged=True,
                )
            )

        _log.debug("Parsed %d polar points from %s", len(points), path)
        return Polar(points=points)

    @staticmethod
    def parse_cp(path: PathLike, point: OperatingPoint) -> CpDistribution:
        """Parse an XFOIL ``CPWR`` pressure-coefficient dump.

        Args:
            path: Path to the Cp text file written by XFOIL.
            point: The operating point at which the Cp was sampled.

        Returns:
            A :class:`CpDistribution` with chordwise stations and Cp values.

        Raises:
            ParsingError: If no parseable ``(x, Cp)`` pairs are found, or if a
                non-blank line is malformed.
        """
        import numpy as np  # local import: parsers are otherwise NumPy-free

        path = Path(path)
        xs: list[float] = []
        cps: list[float] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                fields = stripped.split()
                if len(fields) < 2:
                    raise ParsingError(f"Cp row has < 2 fields: {raw!r}")
                try:
                    xs.append(float(fields[0]))
                    cps.append(float(fields[1]))
                except ValueError as exc:
                    raise ParsingError(f"Cannot parse Cp row {raw!r}: {exc}") from exc

        if not xs:
            raise ParsingError(f"No Cp data points found in {path}.")

        x_arr: FloatArray = np.asarray(xs, dtype=np.float64)
        cp_arr: FloatArray = np.asarray(cps, dtype=np.float64)
        _log.debug("Parsed %d Cp points from %s", x_arr.size, path)
        return CpDistribution(operating_point=point, x=x_arr, cp=cp_arr)
