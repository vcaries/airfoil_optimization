"""Process-level management of the XFOIL binary.

Handles binary discovery, scratch-directory management, subprocess invocation
with a hard timeout, and translation of binary failures into typed exceptions.
This is the only place in the library that owns ``subprocess`` calls; every
other layer drives XFOIL through :class:`XfoilSession`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from aeroforge.core.exceptions import (
    ConvergenceError,
    XfoilExecutionError,
    XfoilNotFoundError,
)
from aeroforge.core.logging import get_logger
from aeroforge.core.types import OperatingPoint
from aeroforge.solver.base import AbstractSolver
from aeroforge.solver.xfoil.parser import XfoilOutputParser
from aeroforge.solver.xfoil.session import XfoilSession

if TYPE_CHECKING:
    from aeroforge.geometry.airfoil import Airfoil
    from aeroforge.solver.xfoil.results import CpDistribution, PolarPoint, WallProfile

PathLike = str | Path
_log = get_logger(__name__)

# Tolerance (degrees) for matching a requested alpha against an XFOIL row.
_ALPHA_MATCH_TOL = 1.0e-3


class XfoilRunner(AbstractSolver):
    """Concrete :class:`AbstractSolver` backed by the XFOIL executable.

    The runner owns the subprocess lifecycle and translates binary failures
    into typed aeroforge exceptions. Per-call solver parameters (iteration
    cap, transition factor, auto-repanel) live as mutable attributes on the
    runner so convergence strategies can adapt them without rebuilding the
    object.

    Args:
        binary: Path to or PATH-resolvable name of the XFOIL executable.
        work_dir: Directory for scratch ``.dat`` / ``.pol`` files. When ``None``
            a per-call temporary directory is created and removed automatically.
        timeout_s: Hard timeout for a single XFOIL invocation, in seconds.
        max_iter: Default viscous iteration cap.
        n_crit: Transition amplification factor.
        repanel: Whether to call XFOIL's ``PANE`` after loading the airfoil.

    Raises:
        XfoilNotFoundError: If ``binary`` cannot be resolved to an executable.
    """

    def __init__(
        self,
        binary: str = "xfoil",
        *,
        work_dir: PathLike | None = None,
        timeout_s: float = 60.0,
        max_iter: int = 200,
        n_crit: float = 9.0,
        repanel: bool = True,
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
        self.max_iter = int(max_iter)
        self.n_crit = float(n_crit)
        self.repanel = bool(repanel)
        # Diagnostic attributes — populated by every ``_run_in`` so callers
        # can inspect the actual transcript + XFOIL stdout/stderr without
        # re-running. Useful for debugging silent dump failures (e.g. CPWR
        # not producing its output file on some XFOIL builds).
        self.last_transcript: str = ""
        self.last_stdout: bytes = b""
        self.last_stderr: bytes = b""
        # Snapshot of the scratch workdir contents right after the XFOIL
        # process exits. Lets the caller see exactly which output files
        # the binary materialised — essential when a dump is missing.
        self.last_workdir_files: list[str] = []
        # Snapshot of the actual file contents (bytes). The workdir is a
        # tempfile.TemporaryDirectory and is wiped as soon as ``_run_in``
        # returns, so without this snapshot the diagnostic layer has no
        # way to look at what XFOIL really wrote.
        self.last_files: dict[str, bytes] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(self, airfoil: Airfoil, point: OperatingPoint) -> PolarPoint:
        """Run XFOIL on ``airfoil`` at the requested ``point``.

        Args:
            airfoil: Geometry to load.
            point: Aerodynamic operating point to evaluate.

        Returns:
            A :class:`PolarPoint` parsed from the XFOIL run.

        Raises:
            ConvergenceError: If XFOIL produced no row matching ``point.alpha``.
            XfoilExecutionError: For process-level failures (timeout, non-zero
                exit, missing output file).
            XfoilNotFoundError: If the binary disappeared between construction
                and the call.
        """
        polar, _, _ = self._run_with(airfoil, point, want_cp=False, want_bl=False)
        return polar

    def analyze_with_dumps(
        self, airfoil: Airfoil, point: OperatingPoint
    ) -> tuple[PolarPoint, CpDistribution | None, WallProfile | None]:
        """Like :meth:`analyze`, but also asks XFOIL to dump Cp and BL data.

        The integrated polar (CL, CD, CM, transition) is returned as before.
        In addition, XFOIL's ``CPWR`` and ``DUMP`` commands are appended to
        the command transcript so we capture the surface Cp distribution
        and the boundary-layer profile (Cf, H, delta*, theta, Ue/Vinf).

        If a dump fails to materialise (for instance because the XFOIL build
        does not support ``DUMP`` at the top level), the corresponding entry
        is returned as ``None`` rather than raising.

        Args:
            airfoil: Geometry to load.
            point: Aerodynamic operating point to evaluate.

        Returns:
            ``(polar, cp, wall)`` where ``cp`` and ``wall`` may be ``None``.

        Raises:
            ConvergenceError: If XFOIL did not converge the requested alpha.
            XfoilExecutionError: For process-level failures.
            XfoilNotFoundError: If the binary disappeared between construction
                and the call.
        """
        return self._run_with(airfoil, point, want_cp=True, want_bl=True)

    def _run_with(
        self,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        want_cp: bool,
        want_bl: bool,
    ) -> tuple[PolarPoint, CpDistribution | None, WallProfile | None]:
        """Common driver shared by :meth:`analyze` and :meth:`analyze_with_dumps`."""
        if self.work_dir is None:
            with tempfile.TemporaryDirectory(prefix="aeroforge-xfoil-") as tmp:
                return self._run_in(Path(tmp), airfoil, point, want_cp=want_cp, want_bl=want_bl)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self._run_in(self.work_dir, airfoil, point, want_cp=want_cp, want_bl=want_bl)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _run_in(
        self,
        workdir: Path,
        airfoil: Airfoil,
        point: OperatingPoint,
        *,
        want_cp: bool = False,
        want_bl: bool = False,
    ) -> tuple[PolarPoint, CpDistribution | None, WallProfile | None]:
        """Execute a single XFOIL invocation inside ``workdir`` and parse it.

        Args:
            workdir: Scratch directory used for the run.
            airfoil: Geometry to load.
            point: Aerodynamic operating point.
            want_cp: If True, also dump the Cp distribution via ``CPWR``.
            want_bl: If True, also dump the BL state via ``DUMP``.

        Returns:
            ``(polar, cp, wall)``. ``cp`` and ``wall`` are ``None`` when the
            corresponding ``want_*`` flag is False or when XFOIL failed to
            materialise that dump (rare; treated as a soft failure so the
            caller still gets the polar).
        """
        dat_path = workdir / "airfoil.dat"
        polar_path = workdir / "polar.pol"
        cp_path = workdir / "cp.dat" if want_cp else None
        bl_path = workdir / "bl.dat" if want_bl else None
        airfoil.to_dat(dat_path)
        # XFOIL refuses to overwrite an existing PACC file; clear it first.
        if polar_path.exists():
            polar_path.unlink()
        for p in (cp_path, bl_path):
            if p is not None and p.exists():
                p.unlink()

        session = XfoilSession(
            airfoil=airfoil,
            operating_points=[point],
            max_iter=self.max_iter,
            n_crit=self.n_crit,
            repanel=self.repanel,
        )
        # XFOIL's FORTRAN free-format parser truncates filenames at the first
        # ``:`` character — so on Windows the absolute path ``C:\Users\...``
        # collapses to just ``C`` and ``LOAD`` fails with "Nonexistent file: C".
        # We launch the subprocess with ``cwd=workdir`` (a few lines below), so
        # XFOIL can resolve the basenames without any path component at all.
        transcript = session.to_command_script(
            dat_path=dat_path.name,
            polar_path=polar_path.name,
            cp_path=cp_path.name if cp_path is not None else None,
            bl_path=bl_path.name if bl_path is not None else None,
        )
        self.last_transcript = transcript

        _log.debug(
            "Invoking XFOIL: alpha=%.3f Re=%.3e Mach=%.3f iter=%d ncrit=%.1f",
            point.alpha,
            point.reynolds,
            point.mach,
            self.max_iter,
            self.n_crit,
        )

        # IMPORTANT — pass stdin as raw bytes with explicit ``\n`` separators.
        # Using ``text=True`` makes Python translate every ``\n`` in the
        # transcript into the platform-native line ending. On Windows that
        # turns ``LOAD\nairfoil.dat\n...`` into ``LOAD\r\nairfoil.dat\r\n...``,
        # and many XFOIL builds (notably MinGW) treat ``LOAD\r`` as a different
        # token from ``LOAD``: the command is silently dropped, XFOIL falls
        # back to the top menu, the polar accumulator is never opened, and the
        # caller sees a mysterious "produced no polar output" :class:`ConvergenceError`.
        # Encoding the transcript ourselves and asking for byte I/O bypasses
        # the universal-newlines translator entirely and behaves identically
        # on Linux, macOS, and Windows.
        # On Windows, ``subprocess.run`` would otherwise flash a console
        # window for every XFOIL call — with thousands of calls per GA run
        # that's a continuous flicker. We suppress it two ways at once:
        #
        # * ``CREATE_NO_WINDOW`` (creationflags): tells Windows not to give
        #   the new process its own console at all.
        # * ``STARTUPINFO`` with ``STARTF_USESHOWWINDOW`` + ``SW_HIDE``: in
        #   case the XFOIL binary tries to call ``AllocConsole()`` or open
        #   its own graphics window despite the previous flag (some MinGW
        #   builds do this on startup), Windows still hides whatever it
        #   asks to show.
        #
        # Both names live in :mod:`subprocess` only on Windows; on POSIX
        # they resolve to ``0`` / ``None`` and are silently ignored.
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = None
        if hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
            startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        try:
            proc = subprocess.run(
                [self.binary],
                input=transcript.encode("ascii", errors="replace"),
                capture_output=True,
                text=False,
                timeout=self.timeout_s,
                cwd=str(workdir),
                check=False,
                creationflags=creationflags,
                startupinfo=startupinfo,
            )
        except subprocess.TimeoutExpired as exc:
            raise XfoilExecutionError(
                f"XFOIL timed out after {self.timeout_s:.1f} s at alpha={point.alpha:.3f} deg."
            ) from exc
        except FileNotFoundError as exc:
            raise XfoilNotFoundError(
                f"XFOIL binary {self.binary!r} could not be launched: {exc}"
            ) from exc
        except OSError as exc:
            raise XfoilExecutionError(
                f"Failed to launch XFOIL binary {self.binary!r}: {exc}"
            ) from exc

        # Stash for post-mortem inspection by callers (e.g. wall-dump
        # diagnostics in the visualisation layer).
        self.last_stdout = proc.stdout if isinstance(proc.stdout, bytes) else b""
        self.last_stderr = proc.stderr if isinstance(proc.stderr, bytes) else b""
        try:
            self.last_workdir_files = sorted(p.name for p in workdir.iterdir())
            self.last_files = {}
            for p in workdir.iterdir():
                if not p.is_file():
                    continue
                try:
                    # Cap to ~256 kB per file to avoid blowing memory on
                    # something unusually big — Cp / BL dumps are < 50 kB.
                    self.last_files[p.name] = p.read_bytes()[:262144]
                except OSError:
                    self.last_files[p.name] = b""
        except OSError:
            self.last_workdir_files = []
            self.last_files = {}

        def _decode(buf: bytes | str | None) -> str:
            # Robust against monkeypatched ``subprocess.run`` fakes that may
            # still hand us ``str`` instead of ``bytes`` in tests.
            if buf is None:
                return ""
            if isinstance(buf, bytes):
                return buf.decode("utf-8", errors="replace")
            return str(buf)

        if proc.returncode != 0:
            tail = (_decode(proc.stderr) or _decode(proc.stdout) or "")[-2000:]
            raise XfoilExecutionError(
                f"XFOIL exited with code {proc.returncode} at "
                f"alpha={point.alpha:.3f} deg.\n--- stderr/stdout tail ---\n{tail}"
            )

        if not polar_path.exists():
            # No polar.pol means either XFOIL couldn't converge OR (much more
            # commonly when this fires for *every* candidate) the command
            # script never reached the PACC accumulator at all. Attach the
            # transcript + stdout/stderr tails so the caller can see where
            # XFOIL actually stopped.
            stdout_tail = _decode(proc.stdout)[-1500:]
            stderr_tail = _decode(proc.stderr)[-500:]
            transcript_preview = transcript if len(transcript) < 600 else transcript[:600] + "..."
            raise ConvergenceError(
                f"XFOIL produced no polar output at alpha={point.alpha:.3f} deg; "
                "the operating point failed to converge.\n"
                f"--- transcript sent to XFOIL ---\n{transcript_preview}\n"
                f"--- XFOIL stdout tail ---\n{stdout_tail}\n"
                f"--- XFOIL stderr tail ---\n{stderr_tail}",
                alpha=point.alpha,
            )

        polar = XfoilOutputParser.parse_polar(polar_path)
        matched: PolarPoint | None = None
        for pt in polar.points:
            if abs(pt.operating_point.alpha - point.alpha) <= _ALPHA_MATCH_TOL:
                matched = pt
                break

        if matched is None:
            raise ConvergenceError(
                f"XFOIL did not converge at alpha={point.alpha:.3f} deg "
                f"(parsed {len(polar.points)} other rows).",
                alpha=point.alpha,
            )

        # Best-effort dump parsing. XFOIL doesn't always honour the
        # filename we hand to CPWR / DUMP — depending on the build, it can
        # fall back to a default derived from the loaded dat file
        # basename (``airfoil.cp`` / ``airfoil.bl`` in our case). To stay
        # robust we look for the file under both our explicit name and
        # any plausible XFOIL default, picking whichever the binary
        # actually produced. We never fail the call because of a missing
        # dump; the caller detects ``None`` and skips that plot.
        def _locate(
            requested: Path | None, default_basenames: tuple[str, ...], glob_pattern: str
        ) -> Path | None:
            if requested is None:
                return None
            if requested.exists():
                return requested
            for name in default_basenames:
                candidate = workdir / name
                if candidate.exists():
                    return candidate
            # Last resort — pick the first matching file XFOIL left behind.
            for candidate in workdir.glob(glob_pattern):
                return candidate
            return None

        cp_result: CpDistribution | None = None
        cp_actual = _locate(cp_path, ("airfoil.cp",), "*.cp")
        if cp_actual is not None:
            try:
                cp_result = XfoilOutputParser.parse_cp(cp_actual, point)
            except Exception as exc:  # noqa: BLE001
                _log.debug("Failed to parse Cp dump %s: %s", cp_actual, exc)

        wall_result: WallProfile | None = None
        bl_actual = _locate(bl_path, ("airfoil.bl",), "*.bl")
        if bl_actual is not None:
            try:
                wall_result = XfoilOutputParser.parse_bl_dump(bl_actual, point)
            except Exception as exc:  # noqa: BLE001
                _log.debug("Failed to parse BL dump %s: %s", bl_actual, exc)

        return matched, cp_result, wall_result
