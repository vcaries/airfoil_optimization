"""Small DSL for emitting XFOIL stdin commands.

Rather than scattering raw command strings throughout the codebase, the
:class:`XfoilCommand` builder produces them centrally. This makes the wrapper
testable (we can compare command transcripts) and gives a single place to
enforce XFOIL's quirky command grammar.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class XfoilCommand:
    """Fluent builder for an XFOIL command script.

    Each method appends to an internal buffer and returns ``self`` so calls
    chain. :meth:`build` returns the final newline-separated transcript that
    can be piped to XFOIL's standard input.

    Example:
        >>> script = (
        ...     XfoilCommand()
        ...     .load("naca2412.dat")
        ...     .pane()
        ...     .oper()
        ...     .viscous(reynolds=1e6, mach=0.0)
        ...     .iter(max_iter=200)
        ...     .alpha(2.0)
        ...     .quit()
        ...     .build()
        ... )
    """

    _lines: list[str] = field(default_factory=list)

    # ----------------------------------------------------------- #
    # Top-level menu navigation
    # ----------------------------------------------------------- #
    def load(self, dat_path: str) -> XfoilCommand:
        """Load an airfoil from a ``.dat`` file."""
        self._lines += ["LOAD", dat_path]
        return self

    def pane(self) -> XfoilCommand:
        """Apply XFOIL's automatic repaneling."""
        self._lines.append("PANE")
        return self

    def oper(self) -> XfoilCommand:
        """Enter the OPER menu."""
        self._lines.append("OPER")
        return self

    # ----------------------------------------------------------- #
    # OPER sub-menu
    # ----------------------------------------------------------- #
    def viscous(self, reynolds: float, mach: float = 0.0) -> XfoilCommand:
        """Switch to viscous mode at the given Reynolds and Mach numbers."""
        self._lines += [
            "VISC",
            f"{reynolds:.6g}",
            "MACH",
            f"{mach:.6g}",
        ]
        return self

    def iter(self, max_iter: int) -> XfoilCommand:
        """Set the maximum number of viscous iterations."""
        self._lines += ["ITER", str(int(max_iter))]
        return self

    def n_crit(self, value: float) -> XfoilCommand:
        """Set the transition amplification factor in VPAR."""
        self._lines += ["VPAR", "N", f"{value:.6g}", ""]
        return self

    def alpha(self, alpha_deg: float) -> XfoilCommand:
        """Solve at a fixed angle of attack (degrees)."""
        self._lines += ["ALFA", f"{alpha_deg:.6g}"]
        return self

    def alpha_sequence(self, start: float, end: float, step: float) -> XfoilCommand:
        """Sweep angle of attack from ``start`` to ``end`` in ``step`` increments."""
        self._lines += ["ASEQ", f"{start:.6g}", f"{end:.6g}", f"{step:.6g}"]
        return self

    def cl(self, cl_value: float) -> XfoilCommand:
        """Solve at a fixed lift coefficient."""
        self._lines += ["CL", f"{cl_value:.6g}"]
        return self

    def dump_cp(self, path: str) -> XfoilCommand:
        """Write the current Cp distribution to ``path``."""
        self._lines += ["CPWR", path]
        return self

    def dump_bl(self, path: str) -> XfoilCommand:
        """Write the current boundary-layer state to ``path``.

        Issued from the OPER menu after a converged viscous solve. The
        resulting file has columns ``s x y Ue/Vinf Dstar Theta Cf H``,
        which is the standard XFOIL BL output format.
        """
        self._lines += ["DUMP", path]
        return self

    def polar_accumulate(self, path: str) -> XfoilCommand:
        """Open a polar-accumulation file at ``path``."""
        self._lines += ["PACC", path, ""]
        return self

    def polar_close(self) -> XfoilCommand:
        """Close the current polar-accumulation file."""
        self._lines.append("PACC")
        return self

    def back(self) -> XfoilCommand:
        """Return to the previous menu."""
        self._lines.append("")
        return self

    def quit(self) -> XfoilCommand:
        """Exit XFOIL."""
        self._lines.append("QUIT")
        return self

    # ----------------------------------------------------------- #
    # Output
    # ----------------------------------------------------------- #
    def build(self) -> str:
        """Return the final newline-separated command transcript.

        Returns:
            The string to feed to XFOIL's standard input.
        """
        return "\n".join(self._lines) + "\n"
