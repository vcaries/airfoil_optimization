"""Command-line entry point for aeroforge.

The CLI is the polished front door for portfolio reviewers who want to try the
project without writing Python. Each subcommand maps to a high-level use case:

* ``aeroforge airfoil naca 2412`` -- generate an airfoil and dump a ``.dat``
  file.
* ``aeroforge polar run NACA2412 --re 1e6 --alphas 0:10:0.5`` -- compute a
  polar sweep through XFOIL.
* ``aeroforge optimize run config.yaml`` -- launch an optimization study.

Requires the ``cli`` extra (``pip install aeroforge[cli]``).
"""

from __future__ import annotations

try:
    import typer  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The aeroforge CLI requires the 'cli' extra. Install it with: pip install aeroforge[cli]"
    ) from exc

from aeroforge import __version__
from aeroforge.geometry import NACA4Generator

app = typer.Typer(
    help="aeroforge -- XFOIL wrapper, geometry engine, and airfoil optimizer.",
    no_args_is_help=True,
)

airfoil_app = typer.Typer(help="Airfoil geometry commands.")
app.add_typer(airfoil_app, name="airfoil")


@app.callback()
def _root(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    """Top-level options."""
    if version:
        typer.echo(f"aeroforge {__version__}")
        raise typer.Exit()


@airfoil_app.command("naca")
def airfoil_naca(
    designation: str = typer.Argument(..., help="NACA 4-digit code, e.g. 2412."),
    n_points: int = typer.Option(120, "--n", help="Stations per surface."),
    output: str = typer.Option("airfoil.dat", "--out", help="Output .dat path."),
) -> None:
    """Generate a NACA 4-digit airfoil and write it to a ``.dat`` file."""
    airfoil = NACA4Generator(designation=designation, n_points=n_points).generate()
    path = airfoil.to_dat(output)
    typer.echo(
        f"Wrote {airfoil.name} ({airfoil.n_points} pts, t_max="
        f"{airfoil.max_thickness:.4f}) to {path}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
