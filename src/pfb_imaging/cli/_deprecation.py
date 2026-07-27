"""Loud deprecation banners for CLI subcommands retired in the next release.

Kept dependency-light (``rich`` only, which the CLI already pulls in) so the
lightweight install and cab generation stay fast. The banner is emitted from the
command-function *body*, never at import/decoration time, so it fires only on a
real invocation and leaves ``pfb --help`` and cab generation untouched.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# The release in which the deprecated subcommands disappear.
REMOVED_IN = "0.1.0"


def warn_deprecated(command: str, replacement: str | None = None) -> None:
    """Print a bold-red, full-width deprecation banner to stderr.

    Args:
        command: The subcommand name as typed on the CLI (e.g. ``"init"``).
        replacement: Suggested replacement command to migrate to, if any.
    """
    body = Text(justify="left")
    body.append(
        f"`pfb {command}` is DEPRECATED and will be REMOVED in pfb-imaging v{REMOVED_IN}.",
        style="bold red",
    )
    if replacement:
        body.append(f"\n\nMigrate to:  {replacement}", style="bold red")

    # Create the Console inside the call so it binds to the current sys.stderr
    # (matters for test capture and any stderr redirection).
    console = Console(stderr=True)
    console.print(
        Panel(
            body,
            title="[bold red]⚠  DEPRECATION WARNING[/bold red]",
            border_style="bold red",
            expand=True,
        )
    )
