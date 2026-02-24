#!/usr/bin/env python3
"""Generate Stimela cab definitions from CLI functions."""

import argparse
import subprocess
from pathlib import Path

from hip_cargo.core.generate_cabs import generate_cabs


def get_image_tag():
    """Get the image tag for the current context.

    During a tbump release, reads the version from the .tbump_version sentinel
    file (written by tbump's before_push hook) and deletes it so that subsequent
    commits revert to branch-based tagging. Otherwise derives the tag from the
    current git branch: 'latest' for main, branch name for feature branches.
    """
    sentinel = Path(".tbump_version")
    if sentinel.exists():
        version = sentinel.read_text().strip()
        sentinel.unlink()
        return version

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        if branch == "main":
            return "latest"
        return branch.replace("/", "-")
    except subprocess.CalledProcessError:
        return "latest"


def main():
    """Generate cabs for all CLI functions in src/hip_cargo/cli."""
    parser = argparse.ArgumentParser(description="Generate Stimela cab definitions")
    parser.add_argument(
        "--version",
        type=str,
        help="Semantic version for the image tag (e.g., 0.1.2). If not provided, uses current branch.",
    )
    args = parser.parse_args()

    # Find all CLI module files
    cli_dir = Path("src/pfb_imaging/cli")
    cli_modules = list(cli_dir.glob("*.py"))

    # Exclude __init__.py
    cli_modules = [m for m in cli_modules if m.name != "__init__.py"]

    if not cli_modules:
        print("No CLI modules found")
        return 0

    # Output directory for cabs
    cabs_dir = Path("src/pfb_imaging/cabs")

    # Determine image tag: use --version if provided, else current branch
    if args.version:
        image_tag = args.version
    else:
        image_tag = get_image_tag()

    image_name = f"ghcr.io/ratt-ru/pfb-imaging:{image_tag}"

    # Generate cabs
    generate_cabs(cli_modules, image=image_name, output_dir=cabs_dir)

    print(f"✓ Generated {len(cli_modules)} cab(s) in {cabs_dir}")
    print(f"✓ Using image: {image_name}")
    return 0


if __name__ == "__main__":
    exit(main())
