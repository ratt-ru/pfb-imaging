#!/usr/bin/env python3
"""Generate Stimela cab definitions from CLI functions."""

import subprocess
from pathlib import Path

from hip_cargo.core.generate_cabs import generate_cabs


def get_current_branch():
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        # Sanitize branch name for use in image tags (replace / with -)
        return branch.replace("/", "-")
    except subprocess.CalledProcessError:
        # Fallback if not in a git repo
        return "latest"


def main():
    """Generate cabs for all CLI functions in src/pfb_imaging/cli."""
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

    # Get current branch for image tag
    branch = get_current_branch()
    image_name = f"ghcr.io/ratt-ru/pfb-imaging:{branch}"
    # Generate cabs
    generate_cabs(cli_modules, image=image_name, output_dir=cabs_dir)

    print(f"✓ Generated {len(cli_modules)} cab(s) in {cabs_dir}")
    print(f"✓ Using image: {image_name}")
    return 0


if __name__ == "__main__":
    exit(main())
