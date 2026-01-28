# cabs/__init__.py
"""Stimela cab definitions for pfb-imaging."""

from pathlib import Path

CAB_DIR = Path(__file__).parent

# Export all available cabs
AVAILABLE_CABS = [p.stem for pattern in ("*.yaml", "*.yml") for p in CAB_DIR.glob(pattern)]


def get_cab_path(name: str) -> Path:
    """Get path to a cab definition file."""
    # Prefer .yaml, but fall back to .yml for backwards compatibility
    for ext in (".yaml", ".yml"):
        path = CAB_DIR / f"{name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cab '{name}' not found")
