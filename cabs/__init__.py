# cabs/__init__.py
"""Stimela cab definitions for pfb-imaging."""
from pathlib import Path

CAB_DIR = Path(__file__).parent

# Export all available cabs
AVAILABLE_CABS = [p.stem for p in CAB_DIR.glob("*.yml")]

def get_cab_path(name: str) -> Path:
    """Get path to a cab definition file."""
    path = CAB_DIR / f"{name}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Cab '{name}' not found")
    return path