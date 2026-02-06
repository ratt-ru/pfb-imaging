"""Test round-trip conversion: CLI function -> cab -> function."""

import tempfile
from pathlib import Path

from hip_cargo.core.generate_cabs import generate_cabs
from hip_cargo.core.generate_function import generate_function


def test_roundtrip():
    """Test round-trip: generate_cabs CLI -> cab -> generated function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cab_dir = tmpdir / "cabs"
        cab_dir.mkdir()

        # CLI -> cabs
        generate_cabs(
            module=[Path("src/pfb_imaging/cli/*.py")],
            output_dir=cab_dir,
            image=None,
        )

        # cab -> CLI
        cabs = list(cab_dir.glob("*.yml"))
        for cab_file in cabs:
            assert cab_file.exists(), "Cab file should be generated"

            generated_file = tmpdir / f"{cab_file.stem}_roundtrip.py"
            generate_function(cab_file, generated_file, config_file=Path("pyproject.toml"))

            # Read generated code
            assert generated_file.exists(), "Generated function should exist"
            generated_code = generated_file.read_text()

            # Compile to check syntax
            compile(generated_code, str(generated_file), "exec")

            # Compare with original (both should have been formatted with ruff)
            module_path = Path("src/pfb_imaging/cli") / f"{cab_file.stem}.py"
            original_code = module_path.read_text()

            # Compare line by line
            original_lines = original_code.splitlines()
            generated_lines = generated_code.splitlines()

            # They should match exactly after normalization
            assert len(original_lines) == len(generated_lines), (
                f"Line count mismatch: original has {len(original_lines)} lines, "
                f"generated has {len(generated_lines)} lines for cab {cab_file.stem}"
            )

            for i, (orig_line, gen_line) in enumerate(zip(original_lines, generated_lines), 1):
                assert orig_line == gen_line, f"Line {i} differs:\n  Original:  {orig_line}\n  Generated: {gen_line}"
