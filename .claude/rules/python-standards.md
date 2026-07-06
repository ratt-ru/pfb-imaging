# Python Standards & CLI Implementation Guidelines

Read this when editing or creating `**/*.py` files.

## 1. Type Hints and Modern Python
* Python 3.10+ features are allowed and encouraged.
* Always use type hints for function signatures.
* Use `from typing import Any` for generic types.
* Use `typing_extensions.Annotated` when needed for forward compatibility, but assume the project only supports Python 3.10+. Do not import from `typing_extensions` unless required.

## 2. Typer Option/Argument Syntax (CRITICAL)
**NEVER** use `None` as a positional argument to `typer.Option()` as it causes AttributeError.

Follow these exact patterns for Typer annotations:
* **Required:** `Annotated[Type, typer.Option(..., help="...")]` (no `= default`).
* **Optional with default:** `Annotated[Type, typer.Option(help="...")] = default`.
* **Optional None:** `Annotated[Type | None, typer.Option(help="...")] = None`.

### 2.1 CLI Help Text Must Round-Trip

`tests/test_roundtrip.py` regenerates every CLI module from its cab and compares
line-by-line, so `help=` strings must match hip-cargo's canonical formatting exactly:
* The generator rewraps help at `". "` sentence boundaries, one sentence per line. A
  single sentence rendering longer than 120 chars makes ruff's E501 unfixable and the
  whole regenerated file comes out unformatted (line-count mismatch) — split long help
  into multiple short sentences.
* Avoid mid-help abbreviations containing periods (`e.g.`) — they are false sentence
  boundaries to the generator.
* After editing any help text, run `uv run pytest tests/test_roundtrip.py` and
  regenerate the cabs.

## 3. Lazy Imports for CLI Modules
CLI modules must remain lightweight. Import heavy dependencies only inside the execution scope (within functions) rather than at the top of the file. This keeps CLI startup fast and allows lightweight installation for cab definitions only.

## 4. Architectural Style & Autonomy
* **Functions vs. Classes:** Prefer functional approaches, pure functions, and explicit over implicit behavior. However, you are empowered to use classes without asking permission when state management is truly beneficial or inheritance/polymorphism is needed.
* **DRY Code:** If writing helper or utility functions reduces complexity and keeps code DRY, implement them. Prefer to avoid one-level-deep functions unless they substantially improve readability. Try to avoid splitting helper functions into separate files unless they are shared across multiple modules.
* **Dependencies:** Keep implementations straightforward and avoid over-engineering. Prefer the standard library first, but if adding a dependency via `uv add` significantly reduces custom boilerplate and complexity, do so.
* **Error Handling:** Be explicit about error cases and let exceptions propagate unless there is a specific reason to catch them. Use `typer.Exit(code=1)` for CLI errors.

## 5. Documentation and Commenting
* Use Google-style docstrings.
* Document Args, Returns, and Raises.
* Keep docstrings concise but informative.
* Add comments when code intent isn't obvious. Do not add long unnecessary comments if the intent is clear; prefer short inline comments in this case.
