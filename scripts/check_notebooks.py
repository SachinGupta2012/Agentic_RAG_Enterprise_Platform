"""Validate that notebook code cells contain syntactically valid Python."""

from pathlib import Path

import nbformat


def iter_code_cells(path: Path) -> list[str]:
    """Return the source of every code cell in a notebook."""

    notebook = nbformat.read(path, as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def strip_ipython_magics(source: str) -> str:
    """Remove line magics and shell escapes that are not valid Python."""

    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )


def main() -> int:
    failures = 0
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        for index, source in enumerate(iter_code_cells(path)):
            try:
                compile(strip_ipython_magics(source), f"{path}:cell{index}", "exec")
            except SyntaxError as exc:
                failures += 1
                print(f"{path} cell {index}: {exc}")
    if failures:
        print(f"{failures} notebook cell(s) failed syntax validation.")
        return 1
    print("All notebook code cells are syntactically valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
