"""Compatibility entry point for the centralized preprocessing CLI."""

from app.preprocessing.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
