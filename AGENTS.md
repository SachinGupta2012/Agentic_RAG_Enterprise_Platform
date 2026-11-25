# Working rules

## Speed and token budget
- Read only the files needed for the task. Use `rg` to locate code; don't open whole notebooks or large files.
- Never print large outputs (data files, logs, full test output). Use `head`, `tail`, `-q`, or summaries.
- Run the narrowest test first (`pytest path::test_name`), then the full suite once at the end.
- Don't re-run checks that already passed unless the code changed.
- Prefer small, targeted edits over rewriting files.

## Stopping rules
- Max 2 attempts to fix the same error. After that, stop and report what failed, what you tried, and your best hypothesis.
- If the task needs a decision I haven't made, ask one short question instead of guessing.
- Don't refactor, rename, or "improve" anything outside the task.

## Reporting
- Start by stating the plan in at most 5 lines.
- After each major step, give a one-line status.
- Always finish with a report: files changed, commands run, tests passed/failed, remaining work. Deliver it even if the task is incomplete.

## Project constraints
- Do NOT open, read, or load `tickets_eval.parquet` or any evaluation data. Use the eval manifest (IDs only) for overlap checks.
- Never run ingestion or contact Qdrant unless I explicitly ask. Use `--dry-run`.
- Environment is managed with `uv` (`uv sync`, `uv run pytest`). Don't use bare `pip`.
- Windows paths and PowerShell. Don't assume bash.
- Preprocessing logic lives in importable functions; notebooks and CLIs only call them.
- Settings go in `core/config.py`, not hard-coded constants.

## Definition of done
- Code changed, relevant tests pass, no eval-file access, short report delivered.