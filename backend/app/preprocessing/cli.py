"""Raw CSV/ZIP preprocessing CLI. No Qdrant writes or held-out-file reads."""

import argparse
import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from app.core.config import PreprocessingSettings
from app.preprocessing.isolation import reject_evaluation_path
from app.preprocessing.pipeline import export, preprocess


def read_raw(path: Path) -> tuple[Any, dict[str, str]]:
    reject_evaluation_path(path)
    if path.suffix.lower() not in {".csv", ".zip"}:
        raise ValueError(
            "Raw input must be CSV or a ZIP containing one CSV; processed splits are not raw sources"
        )
    raw = path.read_bytes()
    data = raw
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(
                    "ZIP must contain exactly one CSV; select an unambiguous raw source"
                )
            reject_evaluation_path(Path(members[0]))
            data = archive.read(members[0])
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding, low_memory=False), {
                "method": "local_csv",
                "path": str(path.resolve()),
                "url": path.resolve().as_uri(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode raw CSV")


def read_huggingface(dataset: str, revision: str) -> tuple[Any, dict[str, str]]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", dataset) or not re.fullmatch(
        r"[0-9a-f]{40}", revision
    ):
        raise ValueError(
            "Hugging Face input requires a public owner/dataset and exact 40-character commit revision"
        )
    from datasets import load_dataset  # type: ignore[import-untyped]

    frame = load_dataset(dataset, split="train", revision=revision, token=False).to_pandas()
    return frame, {
        "method": "huggingface",
        "path": f"{dataset}@{revision}",
        "url": f"https://huggingface.co/datasets/{dataset}/tree/{revision}",
        "sha256": hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        ).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_args = parser.add_mutually_exclusive_group()
    source_args.add_argument("--input", type=Path)
    source_args.add_argument("--hf-dataset")
    parser.add_argument(
        "--revision", help="Exact Hugging Face dataset commit; never an implicit latest revision"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--retrieval-rows", type=int)
    parser.add_argument("--dev-rows", type=int)
    parser.add_argument("--eval-rows", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and prepare splits entirely in memory"
    )
    args = parser.parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "retrieval_rows": args.retrieval_rows,
            "dev_rows": args.dev_rows,
            "eval_rows": args.eval_rows,
            "random_seed": args.seed,
        }.items()
        if value is not None
    }
    try:
        config = PreprocessingSettings(**overrides)
        destination = args.output_dir or config.output_root / datetime.now(UTC).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        if destination.exists():
            raise ValueError("Output directory already exists; refusing to overwrite datasets")
        input_path = args.input if args.input or args.hf_dataset else config.input
        hf_dataset = args.hf_dataset if args.input or args.hf_dataset else config.hf_dataset
        if input_path is None and not hf_dataset:
            raise ValueError(
                "Specify --input/--hf-dataset or configure PREPROCESS_INPUT/PREPROCESS_HF_DATASET"
            )
        raw, source = (
            read_raw(input_path)
            if input_path is not None
            else read_huggingface(hf_dataset or "", args.revision or config.hf_revision or "")
        )
        result = preprocess(raw, config, source)
        print(json.dumps(result.summary, indent=2))
        if not args.dry_run:
            export(result, destination, source)
            print(f"New split artifacts: {destination}")
        else:
            print("Dry run: no output files or Qdrant writes")
        return 0
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f"Preprocessing failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
