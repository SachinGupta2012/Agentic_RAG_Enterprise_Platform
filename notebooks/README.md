# Notebooks

## `01_dataset_ingestion.ipynb`

Downloads, validates, cleans, deduplicates, and splits the customer-support dataset.

### Run on Kaggle

1. Create a Kaggle notebook and attach the dataset
   `tobiasbueck/multilingual-customer-support-tickets`.
2. Import `01_dataset_ingestion.ipynb` into the Kaggle editor.
3. Run all cells. Outputs land in `/kaggle/working/processed/`, and
   `/kaggle/working/processed_tickets.zip` is created for download.

> Kaggle does not expose a Jupyter server endpoint that VS Code can attach to. The practical
> workflows are: run the notebook in the Kaggle editor, or run it locally in VS Code against the
> Kaggle API. This notebook supports both.

### Run locally (VS Code + Jupyter)

1. From the repository root: `uv sync --extra data --extra notebook --extra dev`
2. Install the kernel (once):

   ```bash
   uv run python -m ipykernel install --user --name enterprise-support --display-name "Python (enterprise-support)"
   ```

3. Open the notebook in VS Code and select the `enterprise-support` kernel.
4. Provide Kaggle credentials at `~/.kaggle/kaggle.json` (Kaggle → Account → Create New API Token),
   or let the notebook fall back to the Hugging Face mirror.
5. Run all cells. Outputs land in `data/processed/`.

To start a standalone Jupyter server instead of using the VS Code extension:

```bash
uv run jupyter lab --no-browser
```

Copy the printed `http://127.0.0.1:8888/lab?token=...` URL into VS Code's
**Jupyter: Specify Jupyter Server for Connections** command.

## Conventions

- Notebooks are numbered by pipeline stage.
- Outputs are never committed; `data/` and notebook checkpoints are git-ignored.
- The evaluation split is quarantined. Never ingest `tickets_eval.parquet` into the retrieval index.
