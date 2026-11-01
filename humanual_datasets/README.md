# Humanual dataset reprocessing

This directory rebuilds Humanual training data with this project's persona format.
It does **not** collect the original source data again. Instead, the pipeline:

1. downloads one of Humanual's [preprocessed datasets](https://huggingface.co/collections/snap-stanford/humanual-datasets) from Hugging Face;
2. discards the existing `persona` column;
3. filters and partitions the examples;
4. uses the prompt in `persona_prompt.py` and the model endpoint you select to generate a new persona for each user; and
5. saves the resulting train/validation/test Parquet files locally and, optionally, uploads them to the Hugging Face Hub.

```mermaid
flowchart LR
    A[Humanual preprocessed dataset] --> B[Drop Humanual persona]
    B --> C[Filter examples]
    C --> D[Generate personas with our prompt]
    D --> E[Partition dataset]
    E --> F[Parquet files or Hugging Face dataset]
```

For post-based partitioning, the pipeline creates the split before persona generation and builds personas only from training examples. This avoids using validation or test content to construct a user's persona.

## Prerequisites

- Python with the repository dependencies installed.
- An MLflow server listening at `http://localhost:5000`. The pipeline currently sets this URI in code.
- The persona prompt registered in that MLflow server as `persona-prompt-template`.
- A chat-completion model endpoint supported by LiteLLM. You must pass both its model name and API base URL to the pipeline.
- Access to the selected Humanual dataset on Hugging Face. Set `HF_TOKEN` if the repository or your output repository requires authentication.
- Provider credentials in your environment or an optional `.env` file if the model endpoint requires them. Never commit credentials.

Install the Python dependencies from the repository root. MLflow is installed separately because it is not currently listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m pip install mlflow
```

## Run the pipeline

Run each service or command below from the repository root.

### 1. Start MLflow

In the first terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

MLflow serves its UI and API at `http://127.0.0.1:5000` by default. See the [MLflow tracking server documentation](https://mlflow.org/docs/latest/tracking/server/) for storage and deployment options.

### 2. Register this project's persona prompt

With MLflow still running, open a second terminal and register `persona_prompt.py` in the same tracking server:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m humanual_datasets.persona_prompt
```

PowerShell equivalent:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python -m humanual_datasets.persona_prompt
```

Run this step again whenever `persona_prompt.py` changes. MLflow creates a new prompt version, and persona generation loads the version assigned to the `latest` alias. See the [MLflow Prompt Registry documentation](https://mlflow.org/docs/latest/genai/prompt-registry/).

### 3. Start or select a model endpoint

The endpoint must expose a chat-completions API that LiteLLM supports. For example, a local vLLM server can be started with:

```bash
vllm serve Qwen/Qwen3.8-27B --port 3200
```

That endpoint is addressed as:

- model: `hosted_vllm/Qwen/Qwen3.8-27B`
- API base URL: `http://localhost:3200/v1`

The `hosted_vllm/` prefix tells LiteLLM which provider adapter to use. For another provider, use the corresponding LiteLLM model prefix and endpoint. Refer to the [LiteLLM provider documentation](https://docs.litellm.ai/docs/providers) and [vLLM serving documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart/) for supported configurations.

### 4. Process a Humanual dataset

The following example reprocesses Humanual Opinion. Replace the model and URL with the endpoint selected in the previous step:

```bash
python -m humanual_datasets.process_raw \
    --dataset_name humanual_opinion \
    --splits train \
    --pull_from_hub snap-stanford/humanual-opinion \
    --save_dir data/humanual-opinion-reprocessed \
    --cache_dir data/cache \
    --resume_from_cache \
    --llm_model hosted_vllm/Qwen/Qwen3.8-27B \
    --llm_api_base http://localhost:3200/v1 \
    --persona_history_length 20 \
    --partition_by post \
    --max_concurrent_users 5
```

The two endpoint arguments are the important values to customize:

| Argument | Meaning | Example |
| --- | --- | --- |
| `--llm_model` | LiteLLM provider prefix plus the model served by the endpoint | `hosted_vllm/Qwen/Qwen3.8-27B` |
| `--llm_api_base` | Base URL of the model's OpenAI-compatible API | `http://localhost:3200/v1` |

To use another Humanual dataset, keep `--dataset_name` and `--pull_from_hub` paired:

| Dataset | `--dataset_name` | `--pull_from_hub` |
| --- | --- | --- |
| Book | `humanual_book` | `snap-stanford/humanual-book` |
| Email | `humanual_email` | `snap-stanford/humanual-email` |
| News | `humanual_news` | `snap-stanford/humanual-news` |
| Opinion | `humanual_opinion` | `snap-stanford/humanual-opinion` |
| Politics | `humanual_politics` | `snap-stanford/humanual-politics` |

Choose `--splits` based on the splits or subsets available in the selected Hugging Face dataset. The value supports regular expressions; use `--splits ".*"` to select every available split. Add `--subset_mode` when the dataset organizes sources as Hugging Face subsets rather than splits.

## Useful options

| Option | Purpose |
| --- | --- |
| `--save_dir PATH` | Save the processed splits as local Parquet files. |
| `--push_to_hub ORG/REPO` | Upload the processed dataset to a private Hugging Face repository. |
| `--cache_dir PATH` | Cache downloaded, filtered, and persona-checkpoint data. |
| `--resume_from_cache` | Reuse cached work after interruption. |
| `--persona_history_length N` | Use the earliest `N` examples for each persona. Default: `20`. |
| `--remove_used_persona_rows` | Exclude the examples used to create personas from the final dataset. |
| `--partition_by post` | Split posts before persona generation so only training content informs personas. |
| `--max_concurrent_users N` | Limit concurrent model requests. Lower this if the endpoint runs out of memory or rate-limits requests. |
| `--max_samples N` | Limit rows for a small test run. |
| `--preview_personas` | Exercise persona selection without making model calls. |
| `--overwrite` | Permit replacement of a non-empty Hugging Face output repository. Use carefully. |

If the endpoint needs credentials, place the provider-specific variables in an untracked `.env` file and pass it with `--config path/to/.env`.

## Outputs and recovery

When `--save_dir` is set, the pipeline writes numbered Parquet partitions such as `train0.parquet`, `val0.parquet`, and `test0.parquet` (the exact split names depend on the partitioning strategy). When `--push_to_hub` is set, it also uploads the generated partitions to that dataset repository.

Persona checkpoints are stored below `--cache_dir`, keyed by the processing configuration. Keep the same result-affecting arguments when resuming; changing the model, URL, split fractions, persona history length, or filters produces a different checkpoint key.

## Tradeoffs and security

- Reusing Humanual's preprocessed datasets is faster and more reproducible than scraping every original source, but it inherits the upstream dataset's filtering, schema, licensing, and potential biases.
- Replacing the upstream persona keeps training aligned with this project's prompt and persona schema, but persona quality now depends on the selected model and endpoint.
- Persona generation sends conversation or comment content to the configured model endpoint. Use a trusted endpoint, review the source dataset's license and privacy requirements, and do not send sensitive data to a third-party provider without authorization.
- Model calls may incur provider charges. Start with `--preview_personas` or a small `--max_samples` value, inspect the selected histories, and then run the full job.
- MLflow is bound to loopback in the example. If it must be exposed beyond the local machine, configure authentication, TLS, and MLflow's allowed-host/CORS controls; do not expose an unauthenticated tracking server publicly.
- `--overwrite` can delete the contents of the selected Hugging Face dataset repository before uploading. Confirm the repository name before using it.
