# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`e84-geoai-common` is a library (published to PyPI) of shared Geospatial AI/ML code for Element 84 projects. Most of it is a provider-neutral LLM abstraction over AWS Bedrock, plus geometry/GeoJSON helpers built on shapely. Python >= 3.12, managed with `uv`.

## Commands

Activate the venv first (`source .venv/bin/activate`) or prefix commands with `uv run`.

- Set up / recreate the environment: `scripts/recreate_venv.sh` (runs `uv sync --all-extras`)
- Lint, type-check, and format-check: `scripts/lint.sh` (ruff check, pyright strict, `ruff format --check`)
- Auto-format: `ruff format`
- Run all tests: `scripts/test.sh`
- Run one test: `scripts/test.sh tests/llm/models/test_claude.py -k test_tool_use` (extra args go to pytest)
- Run tests against real Bedrock: `scripts/test.sh --use-real-bedrock-client` (needs AWS credentials)
- Watch mode: `scripts/test.sh --watch`
- Add a dependency: `scripts/add_dependencies.sh <pkg>`. This also regenerates `requirements.txt`, which the Snyk scan reads. After changing dependencies by hand, run `scripts/export_requirements.sh` so `requirements.txt` stays in sync with `uv.lock`.

CI (`.github/workflows/python-test.yml`) runs `scripts/lint.sh` and `scripts/test.sh` with `AWS_DEFAULT_REGION=us-east-1`. Both must pass.

## Code style constraints

- Ruff runs with `select = ['ALL']` (with the exceptions listed in `pyproject.toml`), line length 100, Google-style docstrings.
- Pyright runs in `strict` mode over `src/` and `tests/`, and unknown member, argument, or variable types are errors. boto3 calls that return untyped values need `# type: ignore[reportUnknownMemberType]`, as in the existing code.
- Pydantic models generally use `ConfigDict(strict=True, extra="forbid")`. The exception is models that parse *responses* from providers: these are deliberately lenient so new fields from the API don't break parsing.
- Generics use PEP 695 syntax (`def f[Model: BaseModel](...)`, `class Agent[LLMToolContext]`).

## Architecture

### LLM abstraction (`src/e84_geoai_common/llm/`)

- `core/llm.py` defines the provider-neutral types: `LLM` (an ABC with a single `prompt(messages, inference_cfg) -> LLMAssistantMessage`), `LLMMessage`/`LLMUserMessage`/`LLMAssistantMessage`, the content types (`TextContent`, `Base64ImageContent`, `JSONContent`, `LLMToolUseContent`, `LLMToolResultContent`, `CachePointContent`), `LLMTool` (whose inputs and outputs are Pydantic model *types*), `ExecutableLLMTool`, `LLMToolChoice`, and `LLMInferenceConfig`.
- `models/` contains one implementation per Bedrock API, and each follows the same pattern:
  - Provider-specific Pydantic request and response models (e.g. `ClaudeInvokeLLMRequest`, `ClaudeResponse`).
  - Module-level `_llm_*_to_<provider>_*` functions that translate the neutral types into provider types.
  - `create_request()` builds the provider request from messages + `LLMInferenceConfig`, `invoke_model_with_request()` calls boto3, and `_response_to_llm_message()` translates the response back.
  - `claude.py` (`BedrockClaudeLLM`, uses `invoke_model` with the Anthropic Messages format), `nova.py` (`BedrockNovaLLM`), and `converse/` (`BedrockConverseLLM`, uses the Bedrock Converse API, for Llama and others).
  - Model IDs are module constants (e.g. `CLAUDE_4_5_HAIKU`, which is the default for `BedrockClaudeLLM`). The `*_BEDROCK_MODEL_IDS` dicts are deprecated, so don't add new models to them. Each Claude model constant needs an entry in `_MODEL_CAPABILITIES`, which records whether it supports prompt caching, sampling params (temperature/top_p/top_k), and forced tool choice. Model IDs are normalized (region prefix, bare `anthropic.` IDs, ARNs) before lookup. Claude prompt caching is on by default for the system prompt and tools when the model supports it.
  - Not every `LLMInferenceConfig` option works with every provider. For example, Claude rejects `json_mode` and `response_prefix`, and `structured_output_model` maps to Claude's `output_config`.
- `extraction.py` has `extract_data_from_text` (the default), which gets structured data from any `LLM` by forcing a single `parse_data` tool call. `extract_data_from_text_claude_structured_output` uses Claude's native structured output instead. If a model's root JSON schema is a union (`anyOf`), both functions wrap it in `DataModelWrapper` first.
- `batch.py` provides `BedrockBatchInference`, which runs Bedrock batch inference jobs: it uploads JSONL to S3, creates and polls the job, and parses the results.
- `tracing.py` (at the package root) provides `@timed_function` and `@observe`. `@observe` is a no-op unless the optional `langfuse` extra is installed.

### Testing against LLMs

The test helpers live in the *package* (`src/e84_geoai_common/llm/tests/`) so downstream projects can use them too:
- `make_test_bedrock_runtime_client(responses)` returns a mock `BedrockRuntimeClient` that pops canned responses in order. If `USE_REAL_BEDROCK_RUNTIME_CLIENT=true` (set by `--use-real-bedrock-client`), it returns a real boto3 client instead.
- `claude_response_with_content`, `nova_response_with_content`, and `converse_response_with_content` build canned response bodies.
- `mock_bedrock.py` works the same way for the batch tests (`USE_REAL_BATCH_BEDROCK_CLIENT`, configured through `.env`; see `.env.template`).
- `tests/conftest.py` registers a custom pytest assertion diff for Pydantic models.

Tests should pass both with canned responses and against real Bedrock, so assertions usually exclude non-deterministic fields such as token counts.

### Other modules

- `geometry.py` and `geojson.py`: shapely helpers and Pydantic GeoJSON types.
- `debugging.py`: folium map helpers (needs the `debugging` extra).
- `util.py`: miscellaneous helpers.

`temp/` holds scratch scripts and is ignored by pyright.
