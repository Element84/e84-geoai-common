# Plan: Add Streaming Support to BedrockClaudeLLM

## TL;DR

Refactor `claude.py` into a `claude/` package, then add an async `prompt_stream()` method that uses `aioboto3` and Bedrock's `invoke_model_with_response_stream` to yield fully-typed Pydantic event objects. No raw dicts or strings in the public API.

## Approach

1. **Restructure**: Convert `models/claude.py` → `models/claude/` package (following the existing `converse/` pattern). Move existing code to `claude/claude.py`, add streaming in `claude/streaming.py`.
2. **Native async with aioboto3**: The new `prompt_stream()` is an `async def` generator. It creates an `aioboto3` async client internally (context-managed per call), calls `await client.invoke_model_with_response_stream(...)`, then uses `async for` on the response body's event stream.
3. **Typed events**: Pydantic models for every streaming event type. Parse `chunk["bytes"]` → JSON → discriminated Pydantic model.

## Steps

### Phase 1: Add Dependencies

1. Add `aioboto3` to `pyproject.toml` runtime dependencies
2. Add `types-aiobotocore-bedrock-runtime` to dev extras (type stubs for the async client)
3. Add `pytest-asyncio` to dev extras
4. Run `scripts/refresh_requirements.sh`

### Phase 2: Restructure `claude.py` → `claude/` package

5. Create `src/e84_geoai_common/llm/models/claude/` directory
6. Move `claude.py` → `claude/claude.py` (unchanged content)
7. Create `claude/__init__.py` re-exporting everything (follows `converse/__init__.py` pattern) — all 7 existing import sites remain valid

### Phase 3: Streaming Event Models (`claude/streaming.py`)

8. Pydantic models for each event:
   - Deltas: `ClaudeTextDelta`, `ClaudeInputJsonDelta`
   - Content blocks: `ClaudeStreamTextBlock`, `ClaudeStreamToolUseBlock`
   - Events: `ClaudeStreamMessageStart`, `ClaudeStreamContentBlockStart`, `ClaudeStreamContentBlockDelta`, `ClaudeStreamContentBlockStop`, `ClaudeStreamMessageDelta`, `ClaudeStreamMessageStop`
   - Discriminated union: `ClaudeStreamEvent`
   - Parser: `parse_stream_event(data: dict) -> ClaudeStreamEvent`

### Phase 4: Async `prompt_stream()` (in `claude/claude.py`)

9. New method on `BedrockClaudeLLM`:
   - `async def prompt_stream(...) -> AsyncIterator[ClaudeStreamEvent]`
   - Creates an `aioboto3` async client per-call (`async with session.client(...)`)
   - `await client.invoke_model_with_response_stream(...)`
   - `async for event in response["body"]:` — native async iteration
   - Parses `event["chunk"]["bytes"]` → `parse_stream_event()` → yield
   - Skips `ping`, raises on errors

### Phase 5: Update Exports

10. `claude/__init__.py` exports streaming types
11. `models/__init__.py` exports key streaming types

### Phase 6: Tests

12. Mock async event stream in `mock_bedrock_runtime.py` (`_MockAsyncEventStream` class)
13. `@pytest.mark.asyncio` tests for text streaming, tool use streaming, ping skipping

## Relevant Files

- `src/e84_geoai_common/llm/models/claude.py` → DELETE (becomes package)
- `src/e84_geoai_common/llm/models/claude/__init__.py` — CREATE
- `src/e84_geoai_common/llm/models/claude/claude.py` — CREATE (moved + `prompt_stream`)
- `src/e84_geoai_common/llm/models/claude/streaming.py` — CREATE (event models)
- `src/e84_geoai_common/llm/models/__init__.py` — MODIFY
- `src/e84_geoai_common/llm/tests/mock_bedrock_runtime.py` — MODIFY
- `tests/llm/models/test_claude.py` — MODIFY
- `pyproject.toml` — MODIFY

## Verification

1. `scripts/test.sh` — existing tests pass (import paths preserved)
2. `scripts/lint.sh` — pyright/mypy passes with async + Pydantic models
3. New async streaming tests validate typed event sequences
4. `from e84_geoai_common.llm.models.claude import BedrockClaudeLLM` still works

## Decisions

- `aioboto3` for true async I/O (no thread pool wrapping)
- New deps: `aioboto3` (runtime), `types-aiobotocore-bedrock-runtime` + `pytest-asyncio` (dev)
- Client created per `prompt_stream()` call via async context manager (aiobotocore pools connections)
- Region/config passed from `self.client.meta.region_name` to the async session
- Streaming is Claude-specific (no `LLM` base changes)
- Ping → skip, errors → raise

## Further Considerations

1. **Extended thinking (thinking_delta) events?** — Define the types in `streaming.py` for completeness, yield them as events. Callers can filter.
2. **Convenience `prompt_stream_text()` yielding only text strings?** — Not in initial scope, easy to layer later.
3. **Region/config passthrough to async client?** — The async client should respect the same region/config as the sync one. Pass `region_name` from `self.client.meta.region_name` to the async session.
