from typing import Any

from pydantic import BaseModel

from e84_geoai_common.llm.models.claude import ClaudeInvokeLLMRequest, ClaudeResponse

BATCH_BEDROCK_MODEL_IDS = {
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
}

MINIMUM_BATCH_ENTRIES = 100


# ruff: noqa: N815, N803, N806


class BatchRecordInput(BaseModel):
    """Input record for batch inference."""

    recordId: str
    modelInput: ClaudeInvokeLLMRequest


class BatchRecordOutput(BaseModel):
    """Output record from batch inference."""

    recordId: str
    modelInput: ClaudeInvokeLLMRequest
    modelOutput: ClaudeResponse | None  # type: ignore[name-defined]
    error: dict[str, Any] | None = None


class BatchLLMResults(BaseModel):
    """Container for batch inference results."""

    responses: list[BatchRecordOutput]
    status: str
    message: str


class BatchResponse(BaseModel):
    """Base response class for batch API operations."""

    status: str
    message: str


class JobSubmissionResponse(BatchResponse):
    """Response for job submission operations."""

    jobArn: str | None = None


class JobStatusResponse(BatchResponse):
    """Response for retrieving job status."""

    jobArn: str | None = None


class ResultsRetrievalResponse(BatchResponse):
    """Response for retrieving job results."""

    result: str | None = None


class S3InputDataConfig(BaseModel):
    """Configuration for the S3 input data URI."""

    s3Uri: str


class S3OutputDataConfig(BaseModel):
    """Configuration for the S3 output data URI."""

    s3Uri: str


class InputDataConfig(BaseModel):
    """Wrapper for S3 input data configuration."""

    s3InputDataConfig: S3InputDataConfig


class OutputDataConfig(BaseModel):
    """Wrapper for S3 output data configuration."""

    s3OutputDataConfig: S3OutputDataConfig


class BatchLLMRequest(BaseModel):
    """Request payload for invoking a batch model job."""

    roleArn: str
    jobName: str
    modelId: str
    inputDataConfig: InputDataConfig
    outputDataConfig: OutputDataConfig
