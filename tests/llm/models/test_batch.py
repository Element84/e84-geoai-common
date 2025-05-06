import datetime
import json
import os
import time
from collections.abc import Iterable
from typing import Any, Unpack

import boto3
import pytest
from moto import mock_aws
from mypy_boto3_bedrock import BedrockClient
from mypy_boto3_bedrock.type_defs import (
    CreateModelInvocationJobRequestTypeDef,
    CreateModelInvocationJobResponseTypeDef,
    GetModelInvocationJobRequestTypeDef,
    GetModelInvocationJobResponseTypeDef,
    ListModelInvocationJobsRequestPaginateTypeDef,
    ListModelInvocationJobsResponseTypeDef,
    ModelInvocationJobSummaryTypeDef,
    ResponseMetadataTypeDef,
)
from mypy_boto3_s3 import S3Client

from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMMessage, TextContent
from e84_geoai_common.llm.models.batch import LLM_TO_OUTPUT_MAP, BedrockBatchLLM
from e84_geoai_common.llm.models.claude import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
    ClaudeResponse,
    ClaudeTextContent,
)
from e84_geoai_common.llm.models.nova import BedrockNovaLLM, NovaResponse, NovaTextContent
from e84_geoai_common.llm.tests.mock_bedrock import (
    claude_response_with_content,
    nova_response_with_content,
)

USE_REAL_BATCH_BEDROCK_CLIENT = os.getenv("USE_REAL_BATCH_BEDROCK_CLIENT") == "true"
BATCH_IAM_ROLE_ARN = os.getenv("BATCH_IAM_ROLE_ARN", "example_role")
BATCH_INPUT_S3 = os.getenv(
    "BATCH_INPUT_S3", "s3://example-input-bucket-e84-geoai-common/input.jsonl"
)
BATCH_OUTPUT_S3 = os.getenv("BATCH_OUTPUT_S3", "s3://example-output-bucket-e84-geoai-common/")

job_name: str


@pytest.fixture
def s3_moto_client():
    """Moto S3 mocking."""
    if not USE_REAL_BATCH_BEDROCK_CLIENT:
        with mock_aws():
            yield
    else:
        yield


def batch_claude_output_example(question: str, content: str, lines: int = 5) -> str:
    """Creates a mock claude batch response with the given text."""
    example_response = {
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
            "temperature": 0.0,
        },
        "modelOutput": claude_response_with_content(content),
        "recordId": "RECORD0000000098",
    }

    json_lines: list[str] = []
    for _ in range(lines):
        json_object_string = json.dumps(example_response)

        json_lines.append(json_object_string)
    body = "\n".join(json_lines)
    return body


def batch_nova_output_example(question: str, content: str, lines: int = 5):
    """Creates a mock nova batch response with the given text."""
    example_response = {
        "modelInput": {
            "messages": [{"role": "user", "content": [{"text": question}]}],
            "inferenceConfig": {"max_new_tokens": 1000, "temperature": 0.0},
        },
        "modelOutput": nova_response_with_content(content),
        "recordId": "RECORD0000000009",
    }

    json_lines: list[str] = []
    for _ in range(lines):
        json_object_string = json.dumps(example_response)

        json_lines.append(json_object_string)
    body = "\n".join(json_lines)
    return body


class MockBedrockClient(BedrockClient):
    def __init__(self, batch_response: str, job_name: str) -> None:
        """Creates the mock client with the job name given."""
        self.batch_response = batch_response
        self.job_name = job_name
        self.job_arn = (
            f"arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/{self.job_name}"
        )
        self.response_metadata: ResponseMetadataTypeDef = {
            "RequestId": "req",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {"example": "example"},
            "RetryAttempts": 0,
        }

    def create_model_invocation_job(
        self, **_kwargs: Unpack[CreateModelInvocationJobRequestTypeDef]
    ) -> CreateModelInvocationJobResponseTypeDef:
        mock_create_job_response: CreateModelInvocationJobResponseTypeDef = {
            "jobArn": self.job_arn,
            "ResponseMetadata": self.response_metadata,
        }

        s3_client = boto3.client("s3")  # type: ignore[reportAssignmentType]

        input_parts = BATCH_INPUT_S3.rstrip("/")[5:].split("/", 1)
        output_bucket = BATCH_OUTPUT_S3.rstrip("/")[5:].split("/", 1)[0]
        output_file = input_parts[-1]
        key = f"{self.job_name}/{output_file}.out"

        s3_client.put_object(Bucket=output_bucket, Key=key, Body=self.batch_response)
        return mock_create_job_response

    def get_model_invocation_job(
        self, **_kwargs: Unpack[GetModelInvocationJobRequestTypeDef]
    ) -> GetModelInvocationJobResponseTypeDef:
        example_date = datetime.datetime.now(tz=datetime.UTC)
        response: GetModelInvocationJobResponseTypeDef = {
            "jobArn": self.job_arn,
            "jobName": self.job_name,
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "roleArn": BATCH_IAM_ROLE_ARN,
            "status": "Completed",
            "message": "Job completed successfully.",
            "submitTime": example_date,
            "lastModifiedTime": example_date,
            "endTime": example_date,
            "inputDataConfig": {"s3InputDataConfig": {"s3Uri": BATCH_INPUT_S3}},
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": BATCH_OUTPUT_S3}},
            "clientRequestToken": "mock-token",
            "vpcConfig": {
                "subnetIds": ["subnet-mock12345", "subnet-mock67890"],
                "securityGroupIds": ["sg-mockfedcba", "sg-mock98765"],
            },
            "timeoutDurationInHours": 24,
            "jobExpirationTime": example_date,
            "ResponseMetadata": self.response_metadata,
        }

        return response

    # mock for the paginator used in the get_job_arn method
    class MockListJobsPaginator:
        """Mock the paginator object returned by get_paginator."""

        def __init__(self, pages_to_yield: list[ListModelInvocationJobsResponseTypeDef]) -> None:
            self.pages = pages_to_yield

        def paginate(
            self,
            **kwargs: Unpack[ListModelInvocationJobsRequestPaginateTypeDef],  # noqa: ARG002
        ) -> Iterable[ListModelInvocationJobsResponseTypeDef]:
            yield from self.pages

    def get_paginator(self, operation_name: str) -> Any:  # noqa: ANN401
        """Returns a mock paginator for list_model_invocation_jobs."""
        if operation_name == "list_model_invocation_jobs":
            example_date = datetime.datetime.now(tz=datetime.UTC)

            mock_job_summary_1: ModelInvocationJobSummaryTypeDef = {
                "jobArn": self.job_arn,
                "jobName": self.job_name,
                "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                "roleArn": BATCH_IAM_ROLE_ARN,
                "submitTime": example_date,
                "inputDataConfig": {"s3InputDataConfig": {"s3Uri": BATCH_INPUT_S3}},
                "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": BATCH_OUTPUT_S3}},
                "status": "Completed",
            }
            list_job_pages: ListModelInvocationJobsResponseTypeDef = {
                "invocationJobSummaries": [mock_job_summary_1],
                "ResponseMetadata": self.response_metadata,
            }
            return self.MockListJobsPaginator([list_job_pages])
        return None


def make_test_bedrock_client(
    batch_response: str | None = None,
    job_name: str | None = None,
    *,
    use_real_client: bool = False,
) -> BedrockClient:
    if USE_REAL_BATCH_BEDROCK_CLIENT or use_real_client:
        return boto3.client("bedrock")  # type: ignore[reportUnknownMemberType]
    if batch_response and job_name:
        return MockBedrockClient(batch_response, job_name)
    raise RuntimeError("If not using a real client the responses and job name must be provided")


def test_claude_create_and_run_job(
    s3_moto_client: S3Client,
):
    timestamp_ns = time.time_ns()
    job_name = f"pytest-job-{timestamp_ns}"

    llm = BedrockClaudeLLM(model_id=CLAUDE_BEDROCK_MODEL_IDS["Claude 3.5 Haiku"])
    llm_question = "What is 10+10?"
    llm_response = "10 + 10 = 20"

    conversations: list[list[LLMMessage]] = []
    for _ in range(100):
        conversation = [
            LLMMessage(
                role="user",
                content=[TextContent(text=llm_question)],
            )
        ]
        conversations.append(conversation)

    batch_response = batch_claude_output_example(llm_question, llm_response)

    batch = BedrockBatchLLM(
        client=make_test_bedrock_client(batch_response=batch_response, job_name=job_name),
        llm=llm,
    )

    job_arn = batch.create_job(
        job_name=job_name,
        role_arn=BATCH_IAM_ROLE_ARN,
        input_s3_file_url=BATCH_INPUT_S3,
        output_s3_directory_url=BATCH_OUTPUT_S3,
        create_buckets_if_missing=True,
        conversations=conversations,
        inference_cfg=LLMInferenceConfig(),
    )

    assert job_arn == batch.get_job_arn(job_name)

    batch.wait_for_job_to_finish(job_arn)
    results_response = batch.get_results(job_arn)

    response_type = LLM_TO_OUTPUT_MAP[type(llm)]
    for _, result in enumerate(results_response.responses[:5]):
        assert result.modelOutput is not None
        assert isinstance(result, response_type)
        assert isinstance(result.modelOutput, ClaudeResponse)
        assert result.modelOutput.content[0] == ClaudeTextContent(text=llm_response)


def test_nova_create_and_run_job(
    s3_moto_client: S3Client,
):
    timestamp_ns = time.time_ns()
    job_name = f"pytest-job-{timestamp_ns}"

    llm = BedrockNovaLLM()
    llm_question = "What is 10+10?"
    llm_response = "10 + 10 = 20"

    conversations: list[list[LLMMessage]] = []
    for _ in range(100):
        conversation = [
            LLMMessage(
                role="user",
                content=[TextContent(text=llm_question)],
            )
        ]
        conversations.append(conversation)

    batch_response = batch_nova_output_example(llm_question, llm_response)

    batch = BedrockBatchLLM(
        client=make_test_bedrock_client(batch_response=batch_response, job_name=job_name),
        llm=llm,
    )

    job_arn = batch.create_job(
        job_name=job_name,
        role_arn=BATCH_IAM_ROLE_ARN,
        input_s3_file_url=BATCH_INPUT_S3,
        output_s3_directory_url=BATCH_OUTPUT_S3,
        create_buckets_if_missing=True,
        conversations=conversations,
        inference_cfg=LLMInferenceConfig(),
    )

    assert job_arn == batch.get_job_arn(job_name)

    batch.wait_for_job_to_finish(job_arn)
    results_response = batch.get_results(job_arn)

    response_type = LLM_TO_OUTPUT_MAP[type(llm)]
    for _, result in enumerate(results_response.responses[:5]):
        assert result.modelOutput is not None
        assert isinstance(result, response_type)
        assert isinstance(result.modelOutput, NovaResponse)
        assert result.modelOutput.output.message.content[0] == NovaTextContent(text=llm_response)
