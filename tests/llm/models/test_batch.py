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

from e84_geoai_common.llm.core.llm import LLM, LLMInferenceConfig, LLMMessage, TextContent
from e84_geoai_common.llm.models.batch import BedrockBatchLLM
from e84_geoai_common.llm.models.claude import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
    ClaudeResponse,
    ClaudeTextContent,
)
from e84_geoai_common.llm.models.nova import BedrockNovaLLM, NovaResponse, NovaTextContent

USE_REAL_BATCH_BEDROCK_CLIENT = os.getenv("USE_REAL_BATCH_BEDROCK_CLIENT") == "true"
BATCH_IAM_ROLE_ARN = os.getenv("BATCH_IAM_ROLE_ARN", "example_role")
BATCH_INPUT_S3 = os.getenv(
    "BATCH_INPUT_S3", "s3://example-input-bucket-e84-geoai-common/input.jsonl"
)
BATCH_OUTPUT_S3 = os.getenv("BATCH_OUTPUT_S3", "s3://example-output-bucket-e84-geoai-common/")

job_name: str


@pytest.fixture
def s3_moto_client():
    """Activates moto S3 mocking and provides a boto3 client."""
    if not USE_REAL_BATCH_BEDROCK_CLIENT:
        with mock_aws():
            s3 = boto3.client("s3")  # type: ignore[reportAssignmentType]
            yield s3
            return
    yield


def create_jsonl_output_example(content: list[dict[str, str]], llm: LLM, length: int = 5) -> str:
    base_json_object = None
    if isinstance(llm, BedrockClaudeLLM):
        base_json_object = {
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "What is 10+10?"}]}
                ],
                "temperature": 0.0,
            },
            "modelOutput": {
                "id": "msg_bdrk_011T4WP84btRuGukyRba9E5s",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-haiku-20241022",
                "content": content,
                "stop_reason": "end_turn",
                "stop_sequence": "null",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            "recordId": "RECORD0000000098",
        }
    elif isinstance(llm, BedrockNovaLLM):
        base_json_object = {
            "modelInput": {
                "messages": [{"role": "user", "content": [{"text": "What is 10+10?"}]}],
                "inferenceConfig": {"max_new_tokens": 1000, "temperature": 0.0},
            },
            "modelOutput": {
                "output": {
                    "message": {
                        "content": content,
                        "role": "assistant",
                    }
                },
                "stopReason": "end_turn",
                "usage": {
                    "inputTokens": 9,
                    "outputTokens": 192,
                    "totalTokens": 201,
                    "cacheReadInputTokenCount": 0,
                    "cacheWriteInputTokenCount": 0,
                },
            },
            "recordId": "RECORD0000000009",
        }

    json_lines: list[str] = []
    for _ in range(length):
        json_object_string = json.dumps(base_json_object)

        json_lines.append(json_object_string)
    return "\n".join(json_lines)


class MockBedrockClient(BedrockClient):
    def __init__(self, responses: list[dict[str, Any]], expected_job_name: str, llm: LLM) -> None:
        """Creates the client with the job name given."""
        self.responses = responses
        self.expected_job_name = expected_job_name
        self.llm = llm
        self.job_arn = (
            f"arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/{self.expected_job_name}"
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
            "jobArn": (
                "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/"
                f"{self.expected_job_name}"
            ),
            "ResponseMetadata": self.response_metadata,
        }

        s3_client = boto3.client("s3")  # type: ignore[reportAssignmentType]

        body = create_jsonl_output_example(content=self.responses, llm=self.llm)

        input_parts = BATCH_INPUT_S3.rstrip("/")[5:].split("/", 1)
        output_bucket = BATCH_OUTPUT_S3.rstrip("/")[5:].split("/", 1)[0]
        output_file = input_parts[-1]
        key = f"{self.expected_job_name}/{output_file}.out"

        s3_client.put_object(Bucket=output_bucket, Key=key, Body=body)
        return mock_create_job_response

    def get_model_invocation_job(
        self, **_kwargs: Unpack[GetModelInvocationJobRequestTypeDef]
    ) -> GetModelInvocationJobResponseTypeDef:
        example_date = datetime.datetime.now(tz=datetime.UTC)
        response: GetModelInvocationJobResponseTypeDef = {
            "jobArn": self.job_arn,
            "jobName": self.expected_job_name,
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

    class MockListJobsPaginator:
        """Inner class to mock the paginator object returned by get_paginator."""

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
                "jobName": self.expected_job_name,
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
    responses: list[dict[str, Any]],
    *,
    use_real_client: bool = False,
    expected_job_name: str,
    llm: LLM,
) -> BedrockClient:
    if USE_REAL_BATCH_BEDROCK_CLIENT or use_real_client:
        return boto3.client("bedrock")  # type: ignore[reportUnknownMemberType]
    if expected_job_name:
        return MockBedrockClient(responses, expected_job_name, llm)
    raise RuntimeError("If not using a real client the responses must be provided")


@pytest.mark.parametrize(
    ("llm", "responses", "expected_output"),
    [
        (
            BedrockClaudeLLM(model_id=CLAUDE_BEDROCK_MODEL_IDS["Claude 3.5 Haiku"]),
            [{"type": "text", "text": "10 + 10 = 20"}],
            [ClaudeTextContent(type="text", text="10 + 10 = 20")],
        ),
        (
            BedrockNovaLLM(),
            [{"text": "10 + 10 = 20"}],
            [NovaTextContent(text="10 + 10 = 20")],
        ),
    ],
)
def test_create_and_run_job(
    s3_moto_client: S3Client,  # noqa: ARG001
    llm: LLM,
    responses: list[dict[str, Any]],
    expected_output: dict[str, Any],
):
    timestamp_ns = time.time_ns()
    job_name = f"pytest-job-{timestamp_ns}"

    conversations: list[list[LLMMessage]] = []
    for _ in range(100):
        conversation = [
            LLMMessage(
                role="user",
                content=[TextContent(text="What is 10+10?")],
            )
        ]
        conversations.append(conversation)

    batch = BedrockBatchLLM(
        client=make_test_bedrock_client(responses=responses, expected_job_name=job_name, llm=llm),
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

    for _, result in enumerate(results_response.responses[:5]):
        assert result.modelOutput is not None
        if isinstance(llm, BedrockClaudeLLM):
            assert isinstance(result.modelOutput, ClaudeResponse)
            assert result.modelOutput.content == expected_output
        elif isinstance(llm, BedrockNovaLLM):
            assert isinstance(result.modelOutput, NovaResponse)
            assert result.modelOutput.output.message.content == expected_output
