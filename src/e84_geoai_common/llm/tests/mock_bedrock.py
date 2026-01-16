import datetime
import json
import os
from collections.abc import Iterable
from typing import Any, Unpack

import boto3
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

from e84_geoai_common.llm.tests.mock_bedrock_runtime import (
    claude_response_with_content,
    nova_response_with_content,
)

USE_REAL_BATCH_BEDROCK_CLIENT = os.getenv("USE_REAL_BATCH_BEDROCK_CLIENT") == "true"
BATCH_IAM_ROLE_ARN = os.getenv("BATCH_IAM_ROLE_ARN", "example_role")
BATCH_INPUT_S3 = os.getenv(
    "BATCH_INPUT_S3", "s3://example-input-bucket-e84-geoai-common/input.jsonl"
)
BATCH_OUTPUT_S3 = os.getenv("BATCH_OUTPUT_S3", "s3://example-output-bucket-e84-geoai-common/")


def batch_claude_output_example(
    question: str, content: str, lines: int = 5, record_id_example: str = "RECORD0000000098"
) -> str:
    """Creates a mock claude batch response with the given text."""
    example_response = {
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
            "temperature": 0.0,
        },
        "modelOutput": claude_response_with_content(content),
        "recordId": record_id_example,
    }

    json_lines: list[str] = []
    for _ in range(lines):
        json_object_string = json.dumps(example_response)

        json_lines.append(json_object_string)
    body = "\n".join(json_lines)
    return body


def batch_nova_output_example(question: str, content: str, lines: int = 5) -> str:
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
            """Mock initialize function for Paginator."""
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
    """Creates a BedrockClient for testing."""
    if USE_REAL_BATCH_BEDROCK_CLIENT or use_real_client:
        return boto3.client("bedrock")  # type: ignore[reportUnknownMemberType]
    if batch_response and job_name:
        return MockBedrockClient(batch_response, job_name)
    raise RuntimeError("If not using a real client the responses and job name must be provided")
