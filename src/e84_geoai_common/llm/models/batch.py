from collections.abc import Sequence
from time import sleep
from typing import TYPE_CHECKING, Any, Self, cast

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMMessage
from e84_geoai_common.llm.models.claude import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
    ClaudeInvokeLLMRequest,
    ClaudeResponse,
)
from e84_geoai_common.llm.models.nova import BedrockNovaLLM, NovaInvokeLLMRequest, NovaResponse

if TYPE_CHECKING:
    from mypy_boto3_s3.literals import BucketLocationConstraintType

# Batch inference uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815

ValidBatchLLMs = BedrockClaudeLLM | BedrockNovaLLM


class BatchRecordInput(BaseModel):
    """Single input record for batch inference."""

    recordId: str
    modelInput: ClaudeInvokeLLMRequest | NovaInvokeLLMRequest


class ClaudeBatchRecordOutput(BaseModel):
    """Specific output record for Claude batch inference."""

    model_config = ConfigDict(strict=True, extra="forbid")

    recordId: str
    modelInput: ClaudeInvokeLLMRequest
    modelOutput: ClaudeResponse | None = None
    error: ClaudeInvokeLLMRequest | None = None


class NovaBatchRecordOutput(BaseModel):
    """Specific output record for Nova batch inference."""

    model_config = ConfigDict(strict=True, extra="forbid")

    recordId: str
    modelInput: NovaInvokeLLMRequest
    modelOutput: NovaResponse | None = None
    error: NovaInvokeLLMRequest | None = None


BatchLLMResponseTypes = ClaudeBatchRecordOutput | NovaBatchRecordOutput
BatchLLMResponsesTypes = Sequence[ClaudeBatchRecordOutput] | Sequence[NovaBatchRecordOutput]

LLM_TO_OUTPUT_MAP: dict[type[ValidBatchLLMs], type[BatchLLMResponseTypes]] = {
    BedrockClaudeLLM: ClaudeBatchRecordOutput,
    BedrockNovaLLM: NovaBatchRecordOutput,
}


class BatchLLMResults(BaseModel):
    """All batch results model."""

    responses: BatchLLMResponsesTypes


class S3InputDataConfig(BaseModel):
    """S3 input URI model."""

    s3Uri: str


class S3OutputDataConfig(BaseModel):
    """S3 output URI model."""

    s3Uri: str


class InputDataConfig(BaseModel):
    """Wrapper for S3 input URI model."""

    s3InputDataConfig: S3InputDataConfig


class OutputDataConfig(BaseModel):
    """Wrapper for S3 output URI modoel."""

    s3OutputDataConfig: S3OutputDataConfig


class BatchLLMRequest(BaseModel):
    """Request payload for invoking a batch model job."""

    roleArn: str
    jobName: str
    modelId: str
    inputDataConfig: InputDataConfig
    outputDataConfig: OutputDataConfig


class BedrockBatchLLM(BaseModel):
    # LLM is an abstract, non-pydantic field. arbitrary_types_allowed is needed or else it errors.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: ValidBatchLLMs = Field(
        default_factory=lambda: BedrockClaudeLLM(
            model_id=CLAUDE_BEDROCK_MODEL_IDS["Claude 3.5 Haiku"]
        )
    )
    client: Any = Field(default_factory=lambda: boto3.client("bedrock"))  # type: ignore[reportUnknownMemberType]
    s3_client: Any = Field(default_factory=lambda: boto3.client("s3"))  # type: ignore[reportUnknownMemberType]

    def get_results(self: Self, job_arn: str) -> BatchLLMResults:
        """Returns the results of the job. Returns an error it is not done yet."""
        job_details = self.client.get_model_invocation_job(jobIdentifier=job_arn)
        status = job_details.get("status")

        if status != "Completed":
            message = job_details.get("message")
            error_details = (
                f"get_results called on {job_arn} when status is not 'Completed' "
                f"with status: {status}. Message: '{message}'."
            )
            raise RuntimeError(error_details)

        job_id = job_arn.split("/")[-1]

        input_data_config = job_details.get("inputDataConfig", {})
        s3_input_config = input_data_config.get("s3InputDataConfig", {})
        input_uri = s3_input_config.get("s3Uri")
        input_filename = input_uri.split("/")[-1]

        output_data_config = job_details.get("outputDataConfig", {})
        s3_output_config = output_data_config.get("s3OutputDataConfig", {})
        output_uri = s3_output_config.get("s3Uri")
        path_part = output_uri[5:]  # Remove s3://
        parts = path_part.split("/", 1)  # Split on first /

        output_bucket = parts[0]
        output_prefix_key = ""
        if len(parts) > 1 and parts[1]:
            output_prefix_key = parts[1]

        final_output_key = f"{output_prefix_key}{job_id}/{input_filename}.out"
        llm_instance_type = type(self.llm)
        record_output_model = LLM_TO_OUTPUT_MAP[llm_instance_type]
        return self._validate_results(record_output_model, output_bucket, final_output_key)

    def _validate_results(
        self,
        response_model: type[BatchLLMResponseTypes],
        output_bucket: str,
        key: str,
    ) -> BatchLLMResults:
        response = self.s3_client.get_object(Bucket=output_bucket, Key=key)
        response_body = response["Body"].read().decode("utf-8")

        responses_list: list[BaseModel] = []
        for line in response_body.splitlines():
            record_obj = response_model.model_validate_json(line)
            responses_list.append(record_obj)
        casted_responses = cast("BatchLLMResponsesTypes", responses_list)
        return BatchLLMResults(
            responses=casted_responses,
        )

    def wait_for_job_to_finish(self: Self, job_arn: str) -> None:
        """Returns once the job has either finished or failed."""
        while True:
            response = self.client.get_model_invocation_job(jobIdentifier=job_arn)
            status = response.get("status")

            if status == "Completed":
                break

            if status in ["Failed", "Stopping", "Stopped"]:
                message = response.get("message")
                error_details = (
                    f"Job {job_arn} finished with status: {status}. Message: '{message}'."
                )
                raise RuntimeError(error_details)

            sleep(30)

    def get_job_arn(self: Self, job_name: str) -> str:
        """Returns job arn given the job name."""
        paginator = self.client.get_paginator("list_model_invocation_jobs")

        page_iterator = paginator.paginate(
            nameContains=job_name, sortOrder="Descending", sortBy="CreationTime"
        )

        job_arn_found = None
        for page in page_iterator:
            for job_summary in page.get("invocationJobSummaries", []):
                if job_summary.get("jobName") == job_name:
                    job_arn_found = job_summary.get("jobArn")
                    break
            if job_arn_found:
                break

        if not job_arn_found:
            msg = f"Bedrock Batch Job with name '{job_name}' not found)"
            raise ValueError(msg)

        return job_arn_found

    def create_job(  # noqa: PLR0913
        self: Self,
        job_name: str,
        role_arn: str,
        input_s3_file_url: str,
        output_s3_directory_url: str,
        conversations: list[list[LLMMessage]] | None = None,
        inference_cfg: LLMInferenceConfig | None = None,
        *,
        create_buckets_if_missing: bool,
    ) -> str:
        """Creates and invokes batch job."""
        # check to make sure that these parse and error check correctly
        input_bucket, input_bucket_key, output_bucket = self._parse_s3_urls(
            input_s3_file_url, output_s3_directory_url
        )

        self._ensure_bucket_exists(
            input_bucket,
            create_buckets_if_missing=create_buckets_if_missing,
        )
        self._ensure_bucket_exists(
            output_bucket,
            create_buckets_if_missing=create_buckets_if_missing,
        )

        # Upload files if conversations exist
        if conversations:
            if inference_cfg is None:
                inference_cfg = LLMInferenceConfig()
            model_specific_conversations = self._parse_conversations(conversations, inference_cfg)
            self._upload_conversations(model_specific_conversations, input_bucket, input_bucket_key)

        request = BatchLLMRequest(
            roleArn=role_arn,
            jobName=job_name,
            modelId=self.llm.model_id,
            inputDataConfig=InputDataConfig(
                s3InputDataConfig=S3InputDataConfig(s3Uri=input_s3_file_url)
            ),
            outputDataConfig=OutputDataConfig(
                s3OutputDataConfig=S3OutputDataConfig(s3Uri=output_s3_directory_url)
            ),
        )

        # invoke with request
        response = self.client.create_model_invocation_job(**request.model_dump(exclude_none=True))
        return response.get("jobArn")

    def _parse_s3_urls(
        self: Self, input_s3_file_url: str, output_s3_directory_url: str
    ) -> tuple[str, str, str]:
        """Return the bucket names and key(s) from an S3 URI."""
        if not input_s3_file_url.startswith("s3://") or not input_s3_file_url.startswith("s3://"):
            msg = (
                f"Invalid S3 URI {input_s3_file_url} or {output_s3_directory_url}."
                f"Must start with 's3://'."
            )
            raise ValueError(msg)
        if input_s3_file_url.endswith("/"):
            msg = (
                f"Invalid Input S3 URI {input_s3_file_url}. Must be a filename,"
                "but found a directory."
            )
            raise ValueError(msg)
        if not output_s3_directory_url.endswith("/"):
            msg = (
                f"Invalid Output S3 URI {output_s3_directory_url}. Must be a directory, but found a"
                f"filename. Make sure there is a '/' at the end of the url."
            )
            raise ValueError(msg)
        input_parts = input_s3_file_url[5:].split("/", 1)
        if len(input_parts) <= 1:
            msg = f"Invalid Input S3 URI {input_s3_file_url}. Must have a file path."
            raise ValueError(msg)
        input_bucket = input_parts[0]
        input_key = input_parts[1]

        output_parts = output_s3_directory_url[5:].split("/", 1)
        output_bucket = output_parts[0]

        return input_bucket, input_key, output_bucket

    def _ensure_bucket_exists(self: Self, bucket: str, *, create_buckets_if_missing: bool) -> None:
        """Ensure that the specified bucket exists and creates it if it doesn't and flag is true."""
        try:
            self.s3_client.head_bucket(Bucket=bucket)
        except ClientError as e:
            if create_buckets_if_missing:
                region = self.s3_client.meta.region_name
                if region == "us-east-1":
                    self.s3_client.create_bucket(Bucket=bucket)
                else:
                    region = cast("BucketLocationConstraintType", region)
                    self.s3_client.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={"LocationConstraint": region},
                    )
            else:
                msg = (
                    f"Bucket {bucket} does not exist or you do not have permissions,"
                    f" and create_buckets_if_missing flag is False."
                )
                raise ValueError(msg) from e

    def _parse_conversations(
        self: Self, conversations: list[list[LLMMessage]], inference_config: LLMInferenceConfig
    ) -> list[BatchRecordInput]:
        input_requests: list[BatchRecordInput] = []
        for i, conversation in enumerate(conversations):
            msg = self.llm.create_request(messages=conversation, config=inference_config)
            record = BatchRecordInput(
                recordId=f"RECORD{i:010d}",
                modelInput=msg,
            )
            input_requests.append(record)
        return input_requests

    def _upload_conversations(
        self: Self,
        conversations: list[BatchRecordInput],
        input_bucket: str,
        input_bucket_key: str,
    ) -> None:
        jsonl_str = "\n".join(
            conversation.model_dump_json(exclude_none=True, by_alias=True)
            for conversation in conversations
        )
        self.s3_client.put_object(Bucket=input_bucket, Key=input_bucket_key, Body=jsonl_str)
