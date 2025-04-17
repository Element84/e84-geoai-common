from time import sleep
from typing import Any, Self, cast

import boto3
from botocore.exceptions import ClientError
from mypy_boto3_bedrock.client import BedrockClient
from mypy_boto3_s3.client import S3Client
from mypy_boto3_s3.literals import BucketLocationConstraintType
from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core.llm import LLM, LLMInferenceConfig, LLMMessage
from e84_geoai_common.llm.models.claude import CLAUDE_BEDROCK_MODEL_IDS, BedrockClaudeLLM

# Batch inference uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815


class BatchRecordInput(BaseModel):
    """Single input record for batch inference."""

    recordId: str
    # this could also be ClaudeInvokeLLMRequest | NovaInvokeLLMRequest (not Converse though)
    modelInput: Any


class BatchRecordOutput(BaseModel):
    """Single batch result model."""

    recordId: str
    # This could also be ClaudeInvokeLLMRequest | NovaInvokeLLMRequest
    modelInput: dict[str, Any]
    # This could also be ClaudeResponse... etc.
    modelOutput: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class BatchLLMResults(BaseModel):
    """All batch results model."""

    responses: list[BatchRecordOutput]


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


class Job(BaseModel):
    arn: str


class BedrockBatchLLM(BaseModel):
    # LLM is an abstract, non-pydantic field. arbitrary_types_allowed is needed or else it errors.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LLM = Field(
        default_factory=lambda: BedrockClaudeLLM(
            model_id=CLAUDE_BEDROCK_MODEL_IDS["Claude 3.5 Haiku"]
        )
    )
    client: BedrockClient = Field(default_factory=lambda: boto3.client("bedrock"))  # type: ignore[reportUnknownMemberType]
    s3_client: S3Client = Field(default_factory=lambda: boto3.client("s3"))  # type: ignore[reportUnknownMemberType]

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
        if not input_uri:
            raise ValueError("Input S3 URI not found.")
        input_filename = input_uri.split("/")[-1]

        output_data_config = job_details.get("outputDataConfig", {})
        s3_output_config = output_data_config.get("s3OutputDataConfig", {})
        output_uri = s3_output_config.get("s3Uri")
        if not output_uri:
            raise ValueError("Output S3 URI not found.")
        path_part = output_uri[5:]  # Remove s3://
        parts = path_part.split("/", 1)  # Split on first /

        output_bucket = parts[0]
        output_prefix_key = ""
        if len(parts) > 1 and parts[1]:
            output_prefix_key = parts[1]

        final_output_key = f"{output_prefix_key}{job_id}/{input_filename}.out"

        response = self.s3_client.get_object(Bucket=output_bucket, Key=final_output_key)
        response_body = response["Body"].read().decode("utf-8")

        responses_list: list[BatchRecordOutput] = []
        for line in response_body.splitlines():
            if line.strip():
                record_obj = BatchRecordOutput.model_validate_json(line)
                responses_list.append(record_obj)

        return BatchLLMResults(
            responses=responses_list,
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
    ) -> Job:
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
        return Job(arn=response.get("jobArn"))

    def _parse_s3_urls(
        self: Self, input_s3_file_url: str, output_s3_directory_url: str
    ) -> tuple[str, str, str]:
        """Return the bucket names and key(s) from an S3 URI."""
        if not input_s3_file_url.startswith("s3://") or not input_s3_file_url.startswith("s3://"):
            raise ValueError("Invalid S3 URI; must start with 's3://'.")
        if input_s3_file_url.endswith("/"):
            raise ValueError("Invalid Input S3 URI; must be a filename, but found a directory.")
        if not output_s3_directory_url.endswith("/"):
            raise ValueError("Invalid Output S3 URI; must be a directory, but found a filename.")
        input_parts = input_s3_file_url[5:].split("/", 1)
        if len(input_parts) <= 1:
            raise ValueError("Invalid Input S3 URI; must have a file path")
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
                    region = cast(BucketLocationConstraintType, region)
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
            # create_request isn't recognized
            msg = self.llm.create_request(messages=conversation, config=inference_config)  # type: ignore  # noqa: PGH003
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
