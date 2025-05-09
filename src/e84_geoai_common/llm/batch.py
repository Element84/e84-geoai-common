from time import sleep
from typing import TYPE_CHECKING, Self, cast

import boto3
from botocore.exceptions import ClientError
from mypy_boto3_bedrock import BedrockClient
from mypy_boto3_s3 import S3Client
from pydantic import BaseModel, ConfigDict

from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMMessage
from e84_geoai_common.llm.models.claude import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
)
from e84_geoai_common.llm.models.nova import BedrockNovaLLM

if TYPE_CHECKING:
    from mypy_boto3_s3.literals import BucketLocationConstraintType

# Batch inference uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815

ValidBatchLLMs = BedrockClaudeLLM | BedrockNovaLLM


class BatchRecordInput[RequestModel: BaseModel](BaseModel):
    """Single input record for batch inference."""

    recordId: str
    modelInput: RequestModel


class BatchRecordOutput[RequestModel: BaseModel, ResponseModel: BaseModel](BaseModel):
    """Specific output record for Claude batch inference."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    recordId: str
    modelInput: RequestModel
    modelOutput: ResponseModel | None = None
    error: RequestModel | None = None


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


class BedrockBatchInference[RequestModel: BaseModel, ResponseModel: BaseModel]:
    def __init__(
        self,
        llm: ValidBatchLLMs,
        request_model: type[RequestModel],
        response_model: type[ResponseModel],
        bedrock_client: BedrockClient | None = None,
        s3_client: S3Client | None = None,
    ) -> None:
        """Initalizes BedrockBatchInference.

        Args:
            llm: valid LLM instance.
            request_model: Valid InvokeLLMRequest model (i.e. ClaudeInvokeLLMRequest)
            response_model: Valid Response model (i.e. ClaudeResponse)
            bedrock_client: Optional pre-initialized bedrock boto3 client. Defaults to None.
            s3_client: Optional pre-initialized s3 boto3 client. Defaults to None.
        """
        self.llm = llm or BedrockClaudeLLM(model_id=CLAUDE_BEDROCK_MODEL_IDS["Claude 3.5 Haiku"])
        self.request_model = request_model
        self.response_model = response_model
        self.bedrock_client = bedrock_client or boto3.client("bedrock")  # type: ignore[reportUnknownMemberType]
        self.s3_client = s3_client or boto3.client("s3")  # type: ignore[reportUnknownMemberType]
        self.batch_record_output_model = BatchRecordOutput[self.request_model, self.response_model]

    def get_results(
        self: Self, job_arn: str
    ) -> list[BatchRecordOutput[RequestModel, ResponseModel]]:
        """Returns the results of the job. Returns an error it is not done yet."""
        job_details = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
        status = job_details["status"]

        if status != "Completed":
            message = job_details["message"]
            error_details = (
                f"get_results called on {job_arn} when status is not 'Completed' "
                f"with status: {status}. Message: '{message}'."
            )
            raise RuntimeError(error_details)

        job_id = job_arn.split("/")[-1]

        input_data_config = job_details["inputDataConfig"]
        s3_input_config = input_data_config["s3InputDataConfig"]
        input_uri = s3_input_config["s3Uri"]
        input_filename = input_uri.split("/")[-1]

        output_data_config = job_details["outputDataConfig"]
        s3_output_config = output_data_config["s3OutputDataConfig"]
        output_uri = s3_output_config["s3Uri"]
        path_part = output_uri[5:]  # Remove s3://
        parts = path_part.split("/", 1)  # Split on first /

        output_bucket = parts[0]
        output_prefix_key = ""
        if len(parts) > 1 and parts[1]:
            output_prefix_key = parts[1]

        final_output_key = f"{output_prefix_key}{job_id}/{input_filename}.out"
        return self._validate_results(output_bucket, final_output_key)

    def _validate_results(
        self,
        output_bucket: str,
        key: str,
    ) -> list[BatchRecordOutput[RequestModel, ResponseModel]]:
        response = self.s3_client.get_object(Bucket=output_bucket, Key=key)
        response_body = response["Body"].read().decode("utf-8")
        output_messages = [
            self.batch_record_output_model.model_validate_json(line)
            for line in response_body.splitlines()
        ]

        return output_messages

    def wait_for_job_to_finish(self: Self, job_arn: str) -> None:
        """Returns once the job has either finished or failed."""
        while True:
            response = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
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
        paginator = self.bedrock_client.get_paginator("list_model_invocation_jobs")

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
        response = self.bedrock_client.create_model_invocation_job(
            **request.model_dump(exclude_none=True)
        )
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
    ) -> list[BatchRecordInput[RequestModel]]:
        input_requests: list[BatchRecordInput[RequestModel]] = []
        for i, conversation in enumerate(conversations):
            msg = self.llm.create_request(messages=conversation, config=inference_config)
            record = BatchRecordInput[self.request_model](
                recordId=f"RECORD{i:010d}",
                modelInput=self.request_model.model_validate(msg),
            )
            input_requests.append(record)
        return input_requests

    def _upload_conversations(
        self: Self,
        conversations: list[BatchRecordInput[RequestModel]],
        input_bucket: str,
        input_bucket_key: str,
    ) -> None:
        jsonl_str = "\n".join(
            conversation.model_dump_json(exclude_none=True, by_alias=True)
            for conversation in conversations
        )
        self.s3_client.put_object(Bucket=input_bucket, Key=input_bucket_key, Body=jsonl_str)
