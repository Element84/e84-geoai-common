from time import sleep
from typing import Any, Literal, cast

import boto3
from mypy_boto3_bedrock import BedrockClient
from mypy_boto3_s3 import S3Client
from pydantic import BaseModel, ConfigDict

from e84_geoai_common.embedder.embedder import (
    EmbedderInferenceConfig,
    EmbedderInput,
    EmbedderResponse,
)
from e84_geoai_common.embedder.titan_v2 import TitanV2
from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMMessage
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM
from e84_geoai_common.llm.models.nova import BedrockNovaLLM
from e84_geoai_common.util import ensure_bucket_exists

# Batch inference uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815
MIN_RECORDS_PER_BATCH_JOB = 100
DEFAULT_MAX_RECORDS_PER_BATCH_JOB = 50000

ValidBatchLLMs = BedrockClaudeLLM | BedrockNovaLLM
ValidEmbedders = TitanV2  # Placeholder for future embedder support

# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ModelInvocationJobSummary.html
BatchStatus = Literal[
    "Submitted",
    "Validating",
    "Scheduled",
    "Expired",
    "InProgress",
    "Completed",
    "PartiallyCompleted",
    "Failed",
    "Stopped",
    "Stopping",
]


class BatchInputItem(BaseModel):
    """Preliminary input record for batch before applying model."""

    record_id: str | None = None
    model_input: list[LLMMessage] | EmbedderInput


class BatchRecordInput[RequestModel: BaseModel](BaseModel):
    """Single input record for batch inference."""

    recordId: str
    modelInput: RequestModel


class BatchResponseError(BaseModel):
    """Error model for batch response."""

    errorCode: int
    errorMessage: str
    expired: bool
    retryable: bool


class BatchRecordOutput[RequestModel: BaseModel, ResponseModel: BaseModel](BaseModel):
    """Specific output record for Claude batch inference."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    recordId: str
    modelInput: RequestModel
    modelOutput: ResponseModel | None = None
    error: BatchResponseError | None = None


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
        llm: ValidBatchLLMs | ValidEmbedders,
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
        self.llm = llm
        self.request_model = request_model
        self.response_model = response_model
        self.bedrock_client = bedrock_client or boto3.client("bedrock")  # type: ignore[reportUnknownMemberType]
        self.s3_client = s3_client or boto3.client("s3")  # type: ignore[reportUnknownMemberType]
        self.batch_record_output_model = BatchRecordOutput[self.request_model, self.response_model]

    def get_results(self, job_arn: str) -> list[BatchRecordOutput[RequestModel, ResponseModel]]:
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

        output_uri = job_details["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]

        output_bucket, output_prefix_key = output_uri.replace("s3://", "").split("/", 1)

        response = self.s3_client.list_objects_v2(
            Bucket=output_bucket, Prefix=f"{output_prefix_key}{job_id}/"
        )
        output_uris = [
            obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith("jsonl.out")
        ]

        results: list[BatchRecordOutput[RequestModel, ResponseModel]] = []
        for output_uri in output_uris:
            output_filename = output_uri.split("/")[-1]

            final_output_key = f"{output_prefix_key}{job_id}/{output_filename}"
            results.extend(self._validate_results(output_bucket, final_output_key))
        return results

    def get_llm_results_raw(
        self, job_arn: str
    ) -> list[BatchRecordOutput[RequestModel, LLMMessage]]:
        """Returns the results of the job as LLMMessages. Returns an error it is not done yet."""
        raw_results = self.get_results(job_arn)

        if not isinstance(self.llm, ValidBatchLLMs):
            msg = "get_llm_results_raw can only be called when using an LLM model."
            raise TypeError(msg)

        llm_message_results: list[BatchRecordOutput[RequestModel, LLMMessage]] = []
        for result in raw_results:
            model_output = result.modelOutput
            if model_output is not None:
                model_output = self.llm.response_to_llm_message(
                    model_output,  # pyright: ignore[reportArgumentType]
                    LLMInferenceConfig(),
                )

            llm_result = BatchRecordOutput[self.request_model, LLMMessage](
                recordId=result.recordId,
                modelInput=result.modelInput,
                modelOutput=model_output,
                error=result.error,
            )
            llm_message_results.append(llm_result)

        return llm_message_results

    def get_embedder_results_raw(
        self, job_arn: str
    ) -> list[BatchRecordOutput[RequestModel, EmbedderResponse]]:
        """Returns the results of the job as EmbedderInput. Returns an error it is not done yet."""
        raw_results = self.get_results(job_arn)

        if not isinstance(self.llm, ValidEmbedders):
            msg = "get_embedder_results_raw can only be called when using an Embedder model."
            raise TypeError(msg)

        embedder_results: list[BatchRecordOutput[RequestModel, EmbedderResponse]] = []
        for result in raw_results:
            model_output = result.modelOutput
            if model_output is not None:
                model_output = self.llm.response_to_embedder_response(
                    model_output,  # pyright: ignore[reportArgumentType]
                )

            embedder_result = BatchRecordOutput[self.request_model, EmbedderResponse](
                recordId=result.recordId,
                modelInput=result.modelInput,
                modelOutput=model_output,
                error=result.error,
            )
            embedder_results.append(embedder_result)

        return embedder_results

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

    @staticmethod
    def results_contains_errors(
        results: list[BatchRecordOutput[RequestModel, ResponseModel]],
    ) -> bool:
        """Returns True if any of the results contain errors."""
        return any(result.error is not None for result in results)

    def get_job_status(self, job_arn: str) -> BatchStatus:
        """Returns True if the job is finished, False otherwise."""
        response = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
        return response.get("status")

    def is_completed(self, job_arn: str) -> bool:
        """Returns True if the job is finished, False otherwise."""
        status = self.get_job_status(job_arn)
        return status == "Completed"

    def is_failed(self, job_arn: str) -> bool:
        """Returns True if the job is a terminal state and failed, False otherwise."""
        status = self.get_job_status(job_arn)
        return status in ["Failed", "Stopping", "Stopped"]

    def wait_for_job_to_finish(self, job_arn: str, poll_interval_seconds: int = 30) -> None:
        """Returns once the job has either finished or failed."""
        while True:
            response = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
            status: BatchStatus = response.get("status")

            if status == "Completed":
                break

            if status in ["Failed", "Stopping", "Stopped"]:
                message = response.get("message")
                error_details = (
                    f"Job {job_arn} finished with status: {status}. Message: '{message}'."
                )
                raise RuntimeError(error_details)

            sleep(poll_interval_seconds)

    def get_job_arn(self, job_name: str) -> str:
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
        self,
        job_name: str,
        role_arn: str,
        input_s3_file_url: str,
        output_s3_directory_url: str,
        conversations: list[list[LLMMessage]] | list[BatchInputItem] | None = None,
        inference_cfg: LLMInferenceConfig | EmbedderInferenceConfig | None = None,
        *,
        max_records_per_file: int = 10000,
        max_records_per_job: int = DEFAULT_MAX_RECORDS_PER_BATCH_JOB,  # Bedrock default hard limit
        create_buckets_if_missing: bool = False,
    ) -> str:
        """Creates and invokes batch job."""
        # check to make sure that these parse and error check correctly
        input_bucket, input_bucket_key, output_bucket = self._parse_s3_urls(
            input_s3_file_url, output_s3_directory_url
        )

        input_directory_given = False
        if input_bucket_key == "" or input_bucket_key.endswith("/"):
            input_directory_given = True

        ensure_bucket_exists(
            self.s3_client, input_bucket, create_buckets_if_missing=create_buckets_if_missing
        )
        ensure_bucket_exists(
            self.s3_client, output_bucket, create_buckets_if_missing=create_buckets_if_missing
        )

        # Upload files if conversations exist
        if conversations:
            if len(conversations) > max_records_per_job:
                msg = (
                    f"Number of records {len(conversations)} exceeds the maximum allowed"
                    f" {max_records_per_job}."
                )
                raise ValueError(msg)

            if len(conversations) < MIN_RECORDS_PER_BATCH_JOB:
                msg = (
                    f"Number of records {len(conversations)} is less than the minimum required"
                    f" {MIN_RECORDS_PER_BATCH_JOB}."
                )
                raise ValueError(msg)

            if inference_cfg is None:
                inference_cfg = LLMInferenceConfig()

            model_specific_conversations = self._parse_conversations(conversations, inference_cfg)

            if len(conversations) > max_records_per_file:
                if input_directory_given is False:
                    msg = (
                        "You have more records than the records_per_file limit, but the"
                        "input uri is not a directory so multiple files cannot be created."
                    )
                    raise ValueError(msg)

                for file_idx in range(0, len(model_specific_conversations), max_records_per_file):
                    chunk = model_specific_conversations[file_idx : file_idx + max_records_per_file]
                    chunk_input_key = (
                        f"{input_bucket_key}part{file_idx // max_records_per_file}.jsonl"
                    )
                    self._upload_conversations(chunk, input_bucket, chunk_input_key)
            else:
                if input_directory_given:
                    input_bucket_key = f"{input_bucket_key}input.jsonl"
                self._upload_conversations(
                    model_specific_conversations, input_bucket, input_bucket_key
                )

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
        self, input_s3_file_url: str, output_s3_directory_url: str
    ) -> tuple[str, str, str]:
        """Return the bucket names and key(s) from an S3 URI."""
        if not input_s3_file_url.startswith("s3://") or not output_s3_directory_url.startswith(
            "s3://"
        ):
            msg = (
                f"Invalid S3 URI {input_s3_file_url} or {output_s3_directory_url}."
                f"Must start with 's3://'."
            )
            raise ValueError(msg)
        if not output_s3_directory_url.endswith("/"):
            msg = (
                f"Invalid Output S3 URI {output_s3_directory_url}. Must be a directory, but found a"
                f"filename. Make sure there is a '/' at the end of the url."
            )
            raise ValueError(msg)
        input_parts = input_s3_file_url[5:].split("/", 1)
        input_key = "" if len(input_parts) <= 1 else input_parts[1]
        input_bucket = input_parts[0]

        output_parts = output_s3_directory_url[5:].split("/", 1)
        output_bucket = output_parts[0]

        return input_bucket, input_key, output_bucket

    def _parse_conversations(
        self,
        conversations: list[list[LLMMessage]] | list[BatchInputItem],
        inference_config: LLMInferenceConfig | EmbedderInferenceConfig,
    ) -> list[BatchRecordInput[RequestModel]]:
        input_requests: list[BatchRecordInput[RequestModel]] = []
        for i, conversation in enumerate(conversations):
            record_id = f"RECORD{i:010d}"

            if isinstance(conversation, BatchInputItem):
                messages = conversation.model_input
                if conversation.record_id:
                    record_id = conversation.record_id
            else:
                messages = conversation

            if isinstance(self.llm, ValidBatchLLMs):
                messages = cast("list[LLMMessage]", messages)
                inference_config = cast("LLMInferenceConfig", inference_config)
                msg: Any = self.llm.create_request(messages=messages, config=inference_config)
            else:
                input_text = cast("EmbedderInput", messages)
                inference_config = cast("EmbedderInferenceConfig", inference_config)
                msg: Any = self.llm.create_request(
                    embedder_input=input_text, config=inference_config
                )

            record = BatchRecordInput[self.request_model](
                recordId=record_id,
                modelInput=self.request_model.model_validate(msg),
            )
            input_requests.append(record)
        return input_requests

    def _upload_conversations(
        self,
        conversations: list[BatchRecordInput[RequestModel]],
        input_bucket: str,
        input_bucket_key: str,
    ) -> None:
        jsonl_str = "\n".join(
            conversation.model_dump_json(exclude_none=True, by_alias=True)
            for conversation in conversations
        )
        self.s3_client.put_object(Bucket=input_bucket, Key=input_bucket_key, Body=jsonl_str)
