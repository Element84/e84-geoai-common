import json
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field, PrivateAttr

from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMMessage, TextContent
from e84_geoai_common.llm.models.batch.iam_utils import create_role
from e84_geoai_common.llm.models.batch.models import (
    BATCH_BEDROCK_MODEL_IDS,
    MINIMUM_BATCH_ENTRIES,
    BatchLLMRequest,
    BatchLLMResults,
    BatchRecordInput,
    BatchRecordOutput,
    InputDataConfig,
    JobStatusResponse,
    JobSubmissionResponse,
    OutputDataConfig,
    S3InputDataConfig,
    S3OutputDataConfig,
)
from e84_geoai_common.llm.models.batch.s3_utils import (
    ensure_bucket_exists,
    load_job_arn,
    load_role_arn,
    parse_s3_uri,
    save_job_arn,
    save_role_arn,
)
from e84_geoai_common.llm.models.claude import (
    BedrockClaudeLLM,
)

# Batch inference uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815, N803, N806


class BatchLLM(BaseModel):
    """Wrapper for the Bedrock Batch Inference API."""

    # Internal fields.
    client: Any = Field(default_factory=lambda: boto3.client("bedrock"))  # type: ignore[reportUnknownMemberType]
    s3_client: Any = Field(default_factory=lambda: boto3.client("s3"))  # type: ignore[reportUnknownMemberType]
    create_buckets_if_missing: bool = False
    create_role_if_missing: bool = False

    # API payload fields.
    roleArn: str = ""
    jobName: str
    modelId: str
    inputDataConfig: InputDataConfig
    outputDataConfig: OutputDataConfig
    jobArn: str | None = Field(default=None)

    # Private attributes.
    _input_bucket: str = PrivateAttr()
    _input_key: str = PrivateAttr()
    _output_bucket: str = PrivateAttr()
    _output_key: str = PrivateAttr()

    def __init__(  # noqa: PLR0913
        self,
        *,
        jobName: str,
        inputS3Uri: str,
        outputS3Uri: str,
        roleArn: str | None = None,
        create_buckets_if_missing: bool = False,
        create_role_if_missing: bool = False,
    ) -> None:
        """Initialize the BatchLLM instance."""
        super().__init__(
            roleArn=roleArn or "",
            jobName=jobName,
            modelId=BATCH_BEDROCK_MODEL_IDS["Claude 3 Haiku"],
            inputDataConfig=InputDataConfig(s3InputDataConfig=S3InputDataConfig(s3Uri=inputS3Uri)),
            outputDataConfig=OutputDataConfig(
                s3OutputDataConfig=S3OutputDataConfig(s3Uri=outputS3Uri)
            ),
        )
        self.create_buckets_if_missing = create_buckets_if_missing
        self.create_role_if_missing = create_role_if_missing
        self._input_bucket, self._input_key = parse_s3_uri(inputS3Uri)
        self._output_bucket, self._output_key = parse_s3_uri(outputS3Uri)
        ensure_bucket_exists(
            self.s3_client,
            self._input_bucket,
            create_buckets_if_missing=self.create_buckets_if_missing,
        )
        ensure_bucket_exists(
            self.s3_client,
            self._output_bucket,
            create_buckets_if_missing=self.create_buckets_if_missing,
        )

        self._setup_role()

    def _setup_role(self) -> None:
        """Set up the IAM role for the job."""
        if self.roleArn:
            save_role_arn(self.s3_client, self._input_bucket, self.roleArn)
        else:
            try:
                self.roleArn = load_role_arn(self.s3_client, self._input_bucket)
            except Exception:  # noqa: BLE001
                if self.create_role_if_missing:
                    try:
                        self._create_role()
                    except Exception:  # noqa: BLE001
                        self.roleArn = ""
                else:
                    self.roleArn = ""

    def _create_role(self) -> None:
        """Create the IAM role with S3 access and save its ARN in the input bucket."""
        self.roleArn = create_role(self._input_bucket, self._output_bucket)
        save_role_arn(self.s3_client, self._input_bucket, self.roleArn)

    def upload_input_data(self, data: list[dict[str, Any]], key: str | None = None) -> None:
        """Upload a JSONL file to the input S3 bucket."""
        if len(data) < MINIMUM_BATCH_ENTRIES:
            raise ValueError("Input data must contain at least 100 records.")
        if key is None:
            key = f"{self.jobName}/input.jsonl"
        jsonl_str = "\n".join(json.dumps(record) for record in data)
        self.s3_client.put_object(Bucket=self._input_bucket, Key=key, Body=jsonl_str)
        self.inputDataConfig.s3InputDataConfig.s3Uri = f"s3://{self._input_bucket}/{key}"

    def count_input_records(self) -> int:
        """Count lines in the input S3 file."""
        bucket, key = parse_s3_uri(self.inputDataConfig.s3InputDataConfig.s3Uri)
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read().decode("utf-8")
        count = 0
        for line in body.splitlines():
            if line.strip():
                count += 1
                if count >= MINIMUM_BATCH_ENTRIES:
                    break
        return count

    def _save_job_arn_to_s3(self) -> None:
        """Save the job ARN to a JSON file in the input S3 bucket.

        The file is saved under <jobName>/job_arn.json.
        """
        if not self.jobArn:
            raise ValueError("No job ARN to save.")
        save_job_arn(self.s3_client, self._input_bucket, self.jobName, self.jobArn)

    def _load_job_arn_from_s3(self) -> str:
        """Load the job ARN from the JSON file in the input S3 bucket.

        The file should be under <jobName>/job_arn.json.
        """
        return load_job_arn(self.s3_client, self._input_bucket, self.jobName)

    def invoke_model_with_request(self) -> JobSubmissionResponse:
        """Submit the batch job using a BatchLLMRequest and save the job ARN."""
        if not self.roleArn:
            return JobSubmissionResponse(
                status="Failed",
                message="No valid IAM role ARN found in the input S3 bucket.",
                jobArn=None,
            )
        request = BatchLLMRequest(
            roleArn=self.roleArn,
            jobName=self.jobName,
            modelId=self.modelId,
            inputDataConfig=self.inputDataConfig,
            outputDataConfig=self.outputDataConfig,
        )
        try:
            response = self.client.create_model_invocation_job(
                **request.model_dump(exclude_none=True)
            )
            self.jobArn = response.get("jobArn")
            self._save_job_arn_to_s3()
            return JobSubmissionResponse(
                status="Submitted",
                message="Job submitted successfully.",
                jobArn=self.jobArn,
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ConflictException":
                return JobSubmissionResponse(
                    status="Error",
                    message="The provided job name is currently in use. Use a unique job name.",
                    jobArn=None,
                )
            return JobSubmissionResponse(
                status="Error",
                message=f"Error submitting job: {e}",
                jobArn=None,
            )

    def run_batch_job(self) -> JobSubmissionResponse:
        """Verify input records and submit the batch job."""
        record_count = self.count_input_records()
        if record_count < MINIMUM_BATCH_ENTRIES:
            msg = (
                f"Input data must contain at least {MINIMUM_BATCH_ENTRIES} records. "
                f"Found only {record_count} valid records."
            )
            raise ValueError(msg)
        return self.invoke_model_with_request()

    def get_status(self, jobArn: str | None = None) -> JobStatusResponse:
        """Retrieve the job status from the Batch API."""
        if jobArn is None:
            jobArn = self._load_job_arn_from_s3()
        self.jobArn = jobArn
        try:
            status = self.client.get_model_invocation_job(jobIdentifier=self.jobArn)["status"]
            return JobStatusResponse(
                status=status,
                message="Job status retrieved successfully.",
                jobArn=self.jobArn,
            )
        except ClientError as e:
            return JobStatusResponse(
                status="Error",
                message=f"Error retrieving job status: {e}",
                jobArn=self.jobArn,
            )

    def retrieve_results(self) -> BatchLLMResults:
        """Retrieve the output results from the output S3 bucket."""
        if not self.jobArn:
            self.jobArn = self._load_job_arn_from_s3()
        status_resp = self.get_status(self.jobArn)
        current_status = status_resp.status
        if current_status != "Completed":
            return BatchLLMResults(
                responses=[],
                status=current_status,
                message="Job is not completed yet. Please try again later.",
            )
        job_id = self.jobArn.split("/")[-1]
        result_key = f"{self._output_key.rstrip('/')}/{job_id}/input.jsonl.out"
        try:
            response = self.s3_client.get_object(Bucket=self._output_bucket, Key=result_key)
            result_body = response["Body"].read().decode("utf-8")
            responses_list: list[BatchRecordOutput] = []
            for line in result_body.splitlines():
                if line.strip():
                    record_obj = BatchRecordOutput.model_validate_json(line)
                    responses_list.append(record_obj)
            return BatchLLMResults(
                responses=responses_list,
                status="Completed",
                message="Results retrieved successfully.",
            )
        except ClientError as e:
            return BatchLLMResults(
                responses=[],
                status="Error",
                message=f"Error retrieving results: {e}",
            )

    def dummy_llm_requests(self, total_records: int = 100) -> list["BatchRecordInput"]:
        """Make a hundred dummy LLM requests."""
        msg = [
            LLMMessage(
                role="user",
                content=[TextContent(text="What is 10+10?")],
            )
        ]
        llm = BedrockClaudeLLM()
        claude_msg = llm._create_request(messages=msg, config=LLMInferenceConfig())  # type: ignore  # noqa: PGH003, SLF001
        records: list[BatchRecordInput] = []
        for i in range(total_records):
            record = BatchRecordInput(
                recordId=f"CALL{i:010d}",
                modelInput=claude_msg,
            )
            records.append(record)
        return records


if __name__ == "__main__":
    # Initialize the BatchLLM instance.

    batch = BatchLLM(
        jobName="test8",
        inputS3Uri="s3://brian-batch-test-input-8/",
        outputS3Uri="s3://brian-batch-test-output-8/",
        create_buckets_if_missing=True,
        create_role_if_missing=True,
    )

    # Generate and upload input data.
    example_llm_requests = batch.dummy_llm_requests(total_records=100)
    batch.upload_input_data([r.model_dump(exclude_none=True) for r in example_llm_requests])
    batch_run_response = batch.run_batch_job()
    print("Job submission response:", batch_run_response)  # noqa: T201

    # check job status.
    status_response = batch.get_status()
    print("Job status response:", status_response)  # noqa: T201

    # Retrieve job results.
    results_response = batch.retrieve_results()
    print("Results retrieved:", results_response)  # noqa: T201
