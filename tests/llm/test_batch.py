import time

import pytest
from moto import mock_aws
from mypy_boto3_s3 import S3Client

from e84_geoai_common.llm.batch import BatchInputItem, BedrockBatchInference
from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMMessage, TextContent
from e84_geoai_common.llm.models.claude import (
    CLAUDE_3_5_HAIKU,
    BedrockClaudeLLM,
    ClaudeInvokeLLMRequest,
    ClaudeResponse,
    ClaudeTextContent,
)
from e84_geoai_common.llm.models.nova import (
    BedrockNovaLLM,
    NovaInvokeLLMRequest,
    NovaResponse,
    NovaTextContent,
)
from e84_geoai_common.llm.tests.mock_bedrock import (
    BATCH_IAM_ROLE_ARN,
    BATCH_INPUT_S3,
    BATCH_OUTPUT_S3,
    USE_REAL_BATCH_BEDROCK_CLIENT,
    batch_claude_output_example,
    batch_nova_output_example,
    make_test_bedrock_client,
)
from e84_geoai_common.llm.tests.mock_bedrock_runtime import make_test_bedrock_runtime_client


@pytest.fixture
def s3_moto_client():
    """Moto S3 mocking."""
    if not USE_REAL_BATCH_BEDROCK_CLIENT:
        with mock_aws():
            yield
    else:
        yield


def test_claude_create_and_run_job(
    s3_moto_client: S3Client,
):
    timestamp_ns = time.time_ns()
    job_name = f"pytest-job-{timestamp_ns}"

    bedrock_runtime_client = make_test_bedrock_runtime_client([])
    llm = BedrockClaudeLLM(model_id=CLAUDE_3_5_HAIKU, client=bedrock_runtime_client)
    llm_question = "What is 10+10?"
    llm_response = "10 + 10 = 20"

    conversations: list[BatchInputItem] = []
    for i in range(100):
        conversation = BatchInputItem(
            record_id=f"EXAMPLE_RECORD_{i}",
            model_input=[
                LLMMessage(
                    role="user",
                    content=[TextContent(text=llm_question)],
                )
            ],
        )
        conversations.append(conversation)
    batch_response = batch_claude_output_example(
        llm_question, llm_response, record_id_example="EXAMPLE_RECORD_1"
    )

    batch = BedrockBatchInference(
        request_model=ClaudeInvokeLLMRequest,
        response_model=ClaudeResponse,
        bedrock_client=make_test_bedrock_client(batch_response=batch_response, job_name=job_name),
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

    for _, result in enumerate(results_response[:5]):
        assert result.modelOutput is not None
        assert isinstance(result.modelOutput, ClaudeResponse)
        assert result.modelOutput.content[0] == ClaudeTextContent(text=llm_response)
        assert result.recordId.startswith("EXAMPLE_RECORD_")


def test_nova_create_and_run_job(
    s3_moto_client: S3Client,
):
    timestamp_ns = time.time_ns()
    job_name = f"pytest-job-{timestamp_ns}"

    bedrock_runtime_client = make_test_bedrock_runtime_client([])
    llm = BedrockNovaLLM(client=bedrock_runtime_client)
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

    batch = BedrockBatchInference(
        request_model=NovaInvokeLLMRequest,
        response_model=NovaResponse,
        bedrock_client=make_test_bedrock_client(batch_response=batch_response, job_name=job_name),
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

    for _, result in enumerate(results_response[:5]):
        assert result.modelOutput is not None
        assert isinstance(result.modelOutput, NovaResponse)
        assert result.modelOutput.output.message.content[0] == NovaTextContent(text=llm_response)
        assert result.recordId.startswith("RECORD")  # default value
