from moto import mock_aws

from e84_geoai_common.llm.models.converse import (
    BedrockConverseLLM,
    ConverseResponse,
    ConverseTextContent,
    ConverseUsageInfo,
)


@mock_aws
def test_response_prefix():
    prefix = "__prefix__"
    llm = BedrockConverseLLM()
    content_in = [ConverseTextContent(text="abc"), ConverseTextContent(text="def")]  # noqa: E501
    response_in = ConverseResponse(
        id="",
        content=content_in,
        model="",
        stopReason="end_turn",
        stopSequence=None,
        usage=ConverseUsageInfo(inputTokens=0, outputTokens=0, totalTokens=0),
    )
    response_out = llm._add_prefix_to_response(response_in, prefix=prefix)
    content_out = response_out.content
    assert len(content_out) == len(content_in)
    assert content_out[0].text.startswith(prefix)
    assert content_out[1].text == content_in[1].text
