from moto import mock_aws

from e84_geoai_common.llm.models.nova import (
    BedrockNovaLLM,
    NovaResponse,
    NovaTextContent,
    NovaUsageInfo,
)


@mock_aws
def test_response_prefix():
    prefix = "__prefix__"
    llm = BedrockNovaLLM()
    content_in = [NovaTextContent(text="abc"), NovaTextContent(text="def")]
    response_in = NovaResponse(
        id="",
        content=content_in,
        model="",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=NovaUsageInfo(input_tokens=0, output_tokens=0),
    )
    response_out = llm._add_prefix_to_response(response_in, prefix=prefix)
    content_out = response_out.content
    assert len(content_out) == len(content_in)
    assert content_out[0].text.startswith(prefix)
    assert content_out[1].text == content_in[1].text
