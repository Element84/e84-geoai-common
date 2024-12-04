from moto import mock_aws

from llm.models.claude import (
    BedrockClaudeLLM,
    ClaudeResponse,
    ClaudeTextContent,
    ClaudeUsageInfo,
)


@mock_aws
def test_response_prefix():
    prefix = "__prefix__"
    llm = BedrockClaudeLLM()
    content_in = [ClaudeTextContent(text="abc"), ClaudeTextContent(text="def")]
    response_in = ClaudeResponse(
        id="",
        content=content_in,
        model="",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=ClaudeUsageInfo(input_tokens=0, output_tokens=0),
    )
    response_out = llm._add_prefix_to_response(response_in, prefix=prefix)
    content_out = response_out.content
    assert len(content_out) == len(content_in)
    assert content_out[0].text.startswith(prefix)
    assert content_out[1].text == content_in[1].text
