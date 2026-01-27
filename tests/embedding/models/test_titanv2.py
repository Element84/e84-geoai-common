from e84_geoai_common.embedder.embedder import EmbedderInferenceConfig, EmbedderInput
from e84_geoai_common.embedder.titan_v2 import TitanV2
from e84_geoai_common.llm.tests.mock_bedrock_runtime import (
    make_test_bedrock_runtime_client,
    titanv2_response,
)
from tests.embedding.models.constants import HELLO_WORLD_EMBEDDED_512


def test_temp():
    embedder = TitanV2(
        client=make_test_bedrock_runtime_client(
            [
                titanv2_response(
                    HELLO_WORLD_EMBEDDED_512,
                    {
                        "inputTextTokenCount": 4,
                        "embeddingsByType": {"float": HELLO_WORLD_EMBEDDED_512},
                    },
                )
            ]
        )
    )

    embedder_input = EmbedderInput(input_text="Hello world!")

    embedding_response = embedder.embed(
        embedder_input=embedder_input, config=EmbedderInferenceConfig(dimensions=512)
    )

    assert embedding_response.embedding == HELLO_WORLD_EMBEDDED_512
    assert embedding_response.input_text_token_count == 4
    assert len(embedding_response.embedding) == 512
