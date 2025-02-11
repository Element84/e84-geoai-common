import json

from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.extraction import extract_data_from_text
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM
from e84_geoai_common.llm.tests.mock_bedrock import (
    MockBedrockRuntimeClient,
    claude_response_with_content,
)


class _ShoppingItem(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    name: str = Field(description="The name of an item to buy in single case not plural.")
    quantity: int


class _ShoppingList(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    items: list[_ShoppingItem]


system_prompt = f"""
Extract a shopping list from the request following this schema.

{json.dumps(_ShoppingList.model_json_schema(), indent=2)}
"""


def test_extract():
    user_prompt = "I need four apples and three bananas."
    expected_result = _ShoppingList(
        items=[
            _ShoppingItem(name="apple", quantity=4),
            _ShoppingItem(name="banana", quantity=3),
        ]
    )

    # Drop the opening { as claude will return it.
    mock_resp_json = expected_result.model_dump_json()[1:]
    client = MockBedrockRuntimeClient([claude_response_with_content(mock_resp_json)])
    llm = BedrockClaudeLLM(client=client)
    result = extract_data_from_text(
        llm=llm,
        model_type=_ShoppingList,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    assert result == expected_result
