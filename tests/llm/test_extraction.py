import json
from collections.abc import Callable
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel

from e84_geoai_common.llm.extraction import (
    extract_data_from_text,
    extract_data_from_text_claude_structured_output,
)
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM
from e84_geoai_common.llm.tests.mock_bedrock_runtime import (
    claude_response_with_content,
    make_test_bedrock_runtime_client,
)


@pytest.mark.parametrize(
    "extraction_func",
    [extract_data_from_text, extract_data_from_text_claude_structured_output],
)
def test_extract(extraction_func: Callable[..., BaseModel]):
    class ShoppingItem(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")

        name: str = Field(description="The name of an item to buy in single case not plural.")
        quantity: int

    class ShoppingList(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")

        items: list[ShoppingItem]

    system_prompt = f"""
    Extract a shopping list from the request following this schema.

    {json.dumps(ShoppingList.model_json_schema(), indent=2)}
    """
    user_prompt = "I need four apples and three bananas."
    expected_result = ShoppingList(
        items=[
            ShoppingItem(name="apple", quantity=4),
            ShoppingItem(name="banana", quantity=3),
        ]
    )

    client = make_test_bedrock_runtime_client(
        [_make_mock_response(extraction_func.__name__, expected_result.model_dump(mode="json"))]
    )
    llm = BedrockClaudeLLM(client=client)
    result = extraction_func(
        llm=llm,
        model_type=ShoppingList,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    assert result.model_dump(mode="json") == expected_result.model_dump(mode="json")


@pytest.mark.parametrize(
    "extraction_func",
    [extract_data_from_text, extract_data_from_text_claude_structured_output],
)
def test_extract_with_union_of_basic_types_at_root(extraction_func: Callable[..., BaseModel]):
    class StringOrList(RootModel[str | list[str]]):
        pass

    system_prompt = """Parse a semicolon-delimited string:
- Split the input string by semicolons (;)
- Strip whitespace from each item
- If there is exactly 1 item, return it as a single string
- If there are 2 or more items, return them as a list of strings
    """
    user_prompt = "apple;banana;orange"
    expected_result = StringOrList(root=["apple", "banana", "orange"])

    client = make_test_bedrock_runtime_client(
        [
            _make_mock_response(
                extraction_func.__name__,
                expected_result.model_dump(mode="json"),
                wrap=extraction_func.__name__ == "extract_data_from_text",
            )
        ]
    )
    llm = BedrockClaudeLLM(client=client)
    result = extraction_func(
        llm=llm,
        model_type=StringOrList,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    assert result.model_dump(mode="json") == expected_result.model_dump(mode="json")


@pytest.mark.parametrize(
    "extraction_func",
    [extract_data_from_text, extract_data_from_text_claude_structured_output],
)
def test_extract_with_union_of_models_at_root(
    extraction_func: Callable[..., BaseModel],
):
    class ItemA(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")

        type: Literal["A"] = "A"
        value_a: str

    class ItemB(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")

        type: Literal["B"] = "B"
        value_b: int

    class Item(RootModel[ItemA | ItemB]):
        pass

    system_prompt = f"""
    Extract an item from the request following this schema.

    {json.dumps(Item.model_json_schema(), indent=2)}
    """
    user_prompt = "I have an item of type A with value 'foo'."
    expected_result = {
        "type": "A",
        "value_a": "foo",
    }

    client = make_test_bedrock_runtime_client(
        [_make_mock_response(extraction_func.__name__, expected_result, wrap=True)]
    )
    llm = BedrockClaudeLLM(client=client)
    result = extraction_func(
        llm=llm,
        model_type=Item,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    assert result.model_dump(mode="json") == expected_result


def _make_mock_response(
    extraction_func_name: str, expected_result: dict[str, Any], *, wrap: bool = False
) -> dict[str, Any]:
    if wrap:
        expected_result = {
            "data": expected_result,
        }
    match extraction_func_name:
        case "extract_data_from_text_claude_structured_output":
            return claude_response_with_content([{"text": json.dumps(expected_result)}])
        case "extract_data_from_text":
            return claude_response_with_content(
                [
                    {
                        "id": "tool-use-1",
                        "name": "parse_data",
                        "input": expected_result,
                    }
                ],
                overrides={"stop_reason": "tool_use"},
            )
        case _:
            raise ValueError(f"Unexpected extraction function: {extraction_func_name}")
