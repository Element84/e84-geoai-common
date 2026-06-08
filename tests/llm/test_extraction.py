import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel

from e84_geoai_common.llm.extraction import extract_data_from_text
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM
from e84_geoai_common.llm.tests.mock_bedrock_runtime import (
    claude_response_with_content,
    make_test_bedrock_runtime_client,
)


def test_extract():
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

    mock_resp_json = expected_result.model_dump_json()
    client = make_test_bedrock_runtime_client([claude_response_with_content(mock_resp_json)])
    llm = BedrockClaudeLLM(client=client)
    result = extract_data_from_text(
        llm=llm,
        model_type=ShoppingList,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    assert result.model_dump(mode="json") == expected_result.model_dump(mode="json")


def test_extract_with_union_of_basic_types_at_root():
    class StringOrList(RootModel[str | list[str]]):
        pass

    system_prompt = """
    Split the given delimited string. Return a single string if there is only one item, otherwise,
    a list.
    """
    user_prompt = "apple|banana|orange"
    expected_result = StringOrList(root=["apple", "banana", "orange"])

    mock_resp_json = expected_result.model_dump_json()
    client = make_test_bedrock_runtime_client([claude_response_with_content(mock_resp_json)])
    llm = BedrockClaudeLLM(client=client)
    result = extract_data_from_text(
        llm=llm,
        model_type=StringOrList,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    assert result.model_dump(mode="json") == expected_result.model_dump(mode="json")


def test_extract_with_union_of_models_at_root():
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
        [claude_response_with_content([{"text": json.dumps({"data": expected_result})}])]
    )
    llm = BedrockClaudeLLM(client=client)
    result = extract_data_from_text(
        llm=llm,
        model_type=Item,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    assert result.model_dump(mode="json") == expected_result
