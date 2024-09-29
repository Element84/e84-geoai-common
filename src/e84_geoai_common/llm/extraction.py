from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from e84_geoai_common.llm.core import LLM, InvokeLLMRequest, LLMMessage


Model = TypeVar("Model", bound=BaseModel)


class ExtractDataExample(BaseModel, Generic[Model]):
    """
    Represents an example data extraction scenario that can be used for building system prompts for
    data extraction.

    Attributes:
    - name (str): Name of the example.
    - user_query (str): User's query for data extraction.
    - structure (Model): Data structure to extract.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    name: str
    user_query: str
    structure: Model

    def to_str(self) -> str:
        """
        Returns a formatted string representation of the example data extraction scenario.

        Returns:
        str: Formatted string with example name, user query, and data structure in JSON format.
        """
        query_json = f"```json\n{self.structure.model_dump_json(indent=2, exclude_none=True)}\n```"
        return f'Example: {self.name}\nUser Query: "{self.user_query}"\n\n{query_json}'


def extract_data_from_text(
    *,
    llm: LLM,
    model_type: type[Model],
    system_prompt: str,
    user_prompt: str,
) -> Model:
    """
    Extracts data from text using a Language Model (LLM) by providing system and user prompts.

    Args:
    - llm (LLM): The Language Model instance used for data extraction.
    - model_type (Type[Model]): The type of data model to be used for validation.
    - system_prompt (str): The prompt for the system to process the user input.
    - user_prompt (str): The user input text for data extraction.

    Returns:
    Model: The extracted data model validated against the specified model type.
    """
    request = InvokeLLMRequest(
        system=system_prompt, json_mode=True, messages=[LLMMessage(content=user_prompt)]
    )
    resp = llm.invoke_model_with_request(request)
    try:
        return model_type.model_validate_json(resp)
    except ValidationError as e:
        print("Unable to parse response:", resp)
        raise e
