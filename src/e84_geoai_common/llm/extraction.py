import logging

from pydantic import BaseModel, ConfigDict, ValidationError

from e84_geoai_common.llm.core import (
    LLM,
    LLMInferenceConfig,
    LLMMessage,
    LLMTool,
    LLMToolChoice,
    LLMToolUseContent,
)

log = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Custom exception for errors during data extraction from LLM responses."""


class ExtractDataExample[Model: BaseModel](BaseModel):
    """Example data extraction scenario.

    Attributes:
        name (str): Name of the example.
        user_query (str): User's query for data extraction.
        structure (Model): Data structure to extract.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    name: str
    user_query: str
    structure: Model

    def to_str(self) -> str:
        """Return formatted string representation of the example scenario.

        Returns:
                str: Formatted string with example name, user query, and data
                structure in JSON format.
        """
        json_str = self.structure.model_dump_json(exclude_none=True)
        query_json = f"```json\n{json_str}\n```"
        return f'Example: {self.name}\nUser Query: "{self.user_query}"\n\n{query_json}'


# FUTURE: use built-in structured output support if available
def extract_data_from_text[Model: BaseModel](
    *,
    llm: LLM,
    model_type: type[Model],
    system_prompt: str,
    user_prompt: str,
) -> Model:
    """Extract data from text using an LLM given system and user prompts.

    Args:
        llm (LLM): The Language Model instance used for data extraction.
        model_type (Type[Model]): The type of data model to be used for
            validation.
        system_prompt (str): The prompt for the system to process the user
            input.
        user_prompt (str): The user input text for data extraction.

    Returns:
        Model: The extracted data model validated against the specified model
            type.
    """
    inference_cfg = LLMInferenceConfig(
        system_prompt=system_prompt,
        tools=[
            LLMTool(
                name="parse_data",
                description="Parse the user input into the specified data structure.",
                input_model=model_type,
                output_model=None,
            ),
        ],
        tool_choice=LLMToolChoice(mode="force_specific_tool_use", tool_name="parse_data"),
    )
    messages = [LLMMessage(role="user", content=user_prompt)]
    resp = llm.prompt(messages=messages, inference_cfg=inference_cfg)
    try:
        for content in resp.content:
            if isinstance(content, LLMToolUseContent) and content.name == "parse_data":
                return model_type.model_validate(content.input)
    except (ValidationError, TypeError) as e:
        log.exception("Unable to parse LLM response: %s", resp)
        raise ExtractionError(f"Unable to parse LLM response: {resp}") from e

    raise ExtractionError(f"LLM response did not contain tool use content: {resp}")
