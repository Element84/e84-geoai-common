import logging

from pydantic import BaseModel, ConfigDict, ValidationError

from e84_geoai_common.llm.core import LLM, LLMInferenceConfig, LLMMessage

log = logging.getLogger(__name__)


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
        json_str = self.structure.model_dump_json(indent=2, exclude_none=True)
        query_json = f"```json\n{json_str}\n```"
        return f'Example: {self.name}\nUser Query: "{self.user_query}"\n\n{query_json}'


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
        json_mode=True,
    )
    messages = [LLMMessage(role="user", content=user_prompt)]
    resp = llm.prompt(messages=messages, inference_cfg=inference_cfg)
    try:
        out = model_type.model_validate_json(resp.to_text_only())
    except (ValidationError, TypeError):
        log.exception("Unable to parse LLM response: %s", resp)
        raise
    return out
