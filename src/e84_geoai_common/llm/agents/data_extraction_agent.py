import json
from string import Template
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

from llm.core import LLM, Agent, LLMInferenceConfig, LLMMessage

if TYPE_CHECKING:
    from llm.core import LLM

DATA_EXTRACTION_PROMPT_TEMPLATE = Template("""\
Extract relevant information from the text provided, following the schema \
given below.

Schema:
${schema}

Text:
${text}
""")

ModelT = TypeVar("ModelT", bound=BaseModel)


class DataExtractionAgent(Agent, Generic[ModelT]):
    """Extracts structured information from text following a schema."""

    def __init__(
        self,
        llm: "LLM",
        data_model: type[ModelT],
        inference_cfg: LLMInferenceConfig | None = None,
    ) -> None:
        self.llm = llm
        self.tools = []
        self.data_model = data_model
        if inference_cfg is None:
            inference_cfg = LLMInferenceConfig(
                system_prompt="You are an LLM specializing in extracting "
                "structured information from text.",
                temperature=0,
                json_mode=True,
            )
        self.inference_cfg = inference_cfg
        model_json_schema = json.dumps(
            data_model.model_json_schema(), indent=2
        )
        # partially fill out template
        self._prompt_template: Template = Template(
            DATA_EXTRACTION_PROMPT_TEMPLATE.safe_substitute(
                schema=model_json_schema
            )
        )

    def run(self, text: str) -> ModelT:
        """Extract information from the given text."""
        prompt = self._prompt_template.safe_substitute(text=text)
        response = self.llm.prompt(
            messages=[LLMMessage(role="user", content=prompt)],
            inference_cfg=self.inference_cfg,
            auto_use_tools=True,
        )
        response = response[-1]
        if isinstance(response.content, str):
            output_json = response.content
        else:
            output_json = str(response.content[-1])
        out = self.data_model.model_validate_json(output_json)
        return out

    @property
    def prompt(self) -> str:
        return self._prompt_template.template
