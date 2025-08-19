from abc import ABC, abstractmethod

from e84_geoai_common.llm.core.llm import LLM, ExecutableLLMTool, LLMInferenceConfig


class Agent[LLMToolContext](ABC):
    """A language model with instructions and tools."""

    llm: LLM
    inference_cfg: LLMInferenceConfig
    tools: list[ExecutableLLMTool[LLMToolContext]]

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """The prompt template used by the agent."""
