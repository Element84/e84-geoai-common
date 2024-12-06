from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from e84_geoai_common.llm.core.llm import LLM, LLMInferenceConfig


class Agent(ABC):
    """A language model with instructions and tools."""

    llm: LLM
    inference_cfg: LLMInferenceConfig
    tools: list[Callable[..., Any]]

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """The prompt template used by the agent."""
