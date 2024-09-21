from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMMessage(BaseModel):
    """
    Represents a single message sent to or received from an LLM.

    "assistant" refers to the LLM.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    role: Literal["user", "assistant"] = "user"

    # FUTURE: This could be changed to allow for multiple items following the anthropic content style
    content: str


class InvokeLLMRequest(BaseModel):
    """Represents a request to invoke an LLM and get a response back."""

    model_config = ConfigDict(strict=True, extra="forbid")

    system: str | None = Field(default=None, description="System Prompt")
    max_tokens: int = Field(default=1000, description="Maximum number of output tokens")
    temperature: float = Field(
        default=0, description="Temperature control for randomness"
    )
    top_p: float = Field(default=0, description="Top P for nucleus sampling")
    top_k: int = Field(default=0, description="Top K for sampling")
    json_mode: bool = Field(default=False, description="Turn on/off json mode")
    messages: list[LLMMessage] = Field(
        default_factory=list, description="List of LLM Messages"
    )


class LLM(ABC):
    """
    An abstract base class for interacting with an LLM.
    """

    def invoke_model(
        self,
        *,
        user_prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0,
        top_p: float = 0,
        top_k: int = 0,
        json_mode: bool = False,
    ) -> str:
        """
        This function prepares a request to invoke an LLM and receives the response back.

        Parameters:
        user_prompt (str): The user's prompt to the LLM.
        system (str): An optional system prompt.
        max_tokens (int): Defines the maximum number of output tokens. Default is 1000 tokens.
        temperature (float): Controls randomness in the model's output. A value of 0 means deterministic. Default is 0.
        top_p (float): Defines the cumulative probability below which the possible next tokens are discarded. Default is 0.
        top_k (int): Chooses the next token from the top K probable tokens. Default is 0 which implies no limit.
        json_mode (bool): A flag to specify JSON mode. If True, the model outputs in JSON mode. Default is False.

        Returns:
        str: The LLM's response as a string.
        """
        return self.invoke_model_with_request(
            InvokeLLMRequest(
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                json_mode=json_mode,
                messages=[LLMMessage(content=user_prompt)],
            )
        )

    @abstractmethod
    def invoke_model_with_request(self, request: InvokeLLMRequest) -> str:
        """Invokes the model with the given request"""
        ...
