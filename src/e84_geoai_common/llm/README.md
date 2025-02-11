# Usage

TODO update these docs

## LLM

```py
from e84_geoai_common.llm.core import LLMInferenceConfig, LLMMessage
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM


def get_temperature_celsius() -> float:
    """Returns the current temperature in celsius."""
    return 23.5


llm = BedrockClaudeLLM()

inference_cfg = LLMInferenceConfig(tools=[get_temperature_celsius])
messages = [
    LLMMessage(role="user", content="What is the current temperature?")
]

llm.prompt(
    messages=messages,
    inference_cfg=inference_cfg,
    auto_use_tools=True,
)

# Output:
# ClaudeAssistantMessage(role='assistant', content=[ClaudeTextContent(type='text', text='The current temperature is 23.5 degrees Celsius.')])
```
