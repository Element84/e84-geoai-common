# Usage

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

## Agent

```py
from pydantic import BaseModel, Field
from rich import print

from e84_geoai_common.llm.agents.data_extraction_agent import DataExtractionAgent
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM


class ExtractedLocation(BaseModel):
    """A location extracted from a piece of text."""

    name: str = Field(
        ...,
        description="Name or address of the location.",
        examples=[
            "New York",
            "Germany",
            "1435 Walnut St Philadelphia, PA, USA",
        ],
    )
    role: str = Field(
        ...,
        description="The role of the location in the text.",
        examples=[
            "The city the game was played in.",
            "Country whose national team participated in the game.",
        ],
    )


class ExtractedLocations(BaseModel):
    """List of locations extracted from a piece of text. Can be empty."""

    locations: list[ExtractedLocation] = []


llm = BedrockClaudeLLM()
extraction_agent = DataExtractionAgent(llm, ExtractedLocations)

# source: https://en.wikinews.org/wiki/New_Zealand_defeats_South_Africa_to_win_2024_women%27s_T20_cricket_world_cup
text = """\
On October 20, New Zealand won the 2024 ICC Women's T20 World Cup, defeating South Africa by 32 runs in the tournament's final at the Dubai International Cricket Stadium. This marked the ninth edition of the tournament and New Zealand's first Women's T20 World Cup title.

South Africa won the toss and chose to field first. New Zealand, batting first, posted a total of 158 runs for the loss of five wickets. Amelia Kerr was the top scorer with 43 runs, while Brooke Halliday contributed 38 runs. Nonkululeko Mlaba was the most successful bowler for South Africa, taking two wickets.

In their chase, South Africa struggled to keep up with the required run rate, managing 126 runs for the loss of nine wickets. Laura Wolvaardt, the South African captain, scored 33 runs from 27 balls, but the team could not reach the target. For New Zealand, both Amelia Kerr and Rosemary Mair took three wickets each.

Amelia Kerr was named both Player of the Match and Player of the Tournament for her all-round performance with both bat and ball.

After the match, South African captain Laura Wolvaardt commented, "We had a really good semi-final, the focus was to reset, but we didn't nail our cricket today." New Zealand's Amelia Kerr expressed her thoughts, saying, "I'm a little bit speechless and I'm just so stoked to get the win after all the team has been through."\
"""  # noqa: E501

data = extraction_agent.run(text)
print(data)

# Output:
# ExtractedLocations(
#     locations=[
#         ExtractedLocation(name='New Zealand', role='The country whose national team won the tournament.'),
#         ExtractedLocation(
#             name='South Africa',
#             role='The country whose national team participated in the final and lost.'
#         ),
#         ExtractedLocation(name='Dubai International Cricket Stadium', role='The venue where the final was played.')
#     ]
# )
```
