from contextlib import nullcontext as does_not_raise

import pytest
from pydantic import ValidationError

from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMTool


class TestLLMInferenceConfig:
    def test_json_mode_validation(self):
        with does_not_raise():
            _ = LLMInferenceConfig(json_mode=True, tools=None)

        with does_not_raise():
            _ = LLMInferenceConfig(json_mode=True, tools=[])

        tool = LLMTool(
            name="SomeTool",
            description="",
            input_model=None,
            output_model=None,
            execution_func=None,
        )
        with pytest.raises(ValidationError):
            _ = LLMInferenceConfig(json_mode=True, tools=[tool])
