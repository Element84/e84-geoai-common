import logging
from collections.abc import Iterable

from pydantic import BaseModel

from e84_geoai_common.llm.evaluation.base_eval import (
    BaseEvaluator,
    EvalReport,
    ExperimentConfig,
    SingleEvalCase,
    SingleEvalResult,
)

log = logging.getLogger(__name__)


class ExampleTaskExperimentParams(BaseModel):
    model: str
    temperature: float


class ExampleTaskInput(BaseModel):
    query: str


class ExampleTaskRefOutput(BaseModel):
    response: str


class ExampleTaskOutput(BaseModel):
    response: str


class ExampleTaskAttrs(BaseModel):
    input_attrs: dict[str, int]
    output_attrs: dict[str, int]


class ExampleTaskMetrics(BaseModel):
    format_compliant: bool
    correct: bool


class ExampleTaskAggMetrics(BaseModel):
    pct_format_compliant: float
    pct_correct: float


ExampleTaskExperimentConfig = ExperimentConfig[ExampleTaskExperimentParams]
ExampleTaskSingleEvalCase = SingleEvalCase[ExampleTaskInput, ExampleTaskRefOutput, ExampleTaskAttrs]
ExampleTaskSingleEvalResult = SingleEvalResult[
    ExampleTaskSingleEvalCase, ExampleTaskOutput, ExampleTaskMetrics
]
ExampleTaskEvalReport = EvalReport[
    ExampleTaskExperimentConfig, ExampleTaskSingleEvalResult, ExampleTaskAggMetrics
]

eval_dataset = [
    ExampleTaskSingleEvalCase(
        id="1",
        input=ExampleTaskInput(query="?"),
        reference_output=ExampleTaskRefOutput(response="!"),
        attrs=ExampleTaskAttrs(input_attrs={"a": 1}, output_attrs={"a": 1}),
    ),
    ExampleTaskSingleEvalCase(
        id="2",
        input=ExampleTaskInput(query="???"),
        reference_output=ExampleTaskRefOutput(response="!!!"),
        attrs=ExampleTaskAttrs(input_attrs={"a": 3}, output_attrs={"a": 3}),
    ),
]


class ExampleTaskEvaluator(BaseEvaluator):
    """ExampleTask eval class."""

    def eval_dataset(
        self,
        experiment_config: ExampleTaskExperimentConfig,
        dataset: Iterable[ExampleTaskSingleEvalCase],
    ) -> ExampleTaskEvalReport:
        """Evaluate a dataset using the given experiment config."""
        raise NotImplementedError
