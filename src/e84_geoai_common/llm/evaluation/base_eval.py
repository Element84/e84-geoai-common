import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import fsspec  # type: ignore[reportMissingTypeStubs]
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)


class ExperimentConfig[ParamsT: BaseModel](BaseModel):
    """Name and parameters of the experiment."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    name: str
    params: ParamsT


class SingleEvalCase[
    InputT: str | BaseModel,
    RefOutputT: str | BaseModel,
    AttrsT: BaseModel | None,
](BaseModel):
    """A single evaluation test input and associated reference output."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    id: str
    input: InputT
    reference_output: RefOutputT
    attrs: AttrsT


class SingleEvalResult[
    EvalInputT: SingleEvalCase[Any, Any, Any],
    OutputT: str | BaseModel,
    MetricsT: str | BaseModel,
](BaseModel):
    """Evaluation result for a single test case."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    input: EvalInputT
    output: OutputT
    metrics: MetricsT


class EvalReport[
    ExperimentConfigT: ExperimentConfig[Any],
    SingleEvalResultT: SingleEvalResult[Any, Any, Any],
    AggMetricsT: BaseModel,
](BaseModel):
    """Evaluation report for an experiment."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    experiment_config: ExperimentConfigT
    results: list[SingleEvalResultT]
    aggregated_metrics: AggMetricsT


class BaseEvaluator(ABC):
    """Base eval class."""

    @abstractmethod
    def eval_dataset(
        self,
        experiment_config: ExperimentConfig[Any],
        dataset: Iterable[SingleEvalCase[Any, Any, Any]],
    ) -> EvalReport[Any, Any, Any]:
        """Evaluate a dataset using the given experiment config."""

    def save_eval_report(self, eval_report: EvalReport[Any, Any, Any], path: str) -> None:
        """Save eval report as a JSON file.

        Args:
            eval_report: The eval report.
            path: Local or remote URI to write the JSON file to.
        """
        log.info("Writing eval report to %s", path)
        fs, path = fsspec.url_to_fs(path, auto_mkdir=True)  # type: ignore[reportUnknownMemberType]
        with fs.open(path, "w") as f:
            f.write(eval_report.model_dump_json(indent=2))
