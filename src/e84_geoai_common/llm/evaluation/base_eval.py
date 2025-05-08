import logging
from abc import ABC, abstractmethod
from typing import Any

import fsspec  # type: ignore[reportMissingTypeStubs]
from pydantic import BaseModel

log = logging.getLogger(__name__)


class ExperimentConfig(BaseModel):
    """Name and parameters of the experiment."""

    name: str
    params: dict[str, Any]


class EvalInput(BaseModel):
    """A single evaluation test input and associated reference output."""

    id: str
    input: str | dict[str, Any]
    reference_output: str | dict[str, Any]
    attrs: dict[str, Any] | None


class EvalDataset(BaseModel):
    """An evaluation dataset."""

    items: list[EvalInput]


class SingleEvalResult(BaseModel):
    """Evaluation result for a single test case."""

    input: EvalInput
    output: str | dict[str, Any] | None
    metrics: dict[str, Any]


class EvalReport(BaseModel):
    """Evaluation report for an experiment."""

    experiment_config: ExperimentConfig
    results: list[SingleEvalResult]
    aggregated_metrics: dict[str, Any]


class BaseEvaluator(ABC):
    """Base eval class."""

    @abstractmethod
    def eval_dataset(self, experiment_config: ExperimentConfig, dataset: EvalDataset) -> EvalReport:
        """Evaluate a dataset using the given experiment config."""

    def save_eval_report(self, eval_report: EvalReport, path: str) -> None:
        """Save eval report as a JSON file.

        Args:
            eval_report: The eval report.
            path: Local or remote URI to write the JSON file to.
        """
        log.info("Writing eval report to %s", path)
        fs, path = fsspec.url_to_fs(path, auto_mkdir=True)  # type: ignore[reportUnknownMemberType]
        with fs.open(path, "w") as f:
            f.write(eval_report.model_dump_json(indent=2))
