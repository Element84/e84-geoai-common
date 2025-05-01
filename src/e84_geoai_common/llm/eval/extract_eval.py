"""Defines core code for evaluating the ability to extract data using an LLM."""

import concurrent.futures
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core import LLM

logger = logging.getLogger(__name__)
_TEMP_DIR = Path("temp")


def _load_template(name: str) -> str:
    path = Path(__file__).parent / "templates" / name

    with path.open() as f:
        return f.read()


_EVAL_SUCCESS_TEMPLATE = _load_template("single_eval_success_result.md")
_EVAL_DIFF_TEMPLATE = _load_template("single_eval_diff_result.md")
_FULL_EVAL_TEMPLATE = _load_template("full_eval_result.md")


# TODO REVIEWER Note: One thing I want to avoid with this whole thing is building something that's
# too generic and complicated with inheritance like LangChain. As much as possible this should be
# more like a general library where it's usable in parts and easily understood. It may be a bit too
# generic for my taste currently.

# Thinking this through. Core parts of Evaluation are the following

# Example:
#  - user input
#  - expected result
# Evaluation Result:
#  - Example the result was evaluating
#  - Actual result
#  - Metrics for quality of the evaluation
# EvaluationResults:
#  - A list of the evaluation results
#  - Overall metrics of the entire suite

# We want the ability to do the following:
# - Create a suite of examples for testing
# - Support any kind of pydantic model result
# - Customize the evaluation
# - Customize the metric


TNode = TypeVar("TNode", bound=BaseModel)


class EvalMetric(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    tree_distance: float = Field(
        description=(
            "The tree edit distance between expected and actual nodes. 0 means they are identical."
        )
    )
    diff_explanations: list[str] = Field(
        description="Human-readable explanations of differences between nodes"
    )


class AggregateMetric(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    num_total: int
    num_success: int
    num_failed: int
    num_error: int

    total_tree_distance: float = Field(description=("The combined tree edit distance of all nodes"))


TMetric = TypeVar("TMetric", bound=EvalMetric)

TAggregateMetric = TypeVar("TAggregateMetric", bound=AggregateMetric)


class ExampleEval(BaseModel, Generic[TNode]):
    """A single example evaluating an LLM.

    It defines the input user text and the expected parsed object from the LLM response.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    user_text: str = Field(description="The input user text")
    description: str | None = Field(description="A description of the example.", default=None)
    expected_node: TNode = Field(description="The expected parsed node")


class SingleEvalResult(BaseModel, Generic[TNode, TMetric]):
    """The result of evaluating a single example."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    example: ExampleEval[TNode] = Field(description="The original example to evaluate against")

    error: Exception

    actual: TNode | None = Field(
        description="The actual node produced by the model unless there was an error"
    )

    metric: TMetric | None = Field(
        description="The metric indicating the success of the generation unless there was an error"
    )


class EvalResults(BaseModel, Generic[TNode, TMetric, TAggregateMetric]):
    """The result of evaluating a set of examples."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    evaluations: list[SingleEvalResult[TNode, TMetric]] = Field(
        description="A list of the evaluation results"
    )

    summary_metric: TAggregateMetric = Field(
        description="A metric describing the summary of the results"
    )


class Evaluator(ABC, Generic[TNode, TMetric, TAggregateMetric]):
    """Provides a base class for evaluating the ability to extract data from natural language."""

    @abstractmethod
    def parse(self, llm: LLM, user_text: str) -> TNode:
        """Subclases implement this to parse the text given an LLM and return the parsed object."""
        ...

    @abstractmethod
    def combine_metrics(self, metrics: Iterable[TMetric]) -> TAggregateMetric:
        """TODO docs."""
        ...

    @abstractmethod
    def create_metric(self, node1: TNode, node2: TNode) -> TMetric:
        """TODO docs."""
        ...

    def evaluate(self, llm: LLM, example: ExampleEval[TNode]) -> SingleEvalResult[TNode, TMetric]:
        """Evaluates a single example."""
        logger.info("Evaluating example [%s]", example.user_text)
        actual = self.parse(llm, example.user_text)
        metric = self.create_metric(actual, example.expected_node)
        return SingleEvalResult(example=example, actual=actual, metric=metric)

    def evaluate_examples(
        self, llm: LLM, examples: Sequence[ExampleEval[TNode]], *, max_concurrent: int = 5
    ) -> EvalResults[TNode, TMetric, TAggregateMetric]:
        """Evaluates all of the examples defined in this module in parallel."""
        indexed_evaluations: list[tuple[int, SingleEvalResult[TNode, TMetric]]] = []

        def eval_with_index(
            llm: LLM, example: ExampleEval[TNode], index: int
        ) -> tuple[int, SingleEvalResult[TNode, TMetric]]:
            return (index, self.evaluate(llm, example))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all evaluation tasks to the executor
            future_to_example = {
                executor.submit(eval_with_index, llm, example, index): example
                for index, example in enumerate(examples)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_example):
                example = future_to_example[future]
                try:
                    index, evaluation = future.result()
                    indexed_evaluations.append((index, evaluation))
                except Exception:
                    logger.exception("Evaluation failed for example '%s'", example.user_text)
                    raise

        evaluations = [
            evaluation for _, evaluation in sorted(indexed_evaluations, key=lambda i_e: i_e[0])
        ]

        return EvalResults(
            # Return the evaluations in order
            evaluations=evaluations,
            summary_metric=self.combine_metrics([e.metric for e in evaluations]),
        )
