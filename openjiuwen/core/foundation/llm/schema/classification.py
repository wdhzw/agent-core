# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Typed decision primitives for classification-style model calls.

A classification call asks a model typed questions about a piece of state
and gets back typed answers with calibrated confidence, instead of the
messages-in/free-text-out shape used by chat completion (see
``schema.message``/``schema.generation_response``). Mirrors the
``Noul``/``Choice``/``Score`` primitives and response shapes documented at
https://docs.typesafe.ai/primitives/{noul,choice,score} (TypeSafe's Jev
System One API) field-for-field, including the fields most callers won't
need on every question (``NoulQuestion.criteria``) or every answer
(``probabilities``, ``legend``).
"""

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from openjiuwen.core.foundation.llm.schema.generation_response import GenerationResponse

# TypeSafe documents ``instructions`` and criteria/legend text as
# "a string, object, or array" on every primitive (see
# https://docs.typesafe.ai/api.md and .../primitives/{noul,choice,score}),
# not plain strings. Keep that union everywhere the wire protocol allows it
# instead of narrowing to ``str``, so structured instructions/criteria
# round-trip instead of failing pydantic validation.
RichText = Union[str, dict[str, Any], list[Any]]


class NoulCriteria(BaseModel):
    """What counts as a yes and what counts as a no, for a ``NoulQuestion``."""

    true: RichText = Field(description="What counts as a yes")
    false: RichText = Field(description="What counts as a no")


class NoulQuestion(BaseModel):
    """Asks "is this true?". The answer is a calibrated probability of yes."""

    instructions: RichText = Field(description="The yes/no question, or a statement to judge")
    criteria: Optional[NoulCriteria] = Field(default=None, description="Optional yes/no guidance")


class ChoiceQuestion(BaseModel):
    """Asks "which of these options?". The answer is a choice plus a confidence."""

    instructions: RichText = Field(description="What the question is asking, in natural language")
    criteria: dict[str, RichText] = Field(description="Option name -> description of when it applies")


class ScoreQuestion(BaseModel):
    """Asks "which level?". The answer is a position on a ladder plus a confidence."""

    instructions: RichText = Field(description="What the question is asking, in natural language")
    criteria: list[RichText] = Field(description="Ordered levels, from the low end of the scale to the high end")


ClassificationQuestion = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


class NoulAnswer(BaseModel):
    """Answer to a ``NoulQuestion``: a calibrated probability that the answer is yes."""

    type: Literal["noul"] = "noul"
    noul: float = Field(description="Probability that the answer is yes, in [0, 1]")


class ChoiceAnswer(BaseModel):
    """Answer to a ``ChoiceQuestion``: the selected option plus the full probability distribution."""

    type: Literal["choice"] = "choice"
    choice: str = Field(description="The option with the highest probability")
    confidence: float = Field(description="How peaked probabilities is; low when it's spread across options")
    probabilities: dict[str, float] = Field(description="Full probability distribution across every option")


class ScoreAnswer(BaseModel):
    """Answer to a ``ScoreQuestion``: a fractional position on the level ladder plus its distribution."""

    type: Literal["score"] = "score"
    score: float = Field(description="Probability-weighted mean of the level numbers; can fall between two levels")
    confidence: float = Field(description="How peaked probabilities is; low when it's spread across levels")
    probabilities: dict[int, float] = Field(description="Probability of each level, keyed by level number")
    legend: dict[int, Union[str, dict[str, Any]]] = Field(
        description="Every level number mapped back to its description "
        "(the original criterion string, or the full object for structured criteria)"
    )


ClassificationAnswer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer],
    Field(discriminator="type"),
]


class ClassificationUsage(BaseModel):
    """Token usage (and, when routed through a billing gateway, cost) for one classification call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost: Optional[float] = Field(default=None, description="Set when routed through a billing gateway")

    model_config = ConfigDict(extra="allow")


class ClassificationResponse(GenerationResponse):
    """Response to a classification call: state + typed questions -> typed answers.

    Mirrors the ``GenerationResponse`` family (``ImageGenerationResponse``,
    etc.) but for classification-style providers, which answer typed
    questions instead of generating content. ``answers`` is keyed by the
    question id from the request, same as the TypeSafe wire response (not
    grouped by answer type -- that grouping is a ``langchain-typesafe``
    convenience on top of this shape, not part of the wire protocol).
    """

    id: Optional[str] = Field(default=None, description="Provider-assigned request/generation ID")
    provider: Optional[str] = Field(default=None, description="Provider that served the request")
    answers: dict[str, ClassificationAnswer] = Field(
        default_factory=dict, description="Answers keyed by the question id they respond to"
    )
    usage: ClassificationUsage = Field(default_factory=ClassificationUsage)

    model_config = ConfigDict(extra="allow")
