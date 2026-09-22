# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unit tests for JevModelClient: request/response shape against TypeSafe's
documented wire protocol (https://docs.typesafe.ai/primitives/{noul,choice,score})."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.foundation.llm.model_clients.jev_model_client import JevModelClient
from openjiuwen.core.foundation.llm.schema.classification import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulCriteria,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)


def _make_client(**overrides) -> JevModelClient:
    kwargs = {
        "api_key": "test-key",
        "api_base": "https://api.typesafe.ai",
        "model_name": "jev-1.13",
        **overrides,
    }
    return JevModelClient(**kwargs)


def _mock_async_client(response_json: dict) -> AsyncMock:
    mock_response = MagicMock()
    mock_response.json.return_value = response_json
    mock_response.raise_for_status = MagicMock()

    mock_async_client = AsyncMock()
    mock_async_client.post.return_value = mock_response
    mock_async_client.__aenter__.return_value = mock_async_client
    mock_async_client.__aexit__.return_value = False
    return mock_async_client


class TestJevModelClientConstruction:
    def test_requires_api_key(self):
        with pytest.raises(BaseError):
            JevModelClient(api_key="", api_base="https://api.typesafe.ai")

    def test_requires_api_base(self):
        with pytest.raises(BaseError):
            JevModelClient(api_key="test-key", api_base="")


class TestJevModelClientSystemOne:
    @pytest.mark.asyncio
    async def test_noul_answer(self):
        client = _make_client()
        response_json = {
            "model": "jev-1.13.0",
            "answers": {"is_human_escalation": {"type": "noul", "noul": 0.99}},
            "usage": {"input_tokens": 360, "output_tokens": 39},
        }
        mock_async_client = _mock_async_client(response_json)

        with patch("httpx.AsyncClient", return_value=mock_async_client) as mock_ctor:
            result = await client.system_one(
                state="I have asked three times now. Can I please just talk to a real person?",
                questions={
                    "is_human_escalation": NoulQuestion(instructions="Is the customer asking for a human agent?"),
                },
            )

        mock_ctor.assert_called_once()
        call_args = mock_async_client.post.call_args
        assert call_args.args[0] == "https://api.typesafe.ai/v1/systemone"
        sent_payload = call_args.kwargs["json"]
        assert sent_payload["model"] == "jev-1.13"
        assert sent_payload["questions"]["is_human_escalation"] == {
            "type": "noul",
            "instructions": "Is the customer asking for a human agent?",
        }
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"

        answer = result.answers["is_human_escalation"]
        assert answer.type == "noul"
        assert answer.noul == 0.99
        assert result.usage.input_tokens == 360

    @pytest.mark.asyncio
    async def test_noul_question_sends_optional_criteria(self):
        client = _make_client()
        mock_async_client = _mock_async_client(
            {
                "model": "jev-1.13.0",
                "answers": {"is_repeat_contact": {"type": "noul", "noul": 0.93}},
            }
        )

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            await client.system_one(
                state="I have asked three times now.",
                questions={
                    "is_repeat_contact": NoulQuestion(
                        instructions="Has the customer contacted support about this before?",
                        criteria=NoulCriteria(
                            true="Mentions a prior attempt, ticket, or that they have asked before",
                            false="No sign of any previous contact",
                        ),
                    ),
                },
            )

        sent_questions = mock_async_client.post.call_args.kwargs["json"]["questions"]
        assert sent_questions["is_repeat_contact"]["criteria"] == {
            "true": "Mentions a prior attempt, ticket, or that they have asked before",
            "false": "No sign of any previous contact",
        }

    @pytest.mark.asyncio
    async def test_choice_answer_includes_probabilities(self):
        client = _make_client()
        response_json = {
            "model": "jev-1.13.0",
            "answers": {
                "department": {
                    "type": "choice",
                    "choice": "returns",
                    "confidence": 1.0,
                    "probabilities": {"shipping": 0.0, "returns": 1.0, "billing": 0.0},
                },
            },
        }
        mock_async_client = _mock_async_client(response_json)

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            result = await client.system_one(
                state="My running shoes arrived in the wrong size. Can I swap them for a size 10?",
                questions={
                    "department": ChoiceQuestion(
                        instructions="Which team should handle this?",
                        criteria={
                            "returns": "Exchanges, wrong or damaged items",
                            "shipping": "Delivery status, delays, lost packages",
                            "billing": "Charges, invoices, payment problems",
                        },
                    ),
                },
            )

        answer = result.answers["department"]
        assert isinstance(answer, ChoiceAnswer)
        assert answer.choice == "returns"
        assert answer.confidence == 1.0
        assert answer.probabilities == {"shipping": 0.0, "returns": 1.0, "billing": 0.0}

    @pytest.mark.asyncio
    async def test_score_answer_is_fractional_with_probabilities_and_legend(self):
        client = _make_client()
        response_json = {
            "model": "jev-1.13.0",
            "answers": {
                "bug_severity": {
                    "type": "score",
                    "score": 1.43,
                    "confidence": 0.35,
                    "legend": {
                        "0": "Cosmetic; no impact to functionality",
                        "1": "Broken or degraded feature, but workaround exists",
                        "2": "Blocking issue; no workaround exists",
                    },
                    "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43},
                },
            },
        }
        mock_async_client = _mock_async_client(response_json)

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            result = await client.system_one(
                state="The export button crashes the settings page in Safari.",
                questions={
                    "bug_severity": ScoreQuestion(
                        instructions="How severe is the reported issue?",
                        criteria=[
                            "Cosmetic; no impact to functionality",
                            "Broken or degraded feature, but workaround exists",
                            "Blocking issue; no workaround exists",
                        ],
                    ),
                },
            )

        answer = result.answers["bug_severity"]
        assert isinstance(answer, ScoreAnswer)
        assert answer.score == 1.43
        assert isinstance(answer.score, float)
        assert answer.probabilities == {0: 0.0, 1: 0.57, 2: 0.43}
        assert answer.legend[1] == "Broken or degraded feature, but workaround exists"

    @pytest.mark.asyncio
    async def test_state_converts_messages_nested_inside_a_larger_object(self):
        """A ``BaseMessage`` nested inside a dict/list state must not crash json encoding.

        Regression test: before ``_convert_state`` walked recursively, only a
        top-level ``list[BaseMessage]`` was converted, so this exact shape
        raised ``TypeError: Object of type UserMessage is not JSON
        serializable`` from httpx's ``json=`` encoding.
        """
        from openjiuwen.core.foundation.llm.schema.message import AssistantMessage, UserMessage

        client = _make_client()
        mock_async_client = _mock_async_client(
            {
                "model": "jev-1.13.0",
                "answers": {"urgent": {"type": "noul", "noul": 0.5}},
            }
        )

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            await client.system_one(
                state={
                    "conversation": [
                        UserMessage(content="hi"),
                        AssistantMessage(content="hello, how can I help?"),
                    ],
                    "channel": "web",
                },
                questions={"urgent": NoulQuestion(instructions="Does this need attention?")},
            )

        sent_state = mock_async_client.post.call_args.kwargs["json"]["state"]
        assert sent_state == {
            "conversation": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello, how can I help?"},
            ],
            "channel": "web",
        }

    @pytest.mark.asyncio
    async def test_system_one_rejects_empty_questions(self):
        client = _make_client()
        with pytest.raises(BaseError):
            await client.system_one(state="hello", questions={})

    @pytest.mark.asyncio
    async def test_system_one_wraps_http_errors(self):
        client = _make_client()
        mock_async_client = AsyncMock()
        mock_async_client.post.side_effect = httpx.ConnectError("boom")
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = False

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            with pytest.raises(BaseError):
                await client.system_one(
                    state="hello",
                    questions={"q": NoulQuestion(instructions="is this bad?")},
                )
