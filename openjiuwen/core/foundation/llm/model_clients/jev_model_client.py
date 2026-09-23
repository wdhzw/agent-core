# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Jev model client — TypeSafe's System One classification API.

Jev answers typed questions about a piece of state (``Noul`` / ``Choice`` /
``Score``) instead of generating chat text: a call takes ``state`` +
``questions`` and returns typed ``answers``, not a messages-in/text-out chat
turn. This is a different wire protocol from ``BaseModelClient``'s
messages-in/``AssistantMessage``-out shape, so this client does not subclass
it (mirrors both upstream integrations: OpenRouter routes Jev through a
separate ``/v1/systemone`` endpoint instead of its ``/v1/chat/completions``
dispatch, and LangChain's ``TypeSafeClassifier`` is a standalone class, not
one of the chat models ``init_chat_model`` resolves).

Speaks ``POST {api_base}/v1/systemone`` directly, so it works against
TypeSafe's own endpoint (``https://api.typesafe.ai``) or OpenRouter's
passthrough (``https://openrouter.ai/api``) by just changing ``api_base``.

Takes plain constructor kwargs rather than ``ModelClientConfig``: that type's
``client_provider`` field is validated against the provider registry that
powers ``create_model_client``, a system this client deliberately does not
plug into (see the module docstring above), so requiring it here would just
force callers to supply a provider name that means nothing to this class::

    client = JevModelClient(api_key="...", api_base="https://api.typesafe.ai", model_name="jev-1.13")
    result = await client.system_one(
        state="I was charged twice for my subscription.",
        questions={"refund": NoulQuestion(instructions="Is the customer asking for money back?")},
    )
    result.answers["refund"].noul  # -> 0.98
"""

from typing import Any, Optional

import httpx

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import build_error
from openjiuwen.core.common.logging import LogEventType, llm_logger
from openjiuwen.core.foundation.llm.schema.classification import (
    ChoiceQuestion,
    ClassificationQuestion,
    ClassificationResponse,
    NoulQuestion,
    ScoreQuestion,
)
from openjiuwen.core.foundation.llm.schema.message import BaseMessage

_QUESTION_TYPE_NAMES: dict[type, str] = {
    NoulQuestion: "noul",
    ChoiceQuestion: "choice",
    ScoreQuestion: "score",
}


def _question_to_dict(question: ClassificationQuestion) -> dict[str, Any]:
    for question_cls, type_name in _QUESTION_TYPE_NAMES.items():
        if isinstance(question, question_cls):
            return {"type": type_name, **question.model_dump(exclude_none=True)}
    raise build_error(
        StatusCode.MODEL_INVOKE_PARAM_ERROR,
        error_msg=f"unsupported classification question type: {type(question)!r}",
    )


def _convert_state(value: Any) -> Any:
    """Recursively convert any ``BaseMessage`` found in ``value`` to a role/content dict.

    Mirrors ``langchain-typesafe``'s stated behavior: "A BaseMessage or a
    sequence of messages is converted to role/content JSON, including when
    nested inside a larger object." ``state`` may be a message list at the
    top level, or messages nested inside an arbitrary dict/list structure
    (e.g. ``{"conversation": [UserMessage(...)], "channel": "web"}``); both
    shapes need every ``BaseMessage`` walked and converted, since plain
    ``json.dumps`` (what httpx's ``json=`` uses) cannot serialize one.
    """
    if isinstance(value, BaseMessage):
        return {"role": value.role, "content": value.content}
    if isinstance(value, dict):
        return {key: _convert_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_convert_state(item) for item in value]
    return value


class JevModelClient:
    """Client for TypeSafe's Jev System One classification API."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model_name: Optional[str] = None,
        timeout: float = 360.0,
        verify_ssl: bool = True,
        custom_headers: Optional[dict[str, str]] = None,
    ):
        if not api_key:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="api_key is required for JevModelClient."
            )
        if not api_base:
            raise build_error(
                StatusCode.MODEL_SERVICE_CONFIG_ERROR, error_msg="api_base is required for JevModelClient."
            )
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.custom_headers = custom_headers

    async def system_one(
        self,
        state: Any,
        questions: dict[str, ClassificationQuestion],
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ClassificationResponse:
        """Evaluate ``state`` against a set of typed ``questions``.

        Args:
            state: The context to classify. A string, a JSON-serializable
                dict/list, or a list of ``BaseMessage`` (converted to
                role/content dicts before sending).
            questions: Question id -> typed question (Noul/Choice/Score).
            model: Overrides the ``model_name`` set at construction.
            timeout: Overrides the ``timeout`` set at construction, for this call.

        Returns:
            ClassificationResponse: One typed answer per question, keyed by
            the same question id.
        """
        if not questions:
            raise build_error(StatusCode.MODEL_INVOKE_PARAM_ERROR, error_msg="questions cannot be empty.")

        resolved_model = model or self.model_name
        if not resolved_model:
            raise build_error(StatusCode.MODEL_CONFIG_ERROR, error_msg="The model cannot be None.")

        resolved_state = _convert_state(state)

        payload = {
            "model": resolved_model,
            "state": resolved_state,
            "questions": {qid: _question_to_dict(q) for qid, q in questions.items()},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.custom_headers:
            headers.update(self.custom_headers)

        url = f"{self.api_base.rstrip('/')}/v1/systemone"
        request_timeout = timeout if timeout is not None else self.timeout

        llm_logger.info(
            "Before request Jev system_one, request params ready.",
            event_type=LogEventType.LLM_CALL_START,
            model_name=resolved_model,
            model_provider="jev",
        )
        try:
            async with httpx.AsyncClient(
                timeout=request_timeout,
                verify=self.verify_ssl,
            ) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return ClassificationResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise build_error(
                StatusCode.MODEL_CALL_FAILED,
                error_msg=f"Jev system_one request failed: {exc}",
            ) from exc
