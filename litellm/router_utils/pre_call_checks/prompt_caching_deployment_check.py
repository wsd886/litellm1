"""
Check if prompt caching is valid for a given deployment

Route to previously cached model id, if valid
"""

from typing import Any, Dict, List, Optional, cast

from litellm import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.utils import CallTypes, StandardLoggingPayload
from litellm.utils import is_prompt_caching_valid_prompt

from ..prompt_caching_cache import PromptCachingCache


class PromptCachingDeploymentCheck(CustomLogger):
    STICKY_ROUTING_TTL_SECONDS = 300

    def __init__(self, cache: DualCache):
        self.cache = cache

    @staticmethod
    def _get_sticky_cache_key(model: str, session_id: str) -> str:
        return f"deployment:{model}:session:{session_id}:sticky"

    @staticmethod
    def _extract_session_id(payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if payload is None:
            return None

        for direct_key in ("session_id", "conversation_id", "thread_id"):
            direct_value = payload.get(direct_key)
            if isinstance(direct_value, str) and direct_value.strip():
                return direct_value.strip()

        for nested_key in ("metadata", "litellm_metadata"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                for session_key in ("session_id", "conversation_id", "thread_id"):
                    nested_value = nested.get(session_key)
                    if isinstance(nested_value, str) and nested_value.strip():
                        return nested_value.strip()

        return None

    @staticmethod
    def _get_deployment_by_model_id(
        healthy_deployments: List[dict], model_id: str
    ) -> Optional[dict]:
        for deployment in healthy_deployments:
            deployment_model_id = deployment.get("model_info", {}).get("id")
            if deployment_model_id == model_id:
                return deployment
        return None

    async def _async_get_sticky_model_id(
        self, model: str, session_id: str
    ) -> Optional[str]:
        sticky_cache_key = self._get_sticky_cache_key(model=model, session_id=session_id)
        sticky_result = await self.cache.async_get_cache(key=sticky_cache_key)
        if isinstance(sticky_result, dict):
            sticky_model_id = sticky_result.get("model_id")
            if isinstance(sticky_model_id, str) and sticky_model_id:
                return sticky_model_id
        return None

    async def _async_set_sticky_model_id(
        self, model: str, session_id: str, model_id: str
    ) -> None:
        sticky_cache_key = self._get_sticky_cache_key(model=model, session_id=session_id)
        await self.cache.async_set_cache(
            key=sticky_cache_key,
            value={"model_id": model_id},
            ttl=self.STICKY_ROUTING_TTL_SECONDS,
        )

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[dict]:
        session_id = self._extract_session_id(cast(Optional[Dict[str, Any]], request_kwargs))
        if session_id is not None:
            sticky_model_id = await self._async_get_sticky_model_id(
                model=model, session_id=session_id
            )
            if sticky_model_id is not None:
                sticky_deployment = self._get_deployment_by_model_id(
                    healthy_deployments=healthy_deployments, model_id=sticky_model_id
                )
                if sticky_deployment is not None:
                    return [sticky_deployment]

        if messages is not None and is_prompt_caching_valid_prompt(
            messages=messages,
            model=model,
        ):  # prompt > 1024 tokens
            tools: Optional[List[ChatCompletionToolParam]] = None
            if request_kwargs is not None:
                tools_from_request = request_kwargs.get("tools")
                if isinstance(tools_from_request, list):
                    tools = cast(List[ChatCompletionToolParam], tools_from_request)

            prompt_cache = PromptCachingCache(
                cache=self.cache,
            )

            model_id_dict = await prompt_cache.async_get_model_id(
                messages=cast(List[AllMessageValues], messages),
                tools=tools,
            )
            if model_id_dict is not None:
                model_id = model_id_dict["model_id"]
                for deployment in healthy_deployments:
                    if deployment["model_info"]["id"] == model_id:
                        return [deployment]

        return healthy_deployments

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )

        if standard_logging_object is None:
            return

        call_type = standard_logging_object["call_type"]

        if (
            call_type != CallTypes.completion.value
            and call_type != CallTypes.acompletion.value
            and call_type != CallTypes.anthropic_messages.value
        ):  # only use prompt caching for completion calls
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, CALL TYPE IS NOT COMPLETION or ANTHROPIC MESSAGE"
            )
            return

        model = standard_logging_object["model"]
        messages = standard_logging_object["messages"]
        model_id = standard_logging_object["model_id"]
        tools: Optional[List[ChatCompletionToolParam]] = None
        tools_from_kwargs = kwargs.get("tools")
        if isinstance(tools_from_kwargs, list):
            tools = cast(List[ChatCompletionToolParam], tools_from_kwargs)

        if messages is None or not isinstance(messages, list):
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MESSAGES IS NOT A LIST"
            )
            return
        if model_id is None:
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MODEL ID IS NONE"
            )
            return

        session_id = self._extract_session_id(cast(Optional[Dict[str, Any]], kwargs))
        if session_id is not None:
            await self._async_set_sticky_model_id(
                model=model, session_id=session_id, model_id=model_id
            )

        ## PROMPT CACHING - cache model id, if prompt caching valid prompt + provider
        if is_prompt_caching_valid_prompt(
            model=model,
            messages=cast(List[AllMessageValues], messages),
        ):
            cache = PromptCachingCache(
                cache=self.cache,
            )
            await cache.async_add_model_id(
                model_id=model_id,
                messages=messages,
                tools=tools,
            )

        return
