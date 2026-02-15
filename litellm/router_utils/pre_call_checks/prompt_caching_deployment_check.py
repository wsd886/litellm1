"""
Check if prompt caching is valid for a given deployment

Route to previously cached model id, if valid
"""

import hashlib
import random
import time
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
    def _get_auto_session_id(source_identifier: str) -> str:
        source_hash = hashlib.sha256(source_identifier.encode("utf-8")).hexdigest()[:24]
        return f"auto-{source_hash}"

    @staticmethod
    def _extract_source_identifier(payload: Dict[str, Any]) -> Optional[str]:
        source_candidates = (
            "user",
            "user_id",
            "end_user",
            "user_api_key_hash",
            "user_api_key_token",
            "user_api_key_alias",
            "user_api_key_user_id",
            "user_api_key_team_id",
            "user_api_key",
            "api_key",
        )

        for source_key in source_candidates:
            source_value = payload.get(source_key)
            if isinstance(source_value, str) and source_value.strip():
                return f"top:{source_key}:{source_value.strip()}"

        for nested_key in ("metadata", "litellm_metadata"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                for source_key in source_candidates:
                    source_value = nested.get(source_key)
                    if isinstance(source_value, str) and source_value.strip():
                        return f"{nested_key}:{source_key}:{source_value.strip()}"

        # fallback: proxy request body
        proxy_server_request = payload.get("proxy_server_request")
        if isinstance(proxy_server_request, dict):
            body = proxy_server_request.get("body")
            if isinstance(body, dict):
                for source_key in source_candidates:
                    source_value = body.get(source_key)
                    if isinstance(source_value, str) and source_value.strip():
                        return f"proxy_server_request:body:{source_key}:{source_value.strip()}"

        litellm_params = payload.get("litellm_params")
        if isinstance(litellm_params, dict):
            proxy_server_request = litellm_params.get("proxy_server_request")
            if isinstance(proxy_server_request, dict):
                body = proxy_server_request.get("body")
                if isinstance(body, dict):
                    for source_key in source_candidates:
                        source_value = body.get(source_key)
                        if isinstance(source_value, str) and source_value.strip():
                            return f"litellm_params:proxy_server_request:body:{source_key}:{source_value.strip()}"

        return None

    @staticmethod
    def _get_sticky_cache_key(model: str, session_id: str) -> str:
        return f"deployment:{model}:session:{session_id}:sticky"

    @staticmethod
    def _get_session_sticky_cache_key(session_id: str) -> str:
        return f"deployment:session:{session_id}:sticky"

    @staticmethod
    def _get_sticky_ttl_seconds(session_id: str) -> int:
        _ = session_id
        return PromptCachingDeploymentCheck.STICKY_ROUTING_TTL_SECONDS

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

        # fallback: proxy request body
        proxy_server_request = payload.get("proxy_server_request")
        if isinstance(proxy_server_request, dict):
            body = proxy_server_request.get("body")
            if isinstance(body, dict):
                for session_key in ("session_id", "conversation_id", "thread_id"):
                    nested_value = body.get(session_key)
                    if isinstance(nested_value, str) and nested_value.strip():
                        return nested_value.strip()
                for nested_key in ("metadata", "litellm_metadata"):
                    nested = body.get(nested_key)
                    if isinstance(nested, dict):
                        for session_key in ("session_id", "conversation_id", "thread_id"):
                            nested_value = nested.get(session_key)
                            if isinstance(nested_value, str) and nested_value.strip():
                                return nested_value.strip()

        litellm_params = payload.get("litellm_params")
        if isinstance(litellm_params, dict):
            proxy_server_request = litellm_params.get("proxy_server_request")
            if isinstance(proxy_server_request, dict):
                body = proxy_server_request.get("body")
                if isinstance(body, dict):
                    for session_key in ("session_id", "conversation_id", "thread_id"):
                        nested_value = body.get(session_key)
                        if isinstance(nested_value, str) and nested_value.strip():
                            return nested_value.strip()
                    for nested_key in ("metadata", "litellm_metadata"):
                        nested = body.get(nested_key)
                        if isinstance(nested, dict):
                            for session_key in ("session_id", "conversation_id", "thread_id"):
                                nested_value = nested.get(session_key)
                                if isinstance(nested_value, str) and nested_value.strip():
                                    return nested_value.strip()

        source_identifier = PromptCachingDeploymentCheck._extract_source_identifier(
            payload=payload
        )
        if source_identifier is not None:
            return PromptCachingDeploymentCheck._get_auto_session_id(
                source_identifier=source_identifier
            )

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

    @staticmethod
    def _extract_model_id(payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if payload is None:
            return None

        model_info = payload.get("model_info")
        if isinstance(model_info, dict):
            direct_model_id = model_info.get("id")
            if isinstance(direct_model_id, str) and direct_model_id:
                return direct_model_id

        for nested_key in ("metadata", "litellm_metadata"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                nested_model_info = nested.get("model_info")
                if isinstance(nested_model_info, dict):
                    nested_model_id = nested_model_info.get("id")
                    if isinstance(nested_model_id, str) and nested_model_id:
                        return nested_model_id

        return None

    @staticmethod
    def _get_sticky_model_keys_for_success_event(
        kwargs: Dict[str, Any], standard_logging_object: StandardLoggingPayload
    ) -> List[str]:
        """
        Build candidate model keys used for sticky-routing writes.

        Why:
        - pre-call check often reads with router model alias (e.g. `claude-opus-4-6`)
        - standard logging model may be provider-prefixed (e.g. `bedrock/us...`)
        We write both to avoid key-space mismatch.
        """
        candidate_keys: List[str] = []

        model_group = standard_logging_object.get("model_group")
        if isinstance(model_group, str) and model_group.strip():
            candidate_keys.append(model_group.strip())

        kwargs_model = kwargs.get("model")
        if isinstance(kwargs_model, str) and kwargs_model.strip():
            candidate_keys.append(kwargs_model.strip())

        logging_model = standard_logging_object.get("model")
        if isinstance(logging_model, str) and logging_model.strip():
            candidate_keys.append(logging_model.strip())

        deduped_keys: List[str] = []
        seen = set()
        for key in candidate_keys:
            if key not in seen:
                deduped_keys.append(key)
                seen.add(key)
        return deduped_keys

    async def _async_get_sticky_model_id(
        self, model: str, session_id: str
    ) -> Optional[str]:
        sticky_keys = [
            self._get_sticky_cache_key(model=model, session_id=session_id),
            self._get_session_sticky_cache_key(session_id=session_id),
        ]
        for sticky_cache_key in sticky_keys:
            sticky_result = await self.cache.async_get_cache(key=sticky_cache_key)
            if isinstance(sticky_result, dict):
                sticky_model_id = sticky_result.get("model_id")
                if isinstance(sticky_model_id, str) and sticky_model_id:
                    return sticky_model_id
        return None

    async def _async_set_sticky_model_id(
        self,
        model: str,
        session_id: str,
        model_id: str,
        route_record: Optional[Dict[str, Any]] = None,
    ) -> None:
        sticky_ttl_seconds = self._get_sticky_ttl_seconds(session_id=session_id)
        sticky_value: Dict[str, Any] = {"model_id": model_id}
        if isinstance(route_record, dict):
            sticky_value.update(route_record)
        sticky_keys = [
            self._get_sticky_cache_key(model=model, session_id=session_id),
            self._get_session_sticky_cache_key(session_id=session_id),
        ]
        for sticky_cache_key in sticky_keys:
            await self.cache.async_set_cache(
                key=sticky_cache_key,
                value=sticky_value,
                ttl=sticky_ttl_seconds,
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
                        if session_id is not None:
                            existing_sticky_model_id = (
                                await self._async_get_sticky_model_id(
                                    model=model, session_id=session_id
                                )
                            )
                            if existing_sticky_model_id is None:
                                await self._async_set_sticky_model_id(
                                    model=model, session_id=session_id, model_id=model_id
                                )
                        return [deployment]

        # layer-1 session sticky routing:
        # if session has no sticky route yet, pick a random deployment once and pin for 5 min
        if (
            session_id is not None
            and messages is not None
            and isinstance(healthy_deployments, list)
            and len(healthy_deployments) > 0
        ):
            selected_deployment = random.choice(healthy_deployments)
            selected_model_id = (
                selected_deployment.get("model_info", {}) or {}
            ).get("id")
            if isinstance(selected_model_id, str) and selected_model_id:
                route_record: Dict[str, Any] = {
                    "selected_at_unix": int(time.time()),
                    "selected_model_group": model,
                    "selected_model_name": selected_deployment.get("model_name"),
                    "selected_model_litellm": (
                        selected_deployment.get("litellm_params", {}) or {}
                    ).get("model"),
                }
                await self._async_set_sticky_model_id(
                    model=model,
                    session_id=session_id,
                    model_id=selected_model_id,
                    route_record=route_record,
                )
                return [selected_deployment]

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
            model_id = self._extract_model_id(cast(Optional[Dict[str, Any]], kwargs))
        if model_id is None:
            model_id = self._extract_model_id(
                {"metadata": standard_logging_object.get("metadata")}
            )
        if model_id is None:
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MODEL ID IS NONE"
            )
            return

        session_id = self._extract_session_id(cast(Optional[Dict[str, Any]], kwargs))
        if session_id is None:
            standard_metadata = standard_logging_object.get("metadata")
            if isinstance(standard_metadata, dict):
                session_id = self._extract_session_id({"metadata": standard_metadata})
        if session_id is not None:
            existing_sticky_model_id = await self._async_get_sticky_model_id(
                model=model, session_id=session_id
            )
            if existing_sticky_model_id is None:
                sticky_model_keys = self._get_sticky_model_keys_for_success_event(
                    kwargs=kwargs, standard_logging_object=standard_logging_object
                )
                for sticky_model_key in sticky_model_keys:
                    await self._async_set_sticky_model_id(
                        model=sticky_model_key, session_id=session_id, model_id=model_id
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
