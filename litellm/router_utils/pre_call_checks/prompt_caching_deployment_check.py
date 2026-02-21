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
    ROUTE_USAGE_TTL_SECONDS = 86400

    def __init__(self, cache: DualCache):
        self.cache = cache

    @staticmethod
    def _get_auto_session_id(source_identifier: str) -> str:
        source_hash = hashlib.sha256(source_identifier.encode("utf-8")).hexdigest()[:24]
        return f"auto-{source_hash}"

    @staticmethod
    def _extract_source_identifier(payload: Dict[str, Any]) -> Optional[str]:
        # Prefer API key scoped identifiers for account-level routing fairness.
        source_candidates = (
            "user_api_key_hash",
            "user_api_key_token",
            "user_api_key_alias",
            "user_api_key_user_id",
            "user_api_key_team_id",
            "user_api_key",
            "api_key",
            "end_user",
            "user_id",
            "user",
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
    def _get_source_hash(source_identifier: str) -> str:
        return hashlib.sha256(source_identifier.encode("utf-8")).hexdigest()[:24]

    @classmethod
    def _get_recently_used_route_cache_key(
        cls, model: str, source_identifier: str, model_id: str
    ) -> str:
        source_hash = cls._get_source_hash(source_identifier=source_identifier)
        return f"deployment:{model}:source:{source_hash}:model:{model_id}:used_24h"

    @staticmethod
    def _get_route_usage_ttl_seconds(source_identifier: str) -> int:
        _ = source_identifier
        return PromptCachingDeploymentCheck.ROUTE_USAGE_TTL_SECONDS

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
    def _is_non_retriable_account_error(kwargs: Dict[str, Any]) -> bool:
        exception_obj = kwargs.get("exception")
        traceback_exception = kwargs.get("traceback_exception")
        text_parts: List[str] = []
        if exception_obj is not None:
            text_parts.append(str(exception_obj))
        if isinstance(traceback_exception, str):
            text_parts.append(traceback_exception)
        combined_error_text = " | ".join(text_parts)
        if not combined_error_text:
            return False
        account_error_markers = (
            "INVALID_PAYMENT_INSTRUMENT",
            "Model access is denied",
            "Authentication failed",
            "Operation not allowed",
            "Your AWS Marketplace subscription for this model cannot be completed",
        )
        return any(marker in combined_error_text for marker in account_error_markers)

    @staticmethod
    def _get_sticky_model_keys_for_failure_event(
        kwargs: Dict[str, Any], standard_logging_object: Optional[StandardLoggingPayload]
    ) -> List[str]:
        candidate_keys: List[str] = []

        kwargs_model = kwargs.get("model")
        if isinstance(kwargs_model, str) and kwargs_model.strip():
            candidate_keys.append(kwargs_model.strip())

        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            metadata_model_group = metadata.get("model_group")
            if (
                isinstance(metadata_model_group, str)
                and metadata_model_group.strip()
            ):
                candidate_keys.append(metadata_model_group.strip())

        litellm_metadata = kwargs.get("litellm_metadata")
        if isinstance(litellm_metadata, dict):
            litellm_metadata_model_group = litellm_metadata.get("model_group")
            if (
                isinstance(litellm_metadata_model_group, str)
                and litellm_metadata_model_group.strip()
            ):
                candidate_keys.append(litellm_metadata_model_group.strip())

        if isinstance(standard_logging_object, dict):
            logging_model = standard_logging_object.get("model")
            if isinstance(logging_model, str) and logging_model.strip():
                candidate_keys.append(logging_model.strip())
            logging_model_group = standard_logging_object.get("model_group")
            if isinstance(logging_model_group, str) and logging_model_group.strip():
                candidate_keys.append(logging_model_group.strip())

        deduped_keys: List[str] = []
        seen = set()
        for key in candidate_keys:
            if key not in seen:
                deduped_keys.append(key)
                seen.add(key)
        return deduped_keys

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

    async def _async_has_recently_used_route(
        self, model: str, source_identifier: str, model_id: str
    ) -> bool:
        cache_key = self._get_recently_used_route_cache_key(
            model=model,
            source_identifier=source_identifier,
            model_id=model_id,
        )
        marker = await self.cache.async_get_cache(key=cache_key)
        if marker is None:
            return False
        if isinstance(marker, dict):
            marker_used = marker.get("used")
            if isinstance(marker_used, bool):
                return marker_used
        return True

    async def _async_mark_route_as_recently_used(
        self, model: str, source_identifier: str, model_id: str
    ) -> None:
        cache_key = self._get_recently_used_route_cache_key(
            model=model,
            source_identifier=source_identifier,
            model_id=model_id,
        )
        route_usage_ttl_seconds = self._get_route_usage_ttl_seconds(
            source_identifier=source_identifier
        )
        await self.cache.async_set_cache(
            key=cache_key,
            value={
                "used": True,
                "selected_at_unix": int(time.time()),
            },
            ttl=route_usage_ttl_seconds,
        )

    async def _async_clear_recently_used_route_markers(
        self, model: str, source_identifier: str, model_ids: List[str]
    ) -> None:
        deduped_model_ids = list(dict.fromkeys(model_ids))
        for model_id in deduped_model_ids:
            cache_key = self._get_recently_used_route_cache_key(
                model=model,
                source_identifier=source_identifier,
                model_id=model_id,
            )
            await self.cache.async_delete_cache(key=cache_key)

    async def _async_select_deployment_with_recently_used_markers(
        self,
        model: str,
        healthy_deployments: List[dict],
        source_identifier: Optional[str],
    ) -> Optional[dict]:
        deployments_with_model_ids: List[Dict[str, Any]] = []
        for deployment in healthy_deployments:
            deployment_model_id = deployment.get("model_info", {}).get("id")
            if isinstance(deployment_model_id, str) and deployment_model_id:
                deployments_with_model_ids.append(
                    {
                        "deployment": deployment,
                        "model_id": deployment_model_id,
                    }
                )

        if len(deployments_with_model_ids) == 0:
            return None

        if source_identifier is None:
            return random.choice(
                [row["deployment"] for row in deployments_with_model_ids]
            )

        is_recently_used_by_model_id: Dict[str, bool] = {}
        for row in deployments_with_model_ids:
            deployment_model_id = row["model_id"]
            if deployment_model_id in is_recently_used_by_model_id:
                continue
            is_recently_used_by_model_id[deployment_model_id] = (
                await self._async_has_recently_used_route(
                    model=model,
                    source_identifier=source_identifier,
                    model_id=deployment_model_id,
                )
            )

        unused_deployments = [
            row["deployment"]
            for row in deployments_with_model_ids
            if is_recently_used_by_model_id.get(row["model_id"]) is False
        ]

        if len(unused_deployments) == 0:
            await self._async_clear_recently_used_route_markers(
                model=model,
                source_identifier=source_identifier,
                model_ids=[row["model_id"] for row in deployments_with_model_ids],
            )
            unused_deployments = [row["deployment"] for row in deployments_with_model_ids]

        selected_deployment = random.choice(unused_deployments)
        selected_model_id = selected_deployment.get("model_info", {}).get("id")
        if isinstance(selected_model_id, str) and selected_model_id:
            await self._async_mark_route_as_recently_used(
                model=model,
                source_identifier=source_identifier,
                model_id=selected_model_id,
            )
        return selected_deployment

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[dict]:
        source_identifier: Optional[str] = None
        if request_kwargs is not None:
            source_identifier = self._extract_source_identifier(
                payload=cast(Dict[str, Any], request_kwargs)
            )

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
            selected_deployment = (
                await self._async_select_deployment_with_recently_used_markers(
                    model=model,
                    healthy_deployments=healthy_deployments,
                    source_identifier=source_identifier,
                )
            )
            if selected_deployment is None:
                return healthy_deployments

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
                    "selected_model_usage_ttl_seconds": self.ROUTE_USAGE_TTL_SECONDS,
                    "selected_model_strategy": "unused_24h_first_random",
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

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        if not isinstance(kwargs, dict):
            return
        if not self._is_non_retriable_account_error(kwargs=kwargs):
            return

        session_id = self._extract_session_id(kwargs)
        if session_id is None:
            standard_logging_object = kwargs.get("standard_logging_object")
            if isinstance(standard_logging_object, dict):
                standard_metadata = standard_logging_object.get("metadata")
                if isinstance(standard_metadata, dict):
                    session_id = self._extract_session_id({"metadata": standard_metadata})
        if session_id is None:
            return

        standard_logging_object = kwargs.get("standard_logging_object")
        model_id: Optional[str] = None
        if isinstance(standard_logging_object, dict):
            logged_model_id = standard_logging_object.get("model_id")
            if isinstance(logged_model_id, str) and logged_model_id:
                model_id = logged_model_id
        if model_id is None:
            model_id = self._extract_model_id(kwargs)

        sticky_keys: List[str] = [self._get_session_sticky_cache_key(session_id=session_id)]
        sticky_model_keys = self._get_sticky_model_keys_for_failure_event(
            kwargs=kwargs,
            standard_logging_object=(
                standard_logging_object
                if isinstance(standard_logging_object, dict)
                else None
            ),
        )
        for sticky_model_key in sticky_model_keys:
            sticky_keys.append(
                self._get_sticky_cache_key(model=sticky_model_key, session_id=session_id)
            )

        for sticky_key in sticky_keys:
            try:
                existing_value = await self.cache.async_get_cache(key=sticky_key)
                if model_id is None:
                    await self.cache.async_delete_cache(key=sticky_key)
                    continue
                if isinstance(existing_value, dict):
                    existing_model_id = existing_value.get("model_id")
                    if existing_model_id == model_id:
                        await self.cache.async_delete_cache(key=sticky_key)
            except Exception:
                continue
