from unittest.mock import patch

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import (
    PromptCachingDeploymentCheck,
)


def _build_healthy_deployments() -> list[dict]:
    return [
        {
            "model_name": "test-model",
            "model_info": {"id": "deployment-1"},
            "litellm_params": {"model": "provider/test-model"},
        },
        {
            "model_name": "test-model",
            "model_info": {"id": "deployment-2"},
            "litellm_params": {"model": "provider/test-model"},
        },
        {
            "model_name": "test-model",
            "model_info": {"id": "deployment-3"},
            "litellm_params": {"model": "provider/test-model"},
        },
    ]


@pytest.mark.asyncio
async def test_recently_used_route_marker_prefers_unused_then_resets():
    check = PromptCachingDeploymentCheck(cache=DualCache())
    healthy_deployments = _build_healthy_deployments()
    source_identifier = "top:user_api_key_hash:abc123"

    with patch(
        "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check.random.choice",
        side_effect=lambda options: options[0],
    ):
        selected_1 = await check._async_select_deployment_with_recently_used_markers(
            model="test-model",
            healthy_deployments=healthy_deployments,
            source_identifier=source_identifier,
        )
        selected_2 = await check._async_select_deployment_with_recently_used_markers(
            model="test-model",
            healthy_deployments=healthy_deployments,
            source_identifier=source_identifier,
        )
        selected_3 = await check._async_select_deployment_with_recently_used_markers(
            model="test-model",
            healthy_deployments=healthy_deployments,
            source_identifier=source_identifier,
        )
        selected_4 = await check._async_select_deployment_with_recently_used_markers(
            model="test-model",
            healthy_deployments=healthy_deployments,
            source_identifier=source_identifier,
        )

    assert selected_1 is not None
    assert selected_2 is not None
    assert selected_3 is not None
    assert selected_4 is not None

    assert selected_1["model_info"]["id"] == "deployment-1"
    assert selected_2["model_info"]["id"] == "deployment-2"
    assert selected_3["model_info"]["id"] == "deployment-3"
    # after all 3 routes are marked used, markers reset and selection starts a new round
    assert selected_4["model_info"]["id"] == "deployment-1"

    assert (
        await check._async_has_recently_used_route(
            model="test-model",
            source_identifier=source_identifier,
            model_id="deployment-1",
        )
        is True
    )
    assert (
        await check._async_has_recently_used_route(
            model="test-model",
            source_identifier=source_identifier,
            model_id="deployment-2",
        )
        is False
    )
    assert (
        await check._async_has_recently_used_route(
            model="test-model",
            source_identifier=source_identifier,
            model_id="deployment-3",
        )
        is False
    )


@pytest.mark.asyncio
async def test_async_filter_deployments_uses_unused_route_for_same_source_across_sessions():
    check = PromptCachingDeploymentCheck(cache=DualCache())
    healthy_deployments = _build_healthy_deployments()
    messages = [{"role": "user", "content": "short message"}]

    with patch(
        "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check.random.choice",
        side_effect=lambda options: options[0],
    ):
        filtered_1 = await check.async_filter_deployments(
            model="test-model",
            healthy_deployments=healthy_deployments,
            messages=messages,
            request_kwargs={
                "session_id": "session-1",
                "user_api_key_hash": "account-hash-1",
            },
        )
        filtered_2 = await check.async_filter_deployments(
            model="test-model",
            healthy_deployments=healthy_deployments,
            messages=messages,
            request_kwargs={
                "session_id": "session-2",
                "user_api_key_hash": "account-hash-1",
            },
        )

    assert len(filtered_1) == 1
    assert len(filtered_2) == 1
    assert filtered_1[0]["model_info"]["id"] == "deployment-1"
    assert filtered_2[0]["model_info"]["id"] == "deployment-2"

