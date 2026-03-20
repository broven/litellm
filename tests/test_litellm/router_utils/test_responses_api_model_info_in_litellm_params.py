"""
Test: Responses API passes model_info to litellm_params for failure callback.

Without model_info in litellm_params, the router's deployment_callback_on_failure
cannot identify which deployment failed, so cooldown is never set. This causes
cost-based routing with price=0 to deterministically retry the same failed
deployment on every attempt.

Regression test for: responses API missing model_info in logging litellm_params.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm import Router
from litellm.router_utils.cooldown_handlers import _set_cooldown_deployments


@pytest.mark.asyncio
async def test_responses_api_model_info_passed_to_failure_callback():
    """
    Verify that when a Responses API call fails through the router,
    the failure callback receives model_info with a valid deployment id,
    enabling cooldown to be set on the failed deployment.
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/dep-a",
                    "api_key": "key-a",
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
                "model_info": {"id": "dep-a-id"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/dep-b",
                    "api_key": "key-b",
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
                "model_info": {"id": "dep-b-id"},
            },
        ],
        routing_strategy="cost-based-routing",
        cooldown_time=60,
        allowed_fails=0,
    )

    # Track model_info seen in failure callbacks
    failure_callback_model_infos = []
    original_callback = router.deployment_callback_on_failure

    def tracking_failure_callback(kwargs, completion_response, start_time, end_time):
        litellm_params = kwargs.get("litellm_params", {})
        model_info = litellm_params.get("model_info", {})
        failure_callback_model_infos.append(model_info)
        return original_callback(kwargs, completion_response, start_time, end_time)

    router.deployment_callback_on_failure = tracking_failure_callback
    # Also update the registered callback in litellm
    for i, cb in enumerate(litellm.failure_callback):
        if cb == original_callback:
            litellm.failure_callback[i] = tracking_failure_callback

    call_count = 0

    async def mock_aresponses(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        model = kwargs.get("model", "")

        raise litellm.APIError(
            message="quota_exceeded",
            model=model,
            llm_provider="openai",
            status_code=402,
        )

    with patch("litellm.aresponses", side_effect=mock_aresponses):
        with pytest.raises(Exception):
            await router.aresponses(
                model="test-model",
                input="hello",
            )

    # The failure callback should have been called at least once
    assert len(failure_callback_model_infos) > 0, (
        "Failure callback was never called"
    )

    # Each failure callback invocation should have model_info with a valid id
    for i, model_info in enumerate(failure_callback_model_infos):
        assert isinstance(model_info, dict), (
            f"Callback {i}: model_info should be a dict, got {type(model_info)}"
        )
        assert "id" in model_info and model_info["id"], (
            f"Callback {i}: model_info missing 'id' field. Got: {model_info}"
        )
        assert model_info["id"] in ("dep-a-id", "dep-b-id"), (
            f"Callback {i}: unexpected deployment id: {model_info['id']}"
        )


@pytest.mark.asyncio
async def test_responses_api_cost_routing_zero_price_retries_different_deployments():
    """
    With cost-based routing and all prices=0, verify that the router
    tries different deployments on retry (not the same one repeatedly).

    This is the exact scenario from the original bug: all deployments had
    price=0, and without cooldown the stable sort always selected the
    same first deployment.

    Uses the same pattern as test_cost_routing_retry_all_e2e.py: mock
    litellm.acompletion (which the router captures at creation time) and
    manually set cooldown in the mock.
    """
    model_list = [
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "openai/dep-a",
                "api_key": "key-a",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
            "model_info": {"id": "dep-a-id"},
        },
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "openai/dep-b",
                "api_key": "key-b",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
            "model_info": {"id": "dep-b-id"},
        },
        {
            "model_name": "test-model",
            "litellm_params": {
                "model": "openai/dep-c",
                "api_key": "key-c",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
            "model_info": {"id": "dep-c-id"},
        },
    ]

    models_called = []
    # We need router reference in the mock, so declare it first
    router = None

    async def mock_acompletion(*args, **kwargs):
        model = kwargs.get("model", "")
        models_called.append(model)

        if model == "openai/dep-c":
            return litellm.ModelResponse(
                choices=[{"message": {"content": "success from dep-c"}}]
            )

        # dep-a and dep-b fail — manually set cooldown
        dep_id = "dep-a-id" if "dep-a" in model else "dep-b-id"
        _set_cooldown_deployments(
            litellm_router_instance=router,
            original_exception=litellm.APIError(
                message="quota_exceeded",
                model=model,
                llm_provider="openai",
                status_code=402,
            ),
            exception_status=402,
            deployment=dep_id,
            time_to_cooldown=60,
        )
        raise litellm.APIError(
            message="quota_exceeded",
            model=model,
            llm_provider="openai",
            status_code=402,
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        router = Router(
            model_list=model_list,
            routing_strategy="cost-based-routing",
            cooldown_time=60,
            allowed_fails=0,
        )
        response = await router.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    # Should have called all 3 deployments (not the same one 3 times)
    assert len(models_called) == 3, (
        f"Expected 3 calls to different deployments, got {len(models_called)}: {models_called}"
    )
    # All calls should be to different models (no duplicates)
    assert len(set(models_called)) == 3, (
        f"Expected 3 DIFFERENT deployments, but got duplicates: {models_called}"
    )
