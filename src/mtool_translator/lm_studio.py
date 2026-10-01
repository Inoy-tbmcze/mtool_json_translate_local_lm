"""LM Studio server probing, model lifecycle management, and dynamic model switching."""

from __future__ import annotations

import logging
from typing import Any

from .http_client import HttpRequestError, default_client

logger = logging.getLogger(__name__)

LOAD_CONFIG_KEYS: tuple[str, ...] = (
    "context_length",
    "eval_batch_size",
    "physical_batch_size",
    "parallel",
    "flash_attention",
    "offload_kv_cache_to_gpu",
    "context_checkpoints",
    "speculative_draft_mtp",
    "speculative_draft_simple",
    "speculative_draft_model",
    "speculative_draft_max_tokens",
    "speculative_draft_min_tokens",
    "speculative_draft_min_continue_probability",
)


def get_lm_studio_base_url(api_endpoint: str) -> str:
    """Extracts root server URL from an OpenAI-compatible endpoint."""
    idx = api_endpoint.find("/v1")
    if idx != -1:
        return api_endpoint[:idx]
    parts = api_endpoint.split("/")
    if len(parts) >= 3:
        return f"{parts[0]}//{parts[2]}"
    return "http://127.0.0.1:1234"


def is_lm_studio_available(base_url: str, timeout: float = 2.0) -> bool:
    """Checks whether an LM Studio REST API is available and responding at base_url."""
    models_url = f"{base_url.rstrip('/')}/api/v1/models"
    try:
        resp = default_client.get(models_url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return isinstance(data, dict) and "models" in data
    except (HttpRequestError, ValueError, KeyError):
        pass
    return False


def get_loaded_instances(base_url: str, timeout: float = 5.0) -> list[str]:
    """Retrieves all active loaded model instance IDs in LM Studio."""
    models_url = f"{base_url.rstrip('/')}/api/v1/models"
    instances: list[str] = []
    try:
        resp = default_client.get(models_url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            for model_info in data.get("models", []):
                for instance in model_info.get("loaded_instances", []):
                    inst_id = instance.get("id")
                    if inst_id:
                        instances.append(str(inst_id))
    except (HttpRequestError, ValueError, KeyError) as err:
        logger.warning("Failed to retrieve loaded instances from LM Studio: %s", err)
    return instances


def unload_all_models(base_url: str, timeout: float = 30.0) -> int:
    """Unloads whatever models are currently loaded in LM Studio."""
    instances = get_loaded_instances(base_url, timeout=timeout)
    if not instances:
        return 0

    unload_url = f"{base_url.rstrip('/')}/api/v1/models/unload"
    unloaded_count = 0
    for inst_id in instances:
        try:
            resp = default_client.post(unload_url, json={"instance_id": inst_id}, timeout=timeout)
            if resp.status_code == 200:
                unloaded_count += 1
                logger.info("[LM Studio] Unloaded model instance: %s", inst_id)
            else:
                logger.warning(
                    "[LM Studio] Failed to unload instance '%s' (status %d): %s",
                    inst_id,
                    resp.status_code,
                    resp.text,
                )
        except HttpRequestError as err:
            logger.warning("[LM Studio] Error unloading instance '%s': %s", inst_id, err)

    return unloaded_count


def load_model(
    base_url: str,
    model_key: str,
    config: dict[str, Any] | None = None,
    timeout: float = 180.0,
) -> bool:
    """Loads a specific model into LM Studio with configuration parameters."""
    load_url = f"{base_url.rstrip('/')}/api/v1/models/load"
    payload: dict[str, Any] = {"model": model_key}

    if config:
        for k in LOAD_CONFIG_KEYS:
            if k in config:
                payload[k] = config[k]

    # LM Studio requires that when parallel slots are allocated, context_length is the aggregate
    # across all slots. Guarantee at least 4096 tokens per slot if parallel > 1.
    parallel_slots = payload.get("parallel", 1)
    if parallel_slots > 1 and "context_length" in payload:
        min_aggregate = 4096 * parallel_slots
        if payload["context_length"] < min_aggregate:
            logger.info(
                "[LM Studio] Scaling context_length from %d to %d for %d parallel slots "
                "(4096/slot)",
                payload["context_length"],
                min_aggregate,
                parallel_slots,
            )
            payload["context_length"] = min_aggregate

    # Only include speculative draft params if speculative drafting is actually configured
    if not payload.get("speculative_draft_model"):
        for k in (
            "speculative_draft_model",
            "speculative_draft_simple",
            "speculative_draft_mtp",
            "speculative_draft_max_tokens",
            "speculative_draft_min_tokens",
            "speculative_draft_min_continue_probability",
        ):
            payload.pop(k, None)

    try:
        resp = default_client.post(load_url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            inst_id = data.get("instance_id", model_key)
            load_time = data.get("load_time_seconds", 0.0)
            print(f"[LM Studio] Loaded '{model_key}' (id: {inst_id}, time: {load_time:.2f}s)")
            return True
        logger.error(
            "[LM Studio] Failed to load model '%s' (status %d): %s",
            model_key,
            resp.status_code,
            resp.text,
        )
    except HttpRequestError as err:
        logger.error("[LM Studio] Error loading model '%s': %s", model_key, err)

    return False


def ensure_model_loaded(config: dict[str, Any], stage_name: str = "") -> bool:
    """Checks LM Studio availability, unloads loaded models, and loads the required model."""
    endpoint = config.get("api_endpoint", "http://127.0.0.1:1234/v1/chat/completions")
    base_url = get_lm_studio_base_url(endpoint)

    if not is_lm_studio_available(base_url):
        logger.debug(
            "[LM Studio] Server not detected at %s; skipping model reload for %s.",
            base_url,
            stage_name or "current stage",
        )
        return False

    model_name = config.get("model")
    if not model_name:
        logger.warning(
            "[LM Studio] No model specified in config for %s stage.",
            stage_name or "current",
        )
        return False

    stage_display = stage_name.capitalize() if stage_name else "Pipeline"
    print(f"[{stage_display}] Switching LM Studio model to '{model_name}'...")

    unloaded = unload_all_models(base_url)
    if unloaded > 0:
        print(f"[{stage_display}] Unloaded {unloaded} active model instance(s).")

    success = load_model(base_url, model_name, config)
    if not success:
        error_msg = (
            f"Failed to load required model '{model_name}' into LM Studio for "
            f"{stage_display} stage."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    return True
