# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import VllmConfig


def _verify_connector(
    monkeypatch: pytest.MonkeyPatch,
    connector: str,
    extra_config: dict[str, str] | None = None,
    allocator_env: str = "PYTORCH_CUDA_ALLOC_CONF",
    allocator_value: str = "expandable_segments:True",
) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.setenv(allocator_env, allocator_value)
    config = object.__new__(VllmConfig)
    object.__setattr__(
        config,
        "kv_transfer_config",
        SimpleNamespace(
            kv_connector=connector,
            kv_connector_extra_config=extra_config,
        ),
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(enable_sleep_mode=False),
    )
    config._verify_kv_transfer_compat()


@pytest.mark.parametrize(
    "connector", ["OffloadingConnector", "SimpleCPUOffloadConnector"]
)
def test_expandable_segments_allows_local_cpu_offload(
    monkeypatch: pytest.MonkeyPatch, connector: str
) -> None:
    _verify_connector(monkeypatch, connector)


def test_expandable_segments_still_rejects_registered_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="NixlConnector"):
        _verify_connector(monkeypatch, "NixlConnector")


def test_expandable_segments_rejects_custom_offloading_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="NixlConnector"):
        _verify_connector(
            monkeypatch,
            "OffloadingConnector",
            extra_config={"spec_name": "CustomOffloadingSpec"},
        )


@pytest.mark.parametrize(
    ("allocator_env", "allocator_value"),
    [
        ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:true"),
        ("PYTORCH_ALLOC_CONF", "foo:bar, expandable_segments:1"),
    ],
)
def test_expandable_segments_detection_is_shared_and_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
    allocator_env: str,
    allocator_value: str,
) -> None:
    _verify_connector(
        monkeypatch,
        "SimpleCPUOffloadConnector",
        allocator_env=allocator_env,
        allocator_value=allocator_value,
    )
