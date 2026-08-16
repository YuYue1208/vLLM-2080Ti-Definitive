# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import VllmConfig


def _verify_connector(monkeypatch: pytest.MonkeyPatch, connector: str) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    config = object.__new__(VllmConfig)
    object.__setattr__(
        config,
        "kv_transfer_config",
        SimpleNamespace(kv_connector=connector),
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
