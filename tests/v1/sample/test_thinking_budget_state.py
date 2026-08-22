# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


def _sync_one_request(holder: ThinkingBudgetStateHolder) -> None:
    holder.sync_batch(
        SimpleNamespace(
            removed=[],
            moved=[],
            added=[
                (
                    0,
                    SimpleNamespace(thinking_token_budget=32),
                    [],
                    [],
                )
            ],
        )
    )


def test_natural_end_preserves_next_think_prefix() -> None:
    holder = ThinkingBudgetStateHolder(
        SimpleNamespace(
            reasoning_start_token_ids=[10, 11],
            reasoning_end_token_ids=[20],
        ),
        max_num_seqs=1,
        num_spec_tokens=0,
        device=torch.device("cpu"),
        is_pin_memory=False,
    )
    _sync_one_request(holder)

    # The same accepted batch contains a complete block and the first token of
    # the next multi-token <think> marker.
    holder.update_state([[10, 11, 1, 20, 10]], None)
    state = holder._state[0]
    assert state["in_think"] is False
    assert state["scan_offset"] == 4
    assert state["prev_output_length"] == 5

    # The second token arrives later; scanning from the end-marker boundary
    # must retain the prefix from the previous batch.
    holder.update_state([[10, 11, 1, 20, 10, 11, 2]], None)
    state = holder._state[0]
    assert state["in_think"] is True
    assert state["start_thinking"] == 4
