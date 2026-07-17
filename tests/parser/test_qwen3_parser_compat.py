"""Regression tests for the Qwen3 streaming parser engine contract."""

import json

from vllm.parser.engine.events import EventType
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.qwen3 import _qwen3_arg_converter, qwen3_config


def _parse(chunks: list[str], *, thinking: bool = False):
    engine = StreamingParserEngine(qwen3_config(thinking), tokenizer=None)
    events = []
    for chunk in chunks:
        events.extend(engine.feed(chunk, []))
    events.extend(engine.finish())
    return events


def _values(events, event_type):
    return [event.value for event in events if event.type is event_type]


def test_qwen3_split_tags_and_arguments_flush_at_finish():
    events = _parse(
        [
            "before <tool_",
            "call><function=exec>",
            "<parameter=command>printf 42</parameter>",
            "</function></tool_call>after",
        ]
    )

    assert _values(events, EventType.TOOL_CALL_START)
    assert "exec" == "".join(_values(events, EventType.TOOL_NAME))
    assert "printf 42" in "".join(_values(events, EventType.ARG_VALUE_CHUNK))
    assert len(_values(events, EventType.TOOL_CALL_END)) == 1
    assert "before " in "".join(_values(events, EventType.TEXT_CHUNK))
    assert "after" in "".join(_values(events, EventType.TEXT_CHUNK))


def test_qwen3_reasoning_on_and_off_do_not_leak_tool_markup():
    thinking_events = _parse(
        [
            "<think>",
            "先判断工具参数。",
            "</think><tool_call><function=exec><parameter=command>printf 42</parameter>",
            "</function></tool_call>done",
        ],
        thinking=True,
    )
    assert "先判断工具参数。" == "".join(
        _values(thinking_events, EventType.REASONING_CHUNK)
    )
    assert _values(thinking_events, EventType.REASONING_END)
    assert "<tool_call>" not in "".join(
        _values(thinking_events, EventType.TEXT_CHUNK)
    )

    no_thinking_events = _parse(
        ["<think>ignored</think>plain"],
        thinking=False,
    )
    assert "plain" in "".join(_values(no_thinking_events, EventType.TEXT_CHUNK))


def test_qwen3_multiple_tools_unicode_nested_values_and_partial_converter():
    events = _parse(
        [
            "<tool_call><function=exec><parameter=command>printf '",
            "你好'</parameter></function></tool_call>",
            "<tool_call><function=exec><parameter=command>python -c ",
            "'print(42)'</parameter></function></tool_call>",
        ]
    )
    assert len(_values(events, EventType.TOOL_CALL_START)) == 2
    assert len(_values(events, EventType.TOOL_CALL_END)) == 2
    args = "".join(_values(events, EventType.ARG_VALUE_CHUNK))
    assert "你好" in args
    assert "print(42)" in args

    converted = _qwen3_arg_converter(
        "<parameter=data>{\"items\":[1,{\"值\":\"✓\"}]}</parameter>",
        partial=False,
    )
    assert json.loads(converted) == {
        "data": '{"items":[1,{"值":"✓"}]}'
    }
    partial = _qwen3_arg_converter(
        "<parameter=command>printf 42",
        partial=True,
    )
    assert json.loads(partial) == {"command": "printf 42"}


def test_qwen3_malformed_input_finishes_without_looping():
    events = _parse(
        [
            "<tool_call><function=exec><parameter=command>{\"x\": [1, 2}",
            "still pending",
        ]
    )
    assert events[-1].type is EventType.TOOL_CALL_END
