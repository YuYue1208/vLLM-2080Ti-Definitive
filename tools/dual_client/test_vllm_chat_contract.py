#!/usr/bin/env python3
"""Direct OpenAI Chat Completions contract and prefix-cache smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TOOL = {
    "type": "function",
    "function": {
        "name": "exec",
        "description": "Run a local command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}
STABLE_SYSTEM = (
    "You are a deterministic terminal assistant. Use the provided tool exactly when requested.\n"
    + ("Stable OpenClaw/OpenCode tool prefix. " * 1400)
)
COMMON = {
    "temperature": 0,
    "max_tokens": 256,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}


def request(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    started = time.perf_counter()
    try:
        with urlopen(Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=timeout) as response:
            return json.loads(response.read().decode()), time.perf_counter() - started
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(str(exc)) from exc


def stream(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload, ensure_ascii=False).encode()
    started = time.perf_counter()
    events = []
    with urlopen(Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))
    usage = next((event.get("usage") for event in reversed(events) if event.get("usage")), None)
    tool_fragments = []
    finish = None
    for event in events:
        for choice in event.get("choices", []):
            finish = choice.get("finish_reason") or finish
            tool_fragments.extend((choice.get("delta") or {}).get("tool_calls") or [])
    return {"usage": usage, "events": len(events), "tool_fragments": tool_fragments, "finish_reason": finish}, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8008/v1/chat/completions")
    parser.add_argument("--model", default="thinkingcap-autoround")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    url = args.url

    plain = {**COMMON, "model": args.model, "messages": [{"role": "system", "content": STABLE_SYSTEM}, {"role": "user", "content": "Reply READY."}]}
    tool = {**COMMON, "model": args.model, "messages": [{"role": "system", "content": STABLE_SYSTEM}, {"role": "user", "content": "Use exec with command printf 42."}], "tools": [TOOL], "tool_choice": "required"}
    stream_tool = {**tool, "stream": True, "stream_options": {"include_usage": True}}

    plain_data, plain_time = request(url, plain, args.timeout)
    tool_data, tool_time = request(url, tool, args.timeout)
    stream_data, stream_time = stream(url, stream_tool, args.timeout)
    repeat_data, repeat_time = request(url, tool, args.timeout)

    tool_calls = (tool_data.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
    repeat_usage = repeat_data.get("usage") or {}
    cached = (repeat_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    if not tool_calls:
        raise RuntimeError("non-stream required tool request returned no tool_calls")
    if not stream_data.get("usage"):
        raise RuntimeError("stream request returned no final usage chunk")
    if not stream_data.get("tool_fragments"):
        raise RuntimeError("stream request returned no tool_call deltas")
    if cached <= 0:
        raise RuntimeError("long identical prefix request returned no cached_tokens")

    output = {
        "ok": True,
        "plain": {"elapsed_ms": round(plain_time * 1000, 1), "usage": plain_data.get("usage")},
        "tool": {"elapsed_ms": round(tool_time * 1000, 1), "usage": tool_data.get("usage"), "tool_calls": len(tool_calls)},
        "stream_tool": {"elapsed_ms": round(stream_time * 1000, 1), **stream_data},
        "repeat": {"elapsed_ms": round(repeat_time * 1000, 1), "usage": repeat_usage, "cached_tokens": cached},
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
