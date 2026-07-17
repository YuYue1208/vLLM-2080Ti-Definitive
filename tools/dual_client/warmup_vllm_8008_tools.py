#!/usr/bin/env python3
"""Warm representative vLLM Chat Completions tool shapes without executing tools."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any
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
CLIENT_BODY = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def post(base_url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    return data, time.perf_counter() - started


def stream(base_url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    try:
        with urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                value = line[6:]
                if value != "[DONE]":
                    events.append(json.loads(value))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"stream failed: {exc}") from exc
    final = events[-1] if events else {}
    tool_calls = []
    content = []
    for event in events:
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            tool_calls.extend(delta.get("tool_calls") or [])
    if final.get("usage") is None:
        for event in reversed(events):
            if event.get("usage") is not None:
                final = event
                break
    return {
        "events": len(events),
        "final": final,
        "tool_calls_seen": tool_calls,
        "content": "".join(content),
    }, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8008/v1")
    parser.add_argument("--model", default="thinkingcap-autoround")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    common = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": 256,
        **CLIENT_BODY,
    }
    cases: list[tuple[str, dict[str, Any], bool]] = [
        (
            "plain_nonstream",
            {**common, "messages": [{"role": "system", "content": STABLE_SYSTEM}, {"role": "user", "content": "Reply READY."}]},
            False,
        ),
        (
            "tool_nonstream",
            {
                **common,
                "messages": [{"role": "system", "content": STABLE_SYSTEM}, {"role": "user", "content": "Use exec with command printf 42."}],
                "tools": [TOOL],
                "tool_choice": "required",
            },
            False,
        ),
        (
            "tool_stream",
            {
                **common,
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "system", "content": STABLE_SYSTEM}, {"role": "user", "content": "Use exec with command printf 42."}],
                "tools": [TOOL],
                "tool_choice": "required",
            },
            True,
        ),
    ]

    results: dict[str, Any] = {}
    for name, payload, is_stream in cases:
        started = time.perf_counter()
        data, elapsed = stream(args.url, payload, args.timeout) if is_stream else post(args.url, payload, args.timeout)
        if is_stream:
            usage = (data.get("final") or {}).get("usage") or {}
            tool_calls = data.get("tool_calls_seen") or []
            results[name] = {
                "elapsed_ms": round(elapsed * 1000, 1),
                "events": data.get("events"),
                "usage": usage,
                "tool_calls_seen": len(tool_calls),
            }
            if not usage:
                raise RuntimeError("stream response did not include final usage")
            if name == "tool_stream" and not tool_calls:
                raise RuntimeError("tool_stream response did not include tool_call deltas")
        else:
            choice = (data.get("choices") or [{}])[0]
            usage = data.get("usage") or {}
            results[name] = {
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "tool_calls": len((choice.get("message") or {}).get("tool_calls") or []),
            }
            if not usage:
                raise RuntimeError(f"{name} response did not include usage")
            if name == "tool_nonstream" and not (choice.get("message") or {}).get("tool_calls"):
                raise RuntimeError("tool_nonstream response did not include standard tool_calls")

    # Repeat an identical tool request to expose prefix-cache accounting.
    repeat_payload = {
        **common,
        "messages": [{"role": "system", "content": STABLE_SYSTEM}, {"role": "user", "content": "Use exec with command printf 42."}],
        "tools": [TOOL],
        "tool_choice": "required",
    }
    first, first_elapsed = post(args.url, repeat_payload, args.timeout)
    second, second_elapsed = post(args.url, repeat_payload, args.timeout)
    first_usage = first.get("usage") or {}
    second_usage = second.get("usage") or {}
    second_details = second_usage.get("prompt_tokens_details") or {}
    cached = second_details.get("cached_tokens", 0) or 0
    results["prefix_cache_repeat"] = {
        "first_elapsed_ms": round(first_elapsed * 1000, 1),
        "second_elapsed_ms": round(second_elapsed * 1000, 1),
        "first_usage": first_usage,
        "second_usage": second_usage,
        "cached_tokens": cached,
    }
    if cached <= 0:
        raise RuntimeError("identical repeat request reported no cached_tokens")

    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
