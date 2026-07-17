#!/usr/bin/env python3
"""Run repeated real OpenClaw exec tasks and verify session JSONL truth."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def inspect_session(path: Path) -> tuple[int, int, bool, str]:
    tool_calls = 0
    tool_results = 0
    ok = False
    final_text = ""
    if not path.exists():
        return 0, 0, False, "missing session"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message") or {}
        role = message.get("role")
        content = message.get("content") or []
        if role == "assistant":
            for item in content if isinstance(content, list) else []:
                if item.get("type") == "toolCall" and item.get("name") == "exec":
                    tool_calls += 1
                if item.get("type") == "text":
                    final_text += item.get("text", "")
        elif role == "toolResult":
            tool_results += 1
            text = " ".join(item.get("text", "") for item in content if isinstance(item, dict))
            details = message.get("details") or {}
            if not message.get("isError") and details.get("exitCode") == 0 and "42" in text:
                ok = True
    return tool_calls, tool_results, ok and "RESULT=42" in final_text, final_text[-80:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--node", default="/home/yuyue/.nvm/versions/node/v22.23.1/bin/node")
    parser.add_argument("--cli", default="/home/yuyue/.npm-global/lib/node_modules/openclaw/dist/index.js")
    parser.add_argument("--sessions", default="/home/yuyue/.openclaw/agents/main/sessions")
    parser.add_argument("--output", default="/mnt/models/runtime/results/thinkingcap_autoround_20260715/8008_change_20260715/openclaw-tool-stress-20260716.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt = '请必须使用 exec 执行 python3 -c "print(42)"，只回复 RESULT=42'
    rows = []
    for index in range(1, args.count + 1):
        session_id = f"dual-client-openclaw-stress-20260716-{index:02d}"
        command = [
            args.node,
            args.cli,
            "agent",
            "--session-id",
            session_id,
            "--thinking",
            "off",
            "--json",
            "--timeout",
            "180",
            "--message",
            prompt,
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=240)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        tool_calls, tool_results, success, final_tail = inspect_session(Path(args.sessions) / f"{session_id}.jsonl")
        row = {
            "index": index,
            "session_id": session_id,
            "returncode": completed.returncode,
            "elapsed_ms": elapsed_ms,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "success": bool(completed.returncode == 0 and success),
            "final_tail": final_tail,
        }
        rows.append(row)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    passed = sum(1 for row in rows if row["success"])
    print(json.dumps({"count": len(rows), "passed": passed, "failed": len(rows) - passed, "output": str(output)}, ensure_ascii=False))
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
