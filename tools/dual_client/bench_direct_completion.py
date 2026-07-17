import json
import statistics
import time
import urllib.request

URL = "http://127.0.0.1:8008/v1/completions"
MODEL = "thinkingcap-autoround"
PROMPT = " the" * 4096


def run(i):
    body = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first = None
    token_count = 0
    chunks = 0
    preview = []
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            obj = json.loads(data)
            choices = obj.get("choices") or []
            if not choices:
                continue
            if len(preview) < 3:
                preview.append(choices[0])
            ids = choices[0].get("token_ids") or []
            text = choices[0].get("text") or ""
            if ids or text:
                if first is None:
                    first = time.perf_counter()
                token_count += len(ids) if ids else len(text.split())
                chunks += 1
    end = time.perf_counter()
    decode_s = max(end - (first or start), 1e-6)
    return {
        "run": i,
        "prompt_chars": len(PROMPT),
        "completion_tokens": token_count,
        "chunks": chunks,
        "ttft_ms": round(((first or end) - start) * 1000, 1),
        "elapsed_ms": round((end - start) * 1000, 1),
        "decode_tok_s": round(token_count / decode_s, 2),
        "preview": preview,
    }


results = [run(i) for i in range(4)]
print(json.dumps({
    "results": results,
    "median_decode_tok_s": round(statistics.median(x["decode_tok_s"] for x in results), 2),
    "median_ttft_ms": round(statistics.median(x["ttft_ms"] for x in results), 1),
}, ensure_ascii=False, indent=2))
