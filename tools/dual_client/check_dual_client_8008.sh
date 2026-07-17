#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8008}
MODEL=${MODEL:-thinkingcap-autoround}

health=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "$BASE_URL/health")
models=$(curl -sS --max-time 10 "$BASE_URL/v1/models")
served=$(printf '%s' "$models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')

test "$health" = 200
test "$served" = "$MODEL"
printf 'health_http=%s\nserved_model=%s\n' "$health" "$served"
