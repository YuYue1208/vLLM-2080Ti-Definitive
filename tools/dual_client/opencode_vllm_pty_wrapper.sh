#!/usr/bin/env bash
set -e

BIN="${OPENCODE_BIN:-/usr/local/bin/opencode}"
export OPENCODE_DISABLE_MODELS_FETCH="${OPENCODE_DISABLE_MODELS_FETCH:-true}"
export OPENCODE_DISABLE_AUTOUPDATE="${OPENCODE_DISABLE_AUTOUPDATE:-1}"

if [[ -t 0 && -t 1 ]]; then
  exec "$BIN" "$@"
fi

command_line=$(printf '%q ' "$BIN" "$@")
exec script -qefc "$command_line" /dev/null
