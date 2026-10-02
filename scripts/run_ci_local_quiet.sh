#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${AGENT_CI_LOG_DIR:-${TMPDIR:-/tmp}/agent-ci-logs}"
SHARED_ROOT="${AGENT_CI_SHARED_DIR:-${TMPDIR:-/tmp}/agent-ci-shared-$(id -u)}"
mkdir -p "$LOG_ROOT" "$SHARED_ROOT"
chmod 700 "$LOG_ROOT" "$SHARED_ROOT" 2>/dev/null || true
KEY="$(ROOT_DIR="$ROOT_DIR" python3 -c 'import hashlib, os; print(hashlib.sha256(os.environ["ROOT_DIR"].encode()).hexdigest())')"
LOCK_DIR="$SHARED_ROOT/$KEY.lock"
LOG="$LOG_ROOT/$KEY.log"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "ci_state: shared_in_flight"
  exit 75
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

started="$(date +%s)"
if [ "${AGENT_CI_BACKEND:-policy}" = "act" ]; then
  if bash "$ROOT_DIR/scripts/run_ci_act.sh" >"$LOG" 2>&1; then status=0; else status=$?; fi
else
  if bash "$ROOT_DIR/scripts/ci_local.sh" >"$LOG" 2>&1; then status=0; else status=$?; fi
fi
elapsed=$(( $(date +%s) - started ))
if [ "$status" -eq 0 ]; then
  echo "CI passed (${elapsed}s)"
else
  echo "CI failed (exit ${status}; see ${LOG})" >&2
  python3 "$ROOT_DIR/scripts/ci_log_excerpt.py" "$LOG" >&2 || true
fi
exit "$status"
