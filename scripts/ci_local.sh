#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_FILE="${AGENT_CI_POLICY_FILE:-.agent-ci-policy.yml}"
COMMAND_INVENTORY="${AGENT_CI_COMMAND_INVENTORY:-.localci/product-commands.json}"
cd "$ROOT_DIR"

python3 scripts/validate_policy.py
if [ -f "$COMMAND_INVENTORY" ]; then
  echo "act command coverage:"
  python3 scripts/check_act_command_coverage.py "$COMMAND_INVENTORY"
fi

command_for() {
  local name="$1"
  local command
  command="$(sed -nE "s/^  ${name}:[[:space:]]+(.+)$/\1/p" "$POLICY_FILE" | head -n 1)"
  if [ -z "$command" ] || [ "$command" = "null" ]; then
    return 1
  fi
  printf '%s' "$command"
}

run_product_command() {
  local name="$1"
  local command
  if ! command="$(command_for "$name")"; then
    echo "product ${name}: not configured"
    return 1
  fi
  echo "product ${name}: running"
  bash -o pipefail -c "$command"
  echo "product ${name}: passed"
}

if command_for install >/dev/null; then
  run_product_command install
fi
if command_for full_ci >/dev/null; then
  run_product_command full_ci
else
  for command_name in test typecheck build; do
    if command_for "$command_name" >/dev/null; then
      run_product_command "$command_name"
    fi
  done
fi
