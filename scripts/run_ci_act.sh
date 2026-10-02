#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_VERSION="v0.2.89"
PLATFORM_IMAGE="${AGENT_CI_ACT_PLATFORM_IMAGE:-catthehacker/ubuntu:act-24.04}"
cd "$ROOT_DIR"
command -v act >/dev/null 2>&1 || { echo "act ${EXPECTED_VERSION} is required" >&2; exit 127; }
act --version | grep -Eq "(^|[^0-9])${EXPECTED_VERSION#v}([^0-9]|$)" || { echo "unsupported act version" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "a running Docker daemon is required" >&2; exit 2; }
exec act workflow_dispatch \
  --workflows "$ROOT_DIR/.github/workflows/pre-merge.yml" \
  --platform "ubuntu-24.04=${PLATFORM_IMAGE}" \
  --container-daemon-socket=- \
  --env "AGENT_CI_COMMAND_INVENTORY=${AGENT_CI_COMMAND_INVENTORY:-.localci/product-commands.json}" \
  --input local_ci_passed=true
