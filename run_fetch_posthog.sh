#!/usr/bin/env bash
set -euo pipefail

# Runs the PostHog GitHub fetcher.
#
# Required:
#   - Export `GITHUB_TOKEN` in your environment.
#
# Optional:
#   - Override lookback window with `DAYS`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${DAYS:=90}"
: "${REPO_URL:=https://github.com/PostHog/posthog}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Missing GITHUB_TOKEN env var." >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/posthog_github_fetcher.py" \
  --repo-url "${REPO_URL}" \
  --days "${DAYS}"

