#!/usr/bin/env bash
# Deploy RTTA-AI-Multi-Agent-Ticket-Triage to GCP Vertex AI Agent Engine.
# Requires: GCP_PROJECT, GCP_BUCKET (and optionally GCP_LOCATION).
# Saves the resource name to .env.deployed

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }

# Load .env (local first, then shared RAIRA-AI-Research-Assistant/.env)
if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
elif [[ -f "$ROOT/../RAIRA-AI-Research-Assistant/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/../RAIRA-AI-Research-Assistant/.env"
    set +a
fi

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCP_BUCKET:?set GCP_BUCKET}"
GCP_LOCATION="${GCP_LOCATION:-us-central1}"

bold "Resonance Technologies - deploy to Vertex AI Agent Engine"
echo "  project=${GCP_PROJECT}"
echo "  location=${GCP_LOCATION}"
echo "  bucket=${GCP_BUCKET}"
echo

OUT_LOG="$(mktemp)"
trap 'rm -f "$OUT_LOG"' EXIT

uv run python deploy/vertex_engine_deploy.py 2>&1 | tee "$OUT_LOG"

RESOURCE="$(grep -E '^Deployed:' "$OUT_LOG" | tail -n1 | sed 's/^Deployed:[[:space:]]*//')"
if [[ -z "$RESOURCE" ]]; then
    echo "ERROR: could not parse deployed resource name from deploy output" >&2
    exit 1
fi

{
    echo "VERTEX_ENGINE_RESOURCE_NAME=${RESOURCE}"
    echo "GCP_PROJECT=${GCP_PROJECT}"
    echo "GCP_LOCATION=${GCP_LOCATION}"
    echo "GCP_BUCKET=${GCP_BUCKET}"
} > "$ROOT/.env.deployed"

bold "Done."
echo "Resource name saved to .env.deployed"
echo "  VERTEX_ENGINE_RESOURCE_NAME=${RESOURCE}"
