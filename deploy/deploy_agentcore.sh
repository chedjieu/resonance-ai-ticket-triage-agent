#!/usr/bin/env bash
# Deploy RTTA-AI-Multi-Agent-Ticket-Triage to AWS Bedrock AgentCore.
# Run from anywhere; the script cd's into the project root.

set -euo pipefail

export AGENTCORE_SUPPRESS_RECOMMENDATION="${AGENTCORE_SUPPRESS_RECOMMENDATION:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Windows: AgentCore uses Python shutil.which("zip") — needs zip.cmd/.bat on PATH.
export PATH="$ROOT/deploy/bin:$PATH"
chmod +x "$ROOT/deploy/bin/zip" 2>/dev/null || true
if [[ -d "$ROOT/.venv/Scripts" ]]; then
    cp -f "$ROOT/deploy/bin/zip.cmd" "$ROOT/.venv/Scripts/zip.cmd" 2>/dev/null || true
    cp -f "$ROOT/deploy/bin/zip.bat" "$ROOT/.venv/Scripts/zip.bat" 2>/dev/null || true
fi
if ! command -v zip >/dev/null 2>&1 && ! uv run python -c "import shutil; raise SystemExit(0 if shutil.which('zip') else 1)" 2>/dev/null; then
    echo "ERROR: zip shim not visible on PATH. Try: export PATH=\"$ROOT/deploy/bin:\$PATH\"" >&2
    exit 1
fi

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

# Agent names must be letters, numbers, underscores only (no hyphens).
NAME="${AGENT_NAME:-rtta_ticket_triage}"
# Root entrypoint avoids Windows backslash paths that break Linux ARM64 runtime.
ENTRYPOINT="${ENTRYPOINT:-agentcore_entrypoint.py}"
REGION="${AWS_REGION:-us-east-1}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-600}"
PYTHON_RUNTIME="${PYTHON_RUNTIME:-PYTHON_3_11}"
# Ticket triage uses Postgres memory; skip AgentCore managed memory by default.
DISABLE_MEMORY="${DISABLE_AGENTCORE_MEMORY:-1}"

agentcore_cmd() {
    PATH="$ROOT/deploy/bin:$ROOT/.venv/Scripts:$PATH" uv run agentcore "$@"
}

if ! agentcore_cmd --help >/dev/null 2>&1; then
    echo "agentcore CLI not found — installing bedrock-agentcore-starter-toolkit..."
    uv pip install bedrock-agentcore-starter-toolkit
fi

# Prefer direct_code_deploy (S3 zip) — avoids ECR permissions required by container deploy.
if [[ -z "${DEPLOYMENT_TYPE:-}" ]]; then
    if command -v zip >/dev/null 2>&1; then
        DEPLOYMENT_TYPE="direct_code_deploy"
    else
        DEPLOYMENT_TYPE="direct_code_deploy"
        echo "Note: using deploy/bin/zip shim for direct_code_deploy (no ECR required)."
    fi
fi

echo "Deploying $NAME from $ROOT"
echo "  region=$REGION  entrypoint=$ENTRYPOINT  deployment=$DEPLOYMENT_TYPE"
echo

if [[ "${SKIP_IAM_PREFLIGHT:-0}" != "1" ]]; then
    echo "0. IAM preflight"
    DEPLOYMENT_TYPE="$DEPLOYMENT_TYPE" DISABLE_AGENTCORE_MEMORY="$DISABLE_MEMORY" \
        "$ROOT/deploy/setup_agentcore_iam.sh" || exit 1
    echo
fi

need_reconfigure=0
if [[ -f .bedrock_agentcore.yaml ]]; then
    current_type="$(grep 'deployment_type:' .bedrock_agentcore.yaml | head -1 | awk '{print $2}')"
    if [[ -n "$current_type" && "$current_type" != "$DEPLOYMENT_TYPE" ]]; then
        echo "Switching deployment: $current_type → $DEPLOYMENT_TYPE"
        need_reconfigure=1
    fi
    # Force reconfigure when entrypoint still points at nested deploy/ path (Windows \\ bug).
    if grep -Eq 'deploy[/\\]+agentcore_entrypoint\.py' .bedrock_agentcore.yaml; then
        echo "Entrypoint is nested under deploy/ — moving to project-root agentcore_entrypoint.py"
        need_reconfigure=1
    fi
fi

if [[ "$need_reconfigure" == "1" ]]; then
    echo "Destroying previous agent config (required by AgentCore CLI)..."
    agentcore_cmd destroy --agent "$NAME" --force || true
    rm -f .bedrock_agentcore.yaml
    echo
fi

configure_args=(
    --name "$NAME"
    --entrypoint "$ENTRYPOINT"
    --deployment-type "$DEPLOYMENT_TYPE"
    --idle-timeout "$IDLE_TIMEOUT"
    --region "$REGION"
    --non-interactive
    --language python
)

if [[ "$DISABLE_MEMORY" == "1" ]]; then
    configure_args+=(--disable-memory)
fi

if [[ "$DEPLOYMENT_TYPE" == "direct_code_deploy" ]]; then
    configure_args+=(--runtime "$PYTHON_RUNTIME")
fi

# --create forces container deploy in the toolkit; never use it for direct_code_deploy.
if [[ ! -f .bedrock_agentcore.yaml ]] && [[ "$DEPLOYMENT_TYPE" == "container" ]]; then
    configure_args=(--create "${configure_args[@]}")
fi

echo "1. Configure"
agentcore_cmd configure "${configure_args[@]}"

deploy_args=(--agent "$NAME")
if [[ -n "${POSTGRES_DSN:-}" ]]; then
    deploy_args+=(--env "POSTGRES_DSN=$POSTGRES_DSN")
fi
if [[ -n "${RTTA_MODEL:-}" ]]; then
    deploy_args+=(--env "RTTA_MODEL=$RTTA_MODEL")
fi
if [[ -n "${RTTA_EMBEDDINGS:-}" ]]; then
    deploy_args+=(--env "RTTA_EMBEDDINGS=$RTTA_EMBEDDINGS")
fi
if [[ -n "${AWS_REGION:-}" ]]; then
    deploy_args+=(--env "AWS_REGION=$AWS_REGION")
fi

echo "2. Deploy"
agentcore_cmd deploy "${deploy_args[@]}"

echo "Done. Check status with: uv run agentcore status --agent $NAME"
